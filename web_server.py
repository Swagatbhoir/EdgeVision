"""
web_server.py — EdgeVision ADAS Cockpit Web Server & Telemetry API.

Owns the Flask application, the shared thread-safe telemetry hub, the MJPEG
video stream, the Server-Sent Events (SSE) telemetry feed, the live settings
API, and the ADAS scenario simulator used for UI/UX verification.

This module is intentionally free of camera / YOLO dependencies so the
web layer can be unit tested offline (see tests/test_web_api.py).
"""

import json
import os
import threading
import time
from collections import deque

from flask import Flask, Response, jsonify, render_template, request

# OpenCV is loaded lazily (native DLLs) so the telemetry API stays testable in
# headless/offline environments. The live pipeline always has it available.
# Set EDGEVISION_NO_CV2=1 to force the headless path explicitly.
_CV2 = None

# Smallest valid JPEG, used as the stream keep-alive before the first camera
# frame arrives so the browser <img> tag never flips into a broken state.
BLANK_JPEG = (
    b"\xff\xd8\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.'\")"
    b",(3,#\x12\x13\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b"
    b"\x17\x1b\x1c\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b"
    b"\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x01\x07\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x09\xff"
    b"\xda\x00\x08\x01\x01\x00\x00?\x00\xe3B\x80\xff\xd9"
)


def _cv2():
    global _CV2
    if _CV2 is None:
        if os.environ.get("EDGEVISION_NO_CV2", "") == "1":
            _CV2 = False
            return None
        try:
            import cv2 as _mod
            _CV2 = _mod
        except Exception:
            _CV2 = False
    return _CV2 if _CV2 is not False else None



# ---------------------------------------------------------------------------
# Default telemetry snapshot
# ---------------------------------------------------------------------------

def default_telemetry():
    """A fully-populated NORMAL telemetry payload (also the offline test fixture)."""
    return {
        "timestamp": time.time(),
        "fps": None,
        "inference_ms": None,
        "pipeline_ms": None,
        "hud_render_ms": None,
        "frame_size": [0, 0],
        "track_count": 0,
        "camera_source": "CAM 0",
        "active_alert": None,
        "fcw": {
            "level": "SAFE",
            "distance_m": None,
            "lateral_offset_m": None,
            "ttc": None,
            "lead_class": None,
            "lead_track_id": None,
            "enabled": True,
        },
        "ldw": {
            "departure_direction": "NORMAL",
            "offset": 0.0,
            "status": "NORMAL",
            "lanes_detected": False,
            "alert_message": "",
            "enabled": True,
        },
        "fcdw": {
            "state": "IDLE",
            "alert_active": False,
            "alert_message": "",
            "stopped_duration": 0.0,
            "lead_track_id": None,
            "enabled": True,
        },
        "objects": [],
        "hud_layers": {
            "bboxes": True,
            "lanes": True,
            "radar": True,
            "telemetry": True,
        },
    }


# ---------------------------------------------------------------------------
# Telemetry Hub — thread-safe bridge between pipeline and web clients
# ---------------------------------------------------------------------------

class TelemetryHub:
    """
    Single source of truth for everything the cockpit UI renders.

    The ADAS pipeline calls publish_frame() / publish_telemetry() from the
    camera thread; Flask request handlers and SSE clients only ever read.
    """

    ALERT_HISTORY_MAX = 60
    SIM_DURATION_SECONDS = 8.0

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._telemetry = default_telemetry()
        self._settings = {
            "fcw_enabled": True,
            "ldw_enabled": True,
            "fcdw_enabled": True,
            "cam_height_m": 1.25,
            "cam_pitch_deg": 5.0,
            "ttc_danger_threshold": 1.8,
        }
        self._alert_history = deque(maxlen=self.ALERT_HISTORY_MAX)
        # Sentinel so the very first NORMAL frame still records "engine up",
        # while every subsequent identical state stays quiet.
        self._last_alert_key = "__INIT__"
        self._sim_override = None  # {"payload": dict, "expires_at": float}
        self._sim_lock = threading.Lock()

    # -- writers (pipeline thread) -----------------------------------------

    def publish_frame(self, frame):
        with self._lock:
            self._frame = frame

    def publish_telemetry(self, telemetry):
        """Store a telemetry snapshot and auto-log alert transitions."""
        with self._lock:
            self._telemetry = telemetry

        self._log_alert_transition(telemetry.get("active_alert"))

    def _log_alert_transition(self, alert):
        """Append to the event log whenever the visible alert state changes."""
        if not alert:
            key = None
            label = {"system": "SYSTEM", "level": "NORMAL",
                     "message": "ALL SYSTEMS NORMAL", "subtext": ""}
        else:
            key = (alert.get("system"), alert.get("level"))
            label = alert

        if key == self._last_alert_key:
            return

        self._last_alert_key = key

        with self._lock:
            self._alert_history.append({
                "timestamp": time.time(),
                "system": label.get("system", "SYSTEM"),
                "level": label.get("level", "NORMAL"),
                "message": label.get("message", ""),
                "subtext": label.get("subtext", ""),
            })

    # -- readers (web threads) --------------------------------------------

    def get_frame(self):
        with self._lock:
            return self._frame

    def get_telemetry(self):
        with self._lock:
            payload = json.loads(json.dumps(self._telemetry))  # defensive copy

        # Merge HUD layer toggles so the UI always reflects server-side state
        payload["hud_layers"] = self.get_hud_layers()
        payload["settings"] = self.get_settings()

        override = self._active_sim_override()
        if override is not None:
            payload.update(override)
            payload["simulated"] = True

        return payload

    def get_settings(self):
        with self._lock:
            return dict(self._settings)

    def update_settings(self, updates):
        allowed = set(self._settings.keys())
        with self._lock:
            for key, value in updates.items():
                if key in allowed:
                    self._settings[key] = value
            return dict(self._settings)

    def get_hud_layers(self):
        return dict(self._telemetry.get("hud_layers", {}))

    def set_hud_layer(self, name, enabled):
        with self._lock:
            self._telemetry.setdefault("hud_layers", {})[name] = bool(enabled)

    def get_alert_history(self):
        with self._lock:
            return list(self._alert_history)

    # -- scenario simulator ------------------------------------------------

    def _active_sim_override(self):
        with self._sim_lock:
            if self._sim_override and time.time() < self._sim_override["expires_at"]:
                return self._sim_override["payload"]
            self._sim_override = None
        return None

    def inject_simulation(self, payload):
        with self._sim_lock:
            self._sim_override = {
                "payload": payload,
                "expires_at": time.time() + self.SIM_DURATION_SECONDS,
            }

    def clear_simulation(self):
        with self._sim_lock:
            self._sim_override = None


# ---------------------------------------------------------------------------
# Scenario simulator catalogue (shared by the API and the UI buttons)
# ---------------------------------------------------------------------------

SIM_SCENARIOS = {
    "fcw-danger": {
        "active_alert": {
            "system": "FCW", "level": "DANGER", "message": "FORWARD COLLISION WARNING",
            "subtext": "BRAKE NOW", "color_bgr": [68, 68, 239], "sound_pattern": "CRITICAL",
        },
        "fcw": {"level": "DANGER", "distance_m": 6.5, "lateral_offset_m": 0.1, "ttc": 1.2},
        "ldw": {"departure_direction": "NORMAL", "offset": 0.0, "status": "NORMAL"},
        "fcdw": {"state": "IDLE", "alert_active": False, "stopped_duration": 0.0},
        "objects": [{
            "track_id": 1, "name": "car", "risk": "DANGER", "distance_m": 6.5,
            "lateral_offset_m": 0.1, "ttc": 1.2, "in_ahead": True,
        }],
    },
    "fcw-warning": {
        "active_alert": {
            "system": "FCW", "level": "WARNING", "message": "FORWARD COLLISION WARNING",
            "subtext": "SLOW DOWN", "color_bgr": [22, 115, 249], "sound_pattern": "WARNING",
        },
        "fcw": {"level": "WARNING", "distance_m": 14.2, "lateral_offset_m": 0.0, "ttc": 2.8},
        "ldw": {"departure_direction": "NORMAL", "offset": 0.0, "status": "NORMAL"},
        "fcdw": {"state": "IDLE", "alert_active": False, "stopped_duration": 0.0},
        "objects": [{
            "track_id": 2, "name": "truck", "risk": "WARNING", "distance_m": 14.2,
            "lateral_offset_m": 0.0, "ttc": 2.8, "in_ahead": True,
        }],
    },
    "ldw-left": {
        "active_alert": {
            "system": "LDW", "level": "WARNING", "message": "LANE DEPARTURE WARNING",
            "subtext": "DRIFTING LEFT", "color_bgr": [153, 72, 236], "sound_pattern": "ADVISORY",
        },
        "fcw": {"level": "SAFE", "distance_m": None, "lateral_offset_m": None, "ttc": None},
        "ldw": {"departure_direction": "DRIFTING LEFT", "offset": -0.75, "status": "WARNING_LEFT"},
        "fcdw": {"state": "IDLE", "alert_active": False, "stopped_duration": 0.0},
        "objects": [],
    },
    "fcdw-depart": {
        "active_alert": {
            "system": "FCDW", "level": "INFO", "message": "FRONT VEHICLE MOVING",
            "subtext": "PROCEED WITH CAUTION", "color_bgr": [212, 182, 6], "sound_pattern": "CHIME",
        },
        "fcw": {"level": "SAFE", "distance_m": 5.2, "lateral_offset_m": 0.0, "ttc": None},
        "ldw": {"departure_direction": "NORMAL", "offset": 0.0, "status": "NORMAL"},
        "fcdw": {"state": "DEPARTING", "alert_active": True, "stopped_duration": 12.0,
                 "alert_message": "FRONT VEHICLE MOVING"},
        "objects": [{
            "track_id": 3, "name": "car", "risk": "CAUTION", "distance_m": 5.2,
            "lateral_offset_m": 0.0, "ttc": None, "in_ahead": True,
        }],
    },
    "normal": None,
}


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------

def create_cockpit_app(hub=None, jpeg_quality=80):
    """Build the EdgeVision ADAS cockpit Flask application."""
    hub = hub if hub is not None else TelemetryHub()

    app = Flask(__name__)
    app.config["HUB"] = hub
    app.config["JPEG_QUALITY"] = jpeg_quality

    # -- Cockpit dashboard shell ------------------------------------------
    @app.route("/")
    def index():
        return render_template("dashboard.html")

    # -- MJPEG video stream ----------------------------------------------
    @app.route("/video_feed")
    def video_feed():
        def generate():
            cv2 = _cv2()
            boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            while True:
                frame = hub.get_frame()
                if frame is None or cv2 is None:
                    # No frame yet (or no encoder) — emit a tiny keep-alive
                    # JPEG so the browser <img> tag does not flip to a
                    # broken/errored state while the pipeline starts up.
                    yield boundary + BLANK_JPEG + b"\r\n"
                    time.sleep(0.1)
                    continue

                ok, encoded = cv2.imencode(
                    ".jpg", frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), app.config["JPEG_QUALITY"]],
                )
                if not ok:
                    continue
                yield boundary + encoded.tobytes() + b"\r\n"

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
        )

    # -- Telemetry snapshot ------------------------------------------------
    @app.route("/api/telemetry")
    def api_telemetry():
        return jsonify(hub.get_telemetry())

    # -- Server-Sent Events telemetry stream ------------------------------
    @app.route("/api/stream/telemetry")
    def api_stream_telemetry():
        def events():
            last_sent = 0.0
            while True:
                payload = hub.get_telemetry()
                now = time.time()
                # Only push when the pipeline produced a new frame
                if payload.get("timestamp", 0) > last_sent:
                    last_sent = payload.get("timestamp", 0)
                    yield "data: {0}\n\n".format(json.dumps(payload))
                time.sleep(1.0 / 20.0)  # 20 Hz push cap

        return Response(
            events(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    # -- Live settings / calibration --------------------------------------
    @app.route("/api/settings", methods=["GET", "POST"])
    def api_settings():
        if request.method == "GET":
            return jsonify({"settings": hub.get_settings(), "hud_layers": hub.get_hud_layers()})

        body = request.get_json(silent=True) or {}
        settings_update = {k: v for k, v in body.items() if k in
                           ("fcw_enabled", "ldw_enabled", "fcdw_enabled",
                            "cam_height_m", "cam_pitch_deg", "ttc_danger_threshold")}
        settings = hub.update_settings(settings_update) if settings_update else hub.get_settings()

        for key in ("bboxes", "lanes", "radar", "telemetry"):
            field = "hud_{0}".format(key)
            if field in body:
                hub.set_hud_layer(key, body[field])

        return jsonify({
            "status": "ok",
            "settings": settings,
            "hud_layers": hub.get_hud_layers(),
        })

    # -- Scenario simulator ------------------------------------------------
    @app.route("/api/simulate_alert", methods=["POST"])
    def api_simulate_alert():
        body = request.get_json(silent=True) or {}
        scenario = body.get("scenario", "")

        if scenario == "normal":
            hub.clear_simulation()
            return jsonify({"status": "cleared"})

        if scenario not in SIM_SCENARIOS:
            return jsonify({
                "status": "error",
                "error": "unknown scenario",
                "available": sorted(SIM_SCENARIOS.keys()),
            }), 400

        hub.inject_simulation(SIM_SCENARIOS[scenario])
        return jsonify({"status": "ok", "scenario": scenario,
                        "expires_in": hub.SIM_DURATION_SECONDS})

    # -- Alert event history ----------------------------------------------
    @app.route("/api/alerts/history")
    def api_alert_history():
        return jsonify({"alerts": hub.get_alert_history()})

    return app
