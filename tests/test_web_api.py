"""
tests/test_web_api.py — Offline tests for the EdgeVision ADAS cockpit web layer.

Verifies the dashboard shell, the telemetry snapshot contract, the SSE stream,
live settings, the scenario simulator, and the alert event history. No camera
or YOLO is required: the Flask test client talks straight to a TelemetryHub.
"""

import json
import os
import sys

# The parent process always runs without an OpenCV import; the MJPEG encoder is
# exercised in a spawned child phase when a backend is available.
os.environ["EDGEVISION_NO_CV2"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STREAM_PHASE = os.environ.get("EDGEVISION_WEB_STREAM_PHASE") == "1"

from web_server import (  # noqa: E402
    TelemetryHub,
    create_cockpit_app,
    default_telemetry,
    SIM_SCENARIOS,
)


def check(name, cond):
    if not cond:
        raise AssertionError("FAILED: {0}".format(name))
    print("  PASS  {0}".format(name))


def make_client():
    hub = TelemetryHub()
    app = create_cockpit_app(hub)
    return hub, app.test_client()


def sample_telemetry():
    t = default_telemetry()
    t["fps"] = 27.4
    t["inference_ms"] = 17.9
    t["pipeline_ms"] = 34.2
    t["hud_render_ms"] = 1.8
    t["frame_size"] = [640, 480]
    t["track_count"] = 2
    t["active_alert"] = {
        "system": "FCW", "level": "WARNING",
        "message": "FORWARD COLLISION WARNING", "subtext": "SLOW DOWN",
        "color_bgr": [22, 115, 249], "sound_pattern": "WARNING",
    }
    t["fcw"]["level"] = "WARNING"
    t["fcw"]["distance_m"] = 14.2
    t["fcw"]["ttc"] = 2.8
    t["fcw"]["lateral_offset_m"] = 0.0
    t["fcw"]["lead_track_id"] = 2
    t["ldw"]["departure_direction"] = "NORMAL"
    t["ldw"]["offset"] = 0.05
    t["fcdw"]["state"] = "TRACKING_STOPPED"
    t["fcdw"]["stopped_duration"] = 7.5
    t["objects"] = [{
        "track_id": 2, "name": "truck", "confidence": 0.88,
        "box": [200, 260, 400, 430], "risk": "WARNING",
        "distance_m": 14.2, "lateral_offset_m": 0.0, "ttc": 2.8,
        "in_ahead": True, "in_path": True,
    }]
    return t


# ---------------------------------------------------------------------------
def test_default_telemetry_contract():
    """The NORMAL payload must carry every key the cockpit UI reads."""
    t = default_telemetry()
    for key in ("timestamp", "fps", "inference_ms", "pipeline_ms", "hud_render_ms",
                "frame_size", "track_count", "camera_source", "active_alert",
                "fcw", "ldw", "fcdw", "objects", "hud_layers"):
        check("default telemetry has '{0}'".format(key), key in t)

    check("default FCW level is SAFE", t["fcw"]["level"] == "SAFE")
    check("default LDW direction is NORMAL", t["ldw"]["departure_direction"] == "NORMAL")
    check("default FCDW state is IDLE", t["fcdw"]["state"] == "IDLE")
    check("default alert is None", t["active_alert"] is None)
    check("default object list is empty", t["objects"] == [])


# ---------------------------------------------------------------------------
def test_dashboard_shell_renders():
    _, c = make_client()
    resp = c.get("/")
    body = resp.get_data(as_text=True)

    check("dashboard returns 200", resp.status_code == 200)
    check("dashboard is HTML", resp.mimetype == "text/html")
    check("dashboard is the cockpit shell", "EDGEVISION" in body)
    check("dashboard mounts the live stream", "/video_feed" in body)
    check("dashboard loads the stylesheet", "/static/css/dashboard.css" in body)
    check("dashboard loads the script", "/static/js/dashboard.js" in body)
    check("dashboard has a viewport meta tag", 'name="viewport"' in body)

    # Core instrumentation widgets must all be present
    for widget_id in ("master-alert-banner", "ttc-dial-fill", "val-ttc", "val-dist",
                      "val-lat", "ldw-vehicle-marker", "fcdw-stopped-timer",
                      "objects-tbody", "chip-fps", "chip-latency"):
        check("dashboard exposes '{0}'".format(widget_id), 'id="{0}"'.format(widget_id) in body)

    # Scenario simulator entries
    for scenario in ("fcw-danger", "fcw-warning", "ldw-left", "fcdw-depart", "normal"):
        check("simulator exposes '{0}'".format(scenario),
              'data-sim="{0}"'.format(scenario) in body)


# ---------------------------------------------------------------------------
def test_static_assets_are_served():
    _, c = make_client()
    css = c.get("/static/css/dashboard.css")
    js = c.get("/static/js/dashboard.js")

    check("stylesheet returns 200", css.status_code == 200)
    check("script returns 200", js.status_code == 200)
    check("stylesheet uses the OLED dark background", "#080C14" in css.get_data(as_text=True))
    check("script drives a Web Audio engine", "AudioContext" in js.get_data(as_text=True))


# ---------------------------------------------------------------------------
def test_telemetry_endpoint():
    hub, c = make_client()
    hub.publish_telemetry(sample_telemetry())

    resp = c.get("/api/telemetry")
    body = resp.get_json()

    check("telemetry returns 200", resp.status_code == 200)
    check("telemetry carries fps", body["fps"] == 27.4)
    check("telemetry carries inference time", body["inference_ms"] == 17.9)
    check("telemetry carries HUD render time", body["hud_render_ms"] == 1.8)
    check("telemetry carries frame size", body["frame_size"] == [640, 480])
    check("telemetry carries the active alert",
          body["active_alert"]["message"] == "FORWARD COLLISION WARNING")
    check("telemetry carries FCW distance", body["fcw"]["distance_m"] == 14.2)
    check("telemetry carries FCW TTC", body["fcw"]["ttc"] == 2.8)
    check("telemetry carries LDW offset", body["ldw"]["offset"] == 0.05)
    check("telemetry carries FCDW stopped time", body["fcdw"]["stopped_duration"] == 7.5)
    check("telemetry carries the tracked object", len(body["objects"]) == 1)
    check("telemetry exposes settings for the UI", "settings" in body)
    check("telemetry exposes HUD layer state", "hud_layers" in body)


# ---------------------------------------------------------------------------
def test_telemetry_snapshot_is_defensive():
    """Callers must not be able to mutate the hub's internal state."""
    hub, c = make_client()
    hub.publish_telemetry(sample_telemetry())

    body = c.get("/api/telemetry").get_json()
    body["fcw"]["level"] = "TAMPERED"
    body["objects"].clear()

    fresh = c.get("/api/telemetry").get_json()
    check("mutating a response does not corrupt the hub", fresh["fcw"]["level"] == "WARNING")
    check("object list is not shared between responses", len(fresh["objects"]) == 1)


# ---------------------------------------------------------------------------
def test_settings_get_and_post():
    hub, c = make_client()

    resp = c.get("/api/settings")
    check("settings GET returns 200", resp.status_code == 200)
    check("defaults enable FCW", resp.get_json()["settings"]["fcw_enabled"] is True)

    resp = c.post("/api/settings", json={
        "fcw_enabled": False,
        "ldw_enabled": True,
        "fcdw_enabled": False,
        "cam_height_m": 1.45,
        "cam_pitch_deg": 7.5,
        "ttc_danger_threshold": 2.2,
    })
    body = resp.get_json()

    check("settings POST returns 200", resp.status_code == 200)
    check("settings POST reports ok", body["status"] == "ok")
    check("FCW disable is stored", hub.get_settings()["fcw_enabled"] is False)
    check("FCDW disable is stored", hub.get_settings()["fcdw_enabled"] is False)
    check("camera height is stored", hub.get_settings()["cam_height_m"] == 1.45)
    check("camera pitch is stored", hub.get_settings()["cam_pitch_deg"] == 7.5)
    check("TTC threshold is stored", hub.get_settings()["ttc_danger_threshold"] == 2.2)


# ---------------------------------------------------------------------------
def test_settings_rejects_unknown_fields():
    hub, c = make_client()
    before = dict(hub.get_settings())

    c.post("/api/settings", json={"evil_key": "pwned", "fcw_enabled": False})

    after = hub.get_settings()
    check("unknown fields are not persisted", "evil_key" not in after)
    check("known fields still apply", after["fcw_enabled"] is False)
    check("known defaults are untouched", after["cam_height_m"] == before["cam_height_m"])


# ---------------------------------------------------------------------------
def test_hud_layer_toggles():
    hub, c = make_client()

    c.post("/api/settings", json={"hud_radar": False, "hud_bboxes": False})
    layers = hub.get_hud_layers()
    check("radar layer can be turned off", layers["radar"] is False)
    check("bbox layer can be turned off", layers["bboxes"] is False)
    check("lane layer stays on", layers["lanes"] is True)

    body = c.get("/api/telemetry").get_json()
    check("layer state reaches the dashboard", body["hud_layers"]["radar"] is False)


# ---------------------------------------------------------------------------
def test_alert_history_logging():
    hub, c = make_client()

    calm = default_telemetry()
    hub.publish_telemetry(calm)
    hub.publish_telemetry(calm)  # repeated calm frames must not spam the log

    warn = sample_telemetry()
    hub.publish_telemetry(warn)
    hub.publish_telemetry(warn)  # same alert again — not a new event
    hub.publish_telemetry(calm)

    alerts = c.get("/api/alerts/history").get_json()["alerts"]
    systems = [a["system"] for a in alerts]

    check("history returns a list", isinstance(alerts, list))
    check("calm->warn->calm produces 3 events", len(alerts) == 3)
    check("first event is the system coming up", systems[0] == "SYSTEM")
    check("second event is the FCW warning", systems[1] == "FCW")
    check("warning message is recorded", alerts[1]["message"] == "FORWARD COLLISION WARNING")
    check("third event is the return to normal", systems[2] == "SYSTEM")
    check("normal event has NORMAL level", alerts[2]["level"] == "NORMAL")


# ---------------------------------------------------------------------------
def test_alert_history_is_bounded():
    hub, c = make_client()
    for i in range(hub.ALERT_HISTORY_MAX * 2):
        hub.publish_telemetry(default_telemetry())
        hub.publish_telemetry(sample_telemetry())
        hub.publish_telemetry(default_telemetry())

    alerts = c.get("/api/alerts/history").get_json()["alerts"]
    check("history is bounded", len(alerts) <= hub.ALERT_HISTORY_MAX)


# ---------------------------------------------------------------------------
def test_scenario_simulator():
    hub, c = make_client()

    resp = c.post("/api/simulate_alert", json={"scenario": "fcw-danger"})
    check("simulator accepts a known scenario", resp.status_code == 200)
    check("simulator reports the scenario", resp.get_json()["scenario"] == "fcw-danger")

    body = c.get("/api/telemetry").get_json()
    check("simulated alert reaches the UI", body["active_alert"]["level"] == "DANGER")
    check("simulated alert message is correct",
          body["active_alert"]["message"] == "FORWARD COLLISION WARNING")
    check("simulated alert is flagged", body.get("simulated") is True)
    check("simulated FCW TTC", body["fcw"]["ttc"] == 1.2)

    resp = c.post("/api/simulate_alert", json={"scenario": "nope"})
    check("unknown scenario is rejected", resp.status_code == 400)
    check("rejection lists available scenarios", "available" in resp.get_json())

    resp = c.post("/api/simulate_alert", json={"scenario": "normal"})
    check("normal scenario is accepted", resp.status_code == 200)
    body = c.get("/api/telemetry").get_json()
    check("normal clears the simulation", body.get("simulated") is None)


# ---------------------------------------------------------------------------
def test_all_simulated_scenarios_have_full_payloads():
    """Every button in the simulator drawer must produce a renderable payload."""
    for name, payload in SIM_SCENARIOS.items():
        if payload is None:
            check("'{0}' is the reset scenario".format(name), True)
            continue
        for key in ("active_alert", "fcw", "ldw", "fcdw", "objects"):
            check("scenario '{0}' provides '{1}'".format(name, key), key in payload)

        alert = payload["active_alert"]
        for key in ("system", "level", "message", "subtext", "sound_pattern"):
            check("scenario '{0}' alert has '{1}'".format(name, key), key in alert)

        # The active alert must be JSON-serialisable for the SSE stream
        check("scenario '{0}' is JSON serialisable".format(name),
              json.loads(json.dumps(payload)) is not None)


# ---------------------------------------------------------------------------
def test_alert_wording_matches_readme():
    """Driver-facing wording is part of the project contract (Idea.md §6)."""
    expected = {
        "fcw-danger": "FORWARD COLLISION WARNING",
        "fcw-warning": "FORWARD COLLISION WARNING",
        "ldw-left": "LANE DEPARTURE WARNING",
        "fcdw-depart": "FRONT VEHICLE MOVING",
    }
    for name, message in expected.items():
        check("scenario '{0}' says '{1}'".format(name, message),
              SIM_SCENARIOS[name]["active_alert"]["message"] == message)


# ---------------------------------------------------------------------------
def test_sse_stream_emits_json_events():
    """The SSE route must be a text/event-stream of `data: {...}` frames."""
    hub, c = make_client()
    hub.publish_telemetry(sample_telemetry())

    resp = c.get("/api/stream/telemetry", buffered=False)
    check("SSE returns 200", resp.status_code == 200)
    check("SSE is an event stream", resp.mimetype == "text/event-stream")

    iterator = resp.response.__iter__()
    chunk = next(iterator)
    text = chunk.decode("utf-8", "replace") if isinstance(chunk, bytes) else chunk

    check("SSE emits a data frame", text.startswith("data: "))
    payload = json.loads(text[len("data: "):].strip())
    check("SSE payload is valid JSON telemetry", payload["fcw"]["level"] == "WARNING")

    resp.close()


# ---------------------------------------------------------------------------
def test_hub_is_thread_safe_for_frame_publish():
    """Frame publication must never hand out a torn/None buffer."""
    import threading
    import numpy as np

    hub = TelemetryHub()
    errors = []

    def publisher(n):
        try:
            for i in range(50):
                hub.publish_frame(np.full((48, 64, 3), n, dtype=np.uint8))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def reader():
        try:
            for _ in range(200):
                frame = hub.get_frame()
                if frame is not None and frame.shape != (48, 64, 3):
                    errors.append("torn frame")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=publisher, args=(i,)) for i in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    check("concurrent publish/read raised no errors", not errors)


# ---------------------------------------------------------------------------
# Spawned child phase: real MJPEG encoding
# ---------------------------------------------------------------------------
def test_video_feed_stream_phase():
    """
    The MJPEG route needs a real JPEG encoder, which is probed out of process
    because OpenCV may not be importable in the parent at all.
    """
    import subprocess

    def cv2_importable():
        try:
            return subprocess.run([sys.executable, "-c", "import cv2"], timeout=180).returncode == 0
        except Exception:
            return False

    if not cv2_importable():
        print("  SKIP  MJPEG stream phase (no OpenCV backend in this environment)")
        return

    env = dict(os.environ)
    env["EDGEVISION_WEB_STREAM_PHASE"] = "1"
    env.pop("EDGEVISION_NO_CV2", None)

    res = subprocess.run([sys.executable, __file__], env=env, timeout=300)
    check("MJPEG stream phase passed", res.returncode == 0)


if __name__ == "__main__":
    if STREAM_PHASE:
        import numpy as np
        import web_server

        hub, c = make_client()
        hub.publish_frame(np.zeros((48, 64, 3), dtype=np.uint8))

        resp = c.get("/video_feed", buffered=False)
        check("video feed returns 200", resp.status_code == 200)
        check("video feed is multipart MJPEG",
              resp.mimetype == "multipart/x-mixed-replace")
        check("video feed is not cached", "no-cache" in resp.headers.get("Cache-Control", ""))

        iterator = resp.response.__iter__()
        chunk = next(iterator)
        check("video feed emits a JPEG multipart frame", b"--frame" in chunk)
        check("multipart frame declares image/jpeg", b"Content-Type: image/jpeg" in chunk)
        check("multipart frame carries JPEG SOI marker", b"\xff\xd8" in chunk)

        resp.close()
        print("-" * 60)
        print("ALL WEB STREAM TESTS PASSED")
        sys.exit(0)

    print("=" * 60)
    print(" test_web_api  —  cockpit dashboard, telemetry & simulator")
    print("=" * 60)

    test_default_telemetry_contract()
    test_dashboard_shell_renders()
    test_static_assets_are_served()
    test_telemetry_endpoint()
    test_telemetry_snapshot_is_defensive()
    test_settings_get_and_post()
    test_settings_rejects_unknown_fields()
    test_hud_layer_toggles()
    test_alert_history_logging()
    test_alert_history_is_bounded()
    test_scenario_simulator()
    test_all_simulated_scenarios_have_full_payloads()
    test_alert_wording_matches_readme()
    test_sse_stream_emits_json_events()
    test_hub_is_thread_safe_for_frame_publish()
    test_video_feed_stream_phase()

    print("-" * 60)
    print("ALL WEB API TESTS PASSED")
