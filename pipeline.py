"""
pipeline.py — EdgeVision ADAS per-frame processing pipeline.

Owns the four ADAS engines, the tactical HUD renderer, and the telemetry
snapshot, and turns one camera frame + its YOLO detections into:

  1. an annotated frame (painted in place, ready to display or stream)
  2. a telemetry dict (the exact contract the cockpit dashboard consumes)

main.py stays a thin driver: capture -> model.track() -> process_frame() ->
publish. Keeping the frame body here means the production path is importable
and therefore genuinely testable (see tests/test_cockpit_pipeline.py).
"""

import time

import numpy as np

from fcw import ForwardCollisionWarning
from fcdw import FrontCarDepartureWarning
from ldw import LaneDepartureWarning
from alert_manager import AlertManager
from hud_renderer import HUDRenderer
from web_server import default_telemetry


# Vehicle classes that can act as a lead / queue vehicle.
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

# Display-only "ahead zone" corridor used to decide what to surface as
# in-path in the HUD and the cockpit object table. FCW/FCDW use their own
# stricter filters inside fcw.py / fcdw.py.
AHEAD_ZONE_LEFT = 0.30
AHEAD_ZONE_RIGHT = 0.70
AHEAD_ZONE_MIN_Y = 0.40

# Rolling FPS window length
FPS_WINDOW = 30

# Default driver-facing alert cooldown (seconds)
ALERT_COOLDOWN = 2.0


def is_in_ahead_zone(center_x, y2, frame_w, frame_h):
    """
    True when the object sits in the centre corridor low in the frame,
    i.e. roughly "in our driving path" for display purposes.
    """
    in_x = (AHEAD_ZONE_LEFT * frame_w) <= center_x <= (AHEAD_ZONE_RIGHT * frame_w)
    in_y = y2 >= (AHEAD_ZONE_MIN_Y * frame_h)
    return in_x and in_y


class ADASPipeline:
    """Runs all ADAS subsystems and the cockpit HUD over one frame at a time."""

    def __init__(self, frame_width, frame_height, cam_height_m=1.25,
                 pitch_angle_deg=5.0, enable_lane_detection=True, ipm_config=None):
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.enable_lane_detection = enable_lane_detection

        # ADAS subsystems
        # ipm_config carries the measured 4-point bird's-eye calibration from
        # main.py (src_points, real_world_width_m/height_m, dst_size). Passing
        # None falls back to the module defaults in ipm_distance.py.
        self.fcw = ForwardCollisionWarning(
            frame_width, frame_height,
            cam_height_m=cam_height_m, pitch_angle_deg=pitch_angle_deg,
            ipm_config=ipm_config,
        )
        self.fcdw = FrontCarDepartureWarning(frame_width, frame_height)
        self.ldw = LaneDepartureWarning(frame_width, frame_height)
        self.alerts = AlertManager(alert_cooldown=ALERT_COOLDOWN)

        # Display
        self.hud = HUDRenderer(frame_width, frame_height)
        self.hud_layers = {"bboxes": True, "lanes": True, "radar": True, "telemetry": True}

        # Live driver settings (mirrors the cockpit settings API)
        self.settings = {
            "fcw_enabled": True,
            "ldw_enabled": True,
            "fcdw_enabled": True,
            "cam_height_m": cam_height_m,
            "cam_pitch_deg": pitch_angle_deg,
            "ttc_danger_threshold": 1.8,
        }

        # Performance state
        self.fps_window = []
        self.frame_count = 0
        self.inference_ms = 0.0
        self.pipeline_ms = 0.0
        self.hud_render_ms = 0.0

        # Per-object lane-detection cadence (1 = every frame)
        self.lane_process_every = 1

        # Seam for an alternative lane source. When set, it is called as
        # lane_source(now) and its return value is handed to
        # LaneDepartureWarning.update instead of the raw frame. Leave it as
        # None to feed frames straight through (the normal camera path).
        self.lane_source = None

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def apply_settings(self, settings):
        """Merge a settings dict from the cockpit API into the live state."""
        for key in ("fcw_enabled", "ldw_enabled", "fcdw_enabled"):
            if key in settings:
                self.settings[key] = bool(settings[key])

        geometry_changed = False
        for key in ("cam_height_m", "cam_pitch_deg"):
            if key in settings:
                value = float(settings[key])
                if value != self.settings[key]:
                    self.settings[key] = value
                    geometry_changed = True

        # Re-calibrate the IPM when the mounting geometry changes.
        if geometry_changed:
            self._recalibrate_ipm()

    def _recalibrate_ipm(self):
        """
        No longer a recalibration.

        The IPM used to be a pinhole model built from cam_height_m and
        cam_pitch_deg, so changing those settings in the cockpit drawer
        genuinely rebuilt the geometry. It is now a 4-point measured
        homography that neither value can influence, so this only warns once
        that the dashboard control is inert. The calibration itself is the
        IPM_SRC_POINTS / IPM_REAL_WORLD_* block at the top of main.py.
        """
        if not getattr(self, "_ipm_recalib_warned", False):
            self._ipm_recalib_warned = True
            print()
            print("NOTE: cam_height_m / cam_pitch_deg no longer affect the")
            print("      distance estimate. The IPM is now a 4-point measured")
            print("      homography calibrated in main.py (IPM_SRC_POINTS,")
            print("      IPM_REAL_WORLD_WIDTH_M, IPM_REAL_WORLD_HEIGHT_M).")
            print()

    def set_hud_layers(self, layers):
        for key in self.hud_layers:
            if key in layers:
                self.hud_layers[key] = bool(layers[key])

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame, detections, inference_ms, now, camera_source="CAM 0"):
        """
        Process one frame.

        frame:        BGR numpy image, painted in place with the tactical HUD
        detections:   list of dicts {track_id, class_name, confidence, x1..y2}
        inference_ms: YOLO inference time for this frame (float ms)
        now:          monotonic-ish wall clock timestamp for this frame

        Returns the telemetry dict describing the frame.
        """
        started = time.perf_counter()
        self.frame_count += 1
        self.inference_ms = float(inference_ms)
        h_frame, w_frame = frame.shape[:2]

        # ---- 1. Forward Collision Warning ---------------------------------
        if self.settings["fcw_enabled"]:
            self.fcw.update(detections, now)
        else:
            self.fcw.level = "SAFE"
            self.fcw.primary = None

        # ---- 2. Front Car Departure Warning -------------------------------
        if self.settings["fcdw_enabled"]:
            self.fcdw.update(detections, now)
        else:
            self.fcdw.alert_active = False
            self.fcdw.alert_message = ""
            self.fcdw.state = "IDLE"

        # ---- 3. Lane Departure Warning ------------------------------------
        if self.settings["ldw_enabled"] and self.enable_lane_detection:
            if self.frame_count % self.lane_process_every == 0:
                lane_input = self.lane_source(now) if self.lane_source else frame
                self.ldw.update(lane_input, now)
        else:
            self.ldw.lanes_detected = False

        ldw_direction = ("DRIFTING {0}".format(self.ldw.departure_direction)
                         if self.ldw.alert_active else "NORMAL")

        # ---- 4. Alert arbitration -----------------------------------------
        active_alert = self.alerts.update(
            fcw_level=self.fcw.level,
            fcdw_active=self.fcdw.alert_active,
            fcdw_message=self.fcdw.alert_message,
            ldw_active=self.ldw.alert_active,
            ldw_message=self.ldw.alert_message,
            ldw_dir=self.ldw.departure_direction,
            now=now,
        )

        # ---- 5. Build the display object list -----------------------------
        primary = self.fcw.primary
        lead_id = primary["track_id"] if primary else None

        dist_by_id, lat_by_id, ttc_by_id = {}, {}, {}
        if primary is not None:
            dist_by_id[primary["track_id"]] = primary.get("dist_m")
            lat_by_id[primary["track_id"]] = primary.get("lat_m")
            ttc_by_id[primary["track_id"]] = primary.get("tau")

        tracked_objects = self._build_display_objects(
            detections, lead_id, dist_by_id, lat_by_id, ttc_by_id, w_frame, h_frame)

        # ---- 6. Tactical HUD render ---------------------------------------
        lane_lines = None
        if self.ldw.left_line is not None and self.ldw.right_line is not None:
            lane_lines = (tuple(self.ldw.left_line), tuple(self.ldw.right_line))

        hud_started = time.perf_counter()
        self.hud.render(
            frame,
            tracked_objects=tracked_objects,
            lane_lines=lane_lines,
            ldw_direction=ldw_direction,
            active_alert=active_alert,
            fps=self.current_fps,
            inference_ms=self.inference_ms,
            lead_vehicle_id=lead_id,
            show_bboxes=self.hud_layers["bboxes"],
            show_lanes=self.hud_layers["lanes"],
            show_radar=self.hud_layers["radar"],
            show_telemetry=self.hud_layers["telemetry"],
            camera_source=camera_source,
            now=now,
        )
        self.hud_render_ms = (time.perf_counter() - hud_started) * 1000.0

        # ---- 7. Performance ----------------------------------------------
        self.pipeline_ms = (time.perf_counter() - started) * 1000.0
        if self.pipeline_ms > 0:
            self.fps_window.append(1000.0 / self.pipeline_ms)
            if len(self.fps_window) > FPS_WINDOW:
                self.fps_window.pop(0)

        # ---- 8. Telemetry snapshot ----------------------------------------
        return self._build_telemetry(
            frame=frame,
            active_alert=active_alert,
            tracked_objects=tracked_objects,
            ldw_direction=ldw_direction,
            camera_source=camera_source,
            now=now,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def current_fps(self):
        if not self.fps_window:
            return None
        return float(np.mean(self.fps_window))

    def _build_display_objects(self, detections, lead_id, dist_by_id, lat_by_id,
                               ttc_by_id, frame_w, frame_h):
        """Merge YOLO boxes with IPM metrics and FCW risk for the HUD + UI."""
        objects = []
        for det in detections:
            tid = det["track_id"]
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            in_ahead = is_in_ahead_zone((x1 + x2) // 2, y2, frame_w, frame_h)

            if tid in dist_by_id:
                dist_m = dist_by_id[tid]
                lat_m = lat_by_id.get(tid)
                tau = ttc_by_id.get(tid)
            else:
                # Not the FCW primary: still project it so the BEV radar and
                # the cockpit object table can show every nearby object.
                dist_m, lat_m = self.fcw.ipm.estimate_distance(x1, y1, x2, y2)
                tau = None

            is_lead = (lead_id is not None and tid == lead_id)
            if in_ahead and is_lead and self.fcw.level == "DANGER":
                risk = "DANGER"
            elif in_ahead and is_lead and self.fcw.level == "WARNING":
                risk = "WARNING"
            elif in_ahead and det["class_name"] in VEHICLE_CLASSES:
                risk = "CAUTION"
            else:
                risk = "SAFE"

            objects.append({
                "track_id": tid,
                "name": det["class_name"],
                "confidence": det["confidence"],
                "box": (x1, y1, x2, y2),
                "risk": risk,
                "distance_m": dist_m,
                "lateral_offset_m": lat_m,
                "ttc": tau,
                "in_ahead": in_ahead,
                "in_path": in_ahead,
            })
        return objects

    def _build_telemetry(self, frame, active_alert, tracked_objects,
                         ldw_direction, camera_source, now):
        primary = self.fcw.primary
        lead_id = primary["track_id"] if primary else None

        stopped_duration = 0.0
        if self.fcdw.state == "TRACKING_STOPPED" and self.fcdw.stopped_start_time is not None:
            stopped_duration = max(0.0, now - self.fcdw.stopped_start_time)

        tel = default_telemetry()
        tel.update({
            "timestamp": now,
            "fps": self.current_fps,
            "inference_ms": self.inference_ms,
            "pipeline_ms": self.pipeline_ms,
            "hud_render_ms": self.hud_render_ms,
            "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
            "track_count": len(tracked_objects),
            "camera_source": camera_source,
            "active_alert": active_alert,
            "fcw": {
                "level": self.fcw.level,
                "distance_m": primary.get("dist_m") if primary else None,
                "lateral_offset_m": primary.get("lat_m") if primary else None,
                "ttc": primary.get("tau") if primary else None,
                "lead_class": primary["class_name"] if primary else None,
                "lead_track_id": lead_id,
                "enabled": self.settings["fcw_enabled"],
            },
            "ldw": {
                "departure_direction": ldw_direction,
                "offset": round(self.ldw.lateral_offset, 3),
                "status": self.ldw.status,
                "lanes_detected": self.ldw.lanes_detected,
                "alert_message": self.ldw.alert_message,
                "enabled": self.settings["ldw_enabled"] and self.enable_lane_detection,
            },
            "fcdw": {
                "state": self.fcdw.state,
                "alert_active": self.fcdw.alert_active,
                "alert_message": self.fcdw.alert_message,
                "stopped_duration": stopped_duration,
                "lead_track_id": (self.fcdw.lead_info["track_id"]
                                  if self.fcdw.lead_info else None),
                "enabled": self.settings["fcdw_enabled"],
            },
            "objects": tracked_objects,
            "hud_layers": dict(self.hud_layers),
        })
        return tel
