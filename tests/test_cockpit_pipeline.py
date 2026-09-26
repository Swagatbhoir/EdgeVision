"""
tests/test_cockpit_pipeline.py — End-to-end test of the production ADAS pipeline.

Drives the real ADASPipeline (the exact class main.py runs) over synthetic
frames and detections, then asserts that the annotated frame and the telemetry
payload handed to the cockpit are internally consistent.

main.py itself cannot be imported (it opens a camera and loads YOLO at module
scope), which is precisely why the frame body lives in pipeline.py.
"""

import os
import sys

os.environ["EDGEVISION_NO_CV2"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pipeline import ADASPipeline, is_in_ahead_zone
from web_server import TelemetryHub

W, H = 640, 480
DT = 1.0 / 15.0


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError("FAILED: {0} {1}".format(name, detail))
    print("  PASS  {0} {1}".format(name, detail).rstrip())


def car(cx, cy, w, h, track_id=1, cname="car", conf=0.9):
    return {
        "track_id": track_id,
        "class_name": cname,
        "confidence": conf,
        "x1": int(cx - w / 2), "y1": int(cy - h / 2),
        "x2": int(cx + w / 2), "y2": int(cy + h / 2),
    }


def blank_frame():
    return np.zeros((H, W, 3), np.uint8)


def synthetic_lanes(offset_px=0):
    return {
        "left_line": (120 + offset_px, 456, 240 + int(offset_px * 0.6), 278),
        "right_line": (520 + offset_px, 456, 400 + int(offset_px * 0.6), 278),
    }


def make_pipeline(offset_px=0):
    """
    Production ADASPipeline with a deterministic lane source.

    A black synthetic frame has no Hough lines, so the lane engine is fed
    known line endpoints through the pipeline's documented lane_source seam.
    Everything else on this path is exactly what main.py runs.
    """
    p = ADASPipeline(W, H)
    p.lane_source = lambda now: synthetic_lanes(offset_px)
    return p


# ---------------------------------------------------------------------------
def test_ahead_zone_geometry():
    check("centre corridor is in-path", is_in_ahead_zone(320, 300, W, H))
    check("far-left object is off-path", not is_in_ahead_zone(20, 300, W, H))
    check("far-right object is off-path", not is_in_ahead_zone(620, 300, W, H))
    check("sky-level object is off-path", not is_in_ahead_zone(320, 100, W, H))


# ---------------------------------------------------------------------------
def test_calm_cruise_produces_clean_telemetry():
    """A clear road must yield a NORMAL alert and a renderable payload."""
    p = make_pipeline()
    t = 0.0
    tel = None
    for _ in range(12):
        tel = p.process_frame(blank_frame(), [car(320, 300, 50, 40)],
                              inference_ms=17.0, now=t)
        t += DT

    check("calm cruise yields no active alert", tel["active_alert"] is None)
    check("FCW reports SAFE", tel["fcw"]["level"] == "SAFE")
    check("telemetry has a lead track", tel["fcw"]["lead_track_id"] == 1)
    check("object list is populated", len(tel["objects"]) == 1)
    check("object carries an IPM distance", tel["objects"][0]["distance_m"] is not None)
    check("object carries an IPM lateral offset",
          tel["objects"][0]["lateral_offset_m"] is not None)
    check("frame size is reported", tel["frame_size"] == [W, H])
    check("track count matches detections", tel["track_count"] == 1)
    check("inference time is echoed", tel["inference_ms"] == 17.0)
    check("HUD render time is measured", tel["hud_render_ms"] is not None)
    check("pipeline time is measured", tel["pipeline_ms"] > 0)
    check("fps is computed", tel["fps"] is not None and tel["fps"] > 0)
    check("lane state is reported", tel["ldw"]["departure_direction"] == "NORMAL")
    check("HUD layer state is reported", len(tel["hud_layers"]) == 4)


# ---------------------------------------------------------------------------
def test_frame_is_painted_in_place():
    """process_frame must annotate the caller's frame, not a private copy."""
    p = make_pipeline()
    frame = blank_frame()
    tel = p.process_frame(frame, [car(320, 300, 120, 100)],
                          inference_ms=17.0, now=0.0)

    check("process_frame returns telemetry", isinstance(tel, dict))
    check("frame shape preserved", frame.shape == (H, W, 3))
    check("frame dtype preserved", frame.dtype == np.uint8)


# ---------------------------------------------------------------------------
def test_rapid_approach_escalates_to_danger():
    """A closing lead vehicle must escalate SAFE -> DANGER and stay there."""
    p = make_pipeline()
    t = 0.0
    levels = []
    size = 60

    for _ in range(70):
        tel = p.process_frame(blank_frame(), [car(320, 400 - size * 0.55, size, size)],
                              inference_ms=17.0, now=t)
        levels.append(tel["fcw"]["level"])
        size += 6
        t += DT

    check("escalates to DANGER", tel["fcw"]["level"] == "DANGER")
    check("stays DANGER once reached", levels[-1] == "DANGER")

    order = ["SAFE", "CAUTION", "WARNING", "DANGER"]
    seen = [lvl for lvl in order if lvl in levels]
    check("levels escalate without skipping", seen == sorted(seen, key=order.index))

    check("DANGER alert wins arbitration", tel["active_alert"]["system"] == "FCW")
    check("DANGER alert says BRAKE NOW", tel["active_alert"]["subtext"] == "BRAKE NOW")
    check("DANGER object is flagged in the list",
          any(o["risk"] == "DANGER" for o in tel["objects"]))
    check("DANGER alert carries the CRITICAL sound pattern",
          tel["active_alert"]["sound_pattern"] == "CRITICAL")


# ---------------------------------------------------------------------------
def test_ldw_drift_surfaces_in_telemetry():
    """Lane drift must reach both the alert arbiter and the LDW panel."""
    p = make_pipeline(offset_px=150)
    t = 0.0
    tel = None

    for _ in range(14):
        tel = p.process_frame(blank_frame(), [], inference_ms=17.0, now=t)
        t += DT

    check("LDW reports a left drift", tel["ldw"]["departure_direction"] == "DRIFTING LEFT")
    check("LDW status is WARNING_LEFT", tel["ldw"]["status"] == "WARNING_LEFT")
    check("LDW alert wins arbitration", tel["active_alert"]["system"] == "LDW")
    check("LDW message matches the contract",
          tel["active_alert"]["message"] == "LANE DEPARTURE WARNING")
    check("LDW offset is negative for a left drift", tel["ldw"]["offset"] < 0)
    check("LDW alert uses the ADVISORY sound pattern",
          tel["active_alert"]["sound_pattern"] == "ADVISORY")

    p.lane_source = lambda now: synthetic_lanes(0)
    for _ in range(16):
        tel = p.process_frame(blank_frame(), [], inference_ms=17.0, now=t)
        t += DT

    check("LDW clears after recovery", tel["ldw"]["departure_direction"] == "NORMAL")
    check("no alert after recovery", tel["active_alert"] is None)


# ---------------------------------------------------------------------------
def test_fcdw_departure_surfaces_in_telemetry():
    """A stopped lead vehicle that pulls away must raise FRONT VEHICLE MOVING."""
    p = make_pipeline()
    t = 0.0
    tel = None

    for _ in range(45):
        tel = p.process_frame(blank_frame(), [car(320, 360, 90, 70)],
                              inference_ms=17.0, now=t)
        t += DT

    check("FCDW tracks a stopped lead", tel["fcdw"]["state"] == "TRACKING_STOPPED")
    check("stopped timer accumulates", tel["fcdw"]["stopped_duration"] > 1.0)
    check("no alert while still stopped", not tel["fcdw"]["alert_active"])

    for _ in range(18):
        tel = p.process_frame(blank_frame(), [car(320, 300, 70, 54)],
                              inference_ms=17.0, now=t)
        t += DT

    check("FCDW raises the departure alert", tel["fcdw"]["alert_active"])
    check("FCDW message matches the contract",
          tel["fcdw"]["alert_message"] == "FRONT VEHICLE MOVING")
    check("FCDW alert reaches the arbiter", tel["active_alert"]["system"] == "FCDW")
    check("FCDW alert subtext is the advisory",
          tel["active_alert"]["subtext"] == "PROCEED WITH CAUTION")
    check("FCDW alert uses the CHIME sound pattern",
          tel["active_alert"]["sound_pattern"] == "CHIME")


# ---------------------------------------------------------------------------
def test_alert_priority_ordering_end_to_end():
    """FCW DANGER must outrank every other simultaneous alert."""
    p = make_pipeline(offset_px=150)
    t = 0.0
    size = 60
    tel = None

    for _ in range(70):
        tel = p.process_frame(blank_frame(), [car(320, 400 - size * 0.55, size, size)],
                              inference_ms=17.0, now=t)
        size += 6
        t += DT

    check("FCW DANGER outranks a simultaneous LDW drift",
          tel["active_alert"]["system"] == "FCW")
    check("the active alert is the DANGER one", tel["active_alert"]["level"] == "DANGER")
    check("LDW is still reported underneath",
          tel["ldw"]["departure_direction"] != "NORMAL")


# ---------------------------------------------------------------------------
def test_subsystem_toggles_from_the_dashboard():
    """Disabling a subsystem in the settings drawer must silence it end to end."""
    p = make_pipeline()
    p.apply_settings({"fcw_enabled": False})
    t = 0.0
    size = 60
    tel = None

    for _ in range(40):
        tel = p.process_frame(blank_frame(), [car(320, 400 - size * 0.55, size, size)],
                              inference_ms=17.0, now=t)
        size += 6
        t += DT

    check("disabled FCW stays SAFE", tel["fcw"]["level"] == "SAFE")
    check("disabled FCW produces no alert", tel["active_alert"] is None)
    check("disabled FCW is flagged in telemetry", tel["fcw"]["enabled"] is False)
    check("objects are still tracked for display", len(tel["objects"]) == 1)


# ---------------------------------------------------------------------------
def test_camera_recalibration_changes_ipm_distance():
    """IPM ground distance is proportional to camera mounting height."""
    p = make_pipeline()
    det = [car(320, 380, 120, 110)]
    p.process_frame(blank_frame(), det, inference_ms=17.0, now=0.0)

    dist_low, _ = p.fcw.ipm.estimate_distance(260, 325, 380, 435)
    h_low = p.fcw.ipm.cam_height

    p.apply_settings({"cam_height_m": 2.0})
    dist_high, _ = p.fcw.ipm.estimate_distance(260, 325, 380, 435)

    check("IPM returns a metric distance", dist_low is not None)
    check("IPM distance is physically plausible", 1.0 < dist_low < 60.0,
          "{0:.2f}m".format(dist_low))
    check("camera height reaches the IPM", p.fcw.ipm.cam_height == 2.0)
    check("pitch angle is recalculated",
          abs(p.fcw.ipm.pitch_rad - np.radians(p.fcw.ipm.pitch_deg)) < 1e-9)

    # Pinhole ground plane: Z = f * h / (y - cy), so Z scales with mount height.
    expected = dist_low * (2.0 / h_low)
    check("distance scales with camera height", abs(dist_high - expected) < 0.05,
          "{0:.2f}m -> {1:.2f}m (expected {2:.2f}m)".format(dist_low, dist_high, expected))


# ---------------------------------------------------------------------------
def test_hud_layer_toggle_reaches_the_hud():
    """Turning off HUD layers from the dashboard must be reflected in telemetry."""
    p = make_pipeline()
    p.set_hud_layers({"radar": False, "bboxes": False})

    tel = p.process_frame(blank_frame(), [car(320, 320, 80, 60)],
                          inference_ms=17.0, now=0.0)
    check("radar toggle is reflected in telemetry", tel["hud_layers"]["radar"] is False)
    check("bbox toggle is reflected in telemetry", tel["hud_layers"]["bboxes"] is False)
    check("other layers stay enabled", tel["hud_layers"]["lanes"] is True)


# ---------------------------------------------------------------------------
def test_object_table_rows_are_cockpit_renderable():
    """Every object row must carry the fields the dashboard table binds to."""
    p = make_pipeline()
    t = 0.0
    tel = None

    for _ in range(12):
        tel = p.process_frame(blank_frame(),
                              [car(320, 320, 80, 60, 1, "car"),
                               car(90, 340, 70, 50, 2, "person"),
                               car(560, 350, 60, 45, 3, "truck")],
                              inference_ms=17.0, now=t)
        t += DT

    for obj in tel["objects"]:
        for key in ("track_id", "name", "risk", "distance_m", "lateral_offset_m",
                    "ttc", "in_ahead", "in_path", "box", "confidence"):
            check("object row carries '{0}'".format(key), key in obj)
        check("object box has 4 coordinates", len(obj["box"]) == 4)

    check("all three objects are tracked", len(tel["objects"]) == 3)
    check("only the central corridor object is in-path",
          len([o for o in tel["objects"] if o["in_path"]]) == 1)


# ---------------------------------------------------------------------------
def test_telemetry_is_publishable_to_the_cockpit():
    """The production payload must round-trip through the web telemetry hub."""
    import json

    p = make_pipeline()
    hub = TelemetryHub()

    t = 0.0
    for _ in range(12):
        frame = blank_frame()
        tel = p.process_frame(frame, [car(320, 300, 50, 40)],
                              inference_ms=17.0, now=t)
        hub.publish_frame(frame)
        hub.publish_telemetry(tel)
        t += DT

    served = hub.get_telemetry()
    check("payload is JSON serialisable", json.loads(json.dumps(served)) is not None)
    check("hub serves the same FCW level", served["fcw"]["level"] == tel["fcw"]["level"])
    check("hub serves the track count", served["track_count"] == 1)
    check("hub holds the annotated frame", hub.get_frame() is not None)
    check("hub frame has the right shape", hub.get_frame().shape == (H, W, 3))


if __name__ == "__main__":
    print("=" * 60)
    print(" test_cockpit_pipeline  —  production ADASPipeline frame contract")
    print("=" * 60)

    test_ahead_zone_geometry()
    test_calm_cruise_produces_clean_telemetry()
    test_frame_is_painted_in_place()
    test_rapid_approach_escalates_to_danger()
    test_ldw_drift_surfaces_in_telemetry()
    test_fcdw_departure_surfaces_in_telemetry()
    test_alert_priority_ordering_end_to_end()
    test_subsystem_toggles_from_the_dashboard()
    test_camera_recalibration_changes_ipm_distance()
    test_hud_layer_toggle_reaches_the_hud()
    test_object_table_rows_are_cockpit_renderable()
    test_telemetry_is_publishable_to_the_cockpit()

    print("-" * 60)
    print("ALL COCKPIT PIPELINE TESTS PASSED")
