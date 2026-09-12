"""
tests/test_integration_pipeline.py — End-to-end integration test for all 3 ADAS features.

Simulates end-to-end multi-subsystem driving scenarios without physical camera/YOLO.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fcw import ForwardCollisionWarning
from fcdw import FrontCarDepartureWarning
from ldw import LaneDepartureWarning
from alert_manager import AlertManager

W, H = 640, 480
FPS = 15.0
DT = 1.0 / FPS


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  PASS  {name}")


def synthetic_car(cx=320, cy=350, w=160, h=120, track_id=1, cname="car"):
    return {
        "track_id": track_id,
        "class_name": cname,
        "confidence": 0.9,
        "x1": int(cx - w / 2),
        "y1": int(cy - h / 2),
        "x2": int(cx + w / 2),
        "y2": int(cy + h / 2),
    }


def synthetic_lanes(offset_px=0):
    return {
        "left_line": (120 + offset_px, 456, 240 + int(offset_px * 0.6), 278),
        "right_line": (520 + offset_px, 456, 400 + int(offset_px * 0.6), 278),
    }


def test_scenario_full_driving_mission():
    """Simulates a complete driving mission with various ADAS scenarios."""
    fcw_engine = ForwardCollisionWarning(W, H)
    fcdw_engine = FrontCarDepartureWarning(W, H)
    ldw_engine = LaneDepartureWarning(W, H)
    alert_mgr = AlertManager()

    t = 0.0

    # -------------------------------------------------------------
    # Scenario 1: Calm Highway Driving (Centered, Safe Distance)
    # -------------------------------------------------------------
    for _ in range(15):
        ldw_engine.update(synthetic_lanes(offset_px=0), t)
        fcw_engine.update([synthetic_car(cx=320, cy=300, w=50, h=40)], t) # small box far away
        fcdw_engine.update([], t)

        alert = alert_mgr.update(
            fcw_level=fcw_engine.level,
            fcdw_active=fcdw_engine.alert_active,
            fcdw_message=fcdw_engine.alert_message,
            ldw_active=ldw_engine.alert_active,
            ldw_message=ldw_engine.alert_message,
            ldw_dir=ldw_engine.departure_direction,
            now=t,
        )
        t += DT

    check("Scenario 1: Calm driving produces NORMAL state", alert is None or alert_mgr.active_system == "NORMAL")

    # -------------------------------------------------------------
    # Scenario 2: Unintentional Lane Departure (Drift Left)
    # -------------------------------------------------------------
    for _ in range(12):
        ldw_engine.update(synthetic_lanes(offset_px=150), t) # drift left -> offset ~ -0.75
        fcw_engine.update([], t)
        fcdw_engine.update([], t)

        alert = alert_mgr.update(
            fcw_level=fcw_engine.level,
            fcdw_active=fcdw_engine.alert_active,
            fcdw_message=fcdw_engine.alert_message,
            ldw_active=ldw_engine.alert_active,
            ldw_message=ldw_engine.alert_message,
            ldw_dir=ldw_engine.departure_direction,
            now=t,
        )
        t += DT

    check("Scenario 2: Triggers LDW alert", alert is not None and alert["system"] == "LDW")
    check("Scenario 2: Message is 'LANE DEPARTURE WARNING'", alert["message"] == "LANE DEPARTURE WARNING")

    # Recover back to center
    for _ in range(15):
        ldw_engine.update(synthetic_lanes(offset_px=0), t)
        t += DT

    # -------------------------------------------------------------
    # Scenario 3: Stopped at Traffic Light -> Lead Vehicle Departs
    # -------------------------------------------------------------
    # Stationary lead car for 2.0s
    for _ in range(30):
        ldw_engine.update(synthetic_lanes(offset_px=0), t)
        dets = [synthetic_car(cx=320, cy=360, w=160, h=120)]
        fcw_engine.update(dets, t)
        fcdw_engine.update(dets, t)
        t += DT

    # Lead car departs (moves away)
    fcdw_triggered = False
    for step in range(12):
        cy = 360 - (step + 1) * 4
        h = max(70, 120 - (step + 1) * 3)
        w = int(h * 1.33)
        dets = [synthetic_car(cx=320, cy=cy, w=w, h=h)]

        ldw_engine.update(synthetic_lanes(offset_px=0), t)
        fcw_engine.update(dets, t)
        fcdw_engine.update(dets, t)

        alert = alert_mgr.update(
            fcw_level=fcw_engine.level,
            fcdw_active=fcdw_engine.alert_active,
            fcdw_message=fcdw_engine.alert_message,
            ldw_active=ldw_engine.alert_active,
            ldw_message=ldw_engine.alert_message,
            ldw_dir=ldw_engine.departure_direction,
            now=t,
        )
        t += DT
        if alert is not None and alert["system"] == "FCDW":
            fcdw_triggered = True

    check("Scenario 3: Triggers FCDW 'FRONT VEHICLE MOVING'", fcdw_triggered)

    # -------------------------------------------------------------
    # Scenario 4: Rapid Collision Hazard (FCW Overrides LDW)
    # -------------------------------------------------------------
    # Fast approaching obstacle (h grows from 100 to 300) while drifting
    for step in range(15):
        h = min(400, 150 + (step + 1) * 20)
        w = int(h * 1.2)
        dets = [synthetic_car(cx=320, cy=380, w=w, h=h)]

        ldw_engine.update(synthetic_lanes(offset_px=150), t) # drifting left
        fcw_engine.update(dets, t)                           # collision imminent
        fcdw_engine.update(dets, t)

        alert = alert_mgr.update(
            fcw_level=fcw_engine.level,
            fcdw_active=fcdw_engine.alert_active,
            fcdw_message=fcdw_engine.alert_message,
            ldw_active=ldw_engine.alert_active,
            ldw_message=ldw_engine.alert_message,
            ldw_dir=ldw_engine.departure_direction,
            now=t,
        )
        t += DT

    check("Scenario 4: FCW DANGER takes top priority over LDW", alert is not None and alert["system"] == "FCW")
    check("Scenario 4: Priority is 1 (DANGER)", alert["priority"] == 1)


if __name__ == "__main__":
    print("== test_scenario_full_driving_mission ==")
    test_scenario_full_driving_mission()
    print("-" * 40)
    print("ALL INTEGRATION PIPELINE TESTS PASSED")
    sys.exit(0)
