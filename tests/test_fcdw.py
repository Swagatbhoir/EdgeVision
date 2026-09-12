"""
tests/test_fcdw.py — unit tests for Front Car Departure Warning (FCDW).

Pure Python test suite verifying lead car departure tracking with synthetic data.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fcdw  # noqa: E402

W, H = 640, 480
FPS = 15.0
DT = 1.0 / FPS


def make_car(cx=320, cy=350, w=160, h=120, track_id=1, class_name="car"):
    x1 = int(cx - w / 2)
    x2 = int(cx + w / 2)
    y1 = int(cy - h / 2)
    y2 = int(cy + h / 2)
    return {
        "track_id": track_id,
        "class_name": class_name,
        "confidence": 0.92,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  PASS  {name}")


def test_normal_departure_sequence():
    """Vehicle stopped for 2 seconds, then moves forward/away -> triggers alert."""
    engine = fcdw.FrontCarDepartureWarning(W, H)
    t = 0.0

    # 1. Lead car stopped ahead at y2 = 410, h = 120 for 2.0s (30 frames)
    for _ in range(30):
        engine.update([make_car(cx=320, cy=350, w=160, h=120)], t)
        t += DT

    check("Tracking stopped lead vehicle", engine.state == "TRACKING_STOPPED")
    check("No premature alert", not engine.alert_active)

    # 2. Lead car starts moving away: box moves up, height shrinks
    triggered = False
    for step in range(15):
        # Car moves further away: cy shifts up from 350 to 300, h shrinks from 120 to 95
        cy = 350 - (step + 1) * 3
        h = max(70, 120 - (step + 1) * 2)
        w = int(h * 1.33)
        engine.update([make_car(cx=320, cy=cy, w=w, h=h)], t)
        t += DT
        if engine.alert_active and engine.alert_message == "FRONT VEHICLE MOVING":
            triggered = True
            break

    check("Triggered 'FRONT VEHICLE MOVING' on departure", triggered)
    check("Alert active state", engine.alert_active)


def test_short_stop_does_not_trigger():
    """Vehicle stops for only 0.3s (e.g. slowing down slightly in traffic) -> no departure alert."""
    engine = fcdw.FrontCarDepartureWarning(W, H)
    t = 0.0

    # Stopped for only 5 frames (0.33s < MIN_STOPPED_DURATION 1.5s)
    for _ in range(5):
        engine.update([make_car(cx=320, cy=350, w=160, h=120)], t)
        t += DT

    # Immediately moves away
    for step in range(10):
        cy = 350 - (step + 1) * 4
        h = max(70, 120 - (step + 1) * 3)
        w = int(h * 1.33)
        engine.update([make_car(cx=320, cy=cy, w=w, h=h)], t)
        t += DT

    check("Short stop does not false-trigger departure", not engine.alert_active)


def test_single_frame_jitter_no_trigger():
    """One single jittery frame should not trigger departure."""
    engine = fcdw.FrontCarDepartureWarning(W, H)
    t = 0.0

    # Stopped for 2.0s
    for _ in range(30):
        engine.update([make_car(cx=320, cy=350, w=160, h=120)], t)
        t += DT

    # 1 jitter frame
    engine.update([make_car(cx=320, cy=300, w=100, h=80)], t)
    t += DT
    check("Single jitter frame does not trigger alert", not engine.alert_active)

    # Back to stationary
    for _ in range(5):
        engine.update([make_car(cx=320, cy=350, w=160, h=120)], t)
        t += DT
    check("Returns to stable stopped without alert", not engine.alert_active)


def test_off_corridor_vehicle_ignored():
    """A parked or moving vehicle in the adjacent lane is ignored."""
    engine = fcdw.FrontCarDepartureWarning(W, H)
    t = 0.0

    # Car on far left (cx = 80)
    for _ in range(30):
        engine.update([make_car(cx=80, cy=350, w=160, h=120)], t)
        t += DT

    check("Adjacent lane car ignored as lead", engine.lead_track_id is None)


def test_alert_duration_and_cooldown():
    """Alert automatically expires after ALERT_DURATION."""
    engine = fcdw.FrontCarDepartureWarning(W, H)
    t = 0.0

    for _ in range(30):
        engine.update([make_car(cx=320, cy=350, w=160, h=120)], t)
        t += DT

    for step in range(10):
        cy = 350 - (step + 1) * 4
        h = max(70, 120 - (step + 1) * 3)
        w = int(h * 1.33)
        engine.update([make_car(cx=320, cy=cy, w=w, h=h)], t)
        t += DT
        if engine.alert_active:
            break

    check("Alert active", engine.alert_active)

    # Advance time past ALERT_DURATION (4.0s)
    t += 4.5
    engine.update([], t)
    check("Alert expired after duration", not engine.alert_active)
    check("State reset to IDLE", engine.state == "IDLE")


if __name__ == "__main__":
    tests = [
        test_normal_departure_sequence,
        test_short_stop_does_not_trigger,
        test_single_frame_jitter_no_trigger,
        test_off_corridor_vehicle_ignored,
        test_alert_duration_and_cooldown,
    ]
    failed = 0
    for test in tests:
        name = test.__name__
        print(f"== {name} ==")
        try:
            test()
        except AssertionError as exc:
            print(f"  FAIL {exc}")
            failed += 1
    print("-" * 40)
    if failed:
        print(f"{failed} test(s) FAILED")
        sys.exit(1)
    print("ALL FCDW TESTS PASSED")
    sys.exit(0)
