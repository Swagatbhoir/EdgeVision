"""
tests/test_ldw.py — unit tests for Lane Departure Warning (LDW).

Tests the lane departure geometry, hysteresis, temporal smoothing, and alert states.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ldw  # noqa: E402

W, H = 640, 480
FPS = 15.0
DT = 1.0 / FPS


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  PASS  {name}")


def synthetic_lanes(offset_px=0):
    """
    Returns synthetic lane line endpoints shifted by offset_px.
    Center of frame is 320.
    Standard centered lane:
      Left: bottom (120, 456), top (240, 278)
      Right: bottom (520, 456), top (400, 278)
      Lane center at bottom = (120+520)/2 = 320 (exact vehicle center)
    """
    return {
        "left_line": (120 + offset_px, 456, 240 + int(offset_px * 0.6), 278),
        "right_line": (520 + offset_px, 456, 400 + int(offset_px * 0.6), 278),
    }


def test_centered_lane_driving():
    """Vehicle centered in lane produces NORMAL status with 0 offset."""
    engine = ldw.LaneDepartureWarning(W, H)
    t = 0.0

    for _ in range(15):
        engine.update(synthetic_lanes(offset_px=0), t)
        t += DT

    check("Centered lane produces NORMAL status", engine.status == "NORMAL")
    check("No departure alert", not engine.alert_active)
    check("Lateral offset near zero", abs(engine.lateral_offset) < 0.05)


def test_left_departure_and_recovery():
    """Vehicle drifting left (lane lines shift right relative to camera)."""
    engine = ldw.LaneDepartureWarning(W, H)
    t = 0.0

    # 1. Drive centered for 10 frames
    for _ in range(10):
        engine.update(synthetic_lanes(offset_px=0), t)
        t += DT
    check("Initially NORMAL", engine.status == "NORMAL")

    # 2. Drift left: camera moves left -> lane lines in image move right (+140px)
    # lane center at bottom becomes (120+140 + 520+140)/2 = 460
    # offset = (320 - 460) / 200 = -0.70 (drifting left)
    for _ in range(10):
        engine.update(synthetic_lanes(offset_px=140), t)
        t += DT

    check("Triggered LEFT departure status", engine.status == "WARNING_LEFT")
    check("Triggered 'LANE DEPARTURE WARNING' alert", engine.alert_message == "LANE DEPARTURE WARNING")
    check("Departure direction LEFT", engine.departure_direction == "LEFT")

    # 3. Recover back to center
    for _ in range(15):
        engine.update(synthetic_lanes(offset_px=0), t)
        t += DT

    check("Recovered back to NORMAL after steering back", engine.status == "NORMAL")
    check("Departure alert cleared", not engine.alert_active)


def test_right_departure():
    """Vehicle drifting right (lane lines shift left relative to camera)."""
    engine = ldw.LaneDepartureWarning(W, H)
    t = 0.0

    # Drift right: lane lines shift left (-140px)
    for _ in range(10):
        engine.update(synthetic_lanes(offset_px=-140), t)
        t += DT

    check("Triggered RIGHT departure status", engine.status == "WARNING_RIGHT")
    check("Alert active", engine.alert_active)
    check("Departure direction RIGHT", engine.departure_direction == "RIGHT")


def test_single_spike_noise_rejected():
    """One single frame with bad lane detection does not trigger departure alert."""
    engine = ldw.LaneDepartureWarning(W, H)
    t = 0.0

    for _ in range(10):
        engine.update(synthetic_lanes(offset_px=0), t)
        t += DT

    # 1 corrupted / spike frame
    engine.update(synthetic_lanes(offset_px=180), t)
    t += DT
    check("Single noise spike does not trigger warning", engine.status == "NORMAL")

    # Follow-up centered frames
    for _ in range(5):
        engine.update(synthetic_lanes(offset_px=0), t)
        t += DT
    check("Remains NORMAL", engine.status == "NORMAL")


def test_hysteresis_band():
    """Drifting slightly into the exit band doesn't oscillate status."""
    engine = ldw.LaneDepartureWarning(W, H)
    t = 0.0

    # Start in WARNING_LEFT
    for _ in range(10):
        engine.update(synthetic_lanes(offset_px=140), t)  # offset ~ -0.70
        t += DT
    check("Entered WARNING_LEFT", engine.status == "WARNING_LEFT")

    # Move into hysteresis band (offset ~ -0.45, between 0.35 exit and 0.55 enter)
    for _ in range(8):
        engine.update(synthetic_lanes(offset_px=90), t)
        t += DT
    check("Maintains WARNING_LEFT in hysteresis band", engine.status == "WARNING_LEFT")


if __name__ == "__main__":
    tests = [
        test_centered_lane_driving,
        test_left_departure_and_recovery,
        test_right_departure,
        test_single_spike_noise_rejected,
        test_hysteresis_band,
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
    print("ALL LDW TESTS PASSED")
    sys.exit(0)
