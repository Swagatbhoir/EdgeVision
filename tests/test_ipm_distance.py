"""
tests/test_ipm_distance.py — Unit tests for Inverse Perspective Mapping (IPM) Distance.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipm_distance import IPMDistanceEstimator


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  PASS  {name}")


def test_distance_monotonicity():
    """Points lower in the image (closer to bumper) must have smaller distance."""
    ipm = IPMDistanceEstimator(frame_width=640, frame_height=480, cam_height_m=1.25, pitch_angle_deg=5.0)

    # Box 1: Near bumper (y2 = 460)
    z1, x1 = ipm.estimate_distance(280, 360, 360, 460)
    # Box 2: Mid-range (y2 = 360)
    z2, x2 = ipm.estimate_distance(290, 300, 350, 360)
    # Box 3: Far away (y2 = 280)
    z3, x3 = ipm.estimate_distance(305, 250, 335, 280)

    check("Near distance is positive", z1 is not None and z1 > 0)
    check("Mid distance is greater than near", z2 > z1)
    check("Far distance is greater than mid", z3 > z2)
    check("Realistic car distance range (2m < z1 < 10m)", 2.0 < z1 < 10.0)


def test_lateral_positioning():
    """Left box gives negative X, right box gives positive X, center gives ~0."""
    ipm = IPMDistanceEstimator(frame_width=640, frame_height=480, cam_height_m=1.25, pitch_angle_deg=5.0)

    # Center box (u = 320)
    z_c, x_c = ipm.estimate_distance(280, 300, 360, 380)
    # Left box (u = 120)
    z_l, x_l = ipm.estimate_distance(80, 300, 160, 380)
    # Right box (u = 520)
    z_r, x_r = ipm.estimate_distance(480, 300, 560, 380)

    check("Center lateral offset is near zero", abs(x_c) < 0.05)
    check("Left lateral offset is negative", x_l < -0.5)
    check("Right lateral offset is positive", x_r > 0.5)
    check("Symmetric lateral offsets", abs(abs(x_l) - abs(x_r)) < 0.1)


def test_horizon_boundary_handling():
    """Objects at or above the true horizon should return (None, None)."""
    ipm = IPMDistanceEstimator(frame_width=640, frame_height=480, cam_height_m=1.25, pitch_angle_deg=5.0)
    horizon_y = ipm.get_horizon_y()

    # Above horizon (sky)
    z_sky, x_sky = ipm.estimate_distance(280, 50, 360, horizon_y - 20)
    check("Above horizon returns None", z_sky is None and x_sky is None)


def test_roundtrip_ground_image_projection():
    """Projecting to ground and back to image should recover original pixel coordinate."""
    ipm = IPMDistanceEstimator(frame_width=640, frame_height=480, cam_height_m=1.25, pitch_angle_deg=5.0)

    test_pixels = [(320, 420), (200, 350), (450, 320)]
    for orig_u, orig_v in test_pixels:
        z, x = ipm.project_to_ground(orig_u, orig_v)
        rec_u, rec_v = ipm.project_to_image(x, z)
        check(f"Roundtrip pixel ({orig_u}, {orig_v}) -> ({rec_u}, {rec_v})",
              abs(orig_u - rec_u) <= 1 and abs(orig_v - rec_v) <= 1)


if __name__ == "__main__":
    tests = [
        test_distance_monotonicity,
        test_lateral_positioning,
        test_horizon_boundary_handling,
        test_roundtrip_ground_image_projection,
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
    print("ALL IPM DISTANCE TESTS PASSED")
    sys.exit(0)
