"""
tests/test_fcw.py — offline unit tests for the FCW module.

Runs WITHOUT camera, YOLO or OpenCV: the FCW logic is pure Python, so these
tests verify the risk state machine with synthetic detections.

Run from the project root:
    .\\.venv\\Scripts\\python.exe tests\\test_fcw.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fcw  # noqa: E402

W, H = 640, 480
FPS = 15.0
DT = 1.0 / FPS

# ---------------------------------------------------------------- helpers

class Harness:
    """Feeds synthetic detections into a ForwardCollisionWarning instance."""

    def __init__(self):
        self.fcw = fcw.ForwardCollisionWarning(W, H)
        self.t = 0.0

    def frame(self, detections):
        self.fcw.update(detections, self.t)
        self.t += DT
        return self.fcw.level

    def step_empty(self):
        return self.frame([])


def box(height, bottom_y=470, cx=None, width_ratio=0.33, class_name="car",
        track_id=1, conf=0.9):
    """Build a detection dict: a box of given height, centred at cx."""
    if cx is None:
        cx = W // 2
    w = max(2, int(height * width_ratio))
    x1 = cx - w // 2
    x2 = x1 + w
    return {
        "track_id": track_id,
        "class_name": class_name,
        "confidence": conf,
        "x1": x1,
        "y1": bottom_y - int(height),
        "x2": x2,
        "y2": bottom_y,
    }


def physical_height(t, z0=40.0, v=9.0, kpx=1800.0):
    """Pinhole-camera box height for an object at distance z = z0 - v*t."""
    z = max(z0 - v * t, 1.0)
    return min(int(kpx / z), H)


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  PASS  {name}")


# ---------------------------------------------------------------- tests

def test_escalation_safe_to_danger_by_ttc():
    """A car closing at constant speed must escalate SAFE->CAUTION->WARNING->DANGER
    and stay in DANGER."""
    h = Harness()
    levels = []
    for n in range(80):
        levels.append(h.frame([box(physical_height(h.t))]))

    first = {lv: None for lv in ("CAUTION", "WARNING", "DANGER")}
    for n, lv in enumerate(levels):
        if lv in first and first[lv] is None and lv != "SAFE":
            # n is the index of the first frame at that level
            first[lv] = n

    check("reached DANGER by frame 60",
          first["DANGER"] is not None and first["DANGER"] <= 60)
    check("levels escalate in order", first["CAUTION"] < first["WARNING"] < first["DANGER"])
    check("final level is DANGER", levels[-1] == "DANGER")
    check("stays DANGER once reached", all(lv == "DANGER" for lv in levels[first["DANGER"]:]))


def test_single_spike_frame_does_not_trigger():
    """One noisy frame with a huge box must NOT flicker the warning to DANGER."""
    h = Harness()
    for _ in range(5):
        h.frame([box(80)])
    h.frame([box(400)])  # one-frame spike: would be DANGER by height
    for _ in range(10):
        lv = h.frame([box(80)])
        check("never DANGER after single spike", lv != "DANGER")
    check("still SAFE after spike", h.fcw.level == "SAFE")


def test_off_path_and_above_horizon_excluded():
    """Objects outside the driving corridor or above the horizon must be ignored."""
    h = Harness()
    for _ in range(30):
        h.frame([box(200, cx=30)])            # far left: bottom_x=30 < 96
        h.frame([box(200, bottom_y=100)])     # above horizon (y2 < 168)
    check("off-path objects ignored", h.fcw.level == "SAFE")


def test_wrong_class_ignored():
    """Non-road classes must not trigger FCW."""
    h = Harness()
    for _ in range(30):
        h.frame([box(300, class_name="cat")])
    check("cat never triggers FCW", h.fcw.level == "SAFE")


def test_vanished_object_does_not_clear_instantly():
    """A 2-frame trace miss must not clear DANGER (slow release)."""
    h = Harness()
    for n in range(40):
        h.frame([box(physical_height(h.t, z0=40.0, v=15.0))])
    check("reached DANGER before vanish", h.fcw.level == "DANGER")

    h.frame([] )  # miss 1
    h.frame([])   # miss 2
    check("still DANGER after 2-frame miss", h.fcw.level == "DANGER")

    for _ in range(10):                      # ~0.7 s of nothing
        h.step_empty()
    check("no longer DANGER after 0.7 s of nothing",
          h.fcw.level != "DANGER")

    for _ in range(30):                      # give it plenty of time
        h.step_empty()
    check("eventually back to SAFE", h.fcw.level == "SAFE")


def test_parked_ahead_keeps_caution():
    """A stationary object ahead stays at CAUTION (no flicker)."""
    h = Harness()
    for _ in range(15):
        lv = h.frame([box(100)])
    check("stationary object ahead -> CAUTION", lv == "CAUTION")
    check("is_active True", h.fcw.is_active)
    check("primary reports object", h.fcw.primary is not None
          and h.fcw.primary["track_id"] == 1
          and "tau" in h.fcw.primary and "h_frac" in h.fcw.primary)


def test_hysteresis_exit_band():
    """Dropping a little must NOT clear CAUTION; dropping below the EXIT
    threshold must clear it after 8 confirmed frames."""
    h = Harness()
    for _ in range(10):
        h.frame([box(100)])          # CAUTION
    check("in CAUTION", h.fcw.level == "CAUTION")

    for _ in range(10):
        lv = h.frame([box(90)])      # 90px: below ENTER(86.4)? no, above; in band
    check("stays CAUTION in exit band", lv == "CAUTION")

    for _ in range(5):
        lv = h.frame([box(55)])      # below CAUTION_H_EXIT (62.4)
    check("still CAUTION while confirming SAFE", lv == "CAUTION")

    for _ in range(10):
        lv = h.frame([box(55)])
    check("cleared to SAFE after sustained small object", lv == "SAFE")


def test_vehicle_classes_all_relevant():
    """All configured vehicle/person classes pass the class filter."""
    for cname in ("car", "bus", "truck", "motorcycle", "bicycle", "person"):
        h = Harness()
        for _ in range(5):
            h.frame([box(300, class_name=cname)])
        check(f"{cname} triggers a warning level",
              h.fcw.level in ("CAUTION", "WARNING", "DANGER"))


def test_primary_is_closest():
    """With two relevant objects, the closer (taller) one is primary."""
    h = Harness()
    for _ in range(5):
        h.frame([
            box(150, track_id=1),   # h_frac 0.3125 -> WARNING
            box(280, track_id=2),   # h_frac 0.583 -> DANGER
        ])
    check("primary is the taller object", h.fcw.primary["track_id"] == 2)
    check("level driven by the taller object", h.fcw.level == "DANGER")


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    tests = [
        test_escalation_safe_to_danger_by_ttc,
        test_single_spike_frame_does_not_trigger,
        test_off_path_and_above_horizon_excluded,
        test_wrong_class_ignored,
        test_vanished_object_does_not_clear_instantly,
        test_parked_ahead_keeps_caution,
        test_hysteresis_exit_band,
        test_vehicle_classes_all_relevant,
        test_primary_is_closest,
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
    print("ALL FCW TESTS PASSED")
    sys.exit(0)