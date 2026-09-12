"""
tests/test_alert_manager.py — unit tests for AlertManager priority arbitration.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alert_manager  # noqa: E402


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  PASS  {name}")


def test_normal_quiet_state():
    am = alert_manager.AlertManager()
    alert = am.update(
        fcw_level="SAFE",
        fcdw_active=False,
        fcdw_message="",
        ldw_active=False,
        ldw_message="",
        now=10.0
    )
    check("Quiet produces no active alert", alert is None)
    check("System status is NORMAL", am.active_system == "NORMAL")


def test_fcw_danger_highest_priority():
    """FCW DANGER must override simultaneous LDW and FCDW."""
    am = alert_manager.AlertManager()
    alert = am.update(
        fcw_level="DANGER",
        fcdw_active=True,
        fcdw_message="FRONT VEHICLE MOVING",
        ldw_active=True,
        ldw_message="LANE DEPARTURE WARNING",
        ldw_dir="LEFT",
        now=10.0
    )
    check("Active alert is not None", alert is not None)
    check("System is FCW", am.active_system == "FCW")
    check("Priority is 1", alert["priority"] == 1)
    check("Message is FORWARD COLLISION WARNING", alert["message"] == "FORWARD COLLISION WARNING")


def test_ldw_overrides_fcdw():
    """LDW has higher priority than FCDW and FCW CAUTION."""
    am = alert_manager.AlertManager()
    alert = am.update(
        fcw_level="CAUTION",
        fcdw_active=True,
        fcdw_message="FRONT VEHICLE MOVING",
        ldw_active=True,
        ldw_message="LANE DEPARTURE WARNING",
        ldw_dir="RIGHT",
        now=10.0
    )
    check("System is LDW", am.active_system == "LDW")
    check("Priority is 3", alert["priority"] == 3)
    check("Message is LANE DEPARTURE WARNING", alert["message"] == "LANE DEPARTURE WARNING")


def test_fcdw_overrides_fcw_caution():
    """FCDW has higher priority than mere FCW CAUTION."""
    am = alert_manager.AlertManager()
    alert = am.update(
        fcw_level="CAUTION",
        fcdw_active=True,
        fcdw_message="FRONT VEHICLE MOVING",
        ldw_active=False,
        ldw_message="",
        now=10.0
    )
    check("System is FCDW", am.active_system == "FCDW")
    check("Message is FRONT VEHICLE MOVING", alert["message"] == "FRONT VEHICLE MOVING")


def test_fcw_caution_when_alone():
    """FCW CAUTION displays when no higher alerts are active."""
    am = alert_manager.AlertManager()
    alert = am.update(
        fcw_level="CAUTION",
        fcdw_active=False,
        fcdw_message="",
        ldw_active=False,
        ldw_message="",
        now=10.0
    )
    check("System is FCW", am.active_system == "FCW")
    check("Message is CAUTION", alert["message"] == "CAUTION")


if __name__ == "__main__":
    tests = [
        test_normal_quiet_state,
        test_fcw_danger_highest_priority,
        test_ldw_overrides_fcdw,
        test_fcdw_overrides_fcw_caution,
        test_fcw_caution_when_alone,
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
    print("ALL ALERT MANAGER TESTS PASSED")
    sys.exit(0)
