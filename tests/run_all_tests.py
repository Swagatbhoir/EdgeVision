"""
tests/run_all_tests.py — Master test suite runner for EdgeVision ADAS.

Executes tests for:
  1. IPM Distance Estimation
  2. Forward Collision Warning (FCW)
  3. Front Car Departure Warning (FCDW)
  4. Lane Departure Warning (LDW)
  5. Unified Alert Manager & Arbitration
  6. End-to-End ADAS Integration
  7. Cockpit HUD Renderer
  8. Cockpit Web Dashboard & Telemetry API
  9. Main-Loop Cockpit Pipeline Contract
"""

import sys
import subprocess
import os

TEST_FILES = [
    "tests/test_ipm_distance.py",
    "tests/test_fcw.py",
    "tests/test_fcdw.py",
    "tests/test_ldw.py",
    "tests/test_alert_manager.py",
    "tests/test_integration_pipeline.py",
    "tests/test_hud_renderer.py",
    "tests/test_web_api.py",
    "tests/test_cockpit_pipeline.py",
]

def main():
    print("=" * 60)
    print("  EDGEVISION ADAS — FULL TEST SUITE")
    print("=" * 60)

    py_exe = sys.executable
    passed_suites = 0
    failed_suites = 0

    for test_file in TEST_FILES:
        print(f"\n>>> Running {test_file} ...")
        res = subprocess.run([py_exe, "-X", "utf8", test_file], capture_output=True, text=True)
        print(res.stdout, end="")
        if res.stderr:
            print(res.stderr, end="")

        if res.returncode == 0:
            passed_suites += 1
        else:
            failed_suites += 1
            print(f"!!! SUITE FAILED: {test_file} (exit code {res.returncode})")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed_suites} suites passed, {failed_suites} suites failed.")
    print("=" * 60)

    if failed_suites > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
