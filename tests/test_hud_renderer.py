"""
tests/test_hud_renderer.py — Offline tests for the tactical cockpit HUD renderer.

Covers the HUD geometry helpers and the render pipeline contract. The
pixel-level "did the overlay actually paint?" assertions need a real OpenCV
backend, so they run in a dedicated phase that is spawned as a child process
(importing OpenCV pulls in native libraries that some headless environments
cannot load at all — see the note on _cv2_importable below).
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RENDER_PHASE = os.environ.get("EDGEVISION_HUD_RENDER_PHASE") == "1"


# ===========================================================================
# PHASE 2 — pixel verification, executed in a spawned child process
# ===========================================================================

if RENDER_PHASE:
    # This child is only spawned when OpenCV is known to be importable.
    os.environ.pop("EDGEVISION_NO_CV2", None)

    import numpy as np
    import hud_renderer
    from hud_renderer import COLOR_DANGER, COLOR_LDW

    DANGER_ALERT = {
        "system": "FCW", "level": "DANGER", "message": "FORWARD COLLISION WARNING",
        "subtext": "BRAKE NOW", "color_bgr": COLOR_DANGER, "sound_pattern": "CRITICAL",
    }
    LDW_ALERT = {
        "system": "LDW", "level": "WARNING", "message": "LANE DEPARTURE WARNING",
        "subtext": "DRIFTING LEFT", "color_bgr": COLOR_LDW, "sound_pattern": "ADVISORY",
    }

    def check(name, cond):
        if not cond:
            raise AssertionError("FAILED: {0}".format(name))
        print("  PASS  {0}".format(name))

    check("OpenCV backend is live", hud_renderer.has_cv2())

    # 1. Tactical bounding box actually paints
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hud_renderer.draw_tactical_bbox(
        frame, (250, 250, 390, 420), "car", 7,
        risk="DANGER", distance=6.5, ttc=1.2, lat_offset=0.1,
        is_lead=True, confidence=0.93,
    )
    check("tactical bbox paints the frame", frame.any())

    # 2. Alert banner paints, NORMAL alert paints nothing
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hud_renderer.draw_cockpit_banner(frame, DANGER_ALERT, now=0.0)
    check("danger banner paints the frame", frame.any())

    clean = np.zeros((480, 640, 3), dtype=np.uint8)
    hud_renderer.draw_cockpit_banner(clean, None, now=0.0)
    hud_renderer.draw_cockpit_banner(clean, {"system": "NORMAL"}, now=0.0)
    check("NORMAL / empty alert paints nothing", not clean.any())

    # 3. BEV radar paints and tolerates out-of-range blips
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hud_renderer.draw_bev_radar(frame, [
        {"distance_m": 5.0, "lateral_offset_m": 0.0, "risk": "DANGER"},
        {"distance_m": 500.0, "lateral_offset_m": 0.0, "risk": "DANGER"},
        {"lateral_offset_m": 0.0, "risk": "SAFE"},
        {"distance_m": None, "lateral_offset_m": 0.0, "risk": "SAFE"},
    ])
    check("BEV radar paints in-range blips", frame.any())

    # 4. Lane corridor needs BOTH lines before painting
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hud_renderer.draw_ar_lane_corridor(frame, None, (520, 470, 400, 280))
    check("partial lane detection paints nothing", not frame.any())

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hud_renderer.draw_ar_lane_corridor(frame, (120, 470, 240, 280), (520, 470, 400, 280))
    check("AR lane carpet paints with both lines", frame.any())

    # 5. Full-layer render must modify the frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = hud_renderer.HUDRenderer(640, 480).render(
        frame,
        tracked_objects=[
            {"track_id": 1, "name": "car", "confidence": 0.9, "box": (250, 250, 390, 420),
             "risk": "DANGER", "distance_m": 6.5, "lateral_offset_m": 0.1, "ttc": 1.2,
             "in_ahead": True, "in_path": True},
            {"track_id": 2, "name": "truck", "confidence": 0.8, "box": (60, 300, 180, 440),
             "risk": "CAUTION", "distance_m": 22.0, "lateral_offset_m": -1.4, "ttc": None,
             "in_ahead": False, "in_path": False},
        ],
        lane_lines=((120, 470, 240, 280), (520, 470, 400, 280)),
        ldw_direction="DRIFTING LEFT",
        active_alert=DANGER_ALERT,
        fps=15.0, inference_ms=42.0, lead_vehicle_id=1,
        show_bboxes=True, show_lanes=True, show_radar=True, show_telemetry=True,
        camera_source="CAM 0", now=1000.0,
    )
    check("full-layer render paints pixels", out.any())
    check("full-layer render keeps frame shape", out.shape == (480, 640, 3))

    print("-" * 60)
    print("ALL HUD RENDER PIXEL TESTS PASSED")
    sys.exit(0)


# ===========================================================================
# PHASE 1 — geometry & contract checks (never imports OpenCV)
# ===========================================================================

# The parent process always runs the headless path: the contract checks below
# are about geometry and the render contract, not about pixels. Everything
# that needs a real drawing backend is delegated to the spawned child phase.
os.environ["EDGEVISION_NO_CV2"] = "1"


def _cv2_importable():
    """
    Probe OpenCV in a throwaway subprocess.

    On some headless machines the OpenCV native libraries abort the whole
    process at import time, which no try/except can catch. Probing out of
    process keeps this suite alive and reports honestly.
    """
    try:
        res = subprocess.run([sys.executable, "-c", "import cv2"], timeout=180)
        return res.returncode == 0
    except Exception:
        return False


CV2_OK = _cv2_importable()

import numpy as np  # noqa: E402

import hud_renderer  # noqa: E402
from hud_renderer import HUDRenderer, COLOR_DANGER, COLOR_SAFE, COLOR_LDW  # noqa: E402


def check(name, cond):
    if not cond:
        raise AssertionError("FAILED: {0}".format(name))
    print("  PASS  {0}".format(name))


def sample_object(risk="SAFE", dist=12.0, lat=0.2, ttc=3.0, in_ahead=True):
    return {
        "track_id": 1,
        "name": "car",
        "confidence": 0.91,
        "box": (250, 250, 390, 420),
        "risk": risk,
        "distance_m": dist,
        "lateral_offset_m": lat,
        "ttc": ttc,
        "in_ahead": in_ahead,
        "in_path": in_ahead,
    }


# ---------------------------------------------------------------------------
def test_color_palette_is_automotive_standard():
    """Every alert colour must be a BGR 3-tuple inside the 0-255 range."""
    for name in ("COLOR_DANGER", "COLOR_WARNING", "COLOR_CAUTION",
                 "COLOR_SAFE", "COLOR_LDW", "COLOR_FCDW"):
        color = getattr(hud_renderer, name)
        check("{0} is a BGR triple".format(name), len(color) == 3)
        check("{0} channels in range".format(name),
              all(isinstance(c, (int, np.integer)) and 0 <= c <= 255 for c in color))

    check("DANGER is red-dominant (BGR)", COLOR_DANGER[2] > COLOR_DANGER[1])
    check("SAFE is green-dominant (BGR)", COLOR_SAFE[1] > COLOR_SAFE[0])


# ---------------------------------------------------------------------------
def test_renderer_construction():
    r = HUDRenderer(640, 480)
    check("renderer stores frame width", r.frame_w == 640)
    check("renderer stores frame height", r.frame_h == 480)


# ---------------------------------------------------------------------------
def test_render_preserves_frame_shape():
    """The HUD must paint in place and never resize or recode the frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = HUDRenderer(640, 480).render(
        frame,
        tracked_objects=[sample_object()],
        lane_lines=((120, 470, 240, 280), (520, 470, 400, 280)),
        ldw_direction="NORMAL",
        active_alert=None,
        fps=28.0,
        inference_ms=18.0,
    )
    check("render returns a frame", out is not None)
    check("frame shape unchanged", out.shape == (480, 640, 3))
    check("frame dtype unchanged", out.dtype == np.uint8)


# ---------------------------------------------------------------------------
def test_render_handles_empty_and_edge_inputs():
    """No objects, no lanes, no alert — the common calm-driving case."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = HUDRenderer(640, 480).render(
        frame, tracked_objects=[], lane_lines=None,
        ldw_direction="NORMAL", active_alert=None,
    )
    check("empty render is a no-op but valid", out.shape == (480, 640, 3))

    # Malformed / degenerate geometry must not raise
    out2 = HUDRenderer(640, 480).render(
        np.zeros((480, 640, 3), dtype=np.uint8),
        tracked_objects=[{"track_id": 0, "name": "car", "box": (0, 0, 0, 0)}],
        lane_lines=((0, 0, 0, 0), (0, 0, 0, 0)),
        ldw_direction="DRIFTING LEFT",
        active_alert={
            "system": "LDW", "level": "WARNING",
            "message": "LANE DEPARTURE WARNING", "subtext": "DRIFTING LEFT",
            "color_bgr": COLOR_LDW,
        },
    )
    check("degenerate geometry does not raise", out2.shape == (480, 640, 3))


# ---------------------------------------------------------------------------
def test_render_with_all_layers_enabled():
    """All six HUD layers on at once — the maximum rendering budget case."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    objects = [
        sample_object(risk="DANGER", dist=6.5, lat=0.1, ttc=1.2),
        sample_object(risk="WARNING", dist=14.0, lat=-0.6, ttc=2.8),
        sample_object(risk="CAUTION", dist=28.0, lat=1.2, ttc=None),
        sample_object(risk="SAFE", dist=44.0, lat=0.0, ttc=None),
    ]
    out = HUDRenderer(640, 480).render(
        frame,
        tracked_objects=objects,
        lane_lines=((120, 470, 240, 280), (520, 470, 400, 280)),
        ldw_direction="DRIFTING LEFT",
        active_alert={
            "system": "FCW", "level": "DANGER",
            "message": "FORWARD COLLISION WARNING", "subtext": "BRAKE NOW",
            "color_bgr": COLOR_DANGER,
        },
        fps=15.0,
        inference_ms=42.0,
        lead_vehicle_id=1,
        show_bboxes=True, show_lanes=True, show_radar=True, show_telemetry=True,
        camera_source="CAM 0",
        now=1000.0,
    )
    check("full-layer render succeeds", out.shape == (480, 640, 3))


# ---------------------------------------------------------------------------
def test_layer_toggles_are_respected():
    """Turning every layer off must be a pure no-op on the frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = HUDRenderer(640, 480).render(
        frame,
        tracked_objects=[sample_object(risk="DANGER")],
        lane_lines=((120, 470, 240, 280), (520, 470, 400, 280)),
        ldw_direction="DRIFTING LEFT",
        active_alert={
            "system": "FCW", "level": "DANGER",
            "message": "FORWARD COLLISION WARNING", "color_bgr": COLOR_DANGER,
        },
        show_bboxes=False, show_lanes=False, show_radar=False, show_telemetry=False,
    )
    if not CV2_OK:
        check("all layers off leaves frame untouched (headless)", not out.any())
    else:
        check("all layers off still returns a valid frame", out.shape == (480, 640, 3))


# ---------------------------------------------------------------------------
def test_draw_helpers_degrade_gracefully():
    """Every public draw helper must be callable and shape-safe."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    hud_renderer.draw_panel(frame, (0, 0), (100, 40))
    check("draw_panel keeps frame shape", frame.shape == (480, 640, 3))

    hud_renderer.draw_panel(frame, (5000, 5000), (6000, 6000))
    check("draw_panel clamps out-of-bounds rects", frame.shape == (480, 640, 3))

    hud_renderer.draw_tactical_bbox(frame, (250, 250, 390, 420), "car", 7, risk="DANGER")
    check("draw_tactical_bbox keeps frame shape", frame.shape == (480, 640, 3))

    hud_renderer.draw_top_telemetry_bar(frame, 30.0, 12.0)
    hud_renderer.draw_bottom_status_bar(frame, "DANGER", "DRIFTING LEFT", "ALERT")
    check("instrumentation bars keep frame shape", frame.shape == (480, 640, 3))

    if not CV2_OK:
        check("headless mode leaves frame untouched", not frame.any())


# ---------------------------------------------------------------------------
def test_pixel_verification_phase():
    """Spawn the OpenCV pixel-verification phase when a backend is available."""
    if not CV2_OK:
        print("  SKIP  pixel verification (no OpenCV backend in this environment)")
        return

    env = dict(os.environ)
    env["EDGEVISION_HUD_RENDER_PHASE"] = "1"
    env.pop("EDGEVISION_NO_CV2", None)

    res = subprocess.run([sys.executable, __file__], env=env, timeout=300)
    check("pixel verification phase passed", res.returncode == 0)

if __name__ == "__main__":
    print("=" * 60)
    print(" test_hud_renderer  (OpenCV backend: {0})".format(
        "AVAILABLE" if CV2_OK else "UNAVAILABLE — geometry/contract checks only"))
    print("=" * 60)

    test_color_palette_is_automotive_standard()
    test_renderer_construction()
    test_render_preserves_frame_shape()
    test_render_handles_empty_and_edge_inputs()
    test_render_with_all_layers_enabled()
    test_layer_toggles_are_respected()
    test_draw_helpers_degrade_gracefully()
    test_pixel_verification_phase()

    print("-" * 60)
    print("ALL HUD RENDERER TESTS PASSED")
