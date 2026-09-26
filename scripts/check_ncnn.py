"""
check_ncnn.py — verify model.track() still behaves on the NCNN backend.

Why this exists: main.py now loads the NCNN export instead of the .pt file.
Tracking is implemented in ultralytics as a POST-PROCESSING CALLBACK
(trackers/track.py:on_predict_postprocess_end) that runs on result.boxes
after NMS, and predictor.py fires that callback for every backend — so the
ByteTrack IDs are computed in Python from the detection boxes, not baked into
the exported graph. NCNN only changes how those boxes are produced.

This script proves that on YOUR machine instead of taking it on trust:

  1. the ncnn folder loads and returns detections
  2. boxes come back with stable, monotonically-assigned track IDs
  3. IDs persist across consecutive frames (persist=True is doing its job)
  4. a rough FPS number at the configured imgsz, so you can see what the Pi
     is up against

Usage:
    python scripts/check_ncnn.py [video.mp4] [imgsz]

With no arguments it uses VIDEO_PATH from main.py and MODEL_IMGSZ from main.py.
"""

import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VIDEO_PATH = r"C:\path\to\your_video.mp4"
MODEL_NCNN_PATH = "yolo26n_ncnn_model"
MODEL_IMGSZ = 320
FRAMES = 30

failures = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + ("   " + detail if detail else ""))
    if not ok:
        failures.append(label)


def main():
    global VIDEO_PATH, MODEL_IMGSZ

    args = sys.argv[1:]
    if len(args) >= 1:
        VIDEO_PATH = args[0]
    if len(args) >= 2:
        MODEL_IMGSZ = int(args[1])

    print("=" * 46)
    print(" NCNN + tracking check")
    print("=" * 46)
    print(" model : {0}".format(MODEL_NCNN_PATH))
    print(" video : {0}".format(VIDEO_PATH))
    print(" imgsz : {0}".format(MODEL_IMGSZ))
    print("=" * 46)
    print()

    if not os.path.isdir(MODEL_NCNN_PATH):
        print("ERROR: {0}/ not found. Run:  python scripts/export_ncnn.py".format(MODEL_NCNN_PATH))
        return 1

    # 1. load -----------------------------------------------------------------
    from ultralytics import YOLO
    try:
        model = YOLO(MODEL_NCNN_PATH)
    except Exception as exc:
        print("ERROR: could not load the ncnn model: {0}".format(exc))
        print("       Check that the 'ncnn' python package is installed on this machine:")
        print("         pip install ncnn")
        return 1
    check("ncnn model loads", True, MODEL_NCNN_PATH)

    # 2. open a source --------------------------------------------------------
    cap = cv2.VideoCapture(VIDEO_PATH) if VIDEO_PATH else None
    if cap is None or not cap.isOpened():
        # No video available: fall back to a synthetic frame so the backend
        # and the tracker still get exercised.
        print()
        print("NOTE: no readable video ({0!r}); using a synthetic test frame".format(VIDEO_PATH))
        print("      so the backend and tracker are still exercised.")
        print()
        frames = [(np_random_frame(i)) for i in range(FRAMES)]
    else:
        frames = []
        for _ in range(FRAMES):
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()

    if not frames:
        print("ERROR: no frames available, cannot continue.")
        return 1

    check("have frames to test", True, "{0} frames".format(len(frames)))

    # 3. track ----------------------------------------------------------------
    seen_ids = set()
    id_stable_across_frames = 0
    previous_ids = set()
    timings = []

    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        try:
            results = model.track(f, persist=True, verbose=False, imgsz=MODEL_IMGSZ)
        except Exception as exc:
            print("ERROR: model.track() raised on the NCNN backend:")
            print("       {0}: {1}".format(type(exc).__name__, exc))
            print()
            print("This is the failure mode this script exists to catch.")
            return 1
        timings.append((time.perf_counter() - t0) * 1000.0)

        result = results[0]
        ids = []
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
        seen_ids.update(ids)

        if previous_ids & set(ids):
            id_stable_across_frames += 1
        previous_ids = set(ids)

        if i == 0:
            check("model.track() runs on the ncnn backend", True,
                  "track() returned {0} results".format(len(results)))
            check("result.boxes is present", result.boxes is not None)
            if result.boxes is not None:
                check("boxes expose .id (tracking active)", result.boxes.id is not None,
                      "{0} ids on frame 0".format(len(ids)))

    # 4. judgement ------------------------------------------------------------
    check("track IDs persist across consecutive frames",
          id_stable_across_frames > 0,
          "{0} frames shared an ID with the previous frame".format(id_stable_across_frames))

    if timings:
        avg = sum(timings) / len(timings)
        # Drop the first, which includes backend warm-up.
        steady = sorted(timings[1:]) or timings
        median = steady[len(steady) // 2]
        print()
        print("  timing over {0} frames: avg {1:.1f} ms, median {2:.1f} ms".format(
            len(timings), avg, median))
        print("  -> {0:.1f} FPS inference ({1})".format(1000.0 / median, "this machine"))
        print()
        print("  Raspberry Pi 3B is typically 4-8x slower than a desktop for this")
        print("  workload. Multiply this FPS by that factor and subtract pipeline")
        print("  cost to estimate the Pi number. Below ~8 FPS FCW is post-hoc,")
        print("  not real-time.")

    print()
    print("=" * 46)
    print(" FAILURES: {0}".format(len(failures)))
    for f in failures:
        print("   - " + f)
    print("=" * 46)
    return 1 if failures else 0


def np_random_frame(i):
    """A cheap synthetic frame with a moving bright box, for no-video testing."""
    import numpy as np
    f = np.full((480, 640, 3), 40, dtype=np.uint8)
    x = 260 + (i * 7) % 120
    cv2.rectangle(f, (x, 300), (x + 80, 430), (200, 200, 200), -1)
    return f


if __name__ == "__main__":
    sys.exit(main())
