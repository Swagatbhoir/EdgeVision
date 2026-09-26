"""
export_ncnn.py — one-time NCNN export for the Raspberry Pi 3B.

Run this ONCE on a desktop machine (not on the Pi — it needs PyTorch, PNNX and
a one-off model download), then copy the resulting `yolo26n_ncnn_model/`
folder onto the Pi next to main.py.

    python scripts/export_ncnn.py

Why NCNN: it runs on ARM NEON without PyTorch's CUDA/ROCm machinery and is
roughly 5-10x faster than the .pt model on a Pi 3B. Note that ultralytics
still imports torch for pre/post-processing even with the ncnn backend, so
torch remains a dependency — but the heavy convolutions run in native C++.

The export bakes in an inference resolution. It should MATCH MODEL_IMGSZ in
main.py, otherwise you pay for a resize on every frame inside the runtime.
"""

import os
import sys

# Ultralytics wants to be able to write a config dir on first use.
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.path.expanduser("~"), ".config", "Ultralytics"))

from ultralytics import YOLO

# Keep in sync with MODEL_IMGSZ in main.py.
EXPORT_IMGSZ = 320

WEIGHTS = "yolo26n.pt"
OUT_DIR = "yolo26n_ncnn_model"


def main():
    print("=" * 44)
    print(" EdgeVision ADAS — NCNN one-time export")
    print("=" * 44)
    print(" weights : {0}".format(WEIGHTS))
    print(" imgsz   : {0}   (must match MODEL_IMGSZ in main.py)".format(EXPORT_IMGSZ))
    print(" output  : {0}/".format(OUT_DIR))
    print("=" * 44)
    print()

    model = YOLO(WEIGHTS)
    model.export(format="ncnn", imgsz=EXPORT_IMGSZ)

    print()
    print("=" * 44)
    print(" DONE — copy the {0}/ folder to the Pi".format(OUT_DIR))
    print(" It must sit next to main.py.")
    print("=" * 44)

    if not os.path.isdir(OUT_DIR):
        print()
        print("NOTE: the {0}/ folder was not found in the current".format(OUT_DIR))
        print("working directory. Look for it under runs/{0}/ncnn/".format(WEIGHTS[:-3]))
        print("and copy that folder across, renaming it to {0}.".format(OUT_DIR))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
