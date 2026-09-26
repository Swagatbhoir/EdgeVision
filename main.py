"""
main.py — EdgeVision ADAS: Video -> YOLO26 (NCNN) -> 3 ADAS subsystems -> Cockpit HUD.

Pipeline:
    Pre-recorded video file (VIDEO_PATH)
       |
       v
    YOLO26 nano (yolo26n_ncnn_model) via ultralytics model.track(persist=True)
       |
       v
    ADASPipeline (pipeline.py)
        ForwardCollisionWarning  (fcw.py)   — IPM metric distance + scale-change TTC
        FrontCarDepartureWarning(fcdw.py)   — stationary queue lead departure
        LaneDepartureWarning     (ldw.py)   — Hough lane boundaries + offset
        AlertManager (alert_manager.py)      — priority arbitration + acoustics
        HUDRenderer (hud_renderer.py)        — tactical AR cockpit overlay
       |
       +--> TelemetryHub (web_server.py)    — cockpit web dashboard on :5000
       +--> OpenCV window                   — laptop only (SHOW_LOCAL_WINDOW)

Headless Raspberry Pi 3B:
  Nothing here requires a display. With SHOW_LOCAL_WINDOW = False (the default)
  the Flask stream at http://<pi-ip>:5000 is the only view you need, so the Pi
  can run completely headless. Set it True on the laptop if you want the window.
"""

from ultralytics import YOLO
import cv2
import os
import threading
import time

from pipeline import ADASPipeline
from web_server import TelemetryHub, create_cockpit_app


# ==========================================
# CONFIGURATION
# ==========================================

# ------------------------------------------
# INPUT VIDEO
# ------------------------------------------
# THE ONLY THING YOU NORMALLY EDIT.
# Must point at a pre-recorded video file. This run is video-only: there is no
# live camera / webcam / IP-camera source any more.
#VIDEO_PATH = r"D:\project\ADASS\13588981_3840_2160_30fps.mp4"
VIDEO_PATH = r"D:\project\ADASS\Driving in England Exploring the Beautiful Eton Drive  4K UHD Dashcam Footage - SUPERFM.CO.UK (720p, h264).mp4"

# ------------------------------------------
# MODEL
# ------------------------------------------
# NCNN ONLY. There is deliberately no .pt fallback — run scripts/export_ncnn.py
# once on a desktop machine and copy the yolo26n_ncnn_model folder over.
MODEL_NCNN_PATH = "yolo26n_ncnn_model"

# Inference resolution. 320 is the recommended Raspberry Pi 3B setting:
# roughly 2-3x faster than 640 for a modest accuracy loss on small/distant
# objects. Drop to 256 if you need more speed; raise to 480/640 if you care
# more about accuracy than frame rate. Must be a multiple of 32.
MODEL_IMGSZ = 320

# Run inference on every Nth frame, reusing the previous detections in between.
# 1 = every frame (default, and the only setting validated against the tracking
# logic). 2 roughly doubles the loop rate but halves how often the tracker and
# object history update, which can weaken FCW. Measure before enabling.
INFER_EVERY_N_FRAMES = 1

# Show the cv2.imshow window. False = headless (Raspberry Pi). True = laptop.
# The Flask stream works either way; this is only a local convenience window.
SHOW_LOCAL_WINDOW = False

# Bird's-eye (IPM) picture-in-picture inset. Off by default: warping the full
# frame is the single most expensive per-frame operation in this program and
# the Pi 3B cannot afford it at a usable frame rate. Turn it on while
# calibrating or on the laptop, leave it off on the Pi.
SHOW_IPM_PREVIEW = False

# Lane detection cadence (1 = every frame, 2 = every other frame).
LANE_PROCESS_EVERY_N_FRAMES = 1

# Cockpit web server port.
SERVER_PORT = 5000


# ==========================================
# IPM / BIRD'S-EYE CALIBRATION
# ==========================================
# Everything below is a MEASUREMENT, not a guess. See "HOW TO CALIBRATE IPM"
# in the project notes before trusting any distance reading.

# 4 points on flat ground in the ORIGINAL video frame, clockwise from the
# top-left: (far-left, far-right, near-right, near-left).
# NOTE: these are in PIXELS OF YOUR VIDEO'S ACTUAL RESOLUTION, not of some
# nominal 640x480. The program prints the real resolution at startup — measure
# your points on a frame of that exact size.
IPM_SRC_POINTS = [
    (224, 236),   # far-left
    (416, 236),   # far-right
    (512, 430),   # near-right
    (128, 430),   # near-left
]

# Real-world width in metres between the left and right points.
IPM_REAL_WORLD_WIDTH_M = 4.0

# Real-world depth in metres between the far edge and the near edge.
IPM_REAL_WORLD_HEIGHT_M = 10.0

# Size of the top-down image in pixels. Choose it so both axes share the same
# pixels-per-metre, i.e. IPM_DST_SIZE[0] / WIDTH_M == IPM_DST_SIZE[1] / HEIGHT_M.
# The values below give 400/4.0 == 1000/10.0 == 100 px/m — clean, square
# pixels, 1 metre = 100 pixels in the bird's-eye view.
IPM_DST_SIZE = (400, 1000)

# Flip this to True once you have measured IPM_SRC_POINTS and the two real-world
# distances. While it is False the program still runs, but prints a loud
# reminder on every start so you never trust a placeholder number by accident.
IPM_CALIBRATED = False

IPM_CONFIG = {
    "src_points": IPM_SRC_POINTS,
    "real_world_width_m": IPM_REAL_WORLD_WIDTH_M,
    "real_world_height_m": IPM_REAL_WORLD_HEIGHT_M,
    "dst_size": IPM_DST_SIZE,
}


# ==========================================
# COCKPIT WEB SERVER (Phone 2 / car monitor)
# ==========================================

hub = TelemetryHub()
app = create_cockpit_app(hub)


def start_server(port=SERVER_PORT):
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)


threading.Thread(target=start_server, daemon=True).start()


# ==========================================
# LOAD YOLO26 MODEL (NCNN)
# ==========================================

if not os.path.isdir(MODEL_NCNN_PATH):
    print()
    print("==========================================")
    print(" ERROR: NCNN MODEL NOT FOUND")
    print("==========================================")
    print("Expected folder: {0}".format(MODEL_NCNN_PATH))
    print()
    print("There is no .pt fallback by design. Export it once on a")
    print("desktop machine (not on the Pi):")
    print()
    print("    python scripts/export_ncnn.py")
    print()
    print("or manually:")
    print()
    print("    from ultralytics import YOLO")
    print("    YOLO('yolo26n.pt').export(format='ncnn', imgsz={0})".format(MODEL_IMGSZ))
    print()
    print("Then copy the yolo26n_ncnn_model folder next to main.py.")
    print("==========================================")
    print()
    raise SystemExit(1)

model = YOLO(MODEL_NCNN_PATH)


# ==========================================
# INPUT VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video file:", VIDEO_PATH)
    print("Check the VIDEO_PATH is correct.")
    raise SystemExit(1)

# The IPM source points are in source-frame pixels, so we need the video's
# REAL resolution, not a nominal one. Read it and use it everywhere.
VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480


# ==========================================
# ADAS PIPELINE
# ==========================================

pipeline = ADASPipeline(
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    enable_lane_detection=True,
    ipm_config=IPM_CONFIG,
)
pipeline.lane_process_every = LANE_PROCESS_EVERY_N_FRAMES


# ==========================================
# BIRD'S-EYE INSET (optional, expensive)
# ==========================================

# Fraction of the frame width taken by the inset panel.
IPM_INSET_WIDTH_FRACTION = 0.20


def draw_ipm_inset(frame, ipm, objects):
    """
    Composite a bird's-eye view of the road into a corner of `frame`.

    Draws the warped road plus one dot per tracked object, positioned by its
    calibrated distance. Intended for CALIBRATION, not for driving: it runs
    cv2.warpPerspective over the whole frame every frame, which is the most
    expensive operation in this program and is why SHOW_IPM_PREVIEW defaults
    to False. On the Pi 3B, expect this to cost more than the detector.

    frame:    BGR image, modified in place.
    ipm:      an IPMDistanceEstimator.
    objects:  telemetry["objects"] — each with box / distance_m / lateral_offset_m.
    """
    bev = ipm.warp_birdseye(frame)

    # Scale the tall bird's-eye strip down to a corner-sized panel.
    inset_w = max(1, int(frame.shape[1] * IPM_INSET_WIDTH_FRACTION))
    inset_h = max(1, int(ipm.dst_h * (inset_w / float(ipm.dst_w))))
    if inset_h > frame.shape[0]:
        inset_h = frame.shape[0]
        inset_w = max(1, int(inset_h * (ipm.dst_w / float(ipm.dst_h))))
    bev = cv2.resize(bev, (inset_w, inset_h), interpolation=cv2.INTER_AREA)

    # Object ground-point dots, colour-coded by risk.
    risk_color = {
        "DANGER": (0, 0, 255), "WARNING": (0, 140, 255),
        "CAUTION": (0, 230, 255), "SAFE": (200, 200, 200),
    }
    for obj in objects:
        metres = ipm.metres_to_birdseye(obj.get("lateral_offset_m"), obj.get("distance_m"))
        if metres is None:
            continue
        px, py = metres
        cx = int(px * (inset_w / float(ipm.dst_w)))
        cy = int(py * (inset_h / float(ipm.dst_h)))
        if 0 <= cx < inset_w and 0 <= cy < inset_h:
            cv2.circle(bev, (cx, cy), 3, risk_color.get(obj.get("risk"), (255, 255, 255)), -1)

    cv2.rectangle(bev, (0, 0), (inset_w - 1, inset_h - 1), (255, 255, 255), 1)

    # Composite into the top-right corner, directly on the frame.
    y_off = 0
    x_off = frame.shape[1] - inset_w
    frame[y_off:y_off + inset_h, x_off:x_off + inset_w] = bev


# ==========================================
# STARTUP BANNER
# ==========================================

print()
print("==========================================")
print(" EDGEVISION ADAS — COCKPIT MONITOR")
print("==========================================")
print("Video:       {0}".format(VIDEO_PATH))
print("Resolution:  {0} x {1}   <-- measure IPM points at this size".format(
    VIDEO_WIDTH, VIDEO_HEIGHT))
print("Model:       {0} (ncnn) @ imgsz={1}".format(MODEL_NCNN_PATH, MODEL_IMGSZ))
print("Infer every: {0} frame(s)".format(INFER_EVERY_N_FRAMES))
print("Local window:{0}".format("ON" if SHOW_LOCAL_WINDOW else "OFF (headless)"))
print("IPM preview: {0}".format("ON" if SHOW_IPM_PREVIEW else "OFF"))
print("IPM calib:   {0}".format(pipeline.fcw.ipm.describe()))
print("Dashboard:   http://127.0.0.1:{0}".format(SERVER_PORT))
print("Phone 2:     http://<your-pc-ip>:{0}".format(SERVER_PORT))
print("==========================================")
print()

if not IPM_CALIBRATED:
    print("##########################################")
    print("#  WARNING: IPM IS NOT CALIBRATED        #")
    print("#  The 4 points and metre values above   #")
    print("#  are PLACEHOLDERS. Distances, the BEV   #")
    print("#  radar and the object table are NOT     #")
    print("#  trustworthy until you measure them.    #")
    print("#  See HOW TO CALIBRATE IPM, then set     #")
    print("#  IPM_CALIBRATED = True.                 #")
    print("##########################################")
    print()


# ==========================================
# MAIN LOOP
# ==========================================

frame_index = 0
last_detections = []
last_inference_ms = 0.0

while True:
    ret, frame = cap.read()

    if not ret:
        # Pre-recorded video: rewind and keep going.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read video frame.")
            break

    now = time.time()

    # ---- YOLO detection + tracking -------------------------------------
    # At INFER_EVERY_N_FRAMES == 1 this runs on every frame, exactly as
    # before. Above 1, the previous frame's detections are reused.
    if INFER_EVERY_N_FRAMES <= 1 or (frame_index % INFER_EVERY_N_FRAMES) == 0:
        infer_started = time.perf_counter()
        results = model.track(frame, persist=True, verbose=False, imgsz=MODEL_IMGSZ)
        last_inference_ms = (time.perf_counter() - infer_started) * 1000.0

        detections = []
        result = results[0]
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().numpy()

            for box, track_id, class_id, confidence in zip(boxes, track_ids, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    "track_id": track_id,
                    "class_name": model.names[class_id],
                    "confidence": float(confidence),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
        last_detections = detections

    # ---- ADAS subsystems + tactical HUD (paints `frame` in place) ------
    telemetry = pipeline.process_frame(
        frame,
        detections=last_detections,
        inference_ms=last_inference_ms,
        now=now,
        camera_source="FILE",
    )

    # ---- Push cockpit settings and HUD layer toggles from the dashboard -
    pipeline.apply_settings(hub.get_settings())
    pipeline.set_hud_layers(hub.get_hud_layers())

    # ---- Optional bird's-eye inset (full-frame warp; off on the Pi) ------
    # Runs before publishing so the Flask stream also carries the inset.
    if SHOW_IPM_PREVIEW:
        draw_ipm_inset(frame, pipeline.fcw.ipm, telemetry.get("objects", []))

    # ---- Publish to the cockpit web layer ------------------------------
    hub.publish_frame(frame)
    hub.publish_telemetry(telemetry)

    # ---- Display window (laptop only; guarded for headless Pi) ---------
    if SHOW_LOCAL_WINDOW:
        cv2.imshow("EdgeVision ADAS - Cockpit HUD", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    frame_index += 1


# ==========================================
# SHUTDOWN
# ==========================================

cap.release()
if SHOW_LOCAL_WINDOW:
    cv2.destroyAllWindows()

print()
print("ADAS system stopped.")
