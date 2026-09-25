from ultralytics import YOLO
import cv2
import time
import winsound
import threading
from flask import Flask, Response

from fcw import ForwardCollisionWarning
from fcdw import FrontCarDepartureWarning
from ldw import LaneDepartureWarning
from alert_manager import AlertManager


# ==========================================
# PHONE 2 WEB STREAM SERVER
# ==========================================

app = Flask(__name__)

output_frame = None
frame_lock = threading.Lock()


@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ADAS Dashboard</title>

        <style>
            body {
                margin: 0;
                padding: 0;
                background: black;
                overflow: hidden;
            }

            img {
                width: 100vw;
                height: 100vh;
                object-fit: contain;
            }
        </style>
    </head>

    <body>
        <img src="/video_feed">
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():

    def generate():

        while True:

            with frame_lock:

                if output_frame is None:
                    continue

                success, encoded_image = cv2.imencode(
                    ".jpg",
                    output_frame
                )

            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encoded_image.tobytes()
                + b"\r\n"
            )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def start_server():

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )


# Start Phone 2 server in background
server_thread = threading.Thread(
    target=start_server,
    daemon=True
)

server_thread.start()

print()
print("==========================================")
print(" PHONE 2 ADAS MONITOR")
print("==========================================")
print("Open this on Phone 2:")
print("http://192.168.0.106:5000")
print("==========================================")
print()


# ==========================================
# LOAD YOLO26 MODEL
# ==========================================

model = YOLO("yolo26n.pt")


# ==========================================
# OPEN PHONE 1 CAMERA
# ==========================================

#PHONE_CAMERA_URL = "http://192.168.0.102:8080/video"

cap = cv2.VideoCapture(0)  # Use the default camera (0) for testing
'''
if not cap.isOpened():

    print("ERROR: Could not connect to Phone 1 camera.")
    print()
    print("Check:")
    print("1. Phone 1 IP address")
    print("2. IP Webcam server is running")
    print("3. Laptop and Phone 1 are on the same Wi-Fi")
    exit()


print("Phone 1 camera connected successfully!")
print("YOLO26 ADAS processing started.")
print()
'''

# ==========================================
# OBJECT HISTORY
# ==========================================

object_history = {}


# ==========================================
# ADAS SUB-SYSTEM ENGINES (initialized on
# first frame when frame dimensions are known)
# ==========================================

fcw_engine = None
fcdw_engine = None
ldw_engine = None
alert_mgr = AlertManager(alert_cooldown=2.0)


# ==========================================
# FPS MONITORING
# ==========================================

prev_frame_time = None
fps = 0.0


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read camera frame.")
        break


    current_time = time.time()


    # ======================================
    # YOLO OBJECT TRACKING
    # ======================================

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    result = results[0]


    # ======================================
    # FPS COUNTER (smoothed)
    # ======================================

    global_fps = fps

    if prev_frame_time is not None:

        dt_frame = current_time - prev_frame_time

        if dt_frame > 0:

            instant_fps = 1.0 / dt_frame

            global_fps = 0.9 * global_fps + 0.1 * instant_fps

    fps = global_fps

    prev_frame_time = current_time


    # ======================================
    # DETECTIONS LIST FOR THE FCW ENGINE
    # ======================================

    detections = []


    # ======================================
    # CHECK WHETHER OBJECTS ARE DETECTED
    # ======================================

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        classes = (
            result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )


        # ==================================
        # PROCESS EACH OBJECT
        # ==================================

        for box, track_id, class_id, confidence in zip(
            boxes,
            track_ids,
            classes,
            confidences
        ):

            x1, y1, x2, y2 = map(int, box)


            # ==================================
            # OBJECT CENTER
            # ==================================

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)


            # ==================================
            # BOUNDING BOX DIMENSIONS
            # ==================================

            width = x2 - x1
            height = y2 - y1


            # ==================================
            # OBJECT NAME
            # ==================================

            object_name = model.names[class_id]


            # ==================================
            # CREATE HISTORY
            # ==================================

            if track_id not in object_history:

                object_history[track_id] = []


            object_history[track_id].append(
                (
                    current_time,
                    center_x,
                    center_y,
                    width,
                    height
                )
            )


            # Keep only last 10 measurements

            if len(object_history[track_id]) > 10:

                object_history[track_id].pop(0)


            history = object_history[track_id]


            # ==================================
            # DEFAULT MOVEMENT LABEL
            # ==================================

            movement = "Stationary"


            # ==================================
            # MOVEMENT DETECTION
            # ==================================

            if len(history) >= 2:

                previous = history[-2]

                previous_x = previous[1]
                previous_y = previous[2]


                movement_x = center_x - previous_x
                movement_y = center_y - previous_y


                if abs(movement_x) < 5 and abs(movement_y) < 5:

                    movement = "Stationary"


                elif abs(movement_x) > abs(movement_y):

                    if movement_x > 0:

                        movement = "Moving Right"

                    else:

                        movement = "Moving Left"


                else:

                    if movement_y > 0:

                        movement = "Moving Down"

                    else:

                        movement = "Moving Up"


            # ==================================
            # COLLECT FOR FCW
            # ==================================

            # The FCW engine applies its own filters (driving corridor,
            # horizon, relevant classes), so we simply pass every tracked
            # object on to it.
            detections.append(
                {
                    "track_id": track_id,
                    "class_name": object_name,
                    "confidence": float(confidence),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )


            # ==================================
            # DRAW OBJECT BOX
            # ==================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )


            # ==================================
            # OBJECT INFORMATION
            # ==================================

            cv2.putText(
                frame,
                f"ID {track_id} | {object_name}",
                (
                    x1,
                    max(y1 - 50, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            # ==================================
            # MOVEMENT
            # ==================================

            cv2.putText(
                frame,
                movement,
                (
                    x1,
                    max(y1 - 30, 40)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )


    # ======================================
    # FORGET STALE OBJECT HISTORIES
    # ======================================

    # Track ids accumulate forever otherwise; drop histories that were not
    # seen for more than 5 seconds so memory stays bounded.
    object_history = {
        tid: hist
        for tid, hist in object_history.items()
        if current_time - hist[-1][0] < 5.0
    }


    # ======================================
    # INITIALIZE ADAS ENGINES ON FIRST FRAME
    # ======================================

    h_frame, w_frame = frame.shape[:2]

    if fcw_engine is None:

        fcw_engine = ForwardCollisionWarning(w_frame, h_frame)
        fcdw_engine = FrontCarDepartureWarning(w_frame, h_frame)
        ldw_engine = LaneDepartureWarning(w_frame, h_frame)

        print(f"[ADAS] All sub-systems initialized for {w_frame}x{h_frame} stream.")


    # ======================================
    # 1. LANE DEPARTURE WARNING (LDW) PIPELINE
    # ======================================

    ldw_engine.update(frame, current_time)

    # Draw lane boundary corridor & lines overlay
    frame = ldw_engine.draw_overlay(frame)


    # ======================================
    # 2. FORWARD COLLISION WARNING (FCW)
    # ======================================

    fcw_engine.update(detections, current_time)
    fcw_level = fcw_engine.level       # "SAFE" | "CAUTION" | "WARNING" | "DANGER"
    fcw_primary = fcw_engine.primary    # dict for the closest hazard or None


    # ======================================
    # 3. FRONT CAR DEPARTURE WARNING (FCDW)
    # ======================================

    fcdw_engine.update(detections, current_time)
    fcdw_active = fcdw_engine.alert_active
    fcdw_message = fcdw_engine.alert_message


    # ======================================
    # 4. UNIFIED ALERT ARBITRATION & AUDIO
    # ======================================

    active_alert = alert_mgr.update(
        fcw_level=fcw_level,
        fcdw_active=fcdw_active,
        fcdw_message=fcdw_message,
        ldw_active=ldw_engine.alert_active,
        ldw_message=ldw_engine.alert_message,
        ldw_dir=ldw_engine.departure_direction,
        now=current_time,
    )


    # ======================================
    # 5. FCW PRIMARY TARGET HIGHLIGHT
    # ======================================

    if fcw_primary is not None:

        bx1, by1, bx2, by2 = fcw_primary["box"]

        colour_map = {
            "CAUTION":  (0, 220, 255),
            "WARNING":  (0, 130, 255),
            "DANGER":   (0,   0, 255),
        }
        box_colour = colour_map.get(fcw_level, (255, 255, 255))

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_colour, 3)

        tau = fcw_primary.get("tau")
        tau_text = f"tau: {tau:.1f}s" if tau is not None else "tau: ---"

        cv2.putText(
            frame,
            tau_text,
            (bx1, by2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_colour,
            2,
        )

        h_pct = int(fcw_primary["h_frac"] * 100)
        cv2.putText(
            frame,
            f"prox: {h_pct}%",
            (bx1, by2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_colour,
            2,
        )


    # ======================================
    # 6. UNIFIED ADAS DRIVER ALERT BANNER
    # ======================================

    if active_alert is not None:

        msg = active_alert["message"]
        sub = active_alert.get("subtext", "")
        bgr = active_alert["color_bgr"]

        # Background banner rectangle at top of screen for contrast
        cv2.rectangle(frame, (20, 15), (w_frame - 20, 80), (20, 20, 20), -1)
        cv2.rectangle(frame, (20, 15), (w_frame - 20, 80), bgr, 2)

        cv2.putText(
            frame,
            msg,
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            bgr,
            2,
        )

        if sub:
            cv2.putText(
                frame,
                sub,
                (40, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
            )


    # ======================================
    # 7. ADAS MULTI-SYSTEM STATUS BAR (Bottom)
    # ======================================

    status_bg_y = h_frame - 35
    cv2.rectangle(frame, (0, status_bg_y), (w_frame, h_frame), (15, 15, 15), -1)

    ldw_str = ldw_engine.status if ldw_engine.status != "NORMAL" else "TRACKING" if ldw_engine.lanes_detected else "SEARCHING"
    status_line = f"FCW: {fcw_level}  |  FCDW: {fcdw_engine.state}  |  LDW: {ldw_str}"

    cv2.putText(
        frame,
        status_line,
        (15, h_frame - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )


    # ======================================
    # 8. FPS OVERLAY (Top Right)
    # ======================================

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w_frame - 110, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 120),
        2,
    )


    # ======================================
    # SEND PROCESSED FRAME TO PHONE 2
    # ======================================

    with frame_lock:

        output_frame = frame.copy()


    # ======================================
    # DISPLAY WINDOW ON LAPTOP
    # ======================================

    cv2.imshow(
        "ADAS - YOLO26 Driver Alert System",
        frame
    )


    # ======================================
    # PRESS Q TO EXIT
    # ======================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ==========================================
# RELEASE CAMERA
# ==========================================

cap.release()

cv2.destroyAllWindows()


print()
print("ADAS system stopped.")