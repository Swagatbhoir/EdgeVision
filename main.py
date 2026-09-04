from ultralytics import YOLO
import cv2
import time
import winsound

# ==========================================
# LOAD YOLO26 MODEL
# ==========================================

model = YOLO("yolo26n.pt")

# ==========================================
# OPEN CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

# ==========================================
# OBJECT HISTORY
# ==========================================

object_history = {}

# ==========================================
# DISTANCE CALIBRATION
# ==========================================

# Temporary value.
# We will calibrate this later using your camera.
K = 1000.0

# ==========================================
# ALERT SETTINGS
# ==========================================

last_alert_time = 0
ALERT_COOLDOWN = 2.0


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
    # CHECK WHETHER OBJECTS ARE DETECTED
    # ======================================

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().numpy()

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

            # Object center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Bounding box dimensions
            width = x2 - x1
            height = y2 - y1

            # Object name
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
            # DEFAULT VALUES
            # ==================================

            movement = "Stationary"
            approaching = False
            distance = None
            ttc = None
            risk = "SAFE"

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
            # APPROACHING DETECTION
            # ==================================

            if len(history) >= 5:

                old_height = history[-5][4]

                if old_height > 0:

                    height_change = (
                        (height - old_height)
                        / old_height
                    ) * 100

                    if height_change > 15:

                        approaching = True

            # ==================================
            # APPROXIMATE DISTANCE
            # ==================================

            if height > 0:

                distance = K / height

            # ==================================
            # TIME TO COLLISION (TTC)
            # ==================================

            if len(history) >= 5:

                old_time = history[-5][0]
                old_height = history[-5][4]

                old_distance = K / max(old_height, 1)

                time_difference = current_time - old_time

                if time_difference > 0:

                    closing_speed = (
                        old_distance - distance
                    ) / time_difference

                    # Object is getting closer
                    if closing_speed > 0:

                        ttc = distance / closing_speed

                        # ==================================
                        # RISK CLASSIFICATION
                        # ==================================

                        if ttc < 1.5:

                            risk = "DANGER"

                        elif ttc < 3.0:

                            risk = "WARNING"

                        else:

                            risk = "SAFE"

            # ==================================
            # DRIVER ALERT
            # ==================================

            if risk == "DANGER":

                if current_time - last_alert_time > ALERT_COOLDOWN:

                    winsound.Beep(1000, 500)

                    last_alert_time = current_time

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
                (x1, max(y1 - 50, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Movement

            cv2.putText(
                frame,
                movement,
                (x1, max(y1 - 30, 40)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255), 
                2
            )

            # ==================================
            # DISTANCE
            # ==================================

            if distance is not None:

                cv2.putText(
                    frame,
                    f"Distance: {distance:.1f} m",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

            # ==================================
            # TTC
            # ==================================

            if ttc is not None:

                cv2.putText(
                    frame,
                    f"TTC: {ttc:.1f} sec",
                    (x1, y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

            # ==================================
            # RISK LEVEL
            # ==================================

            cv2.putText(
                frame,
                f"Risk: {risk}",
                (x1, y2 + 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # ==================================
            # APPROACHING
            # ==================================

            if approaching:

                cv2.putText(
                    frame,
                    "APPROACHING",
                    (x1, y2 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

    # ======================================
    # BIG SCREEN WARNING
    # ======================================

    danger_detected = False

    if result.boxes.id is not None:

        # Check all current objects
        for box, track_id, class_id in zip(
            boxes,
            track_ids,
            classes
        ):

            history = object_history.get(track_id, [])

            if len(history) >= 5:

                old_height = history[-5][4]
                current_height = history[-1][4]

                if old_height > 0:

                    old_distance = K / old_height
                    current_distance = K / current_height

                    old_time = history[-5][0]

                    dt = current_time - old_time

                    if dt > 0:

                        closing_speed = (
                            old_distance - current_distance
                        ) / dt

                        if closing_speed > 0:

                            current_ttc = (
                                current_distance
                                / closing_speed
                            )

                            if current_ttc < 1.5:

                                danger_detected = True

    # ======================================
    # DISPLAY WARNING
    # ======================================

    if danger_detected:

        cv2.putText(
            frame,
            "!!! COLLISION WARNING !!!",
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3
        )

    # ======================================
    # DISPLAY WINDOW
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

print("ADAS system stopped.")