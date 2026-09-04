from ultralytics import YOLO
import cv2
import time

# Load YOLO26 model
model = YOLO("yolo26n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Store object history
object_history = {}

# --------------------------------------------------
# DISTANCE CALIBRATION
# --------------------------------------------------
# Example only.
# You MUST calibrate this value with your own camera.
# distance = K / object_height_in_pixels
K = 1000.0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera")
        break

    current_time = time.time()

    # Track all objects
    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    result = results[0]

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().numpy()

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

            # Bounding box size
            width = x2 - x1
            height = y2 - y1

            object_name = model.names[class_id]

            # Create history for new object
            if track_id not in object_history:
                object_history[track_id] = []

            # Save:
            # time, center_x, center_y, width, height
            object_history[track_id].append(
                (
                    current_time,
                    center_x,
                    center_y,
                    width,
                    height
                )
            )

            # Keep last 10 measurements
            if len(object_history[track_id]) > 10:
                object_history[track_id].pop(0)

            movement = "Stationary"
            approaching = False
            distance = None
            ttc = None
            risk = "SAFE"

            history = object_history[track_id]

            # ------------------------------------------
            # MOVEMENT DETECTION
            # ------------------------------------------
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

            # ------------------------------------------
            # APPROACHING DETECTION
            # ------------------------------------------
            if len(history) >= 5:

                old_height = history[-5][4]

                if old_height > 0:

                    height_change = (
                        (height - old_height)
                        / old_height
                    ) * 100

                    if height_change > 15:
                        approaching = True

            # ------------------------------------------
            # APPROXIMATE DISTANCE
            # ------------------------------------------
            if height > 0:

                distance = K / height

            # ------------------------------------------
            # TIME TO COLLISION
            # ------------------------------------------
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

                        # Prototype risk thresholds
                        if ttc < 1.5:

                            risk = "DANGER"

                        elif ttc < 3.0:

                            risk = "WARNING"

                        else:

                            risk = "SAFE"

            # ------------------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            # ------------------------------------------
            # DISPLAY INFORMATION
            # ------------------------------------------

            text = f"ID {track_id} | {object_name}"

            cv2.putText(
                frame,
                text,
                (x1, y1 - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                movement,
                (x1, y1 - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            if distance is not None:

                distance_text = f"Dist: {distance:.1f} m"

                cv2.putText(
                    frame,
                    distance_text,
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

            if ttc is not None:

                ttc_text = f"TTC: {ttc:.1f}s"

                cv2.putText(
                    frame,
                    ttc_text,
                    (x1, y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

            # ------------------------------------------
            # RISK
            # ------------------------------------------

            cv2.putText(
                frame,
                risk,
                (x1, y2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Approaching message
            if approaching:

                cv2.putText(
                    frame,
                    "APPROACHING",
                    (x1, y2 + 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

    # Show frame
    cv2.imshow("ADAS - YOLO26", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()