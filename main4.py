import cv2
import numpy as np
from ultralytics import YOLO

player_model = YOLO("volleyball_detection.pt")
ball_model = YOLO("ball.pt")

cap = cv2.VideoCapture("sample.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

# -------- BALL STATE --------
last_box = None
last_center = None
velocity = np.array([0.0, 0.0])
miss_count = 0
MAX_MISS = 10   # increased memory

alpha = 0.7  # smoothing factor

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # -------- PLAYER TRACKING --------
    player_results = player_model.track(
        frame,
        persist=True,
        conf=0.4,
        tracker="bytetrack.yaml",
        verbose=False
    )

    annotated = player_results[0].plot()

    # -------- BALL DETECTION --------
    results = ball_model.predict(
        frame,
        conf=0.05,
        imgsz=960,
        verbose=False
    )

    detected = False
    current_center = None

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            detected = True

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            current_center = np.array([cx, cy])

            # velocity update
            if last_center is not None:
                velocity = 0.7 * velocity + 0.3 * (current_center - last_center)

            last_center = current_center
            last_box = (x1, y1, x2, y2, conf)
            miss_count = 0

    # -------- PREDICTION WHEN LOST --------
    if not detected and last_center is not None:
        miss_count += 1

        # predict next position
        predicted_center = last_center + velocity * miss_count

        size = 20  # approximate ball size

        x1 = int(predicted_center[0] - size)
        y1 = int(predicted_center[1] - size)
        x2 = int(predicted_center[0] + size)
        y2 = int(predicted_center[1] + size)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            "BALL (PRED)",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    elif detected and last_box is not None:
        x1, y1, x2, y2, conf = last_box

        # smoothing (reduces jitter)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"BALL {conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    # -------- RESET IF LOST TOO LONG --------
    if miss_count > MAX_MISS:
        last_center = None
        velocity = np.array([0.0, 0.0])
        last_box = None

    # -------- SAVE --------
    if out is None:
        h, w = annotated.shape[:2]
        out = cv2.VideoWriter("output.mp4", fourcc, 30, (w, h))

    out.write(annotated)

    cv2.imshow("Volleyball Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()