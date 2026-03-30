import cv2
from ultralytics import YOLO

player_model = YOLO("volleyball_detection.pt")
ball_model = YOLO("ball.pt")

cap = cv2.VideoCapture("sample.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

# 👉 BALL MEMORY
last_ball = None
miss_count = 0
MAX_MISS = 5

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

    annotated_frame = player_results[0].plot()

    # -------- BALL DETECTION --------
    ball_results = ball_model.predict(
        frame,
        conf=0.05,   # IMPORTANT: lower threshold
        imgsz=960,
        verbose=False
    )

    detected = False

    for r in ball_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            last_ball = (x1, y1, x2, y2, conf)
            miss_count = 0
            detected = True

    # -------- MEMORY LOGIC --------
    if not detected:
        miss_count += 1

    # draw last known ball if missing
    if last_ball is not None and miss_count <= MAX_MISS:
        x1, y1, x2, y2, conf = last_ball

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated_frame,
            f"BALL {conf:.2f} (MEM)",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    # reset memory if lost too long
    if miss_count > MAX_MISS:
        last_ball = None

    # -------- SAVE --------
    if out is None:
        h, w = annotated_frame.shape[:2]
        out = cv2.VideoWriter("output.mp4", fourcc, 30, (w, h))

    out.write(annotated_frame)

    cv2.imshow("Volleyball Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()