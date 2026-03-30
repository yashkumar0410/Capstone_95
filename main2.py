import cv2
from ultralytics import YOLO

# Load models
player_model = YOLO("volleyball_detection.pt")
ball_model = YOLO("ball.pt")

# Open video (use 0 for webcam)
cap = cv2.VideoCapture("sample.mp4")

# Optional: save output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # -------- PLAYER TRACKING --------
    player_results = player_model.track(
        frame,
        persist=True,
        conf=0.4,
        tracker="bytetrack.yaml"
    )

    annotated_frame = player_results[0].plot()

    # -------- BALL DETECTION --------
    ball_results = ball_model.predict(
        frame,
        conf=0.15   # low because ball is small
    )

    # Draw BALL manually
    for r in ball_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # RED box for ball
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated_frame,
                f"BALL {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

    # Initialize writer once
    if out is None:
        h, w = annotated_frame.shape[:2]
        out = cv2.VideoWriter("output.mp4", fourcc, 30, (w, h))

    out.write(annotated_frame)

    # Show (works ONLY outside Jupyter)
    cv2.imshow("Volleyball Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()