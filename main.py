import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("volleyball_detection.pt")  # your .pt file

# Open video (or 0 for webcam)
cap = cv2.VideoCapture("sample.mp4")  # change to your video path

# Optional: reduce resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO tracking (THIS is the key for real-time tracking)
    results = model.track(
        frame,
        persist=True,      # keeps IDs across frames
        conf=0.4,
        iou=0.5,
        device=0           # GPU
    )

    # Draw results
    annotated_frame = results[0].plot()

    # Show FPS (optional but useful)
    cv2.imshow("Volleyball Detection + Tracking", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()