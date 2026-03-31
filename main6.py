import cv2
from ultralytics import YOLO

player_model = YOLO("volleyball_detection.pt")

cap = cv2.VideoCapture("sample.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps and fps > 1 else 30

out = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = player_model.track(
        frame,
        persist=True,
        conf=0.4,
        tracker="bytetrack.yaml",
        verbose=False,
        device=0,
        half=False
    )

    annotated_frame = frame.copy()

    for r in results:
        boxes = r.boxes

        if boxes.id is None or len(boxes.id) == 0:
            continue

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy().astype(int)

        for i in range(len(ids)):
            x1, y1, x2, y2 = xyxy[i]
            track_id = ids[i]

            cbx = (x1 + x2) // 2
            cby = (y1 + y2) // 2
            width = x2 - x1

            cv2.ellipse(
                annotated_frame,
                center=(cbx, y2),
                axes=(int(width), int(0.35 * width)),
                angle=0,
                startAngle=-45,
                endAngle=235,
                color=(0, 255, 0),
                thickness=2
            )

            rect_w, rect_h = 40, 20

            x1_rect = cbx - rect_w // 2
            x2_rect = cbx + rect_w // 2
            y1_rect = y2 - rect_h // 2
            y2_rect = y2 + rect_h // 2

            cv2.rectangle(
                annotated_frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                (0, 255, 0),
                cv2.FILLED
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                annotated_frame,
                f"ID: {track_id}",
                (x1_text, y1_rect + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    if out is None:
        h, w = annotated_frame.shape[:2]
        out = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h))

    out.write(annotated_frame)

    cv2.imshow("Volleyball Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()