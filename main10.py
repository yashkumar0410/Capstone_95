import cv2
import torch
from ultralytics import YOLO
import supervision as sv

player_model = YOLO("volleyball_detection.pt")
court_model = YOLO("court_keypoints.pt")

cap = cv2.VideoCapture("sample.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps and fps > 1 else 30

device = 0 if torch.cuda.is_available() else "cpu"

out = None

court_image = cv2.imread("court.png")
court_w, court_h = 300, 150
court_image = cv2.resize(court_image, (court_w, court_h))
alpha = 0.6
margin = 20
actual_width = 18
actual_height = 9

key_points = [
    # left edge
    (0,0),
    (0,int(court_h)),

    # middle line
    (int(court_w/2), court_h),
    (int(court_w/2), 0),

    # left attack line
    (int((9/actual_width)*court_w), court_h),
    (int((9/actual_width)*court_w),0),

    # right edge
    (court_w, int(court_h)),
    (court_w, 0),

    # right attack line
    (int((12/actual_width)*court_w), court_h),
    (int((12/actual_width)*court_w),0),

]


frame_idx = 0
kps = None  # FIX: avoid undefined crash

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = frame.copy()

 
    if frame_idx % 5 == 0:
        court_result = court_model.predict(frame, conf=0.5, verbose=False)[0]
        if court_result.keypoints is not None:
            kps = court_result.keypoints.xy[0].cpu().numpy()

   
    if kps is not None:
        for i, (x, y) in enumerate(kps):
            cv2.circle(annotated_frame, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(
                annotated_frame,
                str(i),
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )


    results = player_model.track(
        frame,
        persist=True,
        conf=0.4,
        tracker="bytetrack.yaml",
        verbose=False,
        device=device,
        half=True,
    )

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
            width = x2 - x1

            cv2.ellipse(
                annotated_frame,
                center=(cbx, y2),
                axes=(int(width), int(0.35 * width)),
                angle=0,
                startAngle=-45,
                endAngle=235,
                color=(0, 255, 0),
                thickness=2,
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
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                annotated_frame,
                f"{track_id}",
                (x1_text, y1_rect + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    # ===== MINI COURT OVERLAY (TOP RIGHT) =====
    h_frame, w_frame = annotated_frame.shape[:2]

    y1 = margin 
    y2 = y1 + court_h

    x2 = w_frame - margin
    x1 = x2 - court_w

    if y2 < h_frame and x2 < w_frame:
        overlay = annotated_frame[y1:y2, x1:x2].copy()

        cv2.addWeighted(
            court_image, alpha,
            overlay, 1 - alpha,
            0,
            annotated_frame[y1:y2, x1:x2]
        )

    for (x, y) in key_points:
        cv2.circle(
            annotated_frame,
            (x1 + x, y1 + y),
            3,
            (0, 0, 255),
            -1
        )

        
    if out is None:
        h, w = annotated_frame.shape[:2]
        out = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h))

    out.write(annotated_frame)

    cv2.imshow("Volleyball Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()