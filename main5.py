import cv2
import numpy as np
from ultralytics import YOLO

from player_tracks_drawer import PlayerTracksDrawer
from ball_tracks_drawer import BallTracksDrawer

# ---------------- MODELS ----------------
player_model = YOLO("volleyball_detection.pt")
ball_model = YOLO("ball.pt")

cap = cv2.VideoCapture("sample.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

# ---------------- DRAWERS ----------------
player_drawer = PlayerTracksDrawer()
ball_drawer = BallTracksDrawer(ball_model)

# ---------------- LOOP ----------------
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

    # -------- BALL + KALMAN --------
    annotated = ball_drawer.draw(annotated)

    # -------- SAVE VIDEO --------
    if out is None:
        h, w = annotated.shape[:2]
        out = cv2.VideoWriter("output.mp4", fourcc, 30, (w, h))

    out.write(annotated)

    # -------- DISPLAY --------
    cv2.imshow("Volleyball Tracking System", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()