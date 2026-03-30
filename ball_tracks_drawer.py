import cv2
import numpy as np


class BallTracksDrawer:
    def __init__(self, model):
        self.model = model

        # Kalman Filter
        self.kalman = cv2.KalmanFilter(4, 2)

        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False
        self.miss_count = 0
        self.MAX_MISS = 15

    def draw(self, frame):

        results = self.model.predict(
            frame,
            conf=0.05,
            imgsz=960,
            verbose=False
        )

        detected = False
        measurement = None

        annotated = frame.copy()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                detected = True

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, "BALL", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Kalman update
        if detected:
            if not self.initialized:
                self.kalman.statePost = np.array([
                    [measurement[0][0]],
                    [measurement[1][0]],
                    [0],
                    [0]
                ], dtype=np.float32)
                self.initialized = True

            self.kalman.correct(measurement)
            self.miss_count = 0
        else:
            self.miss_count += 1

        # prediction
        prediction = self.kalman.predict()
        px, py = int(prediction[0]), int(prediction[1])

        if self.initialized:
            cv2.circle(annotated, (px, py), 8, (0, 0, 255), 2)
            cv2.putText(annotated, "BALL-KF", (px + 5, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if self.miss_count > self.MAX_MISS:
            self.initialized = False

        return annotated