import cv2


def draw_ellipse(frame, bbox, color, track_id):
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    cv2.ellipse(
        frame,
        (cx, cy),
        (int((x2 - x1) / 2), int((y2 - y1) / 3)),
        0,
        0,
        360,
        color,
        2
    )

    cv2.putText(
        frame,
        f"ID:{track_id}",
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )

    return frame


def draw_triangle(frame, bbox, color):
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    top_y = y1 - 10

    pts = np.array([
        [cx, top_y],
        [cx - 10, y1 - 25],
        [cx + 10, y1 - 25]
    ], np.int32)

    cv2.drawContours(frame, [pts], 0, color, -1)

    return frame


class PlayerTracksDrawer:
    def __init__(self, team_1_color=(255, 245, 238), team_2_color=(128, 0, 0)):
        self.default_player_team_id = 1
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    def draw(self, video_frames, tracks, player_assignment, ball_aquisition):

        output_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            if frame_num >= len(tracks):
                output_frames.append(frame)
                continue

            player_dict = tracks[frame_num]
            assignment_dict = player_assignment[frame_num]
            ball_holder = ball_aquisition[frame_num]

            for track_id, player in player_dict.items():

                bbox = player.get("bbox", None)
                if bbox is None:
                    continue

                team_id = assignment_dict.get(track_id, self.default_player_team_id)

                color = self.team_1_color if team_id == 1 else self.team_2_color

                frame = draw_ellipse(frame, bbox, color, track_id)

                if track_id == ball_holder:
                    frame = draw_triangle(frame, bbox, (0, 0, 255))

            output_frames.append(frame)

        return output_frames