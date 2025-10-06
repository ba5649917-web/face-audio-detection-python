# video/face_analyze_video.py
# Detect and analyze faces in a video using MediaPipe.

import cv2
import mediapipe as mp
import math
import json
import numpy as np
from dataclasses import dataclass, asdict

mp_face_mesh = mp.solutions.face_mesh

@dataclass
class FaceEvent:
    face_id: int
    start_time: float
    end_time: float
    events: list

def rotationVectorToEulerAngles(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees([x, y, z])

def run_face_analysis(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter("results/annotated_faces.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    face_events = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        ts = frame_idx / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            for f_lms in res.multi_face_landmarks:
                xs = [int(p.x * W) for p in f_lms.landmark]
                ys = [int(p.y * H) for p in f_lms.landmark]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {len(res.multi_face_landmarks)}", (10, 40), FONT, 1, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 40), FONT, 1, (0, 0, 255), 2)

        writer.write(frame)

    cap.release()
    writer.release()

    result = {
        "video_path": video_path,
        "fps": fps,
        "total_frames": frame_idx
    }
    with open("results/face_analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n✅ Face analysis complete. Results saved to face_analysis.json.")
