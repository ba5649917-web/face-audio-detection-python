# video/face_analyze_video.py
# Detect and analyze faces in a video using MediaPipe.

import cv2
import mediapipe as mp
import math
import json
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def rotationVectorToEulerAngles(rvec):
    """Convert rotation vector to Euler angles (X, Y, Z in degrees)."""
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

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, refine_landmarks=True)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    frame_idx = 0
    multi_face_events = []     # store time intervals when multiple faces are detected
    event_active = False
    current_event = {"start_time": None, "end_time": None, "faces": 0}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        ts = frame_idx / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        face_count = 0
        if res.multi_face_landmarks:
            face_count = len(res.multi_face_landmarks)
            for f_lms in res.multi_face_landmarks:
                xs = [int(p.x * W) for p in f_lms.landmark]
                ys = [int(p.y * H) for p in f_lms.landmark]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {face_count}", (10, 40), FONT, 1, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 40), FONT, 1, (0, 0, 255), 2)

        # ✅ Track when multiple faces appear
        if face_count > 1:
            if not event_active:
                current_event["start_time"] = round(ts, 2)
                current_event["faces"] = face_count
                event_active = True
            else:
                current_event["faces"] = max(current_event["faces"], face_count)
        else:
            if event_active:
                current_event["end_time"] = round(ts, 2)
                multi_face_events.append(current_event.copy())
                current_event = {"start_time": None, "end_time": None, "faces": 0}
                event_active = False

        writer.write(frame)

    # if video ends and event is still active
    if event_active:
        current_event["end_time"] = round(frame_idx / fps, 2)
        multi_face_events.append(current_event)

    cap.release()
    writer.release()

    result = {
        "video_path": video_path,
        "fps": fps,
        "total_frames": frame_idx,
        "multi_face_events": multi_face_events
    }

    with open("results/face_analysis.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n✅ Face analysis complete.")
    print(f"📊 Total frames: {frame_idx}")
    print("👥 Multiple face events:")
    for ev in multi_face_events:
        print(f" - Faces: {ev['faces']} | Start: {ev['start_time']}s | End: {ev['end_time']}s")
