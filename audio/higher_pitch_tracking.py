# audio/higher_pitch_tracking.py
# Detect all disturbances in the audio track of a video.

import numpy as np
import librosa
from moviepy.editor import VideoFileClip

def format_time_s(seconds: float) -> str:
    return f"{seconds:.2f}s"

def detect_all_disturbances(video_path, frame_ms=20, threshold_factor=2.0):
    # 1. Extract audio
    clip = VideoFileClip(video_path)
    audio_path = "results/temp_audio.wav"
    clip.audio.write_audiofile(audio_path, logger=None)

    # 2. Load audio
    y, sr = librosa.load(audio_path, sr=16000)

    # 3. Frame length
    frame_len = int(sr * frame_ms / 1000)
    hop_len = frame_len

    variances = []
    for i in range(0, len(y) - frame_len, hop_len):
        frame = y[i:i + frame_len]
        variance = np.var(frame)
        t_start = i / sr
        t_end = (i + frame_len) / sr
        variances.append((variance, t_start, t_end))

    # 4. Threshold
    avg_var = np.mean([v[0] for v in variances])
    threshold = avg_var * threshold_factor

    # 5. Find disturbances
    disturbed_intervals = []
    current_start, current_end = None, None

    for variance, t_start, t_end in variances:
        if variance > threshold:
            if current_start is None:
                current_start = t_start
            current_end = t_end
        else:
            if current_start is not None:
                disturbed_intervals.append((current_start, current_end))
                current_start, current_end = None, None

    if current_start is not None:
        disturbed_intervals.append((current_start, current_end))

    print("\n📢 Disturbance intervals detected:")
    for idx, (s, e) in enumerate(disturbed_intervals, 1):
        duration = e - s
        print(f" {idx}. {format_time_s(s)} → {format_time_s(e)}   ({duration:.2f}s)")

    return disturbed_intervals
