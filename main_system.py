# main_system.py
# Master script that runs both audio and face analysis on a video file.

from audio import detect_all_disturbances
from video import run_face_analysis

def  main():
    print("🔹 Starting Audio + Face Detection System 🔹")
    
    # Input video path
    video_path = "data/input.mp4"  # change this to your video file path

    print("\n🎧 Step 1: Detecting Audio Disturbances...")
    disturbances = detect_all_disturbances(video_path, frame_ms=20, threshold_factor=2.0)
    print(f"✅ Audio analysis complete! Found {len(disturbances)} disturbance(s).")

    print("\n🎥 Step 2: Running Face Analysis...")
    run_face_analysis(video_path)
    print("✅ Face analysis complete! Annotated video and JSON generated.")

    print("\n🎯 All analyses done! Check your workspace for the output files.\n")

if __name__ == "__main__":
    main()
