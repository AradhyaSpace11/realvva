from ultralytics import YOLO
import cv2
import os
import sys

# --- CONFIG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(ROOT_DIR, "data")
# Start with default
MODEL_PATH = os.path.join(ROOT_DIR, "trained_models/yolo_pose_model.pt")

# Search for latest training run
trained_models_dir = os.path.join(ROOT_DIR, "trained_models")
possible_runs = [d for d in os.listdir(trained_models_dir) if d.startswith("yolo_pose_run") and os.path.isdir(os.path.join(trained_models_dir, d))]
possible_runs.sort(key=lambda x: int(x.replace("yolo_pose_run", "") or "1") if x.replace("yolo_pose_run", "").isdigit() or x == "yolo_pose_run" else 0, reverse=True)

for run in possible_runs:
    run_best = os.path.join(trained_models_dir, run, "weights/best.pt")
    if os.path.exists(run_best):
        MODEL_PATH = run_best
        print(f"Found latest trained model: {MODEL_PATH}")
        break

# Fallback to root if still not found
if not os.path.exists(MODEL_PATH):
    ROOT_MODEL = os.path.join(ROOT_DIR, "yolov8n-pose.pt")
    if os.path.exists(ROOT_MODEL):
        MODEL_PATH = ROOT_MODEL
        print(f"Using Base Model (Not fine-tuned): {MODEL_PATH}")

VIDEO_PATH = os.path.join(DATA_ROOT, "demovideos/demovid1.mp4")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found!")
        print(f"Searched: {MODEL_PATH}")
        print("Please run training_scripts/train_yolo_pose.py")
        return

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        return

    print(f"Loading Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"Processing Video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Run Inference
        # verbose=False to reduce terminal spam
        results = model(frame, verbose=False)

        # Visualize
        # plot() draws bounding boxes and keypoints
        annotated_frame = results[0].plot()

        cv2.imshow("YOLO Pose Inference (Demo)", annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
