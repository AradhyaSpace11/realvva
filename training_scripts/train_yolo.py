from ultralytics import YOLO
import os

# --- CONFIG ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT_DIR, "data/yolo_dataset/data.yaml")
MODEL_OUT = os.path.join(ROOT_DIR, "trained_models/yolo_pose_model.pt")

def train_yolo():
    # Load a model (Pose Model now)
    model = YOLO("yolov8n-pose.pt") 

    # Train the model
    print("Starting YOLO Pose Training...")
    results = model.train(data=DATA_YAML, epochs=100, imgsz=640, project=os.path.dirname(MODEL_OUT), name="yolo_pose_run")
    
    best_weights = os.path.join(os.path.dirname(MODEL_OUT), "yolo_pose_run", "weights", "best.pt")
    
    if os.path.exists(best_weights):
        os.rename(best_weights, MODEL_OUT)
        print(f"Saved best model to {MODEL_OUT}")

if __name__ == "__main__":
    train_yolo()
