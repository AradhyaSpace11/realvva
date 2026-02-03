from ultralytics import YOLO
import os
import yaml

# CONFIG
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # .../yolodetect/train_scripts
YOLO_ROOT = os.path.dirname(CURRENT_DIR) # .../yolodetect
DATASET_DIR = os.path.join(YOLO_ROOT, "data", "dataset")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
MODEL_OUT_DIR = os.path.join(YOLO_ROOT, "models")

def train():
    # 1. Update data.yaml with ABSOLUTE path
    with open(DATA_YAML, 'r') as f:
        config = yaml.safe_load(f)
    
    config['path'] = DATASET_DIR
    
    with open(DATA_YAML, 'w') as f:
        yaml.dump(config, f)
        
    print(f"Updated config: {DATASET_DIR}")

    # 2. Train using yolov8n.pt (Detection model, NOT Pose)
    model = YOLO("yolov8n.pt")
    
    print(f"Starting Training (Detection)...")
    results = model.train(data=DATA_YAML, epochs=100, imgsz=640, project=MODEL_OUT_DIR, name="run")

if __name__ == "__main__":
    train()
