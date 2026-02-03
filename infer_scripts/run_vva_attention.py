import pybullet as p
import pybullet_data
import numpy as np
import cv2
import torch
import torch.nn as nn
import os
import sys
import math
import time
from ultralytics import YOLO

# --- CONFIG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REALVVA_ROOT = os.path.dirname(CURRENT_DIR) # /home/aradhya/realvva
YOLO_ROOT = os.path.join(REALVVA_ROOT, "yolodetect")

# Add paths for utils
sys.path.append(YOLO_ROOT)
from utils.smoothing import CentroidSmoother

# Paths
YOLO_MODEL_PATH = os.path.join(YOLO_ROOT, "models", "run", "weights", "best.pt")
# THIS IS THE NEW ATTENTION MODEL
POLICY_MODEL_PATH = os.path.join(REALVVA_ROOT, "trained_models", "policy_attention.pth")

# Default Video
video_name = "demovid1.mp4"
if len(sys.argv) > 1:
    video_name = sys.argv[1]
    
VIDEO_PATH = os.path.join(REALVVA_ROOT, "data", "demovideos", video_name)

# --- ATTENTION MODEL DEFINITION (Must Match Training) ---
class RobotPolicyAttention(nn.Module):
    def __init__(self, output_dim=6):
        super(RobotPolicyAttention, self).__init__()
        self.embedding = nn.Linear(2, 32)
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        seq = x.view(-1, 7, 2)
        emb = self.embedding(seq)
        attn_output, _ = self.attention(emb, emb, emb)
        out = self.head(attn_output)
        return out

def main():
    if not os.path.exists(POLICY_MODEL_PATH):
        print(f"Error: Attendance Model not found at {POLICY_MODEL_PATH}")
        print("Run: python3 training_scripts/train_policy2_attention.py")
        return

    print(f"="*60)
    print(f"RUNNING VVA (ATTENTION) on: {video_name}")
    print(f"="*60)
    
    yolo_model = YOLO(YOLO_MODEL_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_model = RobotPolicyAttention().to(device)
    policy_model.load_state_dict(torch.load(POLICY_MODEL_PATH, map_location=device))
    policy_model.eval()

    # PyBullet Setup
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    robot = p.loadURDF(os.path.join(REALVVA_ROOT, "3D_models/gripper_arm.urdf"), basePosition=[0, 0, 0], useFixedBase=True)
    
    # Cube (Ghost) - Solid Red
    cube_id = p.loadURDF("cube.urdf", basePosition=[0.4, 0, 0.05], globalScaling=0.05)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1.0])
    
    joints = [0, 1, 2, 3, 4, 5]
    smoother = CentroidSmoother()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0,0,0])

    state_vec = np.zeros(14)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video Finished.")
            break
            
        results = yolo_model(frame, verbose=False, conf=0.05)
        
        found_mask = [False] * 7
        display = frame.copy()
        
        if results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                if cls_id < 7 and not found_mask[cls_id]:
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    
                    # Smooth
                    sx, sy = smoother.update(cls_id, x, y)
                    
                    # Norm
                    H, W = frame.shape[:2]
                    state_vec[cls_id*2] = sx / W
                    state_vec[cls_id*2+1] = sy / H
                    found_mask[cls_id] = True
                    
                    col = (0, 255, 0) if cls_id < 6 else (0, 0, 255)
                    cv2.circle(display, (int(sx), int(sy)), 5, col, -1)
        
        # Policy Forward
        X_tensor = torch.tensor(state_vec, dtype=torch.float32).to(device).unsqueeze(0)
        with torch.no_grad():
            action = policy_model(X_tensor).cpu().numpy()[0]
        
        p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=action, forces=[150]*4 + [60]*2)
        p.stepSimulation()
        
        cv2.putText(display, "VVA ATTENTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        display_half = cv2.resize(display, (0,0), fx=0.6, fy=0.6)
        cv2.imshow("Attention Brain", display_half)
        
        if cv2.waitKey(30) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()
    p.disconnect()

if __name__ == "__main__":
    main()
