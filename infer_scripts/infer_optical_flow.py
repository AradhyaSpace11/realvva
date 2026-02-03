import time
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import torch
import torch.nn as nn
import os
import sys

# --- CONFIG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "trained_models/optical_flow_model.pth")
DATA_ROOT = os.path.join(ROOT_DIR, "data")
DEMO_DIR = os.path.join(DATA_ROOT, "demovideos")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZE = 16 # Must match training
INPUT_DIM = GRID_SIZE * GRID_SIZE * 2

# --- MODEL (Copied from train_optical_flow.py) ---
class FlowPolicy(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, output_dim=6):
        super().__init__()
        # Simple MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim) 
        )

    def forward(self, x):
        return self.net(x)

# --- UTILS ---
def compute_flow_grid(img1, img2):
    # img1, img2 are RGB/BGR
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Dense Flow (Farneback) matches training parameters
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Downsample
    flow_small = cv2.resize(flow, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA)
    
    return flow_small.flatten()

def get_robot_view():
    # Setup camera (Only for visualization if needed, flow uses DEMO video)
    view = p.computeViewMatrixFromYawPitchRoll([0.2, 0, 0.1], 1.2, 45, -40, 0, 2)
    proj = p.computeProjectionMatrixFOV(50, 1.0, 0.1, 4.0)
    _, _, rgb, _, _ = p.getCameraImage(128, 128, view, proj, shadow=0, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    frame = np.reshape(rgb, (128, 128, 4))[:, :, :3]
    return frame

# --- MAIN ---
def main():
    # 1. Select Demo
    print(f"Available Demos in {DEMO_DIR}:")
    demos = [f for f in os.listdir(DEMO_DIR) if f.startswith("demovid") and f.endswith(".mp4")]
    demos.sort()
    for i, d in enumerate(demos):
        print(f"{i+1}: {d}")
    
    try:
        choice = int(input("Select demo number: ")) - 1
        demo_file = os.path.join(DEMO_DIR, demos[choice])
    except:
        print("Invalid selection. Using first demo.")
        if len(demos) > 0:
            demo_file = os.path.join(DEMO_DIR, demos[0])
        else:
            print("No demos found.")
            return

    print(f"Loading Demo: {demo_file}")
    
    # 2. Load Model
    print("Loading Model...")
    model = FlowPolicy().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("Model Loaded.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            return
    else:
        print(f"Model file {MODEL_PATH} not found.")
        return
    model.eval()
    
    # 3. Setup Simulation
    p.connect(p.GUI) # User prefers GUI
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    
    robot = p.loadURDF("../3D_models/gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    
    # Standard Cube at Home
    rand_x = 0.4 
    rand_y = 0.0 
    cube_id = p.loadURDF("cube.urdf", basePosition=[rand_x, rand_y, 0.05], globalScaling=0.05)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])

    # Reset Robot
    joints = [0, 1, 2, 3, 4, 5]
    for i in joints:
        p.resetJointState(robot, i, 0)

    # 4. Pre-load Demo Frames
    print("Pre-loading and resizing demo frames...")
    cap = cv2.VideoCapture(demo_file)
    demo_frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        # Resize to 128x128 for consistency with training preprocessing
        frame = cv2.resize(frame, (128, 128))
        demo_frames.append(frame)
    cap.release()
    print(f"Loaded {len(demo_frames)} frames.")

    print("Starting Optical Flow Inference...")
    
    # 5. Control Loop
    # We iterate t from 0 to N-2
    # At each step t, we compute Flow(Frame_t, Frame_t+1) -> Action_t
    
    for i in range(len(demo_frames) - 1):
        # 1. Get Motion Feature (From Demo)
        img1 = demo_frames[i]
        img2 = demo_frames[i+1]
        
        feat = compute_flow_grid(img1, img2)
        
        # 2. Predict Action
        with torch.no_grad():
            t_feat = torch.tensor(feat).float().unsqueeze(0).to(DEVICE)
            pred_action = model(t_feat)
            deltas = pred_action.cpu().numpy()[0]
            
        print(f"Step {i}: Action={deltas}")
        
        # 3. Apply Action
        joint_states = p.getJointStates(robot, joints)
        current_pos = np.array([s[0] for s in joint_states])
        
        # Apply scaling (1.0 or 1.5, start with 1.0 as flow might be noisy)
        target_pos = current_pos + deltas * 1.5 
        
        # Clamp
        LIMITS = [(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (0, 0.5), (-0.5, 0)]
        for j in range(6):
            target_pos[j] = np.clip(target_pos[j], LIMITS[j][0], LIMITS[j][1])
            
        p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=target_pos, forces=[100]*6)
        p.stepSimulation()
        
        # Optional: Show what we are tracking
        # cv2.imshow("Demo Stream", img1)
        # if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        # Sync with simulation time roughly
        time.sleep(1./30.)

    p.disconnect()
    print("Done.")

if __name__ == "__main__":
    main()
