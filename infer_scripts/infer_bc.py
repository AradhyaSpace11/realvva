import time
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import torch
import torch.nn as nn
import os
import sys
import torchvision.models as models
import torchvision.transforms as transforms

# --- CONFIG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "trained_models/bc_resnet_model.pth")
DATA_ROOT = os.path.join(ROOT_DIR, "data")
DEMO_DIR = os.path.join(DATA_ROOT, "demovideos")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RENDER_W, RENDER_H = 224, 224

# Normalization Constants (Must match training!)
JOINT_MIN = np.array([-3.14, -3.14, -3.14, -3.14, 0.0, -0.5])
JOINT_MAX = np.array([3.14, 3.14, 3.14, 3.14, 0.5, 0.0])
JOINT_RANGE = JOINT_MAX - JOINT_MIN + 1e-6

# --- MODEL (Copied from train_bc.py) ---
class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False) # No need to download weights
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.resnet(x)

# --- UTILS ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_robot_view():
    view = p.computeViewMatrixFromYawPitchRoll([0.2, 0, 0.1], 1.2, 45, -40, 0, 2)
    proj = p.computeProjectionMatrixFOV(50, 1.0, 0.1, 4.0)
    _, _, rgb, _, _ = p.getCameraImage(RENDER_W, RENDER_H, view, proj, shadow=0, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    frame = np.reshape(rgb, (RENDER_H, RENDER_W, 4))[:, :, :3]
    return frame # RGB

# --- MAIN ---
def main():
    # 1. Select Demo (Reference)
    print(f"Available Demos in {DEMO_DIR}:")
    demos = [f for f in os.listdir(DEMO_DIR) if f.startswith("demovid") and f.endswith(".mp4")]
    demos.sort()
    for i, d in enumerate(demos):
        print(f"{i+1}: {d}")
    
    try:
        choice = int(input("Select demo number to visualize: ")) - 1
        demo_file = os.path.join(DEMO_DIR, demos[choice])
    except:
        print("Invalid selection. Using first demo.")
        if len(demos) > 0:
            demo_file = os.path.join(DEMO_DIR, demos[0])
        else:
            print("No demos found.")
            return

    print("Loading Model...")
    model = BCPolicy().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("Model Loaded.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Did you finish training with train_bc.py?")
            return
    else:
        print(f"Model file {MODEL_PATH} not found.")
        return
    model.eval()
    
    # Setup Sim
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    
    robot = p.loadURDF("../3D_models/gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    
# --- MAIN ---
def main():
    # 1. Select Demo (Reference)
    print(f"Available Demos in {DEMO_DIR}:")
    demos = [f for f in os.listdir(DEMO_DIR) if f.startswith("demovid") and f.endswith(".mp4")]
    demos.sort()
    for i, d in enumerate(demos):
        print(f"{i+1}: {d}")
    
    try:
        choice = int(input("Select demo number to visualize: ")) - 1
        demo_file = os.path.join(DEMO_DIR, demos[choice])
    except:
        print("Invalid selection. Using first demo.")
        if len(demos) > 0:
            demo_file = os.path.join(DEMO_DIR, demos[0])
        else:
            print("No demos found.")
            return

    print("Loading Model...")
    model = BCPolicy().to(DEVICE)
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
    
    # Setup Sim
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    
    robot = p.loadURDF("../3D_models/gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    
    # Randomize Cube
    rand_x = 0.4 + np.random.uniform(-0.05, 0.05)
    rand_y = 0.0 + np.random.uniform(-0.1, 0.1)
    cube_id = p.loadURDF("cube.urdf", basePosition=[rand_x, rand_y, 0.05], globalScaling=0.05)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])
    
    joints = [0, 1, 2, 3, 4, 5]
    for i in joints:
        p.resetJointState(robot, i, 0)
        
    print("Starting Behavior Cloning Inference...")
    
    # Load Demo for Visualization
    cap = cv2.VideoCapture(demo_file)
    
    while True:
        # 1. Get View
        img = get_robot_view()
        
        # 2. Predict
        with torch.no_grad():
            t_img = transform(img).unsqueeze(0).to(DEVICE)
            pred_norm = model(t_img).cpu().numpy()[0]
            
            # Denormalize
            target_pos = (pred_norm + 1.0) / 2.0 * JOINT_RANGE + JOINT_MIN
            
        # 3. Apply
        p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=target_pos, forces=[100]*6)
        
        # Reference Demo Frame (Looping)
        ret, frame_demo = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame_demo = cap.read()
            
        # Resize demo to match robot view for side-by-side
        frame_demo = cv2.resize(frame_demo, (RENDER_W, RENDER_H))
        
        # Viz Side-by-Side
        robot_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        combined = np.hstack((robot_bgr, frame_demo))
        
        cv2.imshow("Left: Robot (Autonomous) | Right: Selected Demo (Ref)", combined)
        
        p.stepSimulation()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    p.disconnect()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
