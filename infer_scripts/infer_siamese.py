import time
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import math
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
import sys

# --- CONFIG ---
# Robust path handling
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR) # Go up one level
MODEL_PATH = os.path.join(ROOT_DIR, "trained_models/siamese_model.pth")
DATA_ROOT = os.path.join(ROOT_DIR, "data")
DEMO_DIR = os.path.join(DATA_ROOT, "demovideos")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RENDER_W, RENDER_H = 224, 224 # Match model input
FPS = 30
# The training data deltas were captured at ~15-30Hz. 
# We need to ensure we apply them at a similar rate or scale them.

# --- MODEL DEFINITION (Copied to be standalone) ---
class SiamesePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=False) # No need to download weights, loading state_dict
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 + 6, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, img_robot, img_demo, joints):
        f_robot = self.backbone(img_robot).view(img_robot.size(0), -1)
        f_demo = self.backbone(img_demo).view(img_demo.size(0), -1)
        x = torch.cat([f_robot, f_demo, joints], dim=1)
        return self.fc(x)

# --- UTILS ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_robot_view():
    # Setup camera specifically for the model (Diagonal View used in data collection)
    view = p.computeViewMatrixFromYawPitchRoll([0.2, 0, 0.1], 1.2, 45, -40, 0, 2)
    proj = p.computeProjectionMatrixFOV(50, 1.0, 0.1, 4.0)
    _, _, rgb, _, _ = p.getCameraImage(RENDER_W, RENDER_H, view, proj, shadow=0, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    frame = np.reshape(rgb, (RENDER_H, RENDER_W, 4))[:, :, :3] # RGBA -> RGB
    return frame # RGB (0-255)

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
        demo_file = os.path.join(DEMO_DIR, demos[0])

    print(f"Loading Demo: {demo_file}")
    
    # 2. Load Model
    print("Loading Model...")
    model = SiamesePolicy().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Did you train the model? make sure ../trained_models/siamese_model.pth exists.")
        return
        
    model.eval()
    
    # 3. Setup Simulation
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    
    robot = p.loadURDF("../3D_models/gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    
    # Randomize Cube Position slightly to test generalization (as requested)
    rand_x = 0.4 + np.random.uniform(-0.05, 0.05)
    rand_y = 0.0 + np.random.uniform(-0.1, 0.1)
    cube_id = p.loadURDF("cube.urdf", basePosition=[rand_x, rand_y, 0.05], globalScaling=0.05)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])
    print(f"Cube initialized at {rand_x:.2f}, {rand_y:.2f}")

    # Reset Robot to Home
    joints = [0, 1, 2, 3, 4, 5]
    for i in joints:
        p.resetJointState(robot, i, 0)

    # 4. Pre-load Demo Frames (To prevent lag)
    print("Pre-loading demo frames...")
    cap = cv2.VideoCapture(demo_file)
    demo_frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        # Convert to RGB and Transform immediately
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        demo_frames.append(frame_rgb)
    cap.release()
    print(f"Loaded {len(demo_frames)} frames.")

    # 5. Control Loop (Step-based, not Time-based)
    print("Starting Inference (Fast Mode)...")
    
    # We will simulate at 30Hz, matching the approximate demo FPS
    # We map 1 Sim Step <-> 1 Demo Frame (assuming roughly equal speed)
    # If the robot is too slow, we might need to hold the demo frame for multiple steps.
    
    for i, frame_demo_rgb in enumerate(demo_frames):
        # Get Current Robot State
        frame_robot_rgb = get_robot_view() # RGB
        
        # Get Joint Angles
        joint_states = p.getJointStates(robot, joints)
        current_pos = [s[0] for s in joint_states] # list of 6 floats
        
        # Prepare Inputs
        with torch.no_grad():
            t_robot = transform(frame_robot_rgb).unsqueeze(0).to(DEVICE)
            t_demo = transform(frame_demo_rgb).unsqueeze(0).to(DEVICE)
            t_joints = torch.tensor(current_pos).float().unsqueeze(0).to(DEVICE)
            
            # Inference
            pred_deltas = model(t_robot, t_demo, t_joints)
            deltas = pred_deltas.cpu().numpy()[0]
        
        # Apply Control
        # Position Control
        target_pos = np.array(current_pos) + deltas * 1.5 
        
        # Clamp
        LIMITS = [(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (0, 0.5), (-0.5, 0)]
        for j in range(6):
            target_pos[j] = np.clip(target_pos[j], LIMITS[j][0], LIMITS[j][1])
            
        p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=target_pos, forces=[100]*6)
        p.stepSimulation()
        
        # Visualization (Only Robot View)
        cv2.imshow("Robot Simulation View", cv2.cvtColor(frame_robot_rgb, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        # No sleep needed, runs as fast as GPU/CPU allows
        # This solves the "simulation doing things slowly" issue because
        # we don't move to the next demo frame until the sim step is done.

    cv2.destroyAllWindows()
    p.disconnect()

if __name__ == "__main__":
    main()
