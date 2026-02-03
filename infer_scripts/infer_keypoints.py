import time
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import sys

# --- CONFIG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "trained_models/keypoint_model.pth")
DATA_ROOT = os.path.join(ROOT_DIR, "data")
DEMO_DIR = os.path.join(DATA_ROOT, "demovideos")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RENDER_W, RENDER_H = 128, 128 # Keypoint Model uses 128x128
NUM_KEYPOINTS = 16

# --- MODELS (Copied from train_keypoints.py) ---
class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, temperature=None):
        super().__init__()
        self.height = height
        self.width = width
        self.temperature = temperature
        
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1, 1, width), 
            np.linspace(-1, 1, height)
        )
        self.register_buffer('pos_x', torch.from_numpy(pos_x.reshape(height*width)).float())
        self.register_buffer('pos_y', torch.from_numpy(pos_y.reshape(height*width)).float())

    def forward(self, feature_map):
        B, K, H, W = feature_map.shape
        flat = feature_map.view(B, K, -1)
        
        if self.temperature:
            flat = flat / self.temperature
            
        softmax = F.softmax(flat, dim=-1)
        
        expected_x = torch.sum(self.pos_x * softmax, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax, dim=-1, keepdim=True)
        
        return torch.cat([expected_x, expected_y], dim=-1)

class SpatialAutoencoder(nn.Module):
    def __init__(self, num_keypoints=NUM_KEYPOINTS):
        super().__init__()
        self.k = num_keypoints
        
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=1, padding=2), nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, num_keypoints, 5, stride=2, padding=2)
        )
        
        self.ssm = SpatialSoftmax(16, 16)
        
        self.fc_dec = nn.Linear(num_keypoints * 2, 256)
        self.dec_start = nn.Linear(256, 16*16*64)
        
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid() 
        )

    def forward(self, x):
        feat = self.enc(x)
        points = self.ssm(feat) 
        
        h = F.relu(self.fc_dec(points.view(-1, self.k * 2)))
        h = F.relu(self.dec_start(h))
        h = h.view(-1, 64, 16, 16)
        recon = self.dec(h)
        
        return recon, points

class InverseDynamics(nn.Module):
    def __init__(self, num_keypoints=NUM_KEYPOINTS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_keypoints * 2 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )
        
    def forward(self, p1, p2):
        x = torch.cat([p1.flatten(1), p2.flatten(1)], dim=1)
        return self.net(x)

# --- UTILS ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor() # No normalization in training script, so none here
])

def get_robot_view():
    view = p.computeViewMatrixFromYawPitchRoll([0.2, 0, 0.1], 1.2, 45, -40, 0, 2)
    proj = p.computeProjectionMatrixFOV(50, 1.0, 0.1, 4.0)
    _, _, rgb, _, _ = p.getCameraImage(RENDER_W, RENDER_H, view, proj, shadow=0, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    frame = np.reshape(rgb, (RENDER_H, RENDER_W, 4))[:, :, :3]
    return frame

def draw_keypoints(img, points):
    # points: [K, 2] in range [-1, 1]
    # img: [H, W, 3] uint8
    H, W, _ = img.shape
    img_copy = img.copy()
    
    for k in range(points.shape[0]):
        x = int((points[k, 0] + 1) / 2 * W)
        y = int((points[k, 1] + 1) / 2 * H)
        cv2.circle(img_copy, (x, y), 3, (0, 255, 0), -1)
        
    return img_copy

# --- MAIN ---
def main():
    print(f"Available Demos in {DEMO_DIR}:")
    if os.path.exists(DEMO_DIR):
        demos = [f for f in os.listdir(DEMO_DIR) if f.startswith("demovid") and f.endswith(".mp4")]
        demos.sort()
        for i, d in enumerate(demos):
            print(f"{i+1}: {d}")
        
        try:
            choice = int(input("Select demo number: ")) - 1
            demo_file = os.path.join(DEMO_DIR, demos[choice])
        except:
            print("Invalid/No selection. Using first found or default.")
            if len(demos) > 0:
                demo_file = os.path.join(DEMO_DIR, demos[0])
            else:
                print("No demos found. Exiting.")
                return
    else:
        print("Demo directory not found.")
        return

    print(f"Loading Demo: {demo_file}")
    
    # Load Models
    print("Loading Models...")
    sae = SpatialAutoencoder().to(DEVICE)
    inv = InverseDynamics().to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        sae.load_state_dict(checkpoint['sae'])
        inv.load_state_dict(checkpoint['inv'])
        print("Models Loaded.")
    else:
        print(f"Model file {MODEL_PATH} not found. Ensure you trained the model.")
        return
        
    sae.eval()
    inv.eval()
    
    # Setup Sim (GUI)
    p.connect(p.GUI) 
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    
    robot = p.loadURDF("../3D_models/gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    
    rand_x = 0.4 + np.random.uniform(-0.05, 0.05)
    rand_y = 0.0 + np.random.uniform(-0.1, 0.1)
    cube_id = p.loadURDF("cube.urdf", basePosition=[rand_x, rand_y, 0.05], globalScaling=0.05)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])
    
    joints = [0, 1, 2, 3, 4, 5]
    for i in joints:
        p.resetJointState(robot, i, 0)
        
    # Pre-load Demo
    print("Pre-loading demo...")
    cap = cv2.VideoCapture(demo_file)
    demo_frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        demo_frames.append(frame_rgb)
    cap.release()
    print(f"Loaded {len(demo_frames)} frames.")
    
    print("Starting Inference...")
    
    for i, frame_demo_rgb in enumerate(demo_frames):
        # 1. Get Robot View
        frame_robot_rgb = get_robot_view()
        
        # 2. Get Joint Angles (for Control matching timeframe)
        joint_states = p.getJointStates(robot, joints)
        current_pos = np.array([s[0] for s in joint_states])
        
        # 3. Extract Keypoints & Predict Action
        with torch.no_grad():
            t_robot = transform(frame_robot_rgb).unsqueeze(0).to(DEVICE)
            t_demo = transform(frame_demo_rgb).unsqueeze(0).to(DEVICE)
            
            _, kp_robot = sae(t_robot)
            _, kp_demo = sae(t_demo)
            
            # Predict Action (Robot -> Demo transition)
            # Inverse Dynamics takes (Current, Target) Keypoints -> Action
            pred_action = inv(kp_robot, kp_demo)
            deltas = pred_action.cpu().numpy()[0]
            
        print(f"Step {i}: Action={deltas}")

        # 4. Apply Action
        # Position Control with 1.5 scaling (matching infer_siamese.py)
        target_pos = current_pos + deltas * 1.5 
        
        # Clamp
        LIMITS = [(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (0, 0.5), (-0.5, 0)]
        for j in range(6):
            target_pos[j] = np.clip(target_pos[j], LIMITS[j][0], LIMITS[j][1])
            
        p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=target_pos, forces=[100]*6)
        p.stepSimulation()
            
    p.disconnect()

if __name__ == "__main__":
    main()
