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
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR) # Go up one level
MODEL_PATH = os.path.join(ROOT_DIR, "trained_models/vla_model.pth")
DATA_ROOT = os.path.join(ROOT_DIR, "data")
DEMO_DIR = os.path.join(DATA_ROOT, "demovideos")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RENDER_W, RENDER_H = 224, 224
NUM_BINS = 256
D_MODEL = 256
N_HEAD = 4
N_LAYERS = 4

# --- MODEL DEFINITIONS (Copied from train_vla.py) ---
class ActionTokenizer:
    def __init__(self, min_val=-0.05, max_val=0.05, bins=NUM_BINS):
        self.min_val = min_val
        self.max_val = max_val
        self.bins = bins
        
    def encode(self, x):
        x = torch.clamp(x, self.min_val, self.max_val)
        norm = (x - self.min_val) / (self.max_val - self.min_val)
        ids = (norm * (self.bins - 1)).long()
        return ids 
        
    def decode(self, ids):
        norm = ids.float() / (self.bins - 1)
        x = norm * (self.max_val - self.min_val) + self.min_val
        return x

class TinyVLA(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) 
        self.vis_proj = nn.Linear(512, D_MODEL)
        self.joint_proj = nn.Linear(6, D_MODEL)
        
        layer = nn.TransformerDecoderLayer(d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers=N_LAYERS)
        self.pos_emb = nn.Parameter(torch.randn(1, 3, D_MODEL))
        self.heads = nn.ModuleList([nn.Linear(D_MODEL, NUM_BINS) for _ in range(6)])

    def forward(self, img_r, img_d, joints):
        f_r = self.backbone(img_r).flatten(1)
        f_d = self.backbone(img_d).flatten(1)
        
        emb_r = self.vis_proj(f_r).unsqueeze(1)
        emb_d = self.vis_proj(f_d).unsqueeze(1)
        emb_j = self.joint_proj(joints).unsqueeze(1)
        
        seq = torch.cat([emb_r, emb_d, emb_j], dim=1)
        seq = seq + self.pos_emb 
        out = self.transformer(tgt=seq, memory=seq)
        context = out[:, -1, :] 
        logits = [head(context) for head in self.heads] 
        return logits

# --- UTILS ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_robot_view():
    view = p.computeViewMatrixFromYawPitchRoll([0.2, 0, 0.1], 1.2, 45, -40, 0, 2)
    proj = p.computeProjectionMatrixFOV(50, 1.0, 0.1, 4.0)
    _, _, rgb, _, _ = p.getCameraImage(RENDER_W, RENDER_H, view, proj, shadow=0, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    frame = np.reshape(rgb, (RENDER_H, RENDER_W, 4))[:, :, :3]
    return frame 

# --- MAIN ---
def main():
    # 1. Select Demo
    print(f"Available Demos in {DEMO_DIR}:")
    try:
        demos = [f for f in os.listdir(DEMO_DIR) if f.startswith("demovid") and f.endswith(".mp4")]
        demos.sort()
        for i, d in enumerate(demos):
            print(f"{i+1}: {d}")
        
        choice = int(input("Select demo number: ")) - 1
        demo_file = os.path.join(DEMO_DIR, demos[choice])
    except Exception as e:
        print(f"Error finding demos: {e}")
        return

    print(f"Loading Demo: {demo_file}")
    
    # 2. Load Model
    print("Loading VLA Model...")
    model = TinyVLA().to(DEVICE)
    
    # Hardcoded ranges from training stats (approximate typical joint deltas)
    # Ideally should save these in a config file during training
    # For now using safe default bounds matching the ActionTokenizer default
    tokenizer = ActionTokenizer(min_val=-0.05, max_val=0.05, bins=NUM_BINS)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.eval()
    
    # 3. Setup Simulation
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
    print(f"Cube at {rand_x:.2f}, {rand_y:.2f}")

    joints = [0, 1, 2, 3, 4, 5]
    for i in joints: p.resetJointState(robot, i, 0)

    # 4. Pre-load Demo Frames
    print("Pre-loading demo...")
    cap = cv2.VideoCapture(demo_file)
    demo_frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        demo_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Loaded {len(demo_frames)} frames.")

    # 5. Inference
    print("Starting VLA Inference...")
    
    for i, frame_demo_rgb in enumerate(demo_frames):
        frame_robot_rgb = get_robot_view()
        
        joint_states = p.getJointStates(robot, joints)
        current_pos = [s[0] for s in joint_states]
        
        with torch.no_grad():
            t_robot = transform(frame_robot_rgb).unsqueeze(0).to(DEVICE)
            t_demo = transform(frame_demo_rgb).unsqueeze(0).to(DEVICE)
            t_joints = torch.tensor(current_pos).float().unsqueeze(0).to(DEVICE)
            
            # Forward Pass
            logits_list = model(t_robot, t_demo, t_joints) # List of 6 tensors [1, 256]
            
            # Decode Actions (Argmax -> ID -> Float)
            deltas = []
            for logits in logits_list:
                token_id = torch.argmax(logits, dim=1) # [1]
                val = tokenizer.decode(token_id)
                deltas.append(val.item())
                
            deltas = np.array(deltas)
        
        # Apply Control
        # Important: VLA predicts action bins. These are absolute steps.
        # We might need scaling if the bins are too conservative.
        target_pos = np.array(current_pos) + deltas * 2.0 # Higher gain for VLA
        
        LIMITS = [(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (0, 0.5), (-0.5, 0)]
        for j in range(6):
            target_pos[j] = np.clip(target_pos[j], LIMITS[j][0], LIMITS[j][1])
            
        p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=target_pos, forces=[100]*6)
        p.stepSimulation()
        
        cv2.imshow("VLA Robot View", cv2.cvtColor(frame_robot_rgb, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
    p.disconnect()

if __name__ == "__main__":
    main()
