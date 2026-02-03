import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob

# --- CONFIG ---
BATCH_SIZE = 32
EPOCHS = 50 # Lightweight model converges fast
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "../trained_models/optical_flow_model.pth"
GRID_SIZE = 16 # 16x16 grid for flow downsampling
INPUT_DIM = GRID_SIZE * GRID_SIZE * 2 # x-flow, y-flow per cell

# Data Paths
DATA_ROOT = "../data" # Relative to training_scripts/
CAMVIEW_DIR = os.path.join(DATA_ROOT, "camview")
JOINT_DIR = os.path.join(DATA_ROOT, "jointdata")
OS_DATA_ROOT = "data" # Absolute-ish path check fallback

# --- 1. MODEL ---
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
            nn.Linear(64, output_dim) # Joint Deltas
        )

    def forward(self, x):
        return self.net(x)

# --- 2. DATASET ---
class OpticalFlowDataset(Dataset):
    def __init__(self):
        self.samples = []
        
        # Robust path finding
        j_dir = JOINT_DIR if os.path.exists(JOINT_DIR) else os.path.join(OS_DATA_ROOT, "jointdata")
        c_dir = CAMVIEW_DIR if os.path.exists(CAMVIEW_DIR) else os.path.join(OS_DATA_ROOT, "camview")
        
        joint_files = sorted(glob.glob(os.path.join(j_dir, "*.csv")))
        print(f"Components found: {len(joint_files)} sessions in {j_dir}")

        for j_path in joint_files:
            idx = os.path.basename(j_path).replace("jd", "").replace(".csv", "").replace("_joints", "").replace("rec_", "")
            
            # Find matching video (flexible naming)
            # Logic: jd1.csv -> camview1.mp4
            c_path = os.path.join(c_dir, f"camview{idx}.mp4")

            if os.path.exists(c_path):
                df = pd.read_csv(j_path)
                cap = cv2.VideoCapture(c_path)
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                print(f"DEBUG: {c_path} -> Video Frames: {length}, CSV Rows: {len(df)}")
                
                # We need PAIRS: (Frame_t, Frame_t+1) -> Action_t
                valid_len = min(len(df), length) - 1
                
                for i in range(valid_len):
                    self.samples.append({
                        "video_path": c_path,
                        "frame_idx": i,
                        "action": df.iloc[i][['d0','d1','d2','d3','d4','d5']].values.astype(np.float32)
                    })
        
        print(f"Dataset Size: {len(self.samples)} pairs")

    def __len__(self):
        return len(self.samples)

    def compute_flow_grid(self, img1, img2):
        # Convert to gray
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Dense Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Downsample to Grid
        # flow is HxWx2. We want GRID_SIZE x GRID_SIZE x 2
        H, W = flow.shape[:2]
        
        # Resize flow map naively (average pooling would be better but resize is fast approximation)
        # cv2.resize handles interpolation
        flow_small = cv2.resize(flow, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA)
        
        # Normalize slightly ? Flow magnitude depends on image size/motion speed. 
        # But keeping raw values is okay for limited domain.
        
        return flow_small.flatten()

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        cap = cv2.VideoCapture(s["video_path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, s["frame_idx"])
        
        ret1, img1 = cap.read()
        ret2, img2 = cap.read()
        cap.release()
        
        if not ret1 or not ret2:
            return torch.zeros(INPUT_DIM), torch.zeros(6)
            
        # Resize images to smaller size for faster flow computation (optional but good for speed)
        img1 = cv2.resize(img1, (128, 128))
        img2 = cv2.resize(img2, (128, 128))
        
        flow_feature = self.compute_flow_grid(img1, img2) # [INPUT_DIM]
        
        return torch.tensor(flow_feature).float(), torch.tensor(s["action"])

# --- 3. TRAINING ---
def train():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Training Optical Flow Model on {DEVICE}")
    
    dataset = OpticalFlowDataset()
    if len(dataset) == 0:
        print("No data found! Check 'data' folder.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = FlowPolicy().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for flow_feat, action in dataloader:
            flow_feat, action = flow_feat.to(DEVICE), action.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(flow_feat)
            loss = criterion(preds, action)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg:.6f}")
        
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), MODEL_PATH)
            
    print(f"Done. Saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
