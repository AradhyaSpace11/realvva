import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import glob

# --- 1. CONFIGURATION ---
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "siamese_model.pth"

# Data Paths
DATA_ROOT = "data"
CAMVIEW_DIR = os.path.join(DATA_ROOT, "camview")
DEMO_DIR = os.path.join(DATA_ROOT, "demovideos") 
JOINT_DIR = os.path.join(DATA_ROOT, "jointdata")

# Image Transforms (Standard ResNet Normalization)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 2. DATASET ---
class VisualImitationDataset(Dataset):
    def __init__(self, transform=None):
        self.samples = []
        self.transform = transform
        
        # Find all matching files
        # We assume naming convention: camviewX.mp4 <-> demovidX.mp4 <-> jdX.csv
        # Using jdX.csv as the anchor since it has the length
        joint_files = sorted(glob.glob(os.path.join(JOINT_DIR, "*.csv")))
        
        for j_path in joint_files:
            # Parse ID from filename (e.g., jd1.csv -> 1)
            base = os.path.basename(j_path)
            idx = base.replace("jd", "").replace(".csv", "")
            
            c_path = os.path.join(CAMVIEW_DIR, f"camview{idx}.mp4")
            d_path = os.path.join(DEMO_DIR, f"demovid{idx}.mp4")
            
            if os.path.exists(c_path) and os.path.exists(d_path):
                # Load CSV to check length
                df = pd.read_csv(j_path)
                num_frames = len(df)
                
                # Check video integrity
                cap_c = cv2.VideoCapture(c_path)
                cap_d = cv2.VideoCapture(d_path)
                
                len_c = int(cap_c.get(cv2.CAP_PROP_FRAME_COUNT))
                len_d = int(cap_d.get(cv2.CAP_PROP_FRAME_COUNT))
                
                cap_c.release()
                cap_d.release()
                
                # Verify we have roughly enough frames
                if len_c > 0 and len_d > 0:
                    self.samples.append({
                        "id": idx,
                        "joint_path": j_path,
                        "cam_path": c_path,
                        "demo_path": d_path,
                        "joint_data": df,
                        "len_c": len_c,
                        "len_d": len_d
                    })
                    print(f"Loaded ID {idx}: {num_frames} samples | CamFrames: {len_c} | DemoFrames: {len_d}")

        # Flatten samples: (video_idx, row_idx)
        self.indices = []
        for i, s in enumerate(self.samples):
            # We skip the very last frame because we might predict next delta
            for row_idx in range(len(s["joint_data"])):
                self.indices.append((i, row_idx))
                
        print(f"Total training samples: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx, row_idx = self.indices[idx]
        sample = self.samples[sample_idx]
        
        # 1. Get Joint Data
        # Columns: timestamp,j0...j5,d0...d5
        # We want input: current joints (j0-j5)
        # We want output: deltas (d0-d5)
        row = sample["joint_data"].iloc[row_idx]
        current_joints = row[['j0','j1','j2','j3','j4','j5']].values.astype(np.float32)
        target_deltas = row[['d0','d1','d2','d3','d4','d5']].values.astype(np.float32)
        
        # 2. Get Video Frames
        # Basic Synchronization: Percentage based
        # If we are at row_idx / total_rows, we want the video frame at that same percentage
        n_rows = len(sample["joint_data"])
        pct = row_idx / max(1, n_rows - 1)
        
        frame_c_idx = int(pct * (sample["len_c"] - 1))
        frame_d_idx = int(pct * (sample["len_d"] - 1)) # Demo frame at same progress
        
        # Load the specific frames
        # Opening/Closing capture every time is slow, but safe for low VRAM/memory
        # Optimization: We could keep caps open, but PyTorch dataloaders use multiple workers 
        # which causes issues with OpenCV's non-thread-safe captures.
        # Given small dataset, this is acceptable.
        
        img_robot = self.load_frame(sample["cam_path"], frame_c_idx)
        img_demo = self.load_frame(sample["demo_path"], frame_d_idx)
        
        if self.transform:
            img_robot = self.transform(img_robot)
            img_demo = self.transform(img_demo)
            
        return img_robot, img_demo, torch.tensor(current_joints), torch.tensor(target_deltas)

    def load_frame(self, path, frame_idx):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            # Fallback: return black image
            return np.zeros((224, 224, 3), dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# --- 3. MODEL ARCHITECTURE ---
class SiamesePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Shared Backbone: ResNet18
        resnet = models.resnet18(pretrained=True)
        
        # FREEZE early layers to save gradient memory and avoid overfitting
        for param in list(resnet.parameters())[:-10]: # Keep only last few layers trainable
            param.requires_grad = False
            
        # Remove the classification head (fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Output: [B, 512, 1, 1]
        
        # Fusion & Control Head
        # Input features: 512 (Robot) + 512 (Demo) + 6 (Joints) = 1030
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 + 6, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6) # Output: 6 joint deltas
        )

    def forward(self, img_robot, img_demo, joints):
        # Extract features
        f_robot = self.backbone(img_robot) # [B, 512, 1, 1]
        f_demo = self.backbone(img_demo)   # [B, 512, 1, 1]
        
        # Flatten
        f_robot = f_robot.view(f_robot.size(0), -1)
        f_demo = f_demo.view(f_demo.size(0), -1)
        
        # Concatenate: [Robot Img | Demo Img | Robot Joints]
        x = torch.cat([f_robot, f_demo, joints], dim=1)
        
        # Predict actions
        return self.fc(x)

# --- 4. TRAINING LOOP ---
def train():
    print(f"Using device: {DEVICE}")
    
    # Setup
    dataset = VisualImitationDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    model = SiamesePolicy().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for i, (img_r, img_d, joints, target) in enumerate(dataloader):
            img_r, img_d = img_r.to(DEVICE), img_d.to(DEVICE)
            joints, target = joints.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            
            pred_delta = model(img_r, img_d, joints)
            loss = criterion(pred_delta, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")
        
        # Save Best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Model saved! Best Loss: {best_loss:.6f}")

if __name__ == "__main__":
    train()
