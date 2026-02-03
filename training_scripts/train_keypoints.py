import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
import torch.nn.functional as F

# --- CONFIG ---
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "keypoint_model.pth"
NUM_KEYPOINTS = 16 # Number of feature points to learn
IMG_SIZE = 128 # Smaller size for keypoint bottleneck

# Data Paths
DATA_ROOT = "data"
CAMVIEW_DIR = os.path.join(DATA_ROOT, "camview")
DEMO_DIR = os.path.join(DATA_ROOT, "demovideos")
JOINT_DIR = os.path.join(DATA_ROOT, "jointdata")

# --- 1. MODELS ---

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
        # [B, K, H, W] -> [B, K, H*W]
        B, K, H, W = feature_map.shape
        flat = feature_map.view(B, K, -1)
        
        if self.temperature:
            flat = flat / self.temperature
            
        softmax = F.softmax(flat, dim=-1)
        
        # Expected Value
        expected_x = torch.sum(self.pos_x * softmax, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax, dim=-1, keepdim=True)
        
        # [B, K, 2]
        return torch.cat([expected_x, expected_y], dim=-1)

class SpatialAutoencoder(nn.Module):
    def __init__(self, num_keypoints=NUM_KEYPOINTS):
        super().__init__()
        self.k = num_keypoints
        
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3), nn.ReLU(), # 64
            nn.Conv2d(32, 64, 5, stride=1, padding=2), nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride=2, padding=2), nn.ReLU(), # 32
            nn.Conv2d(64, num_keypoints, 5, stride=2, padding=2) # 16 (Heatmaps)
        )
        
        # Spatial Softmax
        self.ssm = SpatialSoftmax(16, 16)
        
        # Decoder (from points)
        self.fc_dec = nn.Linear(num_keypoints * 2, 256)
        self.dec_start = nn.Linear(256, 16*16*64)
        
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2), # 32
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2), # 64
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2), # 128
            nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid() 
        )

    def forward(self, x):
        # x: [B, 3, 128, 128]
        feat = self.enc(x) # [B, K, 16, 16]
        points = self.ssm(feat) # [B, K, 2]
        
        # Decode
        h = F.relu(self.fc_dec(points.view(-1, self.k * 2)))
        h = F.relu(self.dec_start(h))
        h = h.view(-1, 64, 16, 16)
        recon = self.dec(h)
        
        return recon, points

class InverseDynamics(nn.Module):
    def __init__(self, num_keypoints=NUM_KEYPOINTS):
        super().__init__()
        # Input: Point_Current (K*2) + Point_Next (K*2)
        # Output: Action (6)
        self.net = nn.Sequential(
            nn.Linear(num_keypoints * 2 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6) # Joint Deltas
        )
        
    def forward(self, p1, p2):
        x = torch.cat([p1.flatten(1), p2.flatten(1)], dim=1)
        return self.net(x)

# --- 2. DATASET ---
class KeypointDataset(Dataset):
    def __init__(self, transform=None):
        self.samples = []
        self.transform = transform
        
        joint_files = sorted(glob.glob(os.path.join(JOINT_DIR, "*.csv")))
        
        for j_path in joint_files:
            idx = os.path.basename(j_path).replace("jd", "").replace(".csv", "")
            c_path = os.path.join(CAMVIEW_DIR, f"camview{idx}.mp4")
            
            if os.path.exists(c_path):
                df = pd.read_csv(j_path)
                cap = cv2.VideoCapture(c_path)
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # We need PAIRS of (Frame_t, Frame_t+1, Action_t)
                valid_len = min(len(df), length) - 1
                
                for i in range(valid_len):
                    self.samples.append({
                        "video_path": c_path,
                        "frame_idx": i,
                        "action": df.iloc[i][['d0','d1','d2','d3','d4','d5']].values.astype(np.float32)
                    })
        
        print(f"Dataset Size: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # Load t and t+1
        cap = cv2.VideoCapture(s["video_path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, s["frame_idx"])
        
        ret1, img1 = cap.read()
        ret2, img2 = cap.read()
        cap.release()
        
        if not ret1 or not ret2:
            return torch.zeros(3,128,128), torch.zeros(3,128,128), torch.zeros(6)
            
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, torch.tensor(s["action"])

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# --- 3. TRAINING ---
def train():
    print(f"Training Keypoint Model on {DEVICE}")
    
    dataset = KeypointDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    sae = SpatialAutoencoder().to(DEVICE)
    inv = InverseDynamics().to(DEVICE)
    
    optimizer = optim.Adam(list(sae.parameters()) + list(inv.parameters()), lr=LR)
    
    criterion_recon = nn.MSELoss()
    criterion_action = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        sae.train()
        inv.train()
        total_loss = 0
        
        for img1, img2, action in dataloader:
            img1, img2, action = img1.to(DEVICE), img2.to(DEVICE), action.to(DEVICE)
            
            optimizer.zero_grad()
            
            # SAE Forward (Reconstruction)
            rec1, p1 = sae(img1)
            rec2, p2 = sae(img2)
            
            loss_rec = criterion_recon(rec1, img1) + criterion_recon(rec2, img2)
            
            # INV Forward (Action Prediction from Keypoints)
            # We treat p1, p2 as deterministic features here
            pred_action = inv(p1, p2)
            loss_act = criterion_action(pred_action, action)
            
            # Total Loss
            loss = loss_rec + 10.0 * loss_act # Weight action loss higher since points drift
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Loss: {avg:.4f}")
        
        if avg < best_loss:
            best_loss = avg
            torch.save({
                'sae': sae.state_dict(),
                'inv': inv.state_dict()
            }, MODEL_PATH)
            print("  -> Saved")

if __name__ == "__main__":
    train()
