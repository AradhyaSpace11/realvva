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

# --- CONFIG ---
BATCH_SIZE = 32
EPOCHS = 10 
LR = 1e-4 # Lower LR for finetuning pre-trained model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "../trained_models/bc_resnet_model.pth"

# Data Paths
DATA_ROOT = "../data" 
CAMVIEW_DIR = os.path.join(DATA_ROOT, "camview")
JOINT_DIR = os.path.join(DATA_ROOT, "jointdata")
OS_DATA_ROOT = "data" 

# Normalization Constants (Approximate limits for robot)
# We will normalize output to [-1, 1] for better regression stability
# Joints: [+/-3.14, +/-3.14, +/-3.14, +/-3.14, 0.5, -0.5]
JOINT_MIN = np.array([-3.14, -3.14, -3.14, -3.14, 0.0, -0.5])
JOINT_MAX = np.array([3.14, 3.14, 3.14, 3.14, 0.5, 0.0])
JOINT_RANGE = JOINT_MAX - JOINT_MIN + 1e-6

# --- 1. MODEL ---
class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Load Pretrained ResNet18
        self.resnet = models.resnet18(pretrained=True)
        
        # Replace last FC layer
        # ResNet18 fc input dim is 512
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6) # 6 Joint Positions
        )

    def forward(self, x):
        return self.resnet(x)

# --- 2. DATASET ---
class BCDataset(Dataset):
    def __init__(self, transform=None):
        self.samples = []
        self.transform = transform
        
        j_dir = JOINT_DIR if os.path.exists(JOINT_DIR) else os.path.join(OS_DATA_ROOT, "jointdata")
        c_dir = CAMVIEW_DIR if os.path.exists(CAMVIEW_DIR) else os.path.join(OS_DATA_ROOT, "camview")
        
        joint_files = sorted(glob.glob(os.path.join(j_dir, "*.csv")))
        print(f"Found {len(joint_files)} sessions.")

        for j_path in joint_files:
            idx = os.path.basename(j_path).replace("jd", "").replace(".csv", "")
            c_path = os.path.join(c_dir, f"camview{idx}.mp4")

            if os.path.exists(c_path):
                df = pd.read_csv(j_path)
                cap = cv2.VideoCapture(c_path)
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # We need MAPPING: Frame_t -> Joint_t (Direct Mapping)
                # Or Frame_t -> Joint_t+1 (Next Step Prediction)
                # Let's do Frame_t -> Joint_t. Behavior Cloning usually learns "State -> Action".
                # But here "Action" is "Target Position".
                
                valid_len = min(len(df), length)
                
                for i in range(valid_len):
                    # Extract Joint POSITIONS (j0-j5), not deltas (d0-d5)
                    # The CSV header is: timestamp,j0,j1,j2,j3,j4,j5,d0,d1...
                    joints = df.iloc[i][['j0','j1','j2','j3','j4','j5']].values.astype(np.float32)
                    
                    self.samples.append({
                        "video_path": c_path,
                        "frame_idx": i,
                        "joints": joints
                    })
        
        print(f"Dataset Size: {len(self.samples)} frames")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        cap = cv2.VideoCapture(s["video_path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, s["frame_idx"])
        ret, img = cap.read()
        cap.release()
        
        if not ret:
            return torch.zeros(3, 224, 224), torch.zeros(6)
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
            
        # Normalize Joints to [-1, 1]
        target = (s["joints"] - JOINT_MIN) / JOINT_RANGE # [0, 1]
        target = target * 2.0 - 1.0 # [-1, 1]
        
        return img, torch.tensor(target).float()

# --- 3. TRAINING ---
def train():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Training BC ResNet on {DEVICE}")
    
    # ImageNet normalization is critical for pretrained models
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = BCDataset(transform=transform)
    if len(dataset) == 0:
        print("No data found!")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = BCPolicy().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for imgs, targets in dataloader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            
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
