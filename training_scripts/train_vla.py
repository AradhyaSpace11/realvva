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
import math

# --- 1. CONFIGURATION ---
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 3e-4 # Slightly higher for Transformer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "vla_model.pth"

# VLA Config
NUM_BINS = 256 # Vocab size for each joint action
JOINT_LIMITS = (-0.1, 0.1) # Min/Max delta to verify in CSV
SEQ_LEN = 6 # We predict 6 joint deltas
D_MODEL = 256
N_HEAD = 4
N_LAYERS = 4

# Data Paths (Same as before)
DATA_ROOT = "data"
CAMVIEW_DIR = os.path.join(DATA_ROOT, "camview")
DEMO_DIR = os.path.join(DATA_ROOT, "demovideos") 
JOINT_DIR = os.path.join(DATA_ROOT, "jointdata")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 2. TOKENIZER ---
class ActionTokenizer:
    def __init__(self, min_val=-0.05, max_val=0.05, bins=NUM_BINS):
        self.min_val = min_val
        self.max_val = max_val
        self.bins = bins
        
    def encode(self, x):
        # x: [Batch, 6] continuous
        # Clip to range
        x = torch.clamp(x, self.min_val, self.max_val)
        # Normalize 0-1
        norm = (x - self.min_val) / (self.max_val - self.min_val)
        # Scale to bins
        ids = (norm * (self.bins - 1)).long()
        return ids # [Batch, 6]
        
    def decode(self, ids):
        # ids: [Batch, 6] integers
        norm = ids.float() / (self.bins - 1)
        x = norm * (self.max_val - self.min_val) + self.min_val
        return x

# --- 3. DATASET (Reused logic, added stats) ---
class VLADataset(Dataset):
    def __init__(self, transform=None):
        self.samples = []
        self.transform = transform
        
        joint_files = sorted(glob.glob(os.path.join(JOINT_DIR, "*.csv")))
        
        # Calculate stats for tokenizer
        all_deltas = []
        
        for j_path in joint_files:
            idx = os.path.basename(j_path).replace("jd", "").replace(".csv", "")
            c_path = os.path.join(CAMVIEW_DIR, f"camview{idx}.mp4")
            d_path = os.path.join(DEMO_DIR, f"demovid{idx}.mp4")
            
            if os.path.exists(c_path) and os.path.exists(d_path):
                df = pd.read_csv(j_path)
                
                # Collect deltas for stats
                deltas = df[['d0','d1','d2','d3','d4','d5']].values
                all_deltas.append(deltas)
                
                cap_c = cv2.VideoCapture(c_path)
                cap_d = cv2.VideoCapture(d_path)
                len_c = int(cap_c.get(cv2.CAP_PROP_FRAME_COUNT))
                len_d = int(cap_d.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_c.release()
                cap_d.release()
                
                if len_c > 0 and len_d > 0:
                    self.samples.append({
                        "id": idx, "cam_path": c_path, "demo_path": d_path,
                        "joint_data": df, "len_c": len_c, "len_d": len_d
                    })

        self.indices = []
        for i, s in enumerate(self.samples):
            for row_idx in range(len(s["joint_data"])):
                self.indices.append((i, row_idx))
        
        # Initialize Tokenizer based on data distribution
        all_deltas = np.vstack(all_deltas)
        min_d = np.percentile(all_deltas, 1) # 1st percentile to avoid outliers
        max_d = np.percentile(all_deltas, 99)
        print(f"Data Loaded. Action Range: [{min_d:.4f}, {max_d:.4f}]")
        self.tokenizer = ActionTokenizer(min_d, max_d)

    def __len__(self): 
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx, row_idx = self.indices[idx]
        sample = self.samples[sample_idx]
        
        row = sample["joint_data"].iloc[row_idx]
        current_joints = row[['j0','j1','j2','j3','j4','j5']].values.astype(np.float32)
        target_deltas = row[['d0','d1','d2','d3','d4','d5']].values.astype(np.float32)
        
        # Encode Target Actions to Tokens
        target_tokens = self.tokenizer.encode(torch.tensor(target_deltas))
        
        # Video Frames
        pct = row_idx / max(1, len(sample["joint_data"]) - 1)
        f_c = int(pct * (sample["len_c"] - 1))
        f_d = int(pct * (sample["len_d"] - 1))
        
        img_r = self.load_frame(sample["cam_path"], f_c)
        img_d = self.load_frame(sample["demo_path"], f_d)
        
        if self.transform:
            img_r = self.transform(img_r)
            img_d = self.transform(img_d)
            
        return img_r, img_d, torch.tensor(current_joints), target_tokens

    def load_frame(self, path, f_idx):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret: return np.zeros((224, 224, 3), dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# --- 4. DATA LOADER ---
dataset = VLADataset(transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# --- 5. MODEL: TinyVLA ---
class TinyVLA(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Vision Encoder (ResNet18)
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # [B, 512, 1, 1]
        for p in self.backbone.parameters(): p.requires_grad = False # Freeze
        
        # Project visual features to d_model
        self.vis_proj = nn.Linear(512, D_MODEL)
        
        # 2. Joint Encoder (Continuous -> d_model)
        self.joint_proj = nn.Linear(6, D_MODEL)
        
        # 3. Action Output Heads (6 heads, one for each joint)
        # We model them as 6 separate output tokens
        # Or we can just have one transformer outputting 6 tokens
        # For simplicity in TinyVLA: We output 6 tokens per step
        
        # Transformer Decoder
        layer = nn.TransformerDecoderLayer(d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers=N_LAYERS)
        
        # Positional Embeddings for the sequence of length 3 (RobotImg, DemoImg, Joints)
        self.pos_emb = nn.Parameter(torch.randn(1, 3, D_MODEL))
        
        # Prediction Heads (one for each of the 6 joints)
        # In a real VLA, this would be autoregressive. 
        # Here we parallel predict 6 tokens from the encoded state.
        self.heads = nn.ModuleList([nn.Linear(D_MODEL, NUM_BINS) for _ in range(6)])

    def forward(self, img_r, img_d, joints):
        # Image Features
        f_r = self.backbone(img_r).flatten(1) # [B, 512]
        f_d = self.backbone(img_d).flatten(1) # [B, 512]
        
        # Project
        emb_r = self.vis_proj(f_r).unsqueeze(1) # [B, 1, D]
        emb_d = self.vis_proj(f_d).unsqueeze(1) # [B, 1, D]
        emb_j = self.joint_proj(joints).unsqueeze(1) # [B, 1, D]
        
        # Construct Sequence: [RobotImg, DemoImg, Joints]
        seq = torch.cat([emb_r, emb_d, emb_j], dim=1) # [B, 3, D]
        seq = seq + self.pos_emb # Add position info
        
        # Transformer Pass
        # We treat 'seq' as the memory AND target for simplicity in this non-autoregressive variant
        # (This acts like a powerful cross-attention mixer)
        out = self.transformer(tgt=seq, memory=seq) # [B, 3, D]
        
        # Take the last token (Joints) context to predict actions
        context = out[:, -1, :] # [B, D]
        
        # Predict logits for each joint
        logits = [head(context) for head in self.heads] # List of 6 tensors [B, NUM_BINS]
        return logits

# --- 6. TRAINING ---
def train():
    print(f"VLA Training on {DEVICE}")
    model = TinyVLA().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for i, (img_r, img_d, joints, target_tokens) in enumerate(dataloader):
            img_r, img_d = img_r.to(DEVICE), img_d.to(DEVICE)
            joints, target_tokens = joints.to(DEVICE), target_tokens.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            logits_list = model(img_r, img_d, joints) # List of 6 [B, 256]
            
            # Loss: Sum of CE loss for each joint
            loss = 0
            correct = 0
            for j in range(6):
                l = criterion(logits_list[j], target_tokens[:, j])
                loss += l
                
                # Acc check
                preds = torch.argmax(logits_list[j], dim=1)
                correct += (preds == target_tokens[:, j]).float().mean()
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        avg_acc = correct / (len(dataloader) * 6) # approx acc
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2%}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("  -> Model Saved")

if __name__ == "__main__":
    train()
