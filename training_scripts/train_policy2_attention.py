import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# --- CONFIG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REALVVA_ROOT = os.path.dirname(CURRENT_DIR) # /home/aradhya/realvva

# Data from YOLO approach
DATA_NPZ = os.path.join(REALVVA_ROOT, "yolodetect", "data", "policy_dataset_smoothed.npz")

# Save to trained_models
MODEL_OUT_DIR = os.path.join(REALVVA_ROOT, "trained_models")
os.makedirs(MODEL_OUT_DIR, exist_ok=True)
MODEL_OUT = os.path.join(MODEL_OUT_DIR, "policy_attention.pth")

# --- ATTENTION MODEL ---
class RobotPolicyAttention(nn.Module):
    def __init__(self, output_dim=6):
        super(RobotPolicyAttention, self).__init__()
        
        # 1. Embedding: Map each object's (x,y) to a vector
        # Input: 2 coords per object -> 32 dim embedding
        self.embedding = nn.Linear(2, 32)
        
        # 2. Positional Encoding (Optional but good for order: J0..Target)
        # For simplicity in "bare minimum", we skip explicit pos encoding 
        # because the input order is fixed (J0, J1... Target), so MLP head learns it.
        
        # 3. Self-Attention Block
        # batch_first=True -> (Batch, SeqLen, Dim)
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        
        # 4. Readout / Decision
        # We have 7 objects. After attention, we flatten 7*32 -> 224
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x is [Batch, 14] (flat list of coords)
        
        # View as [Batch, 7 objects, 2 coords]
        # (J0, J1, J2, J3, J4, J5, Target)
        seq = x.view(-1, 7, 2)
        
        # Embed -> [Batch, 7, 32]
        emb = self.embedding(seq)
        
        # Self-Attention
        # Query=emb, Key=emb, Value=emb
        attn_output, _ = self.attention(emb, emb, emb)
        
        # Residual Connection (Optional, but often helps)
        # x = emb + attn_output
        # For bare minimum, just use attn_output
        
        # Decision
        out = self.head(attn_output)
        return out

def train():
    if not os.path.exists(DATA_NPZ):
        print(f"Error: Dataset not found at {DATA_NPZ}")
        return

    print(f"Loading Dataset: {DATA_NPZ}")
    data = np.load(DATA_NPZ)
    X_raw = data['X'] 
    Y_raw = data['Y'] 
    
    X_tensor = torch.tensor(X_raw, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_raw, dtype=torch.float32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training ATTENTION Policy on: {device}")
    
    model = RobotPolicyAttention().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    epochs = 1000
    batch_size = 32
    num_samples = len(X_raw)
    
    X_tensor = X_tensor.to(device)
    Y_tensor = Y_tensor.to(device)
    
    print(f"Starting Training ({epochs} epochs)...")
    for epoch in range(epochs):
        indices = torch.randperm(num_samples)
        epoch_loss = 0.0
        batches = 0
        
        for i in range(0, num_samples, batch_size):
            idxs = indices[i:i+batch_size]
            x_batch = X_tensor[idxs]
            y_batch = Y_tensor[idxs]
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/batches:.6f}")
            
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Attention Policy Saved: {MODEL_OUT}")

if __name__ == "__main__":
    train()
