import json
import os

NOTEBOOK_PATH = "/home/aradhya/realvva/results/Project_Analysis.ipynb"

# --- NEW CELLS CONTENT ---

# 1. Header Cell
cell_header_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Control Subsystem Analysis\n",
        "We evaluate the two trained policies (MLP vs Attention) on the validation dataset.\n",
        "**Metrics:**\n",
        "- **Joint-wise MSE:** Identifying which joints are hardest to control.\n",
        "- **Inference Latency:** Benchmarking FPS for real-time suitability."
    ]
}

# 2. Model Definition & Loading Cell
source_models = [
    "# --- MODEL ARCHITECTURES (Copied for Self-Containment) ---\n",
    "class RobotPolicy(nn.Module):\n",
    "    def __init__(self, input_dim=14, output_dim=6):\n",
    "        super(RobotPolicy, self).__init__()\n",
    "        self.net = nn.Sequential(\n",
    "            nn.Linear(input_dim, 128), nn.ReLU(),\n",
    "            nn.Linear(128, 64), nn.ReLU(),\n",
    "            nn.Linear(64, output_dim)\n",
    "        )\n",
    "    def forward(self, x): return self.net(x)\n",
    "\n",
    "class RobotPolicyAttention(nn.Module):\n",
    "    def __init__(self, output_dim=6):\n",
    "        super(RobotPolicyAttention, self).__init__()\n",
    "        self.embedding = nn.Linear(2, 32)\n",
    "        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)\n",
    "        self.head = nn.Sequential(\n",
    "            nn.Flatten(),\n",
    "            nn.Linear(7 * 32, 128), nn.ReLU(),\n",
    "            nn.Linear(128, 64), nn.ReLU(),\n",
    "            nn.Linear(64, output_dim)\n",
    "        )\n",
    "    def forward(self, x):\n",
    "        seq = x.view(-1, 7, 2)\n",
    "        emb = self.embedding(seq)\n",
    "        attn_output, _ = self.attention(emb, emb, emb)\n",
    "        return self.head(attn_output)\n",
    "\n",
    "# --- LOAD DATA & MODELS ---\n",
    "if os.path.exists(DATASET_NPZ):\n",
    "    print(f\"Loading Data: {DATASET_NPZ}\")\n",
    "    data = np.load(DATASET_NPZ)\n",
    "    X_val = torch.tensor(data['X'], dtype=torch.float32)\n",
    "    Y_val = data['Y']\n",
    "else:\n",
    "    print(\"Dataset not found!\")\n",
    "    X_val, Y_val = None, None\n",
    "\n",
    "# Load MLP\n",
    "model_mlp = RobotPolicy()\n",
    "if os.path.exists(MODEL_MLP_PATH):\n",
    "    model_mlp.load_state_dict(torch.load(MODEL_MLP_PATH, map_location='cpu'))\n",
    "    print(\"MLP Loaded.\")\n",
    "else:\n",
    "    print(f\"MLP not found at {MODEL_MLP_PATH}\")\n",
    "model_mlp.eval()\n",
    "\n",
    "# Load Attention\n",
    "model_attn = RobotPolicyAttention()\n",
    "if os.path.exists(MODEL_ATTN_PATH):\n",
    "    model_attn.load_state_dict(torch.load(MODEL_ATTN_PATH, map_location='cpu'))\n",
    "    print(\"Attention Loaded.\")\n",
    "else:\n",
    "    print(f\"Attention not found at {MODEL_ATTN_PATH}\")\n",
    "model_attn.eval()\n",
    "\n",
    "print(\"Models ready for inference.\")"
]

cell_models = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": source_models
}

# 3. MSE Analysis Cell
source_mse = [
    "# --- JOINT-WISE MSE ANALYSIS ---\n",
    "if X_val is not None:\n",
    "    with torch.no_grad():\n",
    "        pred_mlp = model_mlp(X_val).numpy()\n",
    "        pred_attn = model_attn(X_val).numpy()\n",
    "\n",
    "    # Calculate MSE per joint\n",
    "    mse_mlp = np.mean((Y_val - pred_mlp)**2, axis=0)\n",
    "    mse_attn = np.mean((Y_val - pred_attn)**2, axis=0)\n",
    "\n",
    "    joints = ['J0', 'J1', 'J2', 'J3', 'J4', 'J5']\n",
    "    x = np.arange(len(joints))\n",
    "    width = 0.35\n",
    "\n",
    "    fig, ax = plt.subplots(figsize=(10, 6))\n",
    "    rects1 = ax.bar(x - width/2, mse_mlp, width, label='MLP')\n",
    "    rects2 = ax.bar(x + width/2, mse_attn, width, label='Attention')\n",
    "\n",
    "    ax.set_ylabel('Mean Squared Error (MSE)')\n",
    "    ax.set_title('Control Accuracy by Joint (Lower is Better)')\n",
    "    ax.set_xticks(x)\n",
    "    ax.set_xticklabels(joints)\n",
    "    ax.legend()\n",
    "\n",
    "    # Save and view\n",
    "    plt.savefig('joint_mse.png')\n",
    "    plt.close()\n",
    "    display(Image('joint_mse.png'))\n",
    "    \n",
    "    print(f\"Avg MSE (MLP): {np.mean(mse_mlp):.6f}\")\n",
    "    print(f\"Avg MSE (Attn): {np.mean(mse_attn):.6f}\")"
]

cell_mse = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": source_mse
}

# 4. Latency Benchmark Cell
source_latency = [
    "# --- LATENCY BENCHMARK ---\n",
    "import time\n",
    "\n",
    "def benchmark(model, x, name, loops=1000):\n",
    "    # Warmup\n",
    "    for _ in range(10): model(x)\n",
    "    \n",
    "    t0 = time.time()\n",
    "    with torch.no_grad():\n",
    "        for _ in range(loops):\n",
    "            model(x)\n",
    "    dt = time.time() - t0\n",
    "    avg_time = dt / loops\n",
    "    fps = 1 / avg_time\n",
    "    print(f\"{name}: {avg_time*1000:.3f} ms/inference | {fps:.1f} FPS\")\n",
    "    return fps\n",
    "\n",
    "if X_val is not None:\n",
    "    dummy_input = X_val[0:1] # Single sample inference\n",
    "    print(\"Benchmarking Single Sample Inference (Batch Size=1)...\")\n",
    "    fps_mlp = benchmark(model_mlp, dummy_input, \"MLP\")\n",
    "    fps_attn = benchmark(model_attn, dummy_input, \"Attention\")\n",
    "\n",
    "    # Plot FPS\n",
    "    models = ['MLP', 'Attention']\n",
    "    fps_vals = [fps_mlp, fps_attn]\n",
    "    \n",
    "    plt.figure(figsize=(6, 4))\n",
    "    plt.bar(models, fps_vals, color=['blue', 'orange'])\n",
    "    plt.ylabel('FPS (Hz)')\n",
    "    plt.title('Inference Speed (Higher is Better)')\n",
    "    plt.savefig('latency.png')\n",
    "    plt.close()\n",
    "    display(Image('latency.png'))"
]

cell_latency = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": source_latency
}

# --- APPEND AND WRITE ---
if __name__ == "__main__":
    print(f"Reading {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    # Append new cells
    nb['cells'].extend([cell_header_md, cell_models, cell_mse, cell_latency])

    print("Appending new cells...")
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print("Done. Notebook updated.")
