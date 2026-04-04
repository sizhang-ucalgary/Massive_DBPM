import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder

####################################################################################
# Usage:
# python3 mlp_torch.py input_policy original_policy [--epochs EPOCHS] [--batch_size SIZE] [--lr LR]
# 
# input_policy: Path to the input policy (.npy)
# original_policy: Path to the original policy (.npy) for testing
# --epochs: Number of training epochs (default: 10)
# --batch_size: Training batch size (default: 256)
# --lr: Learning rate (default: 1e-3)
####################################################################################
def main():
    parser = argparse.ArgumentParser(description="PyTorch MLP Policy Miner")
    parser.add_argument("input_policy", help="Path to input policy (.npy)")
    parser.add_argument("original_policy", help="Path to original policy (.npy)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    # Load and Prepare Data
    pp = np.load(args.input_policy)
    op = np.load(args.original_policy)
    k, n, _ = pp.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    # Data Extraction Helper
    def extract_data(tensor):
        if tensor.dtype == np.uint8:
            # Bit-packed Mode B
            g = (tensor >> 4).astype(np.int32)
            d = (tensor & 0x0F).astype(np.int32)
            mask = (g + d > 0)
            a_idx, u_idx, v_idx = np.where(mask)
            # Resolve to majority decision (Deny-by-Default)
            y = np.where(g[a_idx, u_idx, v_idx] > d[a_idx, u_idx, v_idx], 1, 0).astype(np.int32)
            X = np.stack([a_idx, u_idx, v_idx], axis=1).astype(np.int32)
        else:
            # Standard Mode A
            a_idx, u_idx, v_idx = np.where(tensor != 2)
            y = tensor[a_idx, u_idx, v_idx].astype(np.int32)
            X = np.stack([a_idx, u_idx, v_idx], axis=1).astype(np.int32)
        return X, y

    X_tr_raw, y_tr_raw = extract_data(pp)
    X_te_raw, y_te_raw = extract_data(op)

    # Preprocessing (One-Hot Encoding)
    enc = OneHotEncoder(categories=[np.arange(k), np.arange(n), np.arange(n)], sparse_output=False, handle_unknown='ignore')
    X_tr_enc = torch.tensor(enc.fit_transform(X_tr_raw), dtype=torch.float32).to(device)
    y_tr_tensor = torch.tensor(y_tr_raw, dtype=torch.float32).unsqueeze(1).to(device)
    X_te_enc = torch.tensor(enc.transform(X_te_raw), dtype=torch.float32).to(device)

    # Model Definition (Simplified inline)
    model = nn.Sequential(
        nn.Linear(X_tr_enc.shape[1], 128), 
        nn.ReLU(),
        nn.Linear(128, 64), 
        nn.ReLU(),
        nn.Linear(64, 1), 
        nn.Sigmoid()
    ).to(device)
    
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.BCELoss()
    loader = DataLoader(TensorDataset(X_tr_enc, y_tr_tensor), batch_size=args.batch_size, shuffle=True)

    # Training
    print("\n--- PHASE 1: Training ---")
    model.train()
    for _ in range(args.epochs):
        for bx, by in loader:
            opt.zero_grad()
            crit(model(bx), by).backward()
            opt.step()

    # Evaluation
    print("\n--- PHASE 2: Testing ---")
    model.eval()
    with torch.no_grad():
        y_pred = (model(X_te_enc).cpu().numpy() >= 0.5).astype(int).flatten()

    elapsed_time = time.time() - start_time

    # Metrics
    tp = np.sum((y_pred == 1) & (y_te_raw == 1))
    tn = np.sum((y_pred == 0) & (y_te_raw == 0))
    fp = np.sum((y_pred == 1) & (y_te_raw == 0))
    fn = np.sum((y_pred == 0) & (y_te_raw == 1))
    
    accuracy = (tp + tn) / len(y_te_raw) if len(y_te_raw) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Standardized Output
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Final Domains: {params}")
    print(f"Compression Time: {elapsed_time:.2f} s")
    print("-" * 40)

if __name__ == '__main__':
    main()
