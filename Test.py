# ======================================================
# Geo-S4FormerNet v2 — TEST-ONLY Script
# Evaluates on test CSV, restores best validation model,
# prints metrics and saves predictions + scatter plot
# ======================================================

import os, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATv2Conv
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from s4torch import S4Model

warnings.filterwarnings("ignore", message=".*torch-spline-conv.*")
warnings.filterwarnings("ignore")

# ============ Utils ============
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============ Model Components ============
class TemporalHybrid(nn.Module):
    def __init__(self, in_dim, d_model, l_max=60, kernel_size=3, dropout=0.1):
        super().__init__()
        self.s4 = S4Model(
            d_input=in_dim, d_model=d_model, d_output=d_model,
            n_blocks=1, n=d_model, l_max=l_max, collapse=False
        )
        self.tcn = nn.Conv1d(in_dim, d_model, kernel_size=kernel_size, dilation=2, padding=2)
        self.norm, self.drop = nn.LayerNorm(d_model), nn.Dropout(dropout)

    def forward(self, x):
        s4_out = self.s4(x)
        tcn_out = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.drop(self.norm(s4_out + tcn_out))


class ResidualGraphAttention(nn.Module):
    def __init__(self, in_coord_dim=2, hidden_dim=64, heads=2, num_layers=3, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_coord_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, node_coords, edge_index):
        x = self.in_proj(node_coords)
        for conv, ln in zip(self.layers, self.norms):
            res = x
            x = F.elu(conv(x, edge_index))
            x = self.drop(ln(x + res))
        return x


class CrossDomainMixer(nn.Module):
    def __init__(self, d_model, nhead=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2*d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2*d_model, d_model)
        )

    def forward(self, tokens):
        attn_out, _ = self.attn(tokens, tokens, tokens)
        x = self.ln1(tokens + attn_out)
        return self.ln2(x + self.ff(x)).mean(dim=1)


class GeoS4FormerNetV2(nn.Module):
    def __init__(self, seq_len, extra_dim, hidden=64, heads=2, dropout=0.1):
        super().__init__()
        self.temporal = TemporalHybrid(1, hidden, seq_len, dropout=dropout)
        self.spatial = ResidualGraphAttention(2, hidden, heads, 3, dropout)
        self.extra_mlp = nn.Sequential(
            nn.Linear(extra_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden)
        )
        self.cross = CrossDomainMixer(hidden, heads, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1), nn.Softplus()
        )

    def forward(self, x_temp, node_idx, node_coords, edge_index, extras):
        ht = self.temporal(x_temp)[:, -1, :]
        hg = self.spatial(node_coords, edge_index)[node_idx]
        he = self.extra_mlp(extras)
        return self.head(self.cross(torch.stack([ht, hg, he], dim=1)))


# ============ Dataset Loader (TEST) ============
class GridAccidentDatasetTest(Dataset):
    def __init__(self, df, seq_len=60, k_neighbors=8):

        # === Day-of-Week Cyclical Encoding ===
        if "day_of_week" in df.columns:
        # Adjust from 1–7 (Sunday=1 ... Saturday=7) to 0–6
            df["dow_adj"] = (df["day_of_week"] - 1).astype(int)
        # Cyclical features
            df["dow_sin"] = np.sin(2 * np.pi * df["dow_adj"] / 7)
            df["dow_cos"] = np.cos(2 * np.pi * df["dow_adj"] / 7)
        # Drop raw integer columns
            df = df.drop(columns=["day_of_week", "dow_adj"], errors='ignore')

        df = df.sort_values(by=['grid_id', 'date'])
        grid_info = df[['grid_id', 'centroid_lat', 'centroid_lon']].drop_duplicates()
        grid_info = grid_info.dropna(subset=['centroid_lat', 'centroid_lon'])

        self.grid_list = list(grid_info['grid_id'].values)
        self.grid_id_to_idx = {gid: i for i, gid in enumerate(self.grid_list)}

        exclude = ['date', 'time', 'grid_id', 'centroid_lat',
                   'centroid_lon', 'number_of_casualties','number_of_vehicles', 'label']
        feature_cols = [c for c in df.columns if c not in exclude and
                        df[c].dtype in ['float64', 'int64', 'int32']]
        self.feature_cols = feature_cols
        print(f"📊 (TEST) Using {len(feature_cols)} features as extras")

        df['label'] = df.groupby('grid_id')['number_of_casualties'].shift(-1)
        df = df.dropna(subset=['label'])

        sequences, labels, extras_list, node_indices = [], [], [], []
        for grid_id, group in df.groupby('grid_id'):
            group = group.reset_index(drop=True)
            if len(group) < seq_len: continue
            node_idx = self.grid_id_to_idx[grid_id]
            for i in range(len(group) - seq_len + 1):
                w = group.iloc[i:i+seq_len]
                sequences.append(w['number_of_casualties'].values.astype(np.float32))
                labels.append(np.float32(w['label'].values[-1]))
                extras_list.append(w[feature_cols].values[-1].astype(np.float32))
                node_indices.append(node_idx)

        self.x_temp = torch.from_numpy(np.array(sequences)).unsqueeze(-1)
        self.y = torch.from_numpy(np.array(labels)).unsqueeze(1)
        self.extras = torch.from_numpy(np.array(extras_list))
        self.node_idx = torch.from_numpy(np.array(node_indices, dtype=np.int64))

        coords = grid_info[['centroid_lat', 'centroid_lon']].values.astype(np.float32)
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors+1, len(coords))).fit(coords)
        _, idx = nbrs.kneighbors(coords)
        edges = [[i, j] for i in range(idx.shape[0]) for j in idx[i, 1:]]
        edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0))
        self.edge_index = torch.from_numpy(edge_index).contiguous()
        self.x_graph = torch.from_numpy(coords)

        print(f"✅ (TEST) Created {len(self.y):,} sequences from {len(self.grid_list):,} grid cells")
        print(f"🔗 (TEST) Graph: {len(coords)} nodes, {self.edge_index.shape[1]} edges")

    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return self.x_temp[idx], self.extras[idx], self.node_idx[idx], self.y[idx]


# ============ Evaluation Function ============
def evaluate_on_test(test_csv, model_path,
                     seq_len=60, batch_size=64, k_neighbors=8,
                     out_csv="preds.csv", out_png="scatter.png"):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

    checkpoint = torch.load(model_path, map_location=device)
    cfg, norm = checkpoint['config'], checkpoint['normalization']

    df = pd.read_csv(test_csv)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    test_ds = GridAccidentDatasetTest(df, seq_len, k_neighbors)

    if test_ds.extras.shape[1] != cfg['extra_dim']:
        raise ValueError(f"Feature mismatch: TEST={test_ds.extras.shape[1]} vs TRAIN={cfg['extra_dim']}")

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=torch.cuda.is_available())

    model = GeoS4FormerNetV2(
        cfg['seq_len'], cfg['extra_dim'],
        cfg['hidden_dim'], cfg['num_heads'], cfg['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    node_coords, edge_index = test_ds.x_graph.to(device), test_ds.edge_index.to(device)
    mean_x, std_x = norm['mean_x'], norm['std_x']

    print("\n🧪 Running inference...")
    y_true, y_pred = [], []
    bad_batches = 0
    with torch.no_grad():
        for x_temp, extras, node_idx, y in loader:
            x_norm = (x_temp - mean_x) / (std_x + 1e-8)
            x_norm, extras, node_idx, y = x_norm.to(device), extras.to(device), node_idx.to(device), y.to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = model(x_norm, node_idx, node_coords, edge_index, extras)

            if torch.isnan(pred).any() or torch.isinf(pred).any():
                bad_batches += 1
                continue

            y_true.append(y.cpu().numpy()); y_pred.append(pred.cpu().numpy())

    print(f"⚠️ Skipped {bad_batches} bad batches during inference")

    if not y_true or not y_pred:
        print("❌ No valid predictions generated.")
        return {'rmse': float("nan"), 'mae': float("nan"), 'r2': float("nan")}

    y_true, y_pred = np.concatenate(y_true).ravel(), np.concatenate(y_pred).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 TEST Performance — RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).to_csv(out_csv, index=False)
    print(f"💾 Predictions saved to: {out_csv}")

    try:
        plt.figure(figsize=(6,6))
        plt.scatter(y_true, y_pred, alpha=0.4, s=10)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        plt.plot(lims, lims, 'r--')
        plt.xlabel("Actual"); plt.ylabel("Predicted")
        plt.title(f"Scatter (R²={r2:.3f})"); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
        print(f"🖼️ Scatter plot saved to: {out_png}")
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")

    return {'rmse': rmse, 'mae': mae, 'r2': r2}


# ============ Main ============
if __name__ == '__main__':
    evaluate_on_test(
        test_csv="Test_GNO.csv",
        model_path="uk_accident_model_trained.pth",   # <-- your saved checkpoint from training
        out_csv="test_predictions.csv",
        out_png="test_scatter.png"
    )
