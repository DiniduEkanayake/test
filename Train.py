# Geo-S4FormerNet v2 — Training on Pre-Aggregated Dataset
# Includes: Residual GAT, Temporal Hybrid (S4+TCN), Cross-Attention Fusion, Safe Poisson Loss + Validation + Early Stopping

import os, math, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore", message=".*torch-spline-conv.*")
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.nn import GATv2Conv
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from s4torch import S4Model
warnings.filterwarnings('ignore')


# ============ Utils ============

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============ Early Stopping ============

class EarlyStopping:
    def __init__(self, patience=7, delta=0.001, path="best_model.pt"):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = float("inf")
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)  # save best model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ============ Model Components ============

class TemporalHybrid(nn.Module):
    """Temporal encoder = S4 + Dilated TCN (residual fusion)."""
    def __init__(self, in_dim, d_model, l_max=60, kernel_size=3, dropout=0.1):
        super().__init__()
        self.s4 = S4Model(
            d_input=in_dim, d_model=d_model, d_output=d_model,
            n_blocks=1, n=d_model, l_max=l_max, collapse=False
        )
        self.tcn = nn.Conv1d(in_dim, d_model, kernel_size=kernel_size, dilation=2, padding=2)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        s4_out = self.s4(x)
        tcn_out = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        h = s4_out + tcn_out
        h = torch.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6)
        h = self.drop(self.norm(h))
        return h


class ResidualGraphAttention(nn.Module):
    """Stacked GAT with residuals + norm."""
    def __init__(self, in_coord_dim=2, hidden_dim=64, heads=2, num_layers=3, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_coord_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            self.layers.append(conv)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, node_coords, edge_index):
        x = self.in_proj(node_coords)
        for conv, ln in zip(self.layers, self.norms):
            res = x
            x = conv(x, edge_index)
            x = F.elu(x)
            x = ln(x + res)   # residual
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            x = self.drop(x)
        return x

class CrossDomainMixer(nn.Module):
    """Cross-attention fusion of spatial, temporal, and extras."""
    def __init__(self, d_model, nhead=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2*d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, tokens):
        attn_out, _ = self.attn(tokens, tokens, tokens)
        x = self.ln1(tokens + attn_out)
        x = self.ln2(x + self.ff(x))
        return x.mean(dim=1)


class GeoS4FormerNetV2(nn.Module):
    def __init__(self, seq_len, extra_dim, hidden=64, heads=2, dropout=0.1):
        super().__init__()
        self.drop_prob = dropout   # store dropout probability

        self.temporal = TemporalHybrid(
            in_dim=1, d_model=hidden, l_max=seq_len, dropout=dropout
        )
        self.spatial = ResidualGraphAttention(
            in_coord_dim=2, hidden_dim=hidden, heads=heads, num_layers=3, dropout=dropout
        )

        # Extra features MLP (dropout replaced by Identity)
        self.extra_mlp = nn.Sequential(
            nn.Linear(extra_dim, hidden),
            nn.ReLU(),
            nn.Identity(),   # placeholder, dropout applied in forward_extra
            nn.Linear(hidden, hidden),
        )

        self.cross = CrossDomainMixer(d_model=hidden, nhead=heads, dropout=dropout)

        # Prediction head (dropout replaced by Identity)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Identity(),   # placeholder, dropout applied in forward_head
            nn.Linear(hidden // 2, 1),
            nn.Identity()
        )

    # --- Safe forward for extras ---
    def forward_extra(self, extras):
        he = self.extra_mlp[0](extras)   # Linear
        he = self.extra_mlp[1](he)       # ReLU
        he = torch.nan_to_num(he, nan=0.0, posinf=1e6, neginf=-1e6)
        he = F.dropout(he, p=self.drop_prob, training=self.training)
        he = self.extra_mlp[3](he)       # Linear
        return he

    # --- Safe forward for prediction head ---
    def forward_head(self, h_fused):
        h = self.head[0](h_fused)        # Linear
        h = self.head[1](h)              # GELU
        h = torch.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6)
        h = F.dropout(h, p=self.drop_prob, training=self.training)
        h = self.head[3](h)              # Linear
        h = self.head[4](h)              # Identity
        return h

    # --- Main forward ---
    def forward(self, x_temp, node_idx, node_coords, edge_index, extras):
        ht_seq = self.temporal(x_temp)
        ht = ht_seq[:, -1, :]                  # last timestep
        hg_all = self.spatial(node_coords, edge_index)
        hg = hg_all[node_idx]
        he = self.forward_extra(extras)        # safe extra forward
        tokens = torch.stack([ht, hg, he], dim=1)
        h_fused = self.cross(tokens)
        return self.forward_head(h_fused)      # safe head forward


# ============ Dataset Loader ============

class GridAccidentDataset(Dataset):
    def __init__(self, df, seq_len=60, k_neighbors=8, grid_id_to_idx=None):
        df = df.sort_values(by=['grid_id', 'date', 'time'])

        # === Day-of-Week Cyclical Encoding ===
        if "day_of_week" in df.columns:
            # Adjust from 1–7 (Sunday=1 ... Saturday=7) to 0–6
            df["dow_adj"] = (df["day_of_week"] - 1).astype(int)

            # Cyclical features
            df["dow_sin"] = np.sin(2 * np.pi * df["dow_adj"] / 7)
            df["dow_cos"] = np.cos(2 * np.pi * df["dow_adj"] / 7)

            # Drop raw integer columns
            df = df.drop(columns=["day_of_week", "dow_adj"])

        # If mapping provided, use it (ensures consistent indexing)
        if grid_id_to_idx is not None:
            self.grid_id_to_idx = grid_id_to_idx
            self.grid_list = list(grid_id_to_idx.keys())
        else:
            grid_info = df[['grid_id', 'centroid_lat', 'centroid_lon']].drop_duplicates()
            grid_info = grid_info.dropna(subset=['centroid_lat', 'centroid_lon'])
            self.grid_list = list(grid_info['grid_id'].values)
            self.grid_id_to_idx = {gid: i for i, gid in enumerate(self.grid_list)}

        exclude_cols = [
            'date', 'time', 'grid_id',
            'centroid_lat', 'centroid_lon',
            'number_of_casualties',
            'number_of_vehicles',
            'label'
        ]
        all_cols = df.columns.tolist()
        feature_cols = [
            c for c in all_cols if c not in exclude_cols
            and df[c].dtype in ['float64', 'int64', 'int32']
        ]
        print(f"📊 Using {len(feature_cols)} features as extras")

        df['label'] = df.groupby('grid_id')['number_of_casualties'].shift(-1)
        df = df.dropna(subset=['label'])

        sequences, labels, extras_list, node_indices = [], [], [], []
        for grid_id, group in df.groupby('grid_id'):
            group = group.reset_index(drop=True)
            if len(group) < seq_len:
                continue
            node_idx = self.grid_id_to_idx[grid_id]

            for i in range(len(group) - seq_len + 1):
                window = group.iloc[i:i+seq_len]
                sequences.append(window['number_of_casualties'].values.astype(np.float32))
                labels.append(np.float32(window['label'].values[-1]))
                extras_list.append(window[feature_cols].values[-1].astype(np.float32))
                node_indices.append(node_idx)

        self.x_temp = torch.from_numpy(np.array(sequences)).unsqueeze(-1)
        self.y = torch.from_numpy(np.array(labels)).unsqueeze(1)
        self.extras = torch.from_numpy(np.array(extras_list))
        self.node_idx = torch.from_numpy(np.array(node_indices, dtype=np.int64))

        coords = grid_info[['centroid_lat', 'centroid_lon']].values.astype(np.float32)
        self.coords = coords
        if len(coords) > 0:
            nbrs = NearestNeighbors(n_neighbors=min(k_neighbors+1, len(coords))).fit(coords)
            _, idx = nbrs.kneighbors(coords)
            edges = []
            for i in range(idx.shape[0]):
                for j in range(1, idx.shape[1]):
                    edges.append([i, idx[i, j]])
            edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
            self.edge_index = torch.from_numpy(edge_index).contiguous()
            self.x_graph = torch.from_numpy(coords)

            # Sanity check: ensure all node indices are valid
            max_node_idx = self.edge_index.max().item() if self.edge_index.numel() > 0 else -1
            if self.node_idx.max() > max_node_idx:
                raise ValueError("❌ Node index mismatch: dataset has nodes not in edge_index")


        print(f"✅ Created {len(self.y):,} sequences from {len(self.grid_list):,} grid cells")
        print(f"🔗 Graph: {len(coords)} nodes, {self.edge_index.shape[1]} edges")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_temp[idx], self.extras[idx], self.node_idx[idx], self.y[idx]

# ============ Training Utils ============

def compute_stats(loader):
    n, sum_x, sum_x2 = 0, 0.0, 0.0
    sum_y, sum_y2, n_y = 0.0, 0.0, 0
    for x_temp, _, _, y in loader:
        b, l, _ = x_temp.shape
        n += b * l
        sum_x += x_temp.sum().item()
        sum_x2 += (x_temp ** 2).sum().item()
        n_y += y.numel()
        sum_y += y.sum().item()
        sum_y2 += (y ** 2).sum().item()
    mean_x = sum_x / n
    std_x = math.sqrt(max(sum_x2 / n - mean_x ** 2, 1e-8))
    mean_y = sum_y / n_y
    std_y = math.sqrt(max(sum_y2 / n_y - mean_y ** 2, 1e-8))
    return mean_x, std_x, mean_y, std_y

# ============ Validation Function ============

def evaluate(model, val_loader, device, mean_x, std_x, node_coords, edge_index, mean_log_y, std_log_y):
    model.eval()
    val_loss, y_true_all, y_pred_all = 0.0, [], []
    with torch.no_grad():
        for x_temp, extras, node_idx, y in val_loader:
            # Normalize inputs
            x_norm = (x_temp - mean_x) / (std_x + 1e-8)

            # Log-transform targets
            y_log = torch.log(y + 1)

            # Move to device
            x_norm = x_norm.to(device)
            y = y.to(device)
            y_log = y_log.to(device)
            extras = extras.to(device)
            node_idx = node_idx.to(device)

            # Forward pass
            pred = model(x_norm, node_idx, node_coords, edge_index, extras)

            # Log-transform preds
            pred_log = torch.log(torch.clamp(pred, min=1e-8) + 1)

            # Normalize
            y_log_norm = (y_log - mean_log_y) / (std_log_y + 1e-8)
            pred_log_norm = (pred_log - mean_log_y) / (std_log_y + 1e-8)

            # Loss
            loss = F.mse_loss(pred_log_norm, y_log_norm)

            y_true_all.append(y.cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())

    val_loss /= len(val_loader)
    y_true_all = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred_all = np.concatenate(y_pred_all) if y_pred_all else np.array([])

    if len(y_true_all) > 0:
        rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        mae = mean_absolute_error(y_true_all, y_pred_all)
        r2 = r2_score(y_true_all, y_pred_all)
    else:
        rmse = mae = r2 = float('nan')

    return val_loss, rmse, mae, r2

# ============ Training Function ============
def train_uk_accident_model(
    csv_path,
    seq_len=60,
    num_epochs=50,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-2,
    k_neighbors=8,
    hidden_dim=64,
    num_heads=2,
    dropout=0.15,
    model_path="uk_accident_model_trained.pth",
    plot_curves=True,
    device=None,
    accum_steps=2,          # NEW: gradient accumulation steps
    val_interval=1          # NEW: validate every N epochs
):
    set_seed(42)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    torch.autograd.set_detect_anomaly(True)

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')


    # === FIX for NaNs/empties ===
    before = len(df)

# Critical columns (must not be NaN)
    critical_cols = ['number_of_casualties', 'grid_id', 'date', 'time', 'centroid_lat', 'centroid_lon']
    df = df.dropna(subset=critical_cols)

    # Fill numeric NaNs with median, categorical with mode
    for col in df.columns:
        if col in critical_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    after = len(df)
    print(f"🧹 Cleaned dataset: dropped {before - after} rows (critical NaNs), imputed the rest")

    dataset = GridAccidentDataset(df, seq_len=seq_len, k_neighbors=k_neighbors)
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    if len(dataset) == 0:
        print("❌ No valid sequences found.")
        return

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # ENHANCED: DataLoader optimization
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=torch.cuda.is_available(), drop_last=False,
        prefetch_factor=4, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=torch.cuda.is_available(),
        prefetch_factor=4, persistent_workers=True
    )

    mean_x, std_x, mean_y, std_y = compute_stats(train_loader)
    
    # ENHANCED: Compute log-transform stats for count data
    print("🔧 Computing normalization statistics...")
    log_y_values = []
    for _, _, _, y in train_loader:
        log_y = torch.log(y + 1)  # +1 to handle zeros
        log_y_values.append(log_y)
    
    if log_y_values:
        all_log_y = torch.cat(log_y_values)
        mean_log_y = all_log_y.mean()
        std_log_y = all_log_y.std()
        print(f"   X: μ={mean_x:.3f}, σ={std_x:.3f}")
        print(f"   Y: μ={mean_y:.3f}, σ={std_y:.3f}")
        print(f"   Log(Y+1): μ={mean_log_y:.3f}, σ={std_log_y:.3f}")
    else:
        mean_log_y = std_log_y = 0.0

    extra_dim = dataset.extras.shape[1]
    model = GeoS4FormerNetV2(
        seq_len=seq_len,
        extra_dim=extra_dim,
        hidden=hidden_dim,
        heads=num_heads,
        dropout=dropout
    ).to(device)
    
    # ENHANCED: Compile model for PyTorch 2.0+ performance
    # try:
    #     model = torch.compile(model, mode='reduce-overhead')
    #     print("✅ Model compiled for enhanced performance")
    # except Exception as e:
    #     print(f"⚠️ Model compilation failed (using eager mode): {e}")

    node_coords = dataset.x_graph.to(device)
    edge_index = dataset.edge_index.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # FIXED: Scheduler will be stepped once per epoch, not per batch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    use_amp = True

    history = {'train_loss': [], 'train_rmse': [], 'train_mae': [], 'train_r2': [],
               'val_loss': [], 'val_rmse': [], 'val_mae': [], 'val_r2': []}
    
    writer = SummaryWriter(log_dir="runs/GeoS4FormerNet_v2")
    early_stopping = EarlyStopping(patience=7, delta=0.001, path="best_model.pt")

    print(f"\n🚀 Training for {num_epochs} epochs (AMP + Gradient Accumulation x{accum_steps})")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, y_true_all, y_pred_all = 0.0, [], []
        bad_batches = 0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        for i, (x_temp, extras, node_idx, y) in enumerate(train_loader):
            # Normalize inputs
            x_norm = (x_temp - mean_x) / (std_x + 1e-8)

            # Log-transform targets
            y_log = torch.log(y + 1)

            # === Normalize log targets ===
            y_log_norm = (y_log - mean_log_y) / (std_log_y + 1e-8)

            # Move to device
            x_norm = x_norm.to(device)
            y = y.to(device)
            y_log_norm = y_log_norm.to(device)
            extras = extras.to(device)
            node_idx = node_idx.to(device)

            # Sanitize inputs
            x_norm = torch.nan_to_num(x_norm, nan=0.0, posinf=1e6, neginf=-1e6)
            y      = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

            if not x_norm.numel() or not y.numel():
                print(f"⚠️ Skipping invalid batch {i}")
                bad_batches += 1
                continue

            # === Forward pass (AMP) ===
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                pred = model(x_norm, node_idx, node_coords, edge_index, extras)

                # Log-transform + normalize preds
                pred_log = torch.log(torch.clamp(pred, min=1e-8) + 1)
                pred_log_norm = (pred_log - mean_log_y) / (std_log_y + 1e-8)

                # Normalized log-MSE loss
                loss = F.mse_loss(pred_log_norm, y_log_norm) / accum_steps


            if torch.isnan(loss).any() or torch.isnan(pred).any():
                bad_batches += 1
                continue

            # Monitor collapse
            pred_std = pred.std().item()
            if pred_std < 0.01:
                print(f"⚠️ Low prediction variance detected: {pred_std:.6f}")

            # === Backward pass with accumulation ===
            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                if epoch % 10 == 0 and i == 0:
                    print(f"   Epoch {epoch}, Gradient Norm: {total_norm:.4f}")

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * accum_steps
            y_true_all.append(y.cpu().numpy())
            y_pred_all.append(pred.detach().cpu().numpy())

        scheduler.step()

        train_loss /= max(1, (len(train_loader) - bad_batches))
        y_true_all = np.concatenate(y_true_all) if y_true_all else np.array([])
        y_pred_all = np.concatenate(y_pred_all) if y_pred_all else np.array([])

        if len(y_true_all) > 0:
            rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
            mae = mean_absolute_error(y_true_all, y_pred_all)
            r2 = r2_score(y_true_all, y_pred_all)
        else:
            rmse = mae = r2 = float('nan')

        history['train_loss'].append(train_loss)
        history['train_rmse'].append(rmse)
        history['train_mae'].append(mae)
        history['train_r2'].append(r2)

        # === ORIGINAL: Validation (with original logic pattern) ===
        if epoch % val_interval == 0:
            val_loss, val_rmse, val_mae, val_r2 = evaluate(
            model, val_loader, device,
            mean_x, std_x, node_coords, edge_index,
            mean_log_y, std_log_y
        )
        else:
            val_loss = val_rmse = val_mae = val_r2 = float('nan')


        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)

        # === ORIGINAL: Progress reporting with enhanced gradient monitoring ===
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{num_epochs} | {elapsed:5.1f}s | "
              f"Train Loss: {train_loss:.4f} | RMSE: {rmse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f} "
              f"| Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.3f} | Val MAE: {val_mae:.3f} | Val R²: {val_r2:.3f} "
              f"| ⚠️ Skipped {bad_batches} bad batches")
        
        # ENHANCED: Show learning rate and gradient norm info periodically
        if epoch % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"   📊 Current LR: {current_lr:.2e}")

        # === Log metrics to TensorBoard ===
        writer.add_scalar('Learning Rate', current_lr, epoch)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("RMSE/train", rmse, epoch)
        writer.add_scalar("RMSE/val", val_rmse, epoch)
        writer.add_scalar("MAE/train", mae, epoch)
        writer.add_scalar("MAE/val", val_mae, epoch)
        writer.add_scalar("R2/train", r2, epoch)
        writer.add_scalar("R2/val", val_r2, epoch)

        # Log current learning rate
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("LR", current_lr, epoch)

        # === ORIGINAL: Early stopping logic ===
        if epoch % val_interval == 0:
            early_stopping(val_rmse, model)
            if early_stopping.early_stop:
                print("⏹️ Early stopping triggered")
                break

    # === ORIGINAL: Convergence Dashboard (only if enabled) ===
    if plot_curves:
        metrics = ["loss", "rmse", "mae", "r2"]
        titles = ["Loss", "RMSE", "MAE", "R²"]

        plt.figure(figsize=(14, 10))
        for i, (m, title) in enumerate(zip(metrics, titles), 1):
            plt.subplot(2, 2, i)
            plt.plot(history[f"train_{m}"], label=f"Train {title}", marker="o")
            plt.plot(history[f"val_{m}"], label=f"Val {title}", marker="s")
            plt.xlabel("Epoch")
            plt.ylabel(title)
            plt.title(f"Training vs Validation {title}")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig("convergence_dashboard.png", dpi=300)
        plt.show()

    # === ORIGINAL: Load and save best model ===
    model.load_state_dict(torch.load("best_model.pt"))

    # ENHANCED: Save with comprehensive information including log-transform stats
    torch.save({
        'model': model.state_dict(),
        'config': {
            'seq_len': seq_len,
            'extra_dim': extra_dim,
            'hidden_dim': hidden_dim,
            'num_heads': num_heads,
            'dropout': dropout
        },
        'normalization': {
            'mean_x': mean_x, 'std_x': std_x,
            'mean_y': mean_y, 'std_y': std_y,
            'mean_log_y': mean_log_y, 'std_log_y': std_log_y  # Added log-transform stats
        },
        'history': history,  # Added training history
        'hyperparameters': {  # Added hyperparameters for reference
            'lr': lr,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'accum_steps': accum_steps,
        }
    }, model_path)

    print(f"\n✅ Training complete. Model saved to {model_path}")
    
    # === ORIGINAL: Return format maintained ===
    return model, {'rmse': rmse, 'mae': mae, 'r2': r2}

# ============ Main Entry Point ============

if __name__ == "__main__":

    def objective(trial):
    # === Suggested hyperparameters ===
        # Learning-related params
        lr = trial.suggest_loguniform("lr", 1e-4, 2e-3)
        dropout = trial.suggest_uniform("dropout", 0.1, 0.3)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

        # Architecture params
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        seq_len = trial.suggest_categorical("seq_len", [30, 60, 90])
        k_neighbors = trial.suggest_categorical("k_neighbors", [4, 6, 8, 12])

        # Batch size
        batch_size = trial.suggest_categorical("batch_size", [16, 32])


         # === Train with these hyperparameters ===
        _, metrics = train_uk_accident_model(
            csv_path="Train_GNO.csv",
            seq_len=seq_len,
            num_epochs=6,     # shorter runs for tuning
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            k_neighbors=k_neighbors,
            model_path=f"model_trial_{trial.number}.pth",
            plot_curves=False,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        torch.cuda.empty_cache()
        return metrics['rmse']
    # === Run Bayesian Optimisation with pruning ===
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=2,   # let a couple full trials run before pruning
        n_warmup_steps=3,     # require 3 epochs before pruning
        interval_steps=1
    )

# === Run Bayesian optimisation ===
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=3)

    print("\n🎯 Best Trial Results:")
    print(f"  Trial #{study.best_trial.number}")
    print(f"  RMSE: {study.best_trial.value:.4f}")
    print("  Hyperparameters:", study.best_trial.params)

    # === STEP 4: Retrain the best model with more epochs + convergence plots ===
    best_params = study.best_trial.params

    model, final_metrics = train_uk_accident_model(
        csv_path="Train_GNO.csv",
        seq_len=best_params['seq_len'],
        num_epochs=50,    # longer run for convergence
        batch_size=best_params['batch_size'],
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
        hidden_dim=best_params['hidden_dim'],
        num_heads=best_params['num_heads'],
        dropout=best_params['dropout'],
        k_neighbors=best_params['k_neighbors'],
        model_path="final_best_model.pth",
        plot_curves=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )


    print("\n📊 Final Training Metrics (Best Params):")
    print(final_metrics)
    print("✅ Final model saved as final_best_model.pth")



