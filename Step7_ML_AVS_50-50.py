# ------------------------------------------------------------
# NTU Capstone Project: Group Cohesion Estimation using LLM and Deep Learning
# Maryam Ali Aljanobi
# April 2026
#
# File: Step7_ML_AVS_50-50.py
# -------------------------------------------------------------------
# This experiment is set to use different split of the dataset: 50/50 SPLIT (train50/test50) and run on GPU:
# Features used for the ML model (best setup from previous experiments):
# AV + Spectrogram (Reduced) - COHESION ONLY 
#   - wav2vec   : 1024 -> 100 (learned projection)
#   - OpenSMILE :   88 ->  32 (learned projection)
#   - SlowFast  : 2304 -> 100 (learned projection)
#   - Spectrogram: (3,128,512) -> SE-CNN -> 64
#   - Early fusion by concatenation: 100+32+100+64 = 296
#   - Early stopping on VALID MAE
#   - Multi-seed support (default 5 seeds)
#
# -------------------------------------------------------------------

import os
import glob
import json
import random
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------- Args ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="cuda or cpu")
    ap.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    return ap.parse_args()

ARGS = parse_args()

# ---------------- Paths ----------------
ROOT = "/home/msds/maryam007/gce_project"
DATA = os.path.join(ROOT, "inputs")
FEAT = os.path.join(DATA, "features")
RESULTS = os.path.join(ROOT, "results/ML50-50")
os.makedirs(RESULTS, exist_ok=True)

INDEX_PATH = os.path.join(DATA, "data_index.csv")

W2V_PATH = os.path.join(FEAT, "audio_wav2vec.csv")
SMILE_PATH = os.path.join(FEAT, "audio_smile.csv")
SLOWFAST_DIR = os.path.join(FEAT, "slowfast")
SPEC_DIR = os.path.join(FEAT, "spectrogram")

OUT_DIR = os.path.join(RESULTS, "avs_fusion_reduced_multiseed_gpu")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Device ----------------
DEVICE = ARGS.device if (ARGS.device == "cpu" or torch.cuda.is_available()) else "cpu"
PIN_MEMORY = (DEVICE == "cuda")

# ---------------- Config (keep consistent with your step6 baseline) ----------------
SEEDS = [0, 1, 2, 3, 4]

MAX_EPOCHS = 80
PATIENCE = 10
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_EVAL = 256

LR = 1e-3
WEIGHT_DECAY = 1e-4

# Spectrogram preprocess
SPEC_SHAPE = (128, 512)  # (H,W)

# ---------------- Helpers ----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_table_auto(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=None, engine="python")
    return df

def guess_id_col(df: pd.DataFrame) -> str:
    for c in ["VID", "vid", "video_id", "video", "id", "title"]:
        if c in df.columns:
            return c
    return df.columns[0]

def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    return mu, sigma

def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (X - mu) / sigma

def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def mse(y_true, y_pred) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

# Your old MAPE (fraction, not percent)
def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))))

def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)

def corr_metrics(y_true, y_pred):
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan"), float("nan")
    return float(pearsonr(y_true, y_pred)[0]), float(spearmanr(y_true, y_pred)[0])

def pack_metrics(y_true, y_pred):
    p, s = corr_metrics(y_true, y_pred)
    return {
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Pearson": p,
        "Spearman": s,
    }

def format_metrics(m: Dict[str, float]) -> str:
    keys = ["MAE", "MAPE", "MSE", "R2", "Pearson", "Spearman"]
    return "\n".join([f"{k}: {m.get(k, float('nan')):.6f}" for k in keys]) + "\n"

def mean_std_table(all_metrics: List[Dict[str, float]]) -> str:
    keys = ["MAE", "MAPE", "MSE", "R2", "Pearson", "Spearman"]
    lines = []
    for k in keys:
        vals = np.array([m.get(k, np.nan) for m in all_metrics], dtype=np.float64)
        lines.append(f"{k}: {np.nanmean(vals):.6f} ± {np.nanstd(vals):.6f}")
    return "\n".join(lines) + "\n"

# ---------------- Feature loaders ----------------
def load_slowfast_vec(vid: str) -> np.ndarray:
    path = os.path.join(SLOWFAST_DIR, f"{vid}.npz")
    if not os.path.exists(path):
        matches = glob.glob(os.path.join(SLOWFAST_DIR, f"{vid}*.npz"))
        if not matches:
            raise FileNotFoundError(f"SlowFast not found for vid={vid}")
        path = matches[0]

    loader = np.load(path, allow_pickle=True)
    feat = loader["feat"] if "feat" in loader else loader[list(loader.keys())[0]]
    feat = np.array(feat)
    if feat.ndim == 2:
        vec = feat[0] if feat.shape[0] == 1 else feat.mean(axis=0)
    elif feat.ndim == 1:
        vec = feat
    else:
        vec = feat.reshape(-1)
    return vec.astype(np.float32)

def find_spec_path(vid: str) -> str:
    p1 = os.path.join(SPEC_DIR, f"{vid}.jpg")
    if os.path.exists(p1):
        return p1
    matches = glob.glob(os.path.join(SPEC_DIR, vid, "*.jpg"))
    if matches:
        return matches[0]
    matches = glob.glob(os.path.join(SPEC_DIR, f"*{vid}*.jpg"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Spectrogram jpg not found for vid={vid}")

# ---------------- Model ----------------
class FeatureProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )
    def forward(self, x):
        return self.proj(x)

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s.view(b, c, 1, 1)

class SpectrogramSECNN(nn.Module):
    def __init__(self, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.se1 = SEBlock(32)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.se2 = SEBlock(64)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.se3 = SEBlock(128)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.se1(F.relu(self.bn1(self.conv1(x))))
        x = self.se2(F.relu(self.bn2(self.conv2(x))))
        x = self.se3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B,128)
        x = self.drop(x)
        return self.fc(x)  # (B,out_dim)

class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class AVSReducedRegressor(nn.Module):
    def __init__(self, w2v_in: int, smile_in: int, sf_in: int,
                 w2v_out: int = 100, smile_out: int = 32, sf_out: int = 100, spec_out: int = 64):
        super().__init__()
        self.wav_proj = FeatureProjector(w2v_in, w2v_out)
        self.smile_proj = FeatureProjector(smile_in, smile_out)
        self.sf_proj = FeatureProjector(sf_in, sf_out)
        self.spec_enc = SpectrogramSECNN(out_dim=spec_out, dropout=0.1)
        self.regressor = MLPRegressor(w2v_out + smile_out + sf_out + spec_out)

    def forward(self, x_wav, x_smile, x_sf, x_spec):
        w = self.wav_proj(x_wav)
        s = self.smile_proj(x_smile)
        v = self.sf_proj(x_sf)
        sp = self.spec_enc(x_spec)
        z = torch.cat([w, s, v, sp], dim=1)
        return self.regressor(z)

# ---------------- Dataset ----------------
class AVSDataset(Dataset):
    def __init__(self, vids: List[str], y: np.ndarray,
                 Xw: np.ndarray, Xs: np.ndarray, Xv: np.ndarray,
                 spec_paths: List[str]):
        self.vids = vids
        self.y = y.astype(np.float32)
        self.Xw = Xw.astype(np.float32)
        self.Xs = Xs.astype(np.float32)
        self.Xv = Xv.astype(np.float32)
        self.spec_paths = spec_paths

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        vid = self.vids[idx]
        img = Image.open(self.spec_paths[idx]).convert("RGB").resize((SPEC_SHAPE[1], SPEC_SHAPE[0]))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)  # (3,H,W)

        return (
            vid,
            self.Xw[idx],
            self.Xs[idx],
            self.Xv[idx],
            arr,
            self.y[idx]
        )

def collate_fn(batch):
    vids = [b[0] for b in batch]
    xw = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    xs = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
    xv = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
    xsp = torch.tensor(np.stack([b[4] for b in batch]), dtype=torch.float32)
    y = torch.tensor([b[5] for b in batch], dtype=torch.float32)
    return vids, xw, xs, xv, xsp, y

# ---------------- Build split data (same logic as your step6 baseline) ----------------
def build_splits():
    index_df = pd.read_csv(INDEX_PATH)
    index_df["vid"] = index_df["vid"].astype(str).str.strip()

    # wav2vec
    w2v_df = read_table_auto(W2V_PATH)
    w2v_id = guess_id_col(w2v_df)
    w2v_df[w2v_id] = w2v_df[w2v_id].astype(str).str.strip()
    Xw_all = (
        w2v_df.drop(columns=[w2v_id])
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .values.astype(np.float32)
    )
    w2v_vids = w2v_df[w2v_id].tolist()
    w2v_map = {v: i for i, v in enumerate(w2v_vids)}
    w2v_dim = int(Xw_all.shape[1])

    # smile
    sm_df = read_table_auto(SMILE_PATH)
    sm_id = guess_id_col(sm_df)
    sm_df[sm_id] = sm_df[sm_id].astype(str).str.strip()
    Xs_all = (
        sm_df.drop(columns=[sm_id])
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .values.astype(np.float32)
    )
    sm_vids = sm_df[sm_id].tolist()
    sm_map = {v: i for i, v in enumerate(sm_vids)}
    sm_dim = int(Xs_all.shape[1])

    # cache slowfast vectors once
    slowfast_cache = {}
    for v in index_df["vid"].unique().tolist():
        try:
            slowfast_cache[v] = load_slowfast_vec(v)
        except Exception:
            pass
    if len(slowfast_cache) == 0:
        raise RuntimeError("No SlowFast features loaded. Check features/slowfast and naming.")
    sf_dim = int(next(iter(slowfast_cache.values())).shape[0])

    def build_split(split_name: str):
        sub = index_df[index_df["split"] == split_name].copy()
        sub = sub[sub["cohesion"].notna()]

        vids, Xw, Xs, Xv, y, spec_paths = [], [], [], [], [], []
        for _, r in sub.iterrows():
            vid = str(r["vid"]).strip()
            if (vid not in w2v_map) or (vid not in sm_map) or (vid not in slowfast_cache):
                continue
            try:
                sp = find_spec_path(vid)
            except Exception:
                continue

            vids.append(vid)
            Xw.append(Xw_all[w2v_map[vid]])
            Xs.append(Xs_all[sm_map[vid]])
            Xv.append(slowfast_cache[vid])
            y.append(float(r["cohesion"]))
            spec_paths.append(sp)

        Xw = np.vstack(Xw).astype(np.float32)
        Xs = np.vstack(Xs).astype(np.float32)
        Xv = np.vstack(Xv).astype(np.float32)
        y = np.array(y, dtype=np.float32)
        return vids, Xw, Xs, Xv, y, spec_paths

    train = build_split("train")
    valid = build_split("valid")
    test  = build_split("test")

    dims = {"w2v_dim": w2v_dim, "sm_dim": sm_dim, "sf_dim": sf_dim}
    return train, valid, test, dims

@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_vids, preds, trues = [], [], []
    use_amp = (ARGS.amp and DEVICE == "cuda")

    for vids, xw, xs, xv, xsp, y in loader:
        xw = xw.to(DEVICE, non_blocking=True)
        xs = xs.to(DEVICE, non_blocking=True)
        xv = xv.to(DEVICE, non_blocking=True)
        xsp = xsp.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(xw, xs, xv, xsp)

        all_vids.extend(list(vids))
        preds.append(pred.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

    return np.concatenate(trues), np.concatenate(preds), all_vids

def train_one_seed(seed: int, train_data, valid_data, test_data, dims: Dict[str, int]) -> Dict[str, float]:
    set_seed(seed)
    run_dir = os.path.join(OUT_DIR, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    train_vids, Xw_train, Xs_train, Xv_train, y_train, sp_train = train_data
    valid_vids, Xw_valid, Xs_valid, Xv_valid, y_valid, sp_valid = valid_data
    test_vids,  Xw_test,  Xs_test,  Xv_test,  y_test,  sp_test  = test_data

    # Standardize wav/smile/slowfast with TRAIN stats only
    mu_w, sig_w = standardize_fit(Xw_train)
    Xw_train = standardize_apply(Xw_train, mu_w, sig_w)
    Xw_valid = standardize_apply(Xw_valid, mu_w, sig_w)
    Xw_test  = standardize_apply(Xw_test,  mu_w, sig_w)

    mu_s, sig_s = standardize_fit(Xs_train)
    Xs_train = standardize_apply(Xs_train, mu_s, sig_s)
    Xs_valid = standardize_apply(Xs_valid, mu_s, sig_s)
    Xs_test  = standardize_apply(Xs_test,  mu_s, sig_s)

    mu_v, sig_v = standardize_fit(Xv_train)
    Xv_train = standardize_apply(Xv_train, mu_v, sig_v)
    Xv_valid = standardize_apply(Xv_valid, mu_v, sig_v)
    Xv_test  = standardize_apply(Xv_test,  mu_v, sig_v)

    # loaders
    train_ds = AVSDataset(train_vids, y_train, Xw_train, Xs_train, Xv_train, sp_train)
    valid_ds = AVSDataset(valid_vids, y_valid, Xw_valid, Xs_valid, Xv_valid, sp_valid)
    test_ds  = AVSDataset(test_vids,  y_test,  Xw_test,  Xs_test,  Xv_test,  sp_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=ARGS.num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=(ARGS.num_workers > 0),
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        num_workers=ARGS.num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=(ARGS.num_workers > 0),
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        num_workers=ARGS.num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=(ARGS.num_workers > 0),
        collate_fn=collate_fn
    )

    # model
    model = AVSReducedRegressor(
        w2v_in=dims["w2v_dim"],
        smile_in=dims["sm_dim"],
        sf_in=dims["sf_dim"],
        w2v_out=100,
        smile_out=32,
        sf_out=100,
        spec_out=64
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    use_amp = (ARGS.amp and DEVICE == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_valid_mae = float("inf")
    best_state = None
    bad_epochs = 0

    log_lines = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for vids, xw, xs, xv, xsp, yb in train_loader:
            xw = xw.to(DEVICE, non_blocking=True)
            xs = xs.to(DEVICE, non_blocking=True)
            xv = xv.to(DEVICE, non_blocking=True)
            xsp = xsp.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xw, xs, xv, xsp)
                loss = loss_fn(pred, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += float(loss.item())

        yv_true, yv_pred, _ = predict(model, valid_loader)
        v_mae = mae(yv_true, yv_pred)

        line = f"Epoch {epoch:03d} | train_mse={total_loss/len(train_loader):.4f} | valid_mae={v_mae:.4f}"
        print(f"[seed {seed}] {line}")
        log_lines.append(line)

        if v_mae + 1e-6 < best_valid_mae:
            best_valid_mae = v_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"[seed {seed}] Early stopping triggered.")
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # final eval splits
    ytr_true, ytr_pred, tr_vids = predict(model, train_loader)
    yv_true,  yv_pred,  va_vids = predict(model, valid_loader)
    yte_true, yte_pred, te_vids = predict(model, test_loader)

    train_m = pack_metrics(ytr_true, ytr_pred)
    valid_m = pack_metrics(yv_true,  yv_pred)
    test_m  = pack_metrics(yte_true, yte_pred)

    # save preds
    pd.DataFrame({"vid": tr_vids, "y_true": ytr_true, "y_pred": ytr_pred}).to_csv(
        os.path.join(run_dir, "predictions_train.csv"), index=False
    )
    pd.DataFrame({"vid": va_vids, "y_true": yv_true, "y_pred": yv_pred}).to_csv(
        os.path.join(run_dir, "predictions_valid.csv"), index=False
    )
    pd.DataFrame({"vid": te_vids, "y_true": yte_true, "y_pred": yte_pred}).to_csv(
        os.path.join(run_dir, "predictions_test.csv"), index=False
    )

    # save metrics
    with open(os.path.join(run_dir, "metrics_train.txt"), "w", encoding="utf-8") as f:
        f.write(format_metrics(train_m))
    with open(os.path.join(run_dir, "metrics_valid.txt"), "w", encoding="utf-8") as f:
        f.write(format_metrics(valid_m))
    with open(os.path.join(run_dir, "metrics_test.txt"), "w", encoding="utf-8") as f:
        f.write(format_metrics(test_m))

    with open(os.path.join(run_dir, "train_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
        f.write(f"\nBEST_VALID_MAE={best_valid_mae:.6f}\n")

    return test_m

def main():
    for p in [INDEX_PATH, W2V_PATH, SMILE_PATH, SLOWFAST_DIR, SPEC_DIR]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    print("DEVICE:", DEVICE, "| AMP:", bool(ARGS.amp and DEVICE == "cuda"), "| workers:", ARGS.num_workers)

    train_data, valid_data, test_data, dims = build_splits()
    print("Detected dims:", dims)
    print("Loaded sizes:",
          f"train={len(train_data[0])}, valid={len(valid_data[0])}, test={len(test_data[0])}")

    # save config
    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seeds": SEEDS,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_train": BATCH_SIZE_TRAIN,
            "batch_eval": BATCH_SIZE_EVAL,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "spec_shape": SPEC_SHAPE,
            "fused_dim": 100 + 32 + 100 + 64,
            "device": DEVICE,
            "num_workers": ARGS.num_workers,
            "amp": bool(ARGS.amp and DEVICE == "cuda")
        }, f, indent=2)

    all_test = []
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        tm = train_one_seed(seed, train_data, valid_data, test_data, dims)
        all_test.append(tm)

    summary = "TEST metrics across seeds:\n" + mean_std_table(all_test)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "summary_test_mean_std.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

if __name__ == "__main__":
    main()