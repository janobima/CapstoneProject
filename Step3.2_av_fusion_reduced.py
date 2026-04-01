# ============================================================
# NTU Capstone Project: Group Cohesion Estimation using LLM and Deep Learning
# Maryam Ali Aljanobi
# April 2026

# File: Step3.2_av_fusion_reduced.py
# Audio-Visual Fusion with Dimensionality Reduction
#   - wav2vec   : 1024 -> 100 (learned projection)
#   - OpenSMILE :   88 ->  32 (learned projection; inferred from csv)
#   - SlowFast  : 2304 -> 100 (learned projection; inferred from npz)
#   - Early fusion by concatenation (100+32+100=232) + MLP regressor
#   - Early stopping on VALID MAE
#   - Reports MAE/MAPE/MSE/R2 + Pearson/Spearman for train/valid/test
#   - Mean predictor (train mean) for reference
#   - Saves test predictions + metrics text file
# -------------------------------------------------------------------

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------- Paths ----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
FEAT = os.path.join(DATA, "features")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

INDEX_PATH = os.path.join(DATA, "data_index.csv")
W2V_PATH = os.path.join(FEAT, "audio_wav2vec.csv")
SMILE_PATH = os.path.join(FEAT, "audio_smile.csv")
SLOWFAST_DIR = os.path.join(FEAT, "slowfast")

PRED_OUT = os.path.join(RESULTS, "av_fusion_reduced_predictions.csv")
METRICS_OUT = os.path.join(RESULTS, "av_fusion_reduced_metrics.txt")


# ---------------- Helpers ----------------
def read_table_auto(path: str) -> pd.DataFrame:
    """Reads csv/tsv robustly (tab first, then auto)."""
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=None, engine="python")
    return df


def guess_id_col(df: pd.DataFrame) -> str:
    """Tries common ID column names; otherwise uses first column."""
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


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))))


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1.0 - (ss_res / ss_tot))


def corr_metrics(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan, np.nan
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


# ---------------- SlowFast loader ----------------
def load_slowfast_vec(vid: str) -> np.ndarray:
    """
    Expected: features/slowfast/<vid>.npz
    Key: 'feat' (fallback to first key)
    Handles shapes:
      - (1, D) -> take [0]
      - (T, D) -> mean over T
      - (D,)   -> use as-is
    """
    path = os.path.join(SLOWFAST_DIR, f"{vid}.npz")
    if not os.path.exists(path):
        matches = glob.glob(os.path.join(SLOWFAST_DIR, f"{vid}*.npz"))
        if not matches:
            raise FileNotFoundError(f"SlowFast not found for vid={vid}")
        path = matches[0]

    loader = np.load(path, allow_pickle=True)
    if "feat" in loader:
        feat = loader["feat"]
    else:
        keys = list(loader.keys())
        feat = loader[keys[0]]

    feat = np.array(feat)
    if feat.ndim == 2:
        if feat.shape[0] == 1:
            vec = feat[0]
        else:
            vec = feat.mean(axis=0)
    elif feat.ndim == 1:
        vec = feat
    else:
        vec = feat.reshape(-1)

    return vec.astype(np.float32)


# ---------------- Model ----------------
class FeatureProjector(nn.Module):
    """Learnable dimensionality reduction."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.proj(x)


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


class AVFusionReducedRegressor(nn.Module):
    def __init__(self, w2v_in: int, smile_in: int, sf_in: int,
                 w2v_out: int = 100, smile_out: int = 32, sf_out: int = 100):
        super().__init__()
        self.wav_proj = FeatureProjector(w2v_in, w2v_out)
        self.smile_proj = FeatureProjector(smile_in, smile_out)
        self.sf_proj = FeatureProjector(sf_in, sf_out)
        self.regressor = MLPRegressor(w2v_out + smile_out + sf_out)

    def forward(self, x_wav, x_smile, x_sf):
        w = self.wav_proj(x_wav)
        s = self.smile_proj(x_smile)
        v = self.sf_proj(x_sf)
        z = torch.cat([w, s, v], dim=1)
        return self.regressor(z)


# ---------------- Main ----------------
def main():
    # 1) Load master index
    index_df = pd.read_csv(INDEX_PATH)
    index_df["vid"] = index_df["vid"].astype(str).str.strip()

    # 2) Load wav2vec
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

    # 3) Load OpenSMILE
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

    # 4) Cache SlowFast (avoid repeated disk reads)
    slowfast_cache = {}
    for v in index_df["vid"].unique().tolist():
        try:
            slowfast_cache[v] = load_slowfast_vec(v)
        except Exception:
            pass

    if len(slowfast_cache) == 0:
        raise RuntimeError("No SlowFast features were loaded. Check features/slowfast folder and naming.")

    sf_dim = int(next(iter(slowfast_cache.values())).shape[0])

    print(f"Detected dims: wav2vec={w2v_dim}, OpenSMILE={sm_dim}, SlowFast={sf_dim}")

    # 5) Build split arrays
    def build_split(split_name: str):
        sub = index_df[index_df["split"] == split_name].copy()
        sub = sub[sub["cohesion"].notna()]

        Xw, Xs, Xv, y, vids = [], [], [], [], []
        for _, r in sub.iterrows():
            vid = str(r["vid"]).strip()
            if (vid in w2v_map) and (vid in sm_map) and (vid in slowfast_cache):
                Xw.append(Xw_all[w2v_map[vid]])
                Xs.append(Xs_all[sm_map[vid]])
                Xv.append(slowfast_cache[vid])
                y.append(float(r["cohesion"]))  # IMPORTANT: float to avoid Long dtype
                vids.append(vid)

        Xw = np.vstack(Xw).astype(np.float32)
        Xs = np.vstack(Xs).astype(np.float32)
        Xv = np.vstack(Xv).astype(np.float32)
        y = np.array(y, dtype=np.float32)
        return vids, Xw, Xs, Xv, y

    train_vids, Xw_train, Xs_train, Xv_train, y_train = build_split("train")
    valid_vids, Xw_valid, Xs_valid, Xv_valid, y_valid = build_split("valid")
    test_vids,  Xw_test,  Xs_test,  Xv_test,  y_test  = build_split("test")

    print("Loaded shapes:")
    print("Train:", Xw_train.shape, Xs_train.shape, Xv_train.shape, y_train.shape)
    print("Valid:", Xw_valid.shape, Xs_valid.shape, Xv_valid.shape, y_valid.shape)
    print("Test :", Xw_test.shape,  Xs_test.shape,  Xv_test.shape,  y_test.shape)

    # 6) Standardize each modality using TRAIN stats only
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

    # 7) Torch loaders (IMPORTANT: float32 everywhere)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(Xw_train).float(),
            torch.from_numpy(Xs_train).float(),
            torch.from_numpy(Xv_train).float(),
            torch.from_numpy(y_train).float(),
        ),
        batch_size=64,
        shuffle=True,
    )

    valid_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(Xw_valid).float(),
            torch.from_numpy(Xs_valid).float(),
            torch.from_numpy(Xv_valid).float(),
            torch.from_numpy(y_valid).float(),
        ),
        batch_size=256,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(Xw_test).float(),
            torch.from_numpy(Xs_test).float(),
            torch.from_numpy(Xv_test).float(),
            torch.from_numpy(y_test).float(),
        ),
        batch_size=256,
        shuffle=False,
    )

    # 8) Model + optim
    model = AVFusionReducedRegressor(w2v_in=w2v_dim, smile_in=sm_dim, sf_in=sf_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # 9) Train with early stopping on valid MAE
    best_valid_mae = float("inf")
    best_state = None
    patience = 10
    bad_epochs = 0
    max_epochs = 80

    def predict(loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xw, xs, xv, yb in loader:
                xw = xw.to(device)
                xs = xs.to(device)
                xv = xv.to(device)
                pred = model(xw, xs, xv)
                preds.append(pred.cpu().numpy())
                trues.append(yb.cpu().numpy())
        return np.concatenate(trues), np.concatenate(preds)

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for xw, xs, xv, yb in train_loader:
            xw = xw.to(device)
            xs = xs.to(device)
            xv = xv.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred = model(xw, xs, xv)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        yv_true, yv_pred = predict(valid_loader)
        v_mae = mae(yv_true, yv_pred)
        print(f"Epoch {epoch:03d} | train_mse={total_loss/len(train_loader):.4f} | valid_mae={v_mae:.4f}")

        if v_mae + 1e-6 < best_valid_mae:
            best_valid_mae = v_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 10) Final evaluation
    ytr_true, ytr_pred = predict(train_loader)
    yv_true,  yv_pred  = predict(valid_loader)
    yte_true, yte_pred = predict(test_loader)

    train_metrics = pack_metrics(ytr_true, ytr_pred)
    valid_metrics = pack_metrics(yv_true,  yv_pred)
    test_metrics  = pack_metrics(yte_true, yte_pred)

    # Mean predictor baseline (using train mean)
    mean_train = float(ytr_true.mean())
    yte_pred_mean = np.full_like(yte_true, mean_train)
    mean_test = {
        "MAE": mae(yte_true, yte_pred_mean),
        "MAPE": mape(yte_true, yte_pred_mean),
        "MSE": mse(yte_true, yte_pred_mean),
        "R2": r2_score(yte_true, yte_pred_mean),
    }

    # 11) Save predictions (test)
    pred_df = pd.DataFrame({
        "vid": test_vids,
        "y_true": yte_true,
        "y_pred": yte_pred,
        "abs_error": np.abs(yte_true - yte_pred),
    })
    pred_df.to_csv(PRED_OUT, index=False)

    # 12) Save metrics
    metrics_text = f"""
Audio-Visual Fusion Baseline (Reduced) - MLP Regressor
-----------------------------------------------------
Samples: train={len(train_vids)}, valid={len(valid_vids)}, test={len(test_vids)}
Dims: wav2vec={w2v_dim}->100 | OpenSMILE={sm_dim}->32 | SlowFast={sf_dim}->100 | fused=232

Train:
  MAE={train_metrics['MAE']:.4f} | MAPE={train_metrics['MAPE']:.4f} | MSE={train_metrics['MSE']:.4f} | R2={train_metrics['R2']:.4f}
  Pearson={train_metrics['Pearson']:.4f} | Spearman={train_metrics['Spearman']:.4f}

Valid:
  MAE={valid_metrics['MAE']:.4f} | MAPE={valid_metrics['MAPE']:.4f} | MSE={valid_metrics['MSE']:.4f} | R2={valid_metrics['R2']:.4f}
  Pearson={valid_metrics['Pearson']:.4f} | Spearman={valid_metrics['Spearman']:.4f}

Test:
  MAE={test_metrics['MAE']:.4f} | MAPE={test_metrics['MAPE']:.4f} | MSE={test_metrics['MSE']:.4f} | R2={test_metrics['R2']:.4f}
  Pearson={test_metrics['Pearson']:.4f} | Spearman={test_metrics['Spearman']:.4f}

Mean predictor (test reference, using train mean={mean_train:.4f}):
  MAE={mean_test['MAE']:.4f} | MAPE={mean_test['MAPE']:.4f} | MSE={mean_test['MSE']:.4f} | R2={mean_test['R2']:.4f}
""".strip()

    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        f.write(metrics_text + "\n")

    print("\n=== DONE ===")
    print(metrics_text)
    print("\nSaved predictions:", PRED_OUT)
    print("Saved metrics:", METRICS_OUT)


if __name__ == "__main__":
    main()
