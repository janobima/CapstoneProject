# ============================================================
# NTU Capstone Project: Group Cohesion Estimation using LLM and Deep Learning
# Maryam Ali Aljanobi
# April 2026

# File: Step2.3_wav2vec_baseline_reduced.py
# Unimodal Wav2Vec baseline with dimensionality reduction
# - Loads data_index.csv
# - Loads audio_wav2vec.csv
# - Builds train/valid/test tensors (X=1024-d wav2vec, y=cohesion)
# - Applies a learnable projection: 1024 -> 100 
# - Trains an MLP regressor with early stopping on valid MAE
# - Reports MAE/MAPE/MSE/R2 + Pearson/Spearman for train/valid/test
# - Also reports Mean Predictor baseline (train-mean) for reference
# - Saves test predictions + metrics text file
# ------------------------------------------------------------

import os
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

PRED_OUT = os.path.join(RESULTS, "wav2vec_baseline_reduced_predictions.csv")
METRICS_OUT = os.path.join(RESULTS, "wav2vec_baseline_reduced_metrics.txt")


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


def standardize_fit(train_X: np.ndarray):
    mu = train_X.mean(axis=0, keepdims=True)
    sigma = train_X.std(axis=0, keepdims=True)
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


# ---------------- Model ----------------
class FeatureProjector(nn.Module):
    """Learnable dimensionality reduction (1024 -> 100), GCE-style."""
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


class Wav2VecReducedRegressor(nn.Module):
    def __init__(self, w2v_in: int = 1024, w2v_out: int = 100):
        super().__init__()
        self.wav_proj = FeatureProjector(w2v_in, w2v_out)
        self.regressor = MLPRegressor(w2v_out)

    def forward(self, x_wav):
        z = self.wav_proj(x_wav)
        return self.regressor(z)


# ---------------- Main ----------------
def main():
    # 1) Load master index
    index_df = pd.read_csv(INDEX_PATH)
    index_df["vid"] = index_df["vid"].astype(str).str.strip()

    # 2) Load wav2vec table
    w2v_df = read_table_auto(W2V_PATH)
    id_col = guess_id_col(w2v_df)
    w2v_df[id_col] = w2v_df[id_col].astype(str).str.strip()

    X_all = (
        w2v_df.drop(columns=[id_col])
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .values.astype(np.float32)
    )
    vids_all = w2v_df[id_col].tolist()
    w2v_map = {v: i for i, v in enumerate(vids_all)}

    # Infer input dim from file
    w2v_dim = int(X_all.shape[1])
    print("Detected wav2vec dim:", w2v_dim)

    # 3) Build split arrays
    def build_split(split_name: str):
        sub = index_df[index_df["split"] == split_name].copy()
        sub = sub[sub["cohesion"].notna()]
        vids = sub["vid"].tolist()

        X, y, keep = [], [], []
        for v in vids:
            if v in w2v_map:
                X.append(X_all[w2v_map[v]])
                y.append(float(sub.loc[sub["vid"] == v, "cohesion"].values[0]))
                keep.append(v)

        X = np.vstack(X).astype(np.float32)
        y = np.array(y, dtype=np.float32)
        return keep, X, y

    train_vids, X_train, y_train = build_split("train")
    valid_vids, X_valid, y_valid = build_split("valid")
    test_vids,  X_test,  y_test  = build_split("test")

    print("Loaded shapes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Valid:", X_valid.shape, y_valid.shape)
    print("Test :", X_test.shape,  y_test.shape)

    # 4) Standardize on train only
    mu, sigma = standardize_fit(X_train)
    X_train = standardize_apply(X_train, mu, sigma)
    X_valid = standardize_apply(X_valid, mu, sigma)
    X_test  = standardize_apply(X_test,  mu, sigma)

    # 5) Torch loaders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=64, shuffle=True
    )
    valid_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid)),
        batch_size=256, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=256, shuffle=False
    )

    # 6) Model + optim
    model = Wav2VecReducedRegressor(w2v_in=w2v_dim, w2v_out=100).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # 7) Train with early stopping on valid MAE
    best_valid_mae = float("inf")
    best_state = None
    patience = 10
    bad_epochs = 0
    max_epochs = 80

    def predict(loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                pred = model(xb)
                preds.append(pred.cpu().numpy())
                trues.append(yb.cpu().numpy())
        return np.concatenate(trues), np.concatenate(preds)

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        # validation
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

    # 8) Final evaluation
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

    # 9) Save predictions (test)
    pred_df = pd.DataFrame({
        "vid": test_vids,
        "y_true": yte_true,
        "y_pred": yte_pred,
        "abs_error": np.abs(yte_true - yte_pred),
    })
    pred_df.to_csv(PRED_OUT, index=False)

    # 10) Save metrics
    metrics_text = f"""
Wav2Vec Baseline (Reduced 1024->100) - MLP Regressor
---------------------------------------------------
Samples: train={len(train_vids)}, valid={len(valid_vids)}, test={len(test_vids)}
Input dim: wav2vec={w2v_dim} -> proj=100

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
