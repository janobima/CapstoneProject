# ============================================================
# NTU Capstone Project: Group Cohesion Estimation using LLM and Deep Learning
# Maryam Ali Aljanobi
# April 2026
#
# File: Step5.1_LLM_Make_JSON.py
# ------------------------------------------------------------
# Builds LLM input JSONL for GCE cohesion prediction using:
#   - data/data_index.csv (clip id + split + labels + availability flags)
#   - data/Group_Cohesion_Video_Collection.xlsx (Sheet1: num_of_people + keyword)
#   - data/audio_smile.csv (OpenSMILE / GeMAPS-style)
#   - data/features/audio_wav2vec.csv (1024-d embeddings, column: VID)
#   - data/features/slowfast/<vid>.npz (key: feat, shape (1,2304))
#   - data/features/facial/<vid>/*.npz (keys: fr (n,512), fer (n,1408))#
#   - All discretization thresholds (low/mid/high) computed on TRAIN only (P33/P66).
#
# JSON features:
#   Audio (OpenSMILE):
#       - speech_activity_level
#       - speech_variability
#       - tone_variability
#   Audio (wav2vec):
#       - audio_strength
#       - audio_spread
#       - audio_peaks
#   Visual (SlowFast):
#       - motion_strength
#       - motion_variation
#       - motion_spikes
#   Facial:
#       - face_presence_ratio
#       - faces_per_frame
#       - face_stability
#       - expression_change
# ------------------------------------------------------------

import os
import json
import glob
import numpy as np
import pandas as pd


# =====================
# PATHS
# =====================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
FEAT = os.path.join(DATA, "features")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

INDEX_PATH = os.path.join(DATA, "data_index.csv")
META_XLSX_PATH = os.path.join(DATA, "Group_Cohesion_Video_Collection.xlsx")

W2V_PATH = os.path.join(FEAT, "audio_wav2vec.csv")
SMILE_PATH = os.path.join(FEAT, "audio_smile.csv")
SLOWFAST_ROOT = os.path.join(FEAT, "slowfast")
FACIAL_ROOT = os.path.join(FEAT, "facial")

OUT_JSONL = os.path.join(RESULTS, "llm_inputs_v1_all_modalities.jsonl")
OUT_LABELS = os.path.join(RESULTS, "llm_eval_labels.csv")
OUT_THRESHOLDS = os.path.join(RESULTS, "llm_binning_thresholds_v1.json")

FACIAL_SAMPLE_K = 9
Q33 = 0.333
Q66 = 0.666


# ---------------------------
# Helpers: train-only standardization + binning
# ---------------------------
def fit_standardizer(train_vals: pd.Series):
    train_vals = train_vals.dropna().astype(float)
    if len(train_vals) == 0:
        return 0.0, 1.0
    mu = float(train_vals.mean())
    sd = float(train_vals.std(ddof=0))
    if sd <= 1e-12:
        sd = 1.0
    return mu, sd


def apply_standardizer(vals: pd.Series, mu: float, sd: float) -> pd.Series:
    vals = pd.to_numeric(vals, errors="coerce")
    return (vals - mu) / sd


def fit_terciles(train_vals: pd.Series):
    train_vals = pd.to_numeric(train_vals, errors="coerce").dropna().astype(float)
    if len(train_vals) == 0:
        return np.nan, np.nan
    return float(train_vals.quantile(Q33)), float(train_vals.quantile(Q66))


def bin_tercile(x: float, p33: float, p66: float) -> str:
    if pd.isna(x) or np.isnan(p33) or np.isnan(p66):
        return "unknown"
    if x <= p33:
        return "low"
    if x <= p66:
        return "mid"
    return "high"


# ---------------------------
# Facial aggregation
# ---------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    eps = 1e-9
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _even_sample(items, k: int):
    if k is None or k <= 0 or len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, num=k)
    idx = np.round(idx).astype(int)
    return [items[i] for i in idx]


def aggregate_facial_folder(folder_path: str, sample_k_frames: int = FACIAL_SAMPLE_K) -> dict:
    files = sorted(glob.glob(os.path.join(folder_path, "*.npz")))
    if len(files) == 0:
        return {
            "face_presence_ratio": np.nan,
            "faces_per_frame": np.nan,
            "face_stability": np.nan,
            "expression_change": np.nan,
            "num_frames_used": 0.0,
        }

    files = _even_sample(files, sample_k_frames)

    fr_means, fer_means, faces_per_frame = [], [], []

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        fr = d["fr"]   # (n_faces, 512)
        fer = d["fer"] # (n_faces, 1408)

        n_faces = int(fr.shape[0])
        faces_per_frame.append(n_faces)

        if n_faces > 0:
            fr_means.append(fr.mean(axis=0).astype(np.float32))
            fer_means.append(fer.mean(axis=0).astype(np.float32))
        else:
            fr_means.append(None)
            fer_means.append(None)

    faces_per_frame = np.array(faces_per_frame, dtype=float)
    face_presence_ratio = float(np.mean(faces_per_frame > 0))
    faces_per_frame_mean = float(np.mean(faces_per_frame))

    def consecutive_cosine(vecs):
        sims, prev = [], None
        for v in vecs:
            if v is None:
                prev = None
                continue
            if prev is not None:
                sims.append(_cosine(prev, v))
            prev = v
        return float(np.mean(sims)) if len(sims) else np.nan

    def consecutive_l2(vecs):
        dists, prev = [], None
        for v in vecs:
            if v is None:
                prev = None
                continue
            if prev is not None:
                dists.append(_l2(prev, v))
            prev = v
        return float(np.mean(dists)) if len(dists) else np.nan

    return {
        "face_presence_ratio": face_presence_ratio,
        "faces_per_frame": faces_per_frame_mean,
        "face_stability": consecutive_cosine(fr_means),
        "expression_change": consecutive_l2(fer_means),
        "num_frames_used": float(len(files)),
    }


# ---------------------------
# (SlowFast + wav2vec)
# ---------------------------
def embedding_strength(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(np.linalg.norm(x))


def embedding_spread(x: np.ndarray, topk: int = 32) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    ax = np.abs(x)
    denom = float(np.sum(ax)) + 1e-12
    k = min(topk, ax.size)
    top = float(np.sum(np.partition(ax, -k)[-k:]))
    return float(1.0 - top / denom)


def embedding_peaks(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    ax = np.abs(x)
    m = float(np.mean(ax)) + 1e-12
    return float(np.max(ax) / m)


# ---------------------------
# Main
# ---------------------------
def main():
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

    # ---- Load index
    idx = pd.read_csv(INDEX_PATH)
    idx["vid"] = idx["vid"].astype(str)
    idx["split"] = idx["split"].astype(str)

    # Save labels separately (NEVER fed to LLM)
    labels = idx[["vid", "split", "cohesion"]].rename(
        columns={"vid": "clip_id", "cohesion": "y_true"}
    )
    labels.to_csv(OUT_LABELS, index=False)

    # ---- Load metadata Excel (Sheet1)
    meta = pd.read_excel(META_XLSX_PATH, sheet_name="Sheet1")
    meta["title"] = meta["title"].astype(str)
    meta = meta.rename(columns={"num_of_people": "num_people", "keyword": "interaction_keyword"})

    df = idx.merge(
        meta[["title", "num_people", "interaction_keyword"]],
        left_on="vid",
        right_on="title",
        how="left"
    ).drop(columns=["title"])

    print("[Sanity] Missing num_people after meta merge:", float(df["num_people"].isna().mean()))

    # ---- Load OpenSMILE
    smile = pd.read_csv(SMILE_PATH)
    smile["VID"] = smile["VID"].astype(str)

    df = df.merge(smile, left_on="vid", right_on="VID", how="left")
    if "VID" in df.columns:
        df = df.drop(columns=["VID"])

    # ---- Load wav2vec and compute summary measures
    w2v = pd.read_csv(W2V_PATH)
    w2v["VID"] = w2v["VID"].astype(str)

    w2v_feat_cols = [c for c in w2v.columns if c != "VID"]
    w2v_mat = w2v[w2v_feat_cols].to_numpy(dtype=np.float32)

    w2v["audio_strength"] = np.linalg.norm(w2v_mat, axis=1)
    w2v["audio_spread"] = [embedding_spread(row) for row in w2v_mat]
    w2v["audio_peaks"] = [embedding_peaks(row) for row in w2v_mat]

    df = df.merge(
        w2v[["VID", "audio_strength", "audio_spread", "audio_peaks"]],
        left_on="vid",
        right_on="VID",
        how="left"
    ).drop(columns=["VID"])

    # ---- SlowFast summary measures
    slow_rows = []
    for vid in df["vid"].tolist():
        fp = os.path.join(SLOWFAST_ROOT, f"{vid}.npz")
        if not os.path.exists(fp):
            slow_rows.append({
                "vid": vid,
                "motion_strength": np.nan,
                "motion_variation": np.nan,
                "motion_spikes": np.nan,
            })
            continue

        d = np.load(fp, allow_pickle=True)
        x = d["feat"].reshape(-1)
        slow_rows.append({
            "vid": vid,
            "motion_strength": embedding_strength(x),
            "motion_variation": embedding_spread(x),
            "motion_spikes": embedding_peaks(x),
        })

    df = df.merge(pd.DataFrame(slow_rows), on="vid", how="left")

    # ---- Facial aggregation
    face_rows = []
    for vid in df["vid"].tolist():
        folder = os.path.join(FACIAL_ROOT, vid)
        face_feats = aggregate_facial_folder(folder, sample_k_frames=FACIAL_SAMPLE_K)
        face_feats["vid"] = vid
        face_rows.append(face_feats)

    df = df.merge(pd.DataFrame(face_rows), on="vid", how="left")

    # ---------------------------
    # OpenSMILE indicators with TRAIN-only standardization
    # ---------------------------
    required_smile = [
        "equivalentSoundLevel_dBp",
        "VoicedSegmentsPerSec",
        "MeanVoicedSegmentLengthSec",
        "StddevVoicedSegmentLengthSec",
        "loudness_sma3_percentile80.0",
        "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    ]
    for c in required_smile:
        if c not in df.columns:
            raise KeyError(f"Missing OpenSMILE column: {c}")

    train_mask = df["split"] == "train"

    for c in required_smile:
        mu, sd = fit_standardizer(df.loc[train_mask, c])
        df[c + "_z"] = apply_standardizer(df[c], mu, sd)

    df["speech_activity_level"] = (
        df["equivalentSoundLevel_dBp_z"] +
        df["VoicedSegmentsPerSec_z"] +
        df["MeanVoicedSegmentLengthSec_z"]
    )

    df["speech_variability"] = df["StddevVoicedSegmentLengthSec_z"]

    df["tone_variability"] = (
        0.5 * df["F0semitoneFrom27.5Hz_sma3nz_stddevNorm_z"] +
        0.5 * df["loudness_sma3_percentile80.0_z"]
    )

    # ---------------------------
    # Train-only binning
    # ---------------------------
    indicators = [
        "speech_activity_level",
        "speech_variability",
        "tone_variability",
        "audio_strength",
        "audio_spread",
        "audio_peaks",
        "motion_strength",
        "motion_variation",
        "motion_spikes",
        "face_presence_ratio",
        "faces_per_frame",
        "face_stability",
        "expression_change",
    ]

    thresholds = {}
    for c in indicators:
        p33, p66 = fit_terciles(df.loc[train_mask, c])
        thresholds[c] = {
            "p33": None if np.isnan(p33) else p33,
            "p66": None if np.isnan(p66) else p66,
        }
        df[c + "_bin"] = df[c].apply(lambda x: bin_tercile(x, p33, p66))

    with open(OUT_THRESHOLDS, "w", encoding="utf-8") as f:
        json.dump(
            {
                "binning_rule": "Bins computed from training-set percentiles only: low <= P33, mid = P33–P66, high > P66.",
                "indicators": thresholds,
            },
            f,
            indent=2,
        )

    # ---------------------------
    # Write JSONL
    # ---------------------------
    def is_missing(mod: str, row) -> bool:
        if mod == "audio_smile":
            flag = int(row.get("has_audio_smile", 0)) == 1
            return (not flag) or (row["speech_activity_level_bin"] == "unknown")
        if mod == "audio_wav2vec":
            flag = int(row.get("has_audio_wav2vec", 0)) == 1
            return (not flag) or pd.isna(row["audio_strength"])
        if mod == "slowfast":
            flag = int(row.get("has_slowfast", 0)) == 1
            return (not flag) or pd.isna(row["motion_strength"])
        if mod == "facial":
            flag = int(row.get("has_facial", 0)) == 1
            return (not flag) or float(row.get("num_frames_used", 0.0)) == 0.0
        return True

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            missing = []
            if is_missing("audio_smile", r):
                missing.append("audio_smile")
            if is_missing("audio_wav2vec", r):
                missing.append("audio_wav2vec")
            if is_missing("slowfast", r):
                missing.append("slowfast")
            if is_missing("facial", r):
                missing.append("facial")

            item = {
                "clip_id": r["vid"],
                "split": r["split"],
                "group_context": {
                    "num_people": None if pd.isna(r.get("num_people")) else int(r.get("num_people")),
                    "interaction_keyword": None if pd.isna(r.get("interaction_keyword")) else str(r.get("interaction_keyword")),
                },
                "evidence_summary": {
                    "audio_smile": {
                        "speech_activity_level": r["speech_activity_level_bin"],
                        "speech_variability": r["speech_variability_bin"],
                        "tone_variability": r["tone_variability_bin"],
                    },
                    "audio_wav2vec": {
                        "audio_strength": r["audio_strength_bin"],
                        "audio_spread": r["audio_spread_bin"],
                        "audio_peaks": r["audio_peaks_bin"],
                    },
                    "visual_slowfast": {
                        "motion_strength": r["motion_strength_bin"],
                        "motion_variation": r["motion_variation_bin"],
                        "motion_spikes": r["motion_spikes_bin"],
                    },
                    "face_behavior": {
                        "face_presence_ratio": r["face_presence_ratio_bin"],
                        "faces_per_frame": r["faces_per_frame_bin"],
                        "face_stability": r["face_stability_bin"],
                        "expression_change": r["expression_change_bin"],
                    },
                },
                "missing_modalities": missing,
                "reference_note": {
                    "binning_rule": "Bins computed from training-set percentiles only: low <= P33, mid = P33–P66, high > P66.",
                    "embedding_note": "wav2vec and SlowFast are summarized using simple signal-based indicators (strength, spread, peaks).",
                    "facial_note": f"Facial features aggregated from up to {FACIAL_SAMPLE_K} evenly sampled frames; mean-over-faces per frame; stability and change computed across consecutive sampled frames.",
                },
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Wrote JSONL:", OUT_JSONL)
    print("Wrote labels CSV:", OUT_LABELS)
    print("Wrote thresholds JSON:", OUT_THRESHOLDS)

    print("\n--- Example JSON rows ---")
    with open(OUT_JSONL, "r", encoding="utf-8") as f:
        for _ in range(2):
            line = f.readline().strip()
            if not line:
                break
            print(line[:500] + ("..." if len(line) > 500 else ""))


if __name__ == "__main__":
    main()