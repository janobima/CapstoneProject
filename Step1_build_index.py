# ============================================================
# NTU Capstone Project: Group Cohesion Estimation using LLM and Deep Learning
# Maryam Ali Aljanobi
# April 2026

# File: step1_build_index.py
# Description: Builds a unified dataset index by merging split information,
# labels, and multimodal feature availability into a single CSV file.
# It also validates feature completeness across modalities and summarizes
# dataset coverage for downstream modeling.
# ============================================================

import os
import pickle
import glob
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
FEAT = os.path.join(DATA, "features")

LABELS_TSV = os.path.join(DATA, "labels.tsv")
LABELS_CSV = os.path.join(DATA, "labels.csv")
OUT_PATH = os.path.join(DATA, "data_index.csv")

def load_ids(name: str):
    path = os.path.join(DATA, name)
    with open(path, "rb") as f:
        return pickle.load(f)

def read_table_auto(path):
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=None, engine="python")
    return df

def guess_id_col(df):
    for c in ["VID", "vid", "video_id", "video", "id", "title"]:
        if c in df.columns:
            return c
    return df.columns[0]

def exists_slowfast(vid):
    return os.path.exists(os.path.join(FEAT, "slowfast", f"{vid}.npz"))

def exists_spectrogram(vid):
    return os.path.exists(os.path.join(FEAT, "spectrogram", f"{vid}.jpg"))

def exists_facial(vid):
    d = os.path.join(FEAT, "facial", vid)
    return os.path.isdir(d) and len(glob.glob(os.path.join(d, "*.npz"))) > 0

train_ids = load_ids("train")
valid_ids = load_ids("valid")
test_ids  = load_ids("test")

split_map = {}
for v in train_ids: split_map[str(v)] = "train"
for v in valid_ids: split_map[str(v)] = "valid"
for v in test_ids:  split_map[str(v)] = "test"

all_ids = train_ids + valid_ids + test_ids
all_ids = [str(x).strip() for x in all_ids]

if os.path.exists(LABELS_TSV):
    labels_df = pd.read_csv(LABELS_TSV, sep="\t")
elif os.path.exists(LABELS_CSV):
    labels_df = pd.read_csv(LABELS_CSV)
else:
    raise FileNotFoundError("Could not find labels.tsv or labels.csv in data/")

for c in ["video_code", "title", "emotion"]:
    if c in labels_df.columns:
        labels_df[c] = labels_df[c].astype(str).str.strip()
labels_df["cohesion"] = pd.to_numeric(labels_df["cohesion"], errors="coerce")

labels_by_title = labels_df.set_index("title")

audio_smile_path = os.path.join(FEAT, "audio_smile.csv")
audio_w2v_path   = os.path.join(FEAT, "audio_wav2vec.csv")

audio_smile = read_table_auto(audio_smile_path)
audio_w2v   = read_table_auto(audio_w2v_path)

smile_id_col = guess_id_col(audio_smile)
w2v_id_col   = guess_id_col(audio_w2v)

audio_smile[smile_id_col] = audio_smile[smile_id_col].astype(str).str.strip()
audio_w2v[w2v_id_col]     = audio_w2v[w2v_id_col].astype(str).str.strip()

smile_set = set(audio_smile[smile_id_col])
w2v_set   = set(audio_w2v[w2v_id_col])

rows = []
for vid in all_ids:
    if vid in labels_by_title.index:
        cohesion = labels_by_title.loc[vid, "cohesion"]
        emotion  = labels_by_title.loc[vid, "emotion"]
        video_code = labels_by_title.loc[vid, "video_code"]
    else:
        cohesion = None
        emotion  = None
        video_code = None

    row = {
        "vid": vid,
        "split": split_map.get(vid, "unknown"),
        "video_code": video_code,
        "emotion": emotion,
        "cohesion": cohesion,
        "has_audio_smile": int(vid in smile_set),
        "has_audio_wav2vec": int(vid in w2v_set),
        "has_slowfast": int(exists_slowfast(vid)),
        "has_spectrogram": int(exists_spectrogram(vid)),
        "has_facial": int(exists_facial(vid)),
    }
    rows.append(row)

index_df = pd.DataFrame(rows)
index_df.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("\n=== Split counts ===")
print(index_df["split"].value_counts())

print("\n=== Label availability ===")
print("rows with cohesion:", index_df["cohesion"].notna().sum(), "/", len(index_df))

print("\n=== Feature availability (overall) ===")
feat_cols = ["has_audio_smile", "has_audio_wav2vec", "has_slowfast", "has_spectrogram", "has_facial"]
print(index_df[feat_cols].mean().sort_values(ascending=False).apply(lambda x: f"{x*100:.1f}%"))

print("\n=== How many samples have BOTH audio features? ===")
both_audio = (index_df["has_audio_smile"] == 1) & (index_df["has_audio_wav2vec"] == 1)
print("both audio:", both_audio.sum(), "/", len(index_df))

print("\n=== How many samples have ALL features? ===")
all_feats = both_audio & (index_df["has_slowfast"] == 1) & (index_df["has_spectrogram"] == 1) & (index_df["has_facial"] == 1)
print("all features:", all_feats.sum(), "/", len(index_df))

print("\n=== Train-only availability (quick) ===")
train_df = index_df[index_df["split"] == "train"]
print(train_df[feat_cols].mean().sort_values(ascending=False).apply(lambda x: f"{x*100:.1f}%"))