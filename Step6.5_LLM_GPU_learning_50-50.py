# ------------------------------------------------------------
# NTU Capstone Project: Group Cohesion Estimation using LLM and Deep Learning
# Maryam Ali Aljanobi
# April 2026
#
# File: Step6.5_LLM_GPU_learning_50-50.py
#
# This experiment is set to use different split of the dataset: 50/50 SPLIT (train50/test50):
# - Loads JSONL evidence + labels CSV (clip_id, split, y_true)
# - Builds a NEW  50/50 split across all labeled clips present in JSONL
#    - Fallback stratify: bucket(y_true) if some classes are too small (Low Cohesion)
# - Uses train50 to get K similar samples, predicts test50 via retrieval few-shot
# - output:
#    - labels_with_split50.csv  (to use in the ML experiment for compirability)
#    - predictions.jsonl, predictions.csv, metrics.json, failures_raw.jsonl
#
# Example:
# python Step6.5_LLM_GPU_learning_50-50.py \
#   --jsonl $/gce_project/results/llm_inputs_v1_all_modalities.jsonl \
#   --labels $/gce_project/results/llm_eval_labels.csv \
#   --outdir $/gce_project/results/llm_runs/qwen3b_retrieval_split50_k16 \
#   --model Qwen/Qwen2.5-3B-Instruct \
#   --k 16 \
#   --seed 0 \
#   --device cuda --dtype fp16 --batch_size 4 --max_new_tokens 140 --resume
#


import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1800"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
os.environ["HF_HUB_MAX_RETRIES"] = "50"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import argparse
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr


# =========================
# Prompt (retrieval few-shot)
# =========================
SYSTEM_PROMPT = """You are an annotation model for a research study on GROUP COHESION.
You will receive ONE clip described ONLY by a JSON object of computed evidence fields.

Task: predict cohesion as an integer 1–7.

Hard rules:
- Use ONLY the provided JSON fields. Do not invent details.
- Do NOT claim you saw/heard anything beyond the evidence fields.
- Output MUST be a SINGLE valid JSON object on ONE LINE. No markdown. No extra text.
- Keep "rationale" to <= 2 short sentences.
- Keep "evidence_used" to <= 6 items (pick the most important fields).
- If evidence is missing/uncertain/contradictory, still output your best estimate but lower confidence.

Scale: 1 = very low cohesion, 7 = very high cohesion.

Return EXACT schema:
{"clip_id":"...","predicted_cohesion":1,"confidence":"low|medium|high","evidence_used":["..."],"rationale":"...","failure_flag":false}
"""

FEWSHOT_USER_TEMPLATE = """You will be given {k} labeled examples, then one unlabeled clip.
Learn the mapping from evidence fields to cohesion score.

LABELED EXAMPLES:
{examples}

UNLABELED CLIP (predict cohesion 1–7):
{query_json}
"""


# =========================
# IO helpers
# =========================
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_labels_df(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    # expect at least clip_id and y_true
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("clip_id", cols.get("vid", df.columns[0]))
    y_col = cols.get("y_true", cols.get("cohesion", df.columns[-1]))

    df = df.rename(columns={id_col: "clip_id", y_col: "y_true"})
    df["clip_id"] = df["clip_id"].astype(str)
    df["y_true"] = df["y_true"].astype(float)
    return df


def load_done_clip_ids(pred_jsonl_path: str) -> set:
    done = set()
    if not os.path.exists(pred_jsonl_path):
        return done
    with open(pred_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = obj.get("clip_id")
                if cid:
                    done.add(str(cid))
            except Exception:
                continue
    return done


# =========================
# Feature vectorization for retrieval
# =========================
BIN_MAP = {"low": 0.0, "mid": 0.5, "medium": 0.5, "high": 1.0}

def to_num(v: Any) -> float:
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).strip().lower()
    if s in BIN_MAP:
        return BIN_MAP[s]
    try:
        return float(s)
    except Exception:
        return np.nan


def get_by_path(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def build_feature_key_list() -> List[str]:
    return [
        "group_context.num_people",
    
        "evidence_summary.audio_smile.speech_activity_level",
        "evidence_summary.audio_smile.speech_variability",
        "evidence_summary.audio_smile.tone_variability",
    
        "evidence_summary.audio_wav2vec.audio_strength",
        "evidence_summary.audio_wav2vec.audio_spread",
        "evidence_summary.audio_wav2vec.audio_peaks",
    
        "evidence_summary.visual_slowfast.motion_strength",
        "evidence_summary.visual_slowfast.motion_variation",
        "evidence_summary.visual_slowfast.motion_spikes",
    
        "evidence_summary.face_behavior.face_presence_ratio",
        "evidence_summary.face_behavior.faces_per_frame",
        "evidence_summary.face_behavior.face_stability",
        "evidence_summary.face_behavior.expression_change",
    ]


def flatten_fields_for_vector(row: Dict[str, Any], feature_keys: List[str]) -> np.ndarray:
    vec = [to_num(get_by_path(row, k)) for k in feature_keys]
    return np.asarray(vec, dtype=np.float32)


def impute_and_normalize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = X.astype(np.float32)
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)

    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])

    col_std = X.std(axis=0)
    col_std = np.where(col_std < 1e-6, 1.0, col_std)

    Xn = (X - col_mean) / col_std
    return Xn, col_mean, col_std


# =========================
# 50/50 split (stratified)
# =========================
def y_bucket(y: float) -> str:
    # only for fallback stratification if classes too small
    y = float(y)
    if y <= 2:
        return "low"
    if y <= 5:
        return "mid"
    return "high"


def build_split50(labels_df, seed: int):
    """
    Returns labels_df with a new column split50 in {"train50","test50"}.
    Stratify on y_true if possible; otherwise stratify on bucket(y_true).
    """
    # Try stratify by exact y_true (rounded to int)
    y_int = labels_df["y_true"].round().astype(int)
    counts = y_int.value_counts()
    can_stratify_exact = (counts.min() >= 2) and (counts.shape[0] >= 2)

    if can_stratify_exact:
        strat = y_int
        strat_name = "y_true (rounded int 1..7)"
    else:
        strat = labels_df["y_true"].apply(y_bucket)
        strat_name = "bucket(y_true) fallback"

    train_ids, test_ids = train_test_split(
        labels_df["clip_id"].values,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
        stratify=strat
    )

    split50 = {cid: "train50" for cid in train_ids}
    split50.update({cid: "test50" for cid in test_ids})

    out = labels_df.copy()
    out["split50"] = out["clip_id"].map(split50)
    out.attrs["split50_stratify_used"] = strat_name
    return out


# =========================
# LLM utilities
# =========================
def parse_dtype(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def load_llm(model_name: str, device: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if device.startswith("cuda") else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, dtype=dtype)
    model.eval()
    return tokenizer, model


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1].strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    blocks = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for b in reversed(blocks):
        try:
            return json.loads(b)
        except Exception:
            continue
    return None


def clamp_int_1_7(x: Any) -> int:
    try:
        v = int(round(float(x)))
    except Exception:
        v = 4
    return max(1, min(7, v))


def normalize_confidence(x: Any) -> str:
    s = str(x).strip().lower()
    if s in {"low", "medium", "high"}:
        return s
    if "high" in s:
        return "high"
    if "med" in s:
        return "medium"
    return "low"


def compact_clip_json(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "clip_id": row.get("clip_id"),
        "missing_modalities": row.get("missing_modalities"),
        "group_context": row.get("group_context", {}),
        "evidence_summary": row.get("evidence_summary", {}),
    }
    rn = row.get("reference_note", {})
    if isinstance(rn, dict) and "binning_rule" in rn:
        out["reference_note"] = {"binning_rule": rn.get("binning_rule")}
    return out


def build_fewshot_user_content(
    train_examples: List[Tuple[Dict[str, Any], float]],
    query_row: Dict[str, Any],
    k: int
) -> str:
    lines = []
    for ex_row, y in train_examples:
        ex_json = compact_clip_json(ex_row)
        lines.append(
            f"- example_json: {json.dumps(ex_json, ensure_ascii=False)}\n"
            f"  label_y_true: {int(round(float(y)))}"
        )
    examples_block = "\n".join(lines)
    query_json = compact_clip_json(query_row)
    return FEWSHOT_USER_TEMPLATE.format(
        k=k,
        examples=examples_block,
        query_json=json.dumps(query_json, ensure_ascii=False),
    )


@torch.no_grad()
def run_batch_prompts(tokenizer, model, prompts: List[str], max_new_tokens: int) -> List[str]:
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    prompt_len = int(enc["input_ids"].shape[1])

    out_ids = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    raws = []
    for i in range(len(prompts)):
        gen = out_ids[i, prompt_len:]
        raws.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    return raws


# =========================
# Metrics
# =========================
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    denom = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    r2 = r2_score(y_true, y_pred)

    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        pear = np.nan
        spear = np.nan
    else:
        pear = float(pearsonr(y_true, y_pred)[0])
        spear = float(spearmanr(y_true, y_pred)[0])

    return {
        "MAE": float(mae),
        "MAPE_%": float(mape),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Pearson": pear,
        "Spearman": spear,
    }


def mean_baseline(y_true: np.ndarray) -> np.ndarray:
    mu = float(np.mean(y_true))
    return np.full_like(y_true, mu, dtype=float)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=140)

    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pred_jsonl = os.path.join(args.outdir, "predictions.jsonl")
    fail_jsonl = os.path.join(args.outdir, "failures_raw.jsonl")
    metrics_path = os.path.join(args.outdir, "metrics.json")
    pred_csv = os.path.join(args.outdir, "predictions.csv")
    labels_split50_csv = os.path.join(args.outdir, "labels_with_split50.csv")

    # Load JSONL + index by clip_id
    rows = read_jsonl(args.jsonl)
    row_by_id = {str(r.get("clip_id")): r for r in rows if r.get("clip_id") is not None}

    # Load labels
    labels_df = load_labels_df(args.labels)

    # Keep only labeled clips that exist in JSONL
    labels_df = labels_df[labels_df["clip_id"].isin(row_by_id.keys())].copy()
    labels_df = labels_df.dropna(subset=["clip_id", "y_true"])
    labels_df["y_true"] = labels_df["y_true"].astype(float)

    if len(labels_df) < 10:
        raise ValueError(f"Too few labeled clips after matching JSONL: {len(labels_df)}")

    # Build stratified split50
    labels_df = build_split50(labels_df, seed=args.seed)
    strat_used = labels_df.attrs.get("split50_stratify_used", "unknown")

    # Save labels_with_split50.csv for reporting + reproducibility
    try:
        import pandas as pd
        labels_df.to_csv(labels_split50_csv, index=False)
        print(f"Wrote split file: {labels_split50_csv} | stratify={strat_used}")
    except Exception as e:
        print("[warn] could not write labels_with_split50.csv:", str(e))

    # Build train/test rows
    train_ids = labels_df[labels_df["split50"] == "train50"]["clip_id"].tolist()
    test_ids = labels_df[labels_df["split50"] == "test50"]["clip_id"].tolist()

    train_rows = [row_by_id[cid] for cid in train_ids]
    test_rows = [row_by_id[cid] for cid in test_ids]

    # labels map
    labels_map = {row["clip_id"]: float(row["y_true"]) for _, row in labels_df.iterrows()}

    print(f"Split50: train50={len(train_rows)} | test50={len(test_rows)} | stratify_used={strat_used}")

    # Resume
    if args.resume:
        done = load_done_clip_ids(pred_jsonl)
        if done:
            before = len(test_rows)
            test_rows = [r for r in test_rows if str(r.get("clip_id")) not in done]
            print(f"[resume] Found {len(done)} completed clip_ids. Remaining test: {len(test_rows)} (was {before})")

    if len(test_rows) == 0:
        print("Nothing to run.")
        return

    # Retrieval vectors
    feature_keys = build_feature_key_list()
    X_train = np.stack([flatten_fields_for_vector(r, feature_keys) for r in train_rows], axis=0)
    X_test = np.stack([flatten_fields_for_vector(r, feature_keys) for r in test_rows], axis=0)

    X_train_n, mean_, std_ = impute_and_normalize(X_train)
    X_test_n = (np.where(np.isnan(X_test), mean_, X_test) - mean_) / std_

    nn = NearestNeighbors(n_neighbors=min(args.k, len(train_rows)), metric="cosine")
    nn.fit(X_train_n)
    dists, idxs = nn.kneighbors(X_test_n, return_distance=True)

    # Load LLM
    dtype = parse_dtype(args.dtype)
    if args.device.lower() == "cpu":
        dtype = torch.float32
    print(f"Loading model={args.model} | device={args.device} | dtype={dtype} | k={args.k}")
    tokenizer, model = load_llm(args.model, device=args.device, dtype=dtype)

    # Streaming files
    pred_f = open(pred_jsonl, "a", encoding="utf-8")
    fail_f = open(fail_jsonl, "a", encoding="utf-8")

    failures_count = 0

    try:
        for start in range(0, len(test_rows), args.batch_size):
            batch_rows = test_rows[start:start + args.batch_size]
            batch_prompts = []

            for bi, qrow in enumerate(batch_rows):
                qi = start + bi
                neigh_ids = idxs[qi]

                examples: List[Tuple[Dict[str, Any], float]] = []
                for ti in neigh_ids:
                    tr = train_rows[int(ti)]
                    cid = str(tr.get("clip_id"))
                    y = labels_map[cid]
                    examples.append((tr, y))

                user_content = build_fewshot_user_content(examples, qrow, k=len(examples))
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batch_prompts.append(prompt)

            raws = run_batch_prompts(tokenizer, model, batch_prompts, max_new_tokens=args.max_new_tokens)

            for qrow, raw in zip(batch_rows, raws):
                parsed = extract_json_object(raw)

                if parsed is None:
                    out = {
                        "clip_id": qrow.get("clip_id", "UNKNOWN"),
                        "predicted_cohesion": 4,
                        "confidence": "low",
                        "evidence_used": [],
                        "rationale": "Parsing failed; model did not return valid JSON.",
                        "failure_flag": True,
                    }
                    pred_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    fail_f.write(json.dumps({"clip_id": out["clip_id"], "raw_text": raw}, ensure_ascii=False) + "\n")
                    failures_count += 1
                    continue

                out = {
                    "clip_id": parsed.get("clip_id", qrow.get("clip_id", "UNKNOWN")),
                    "predicted_cohesion": clamp_int_1_7(parsed.get("predicted_cohesion", 4)),
                    "confidence": normalize_confidence(parsed.get("confidence", "low")),
                    "evidence_used": (parsed.get("evidence_used", []) or [])[:6],
                    "rationale": str(parsed.get("rationale", ""))[:600],
                    "failure_flag": bool(parsed.get("failure_flag", False)),
                }
                pred_f.write(json.dumps(out, ensure_ascii=False) + "\n")

            pred_f.flush()
            fail_f.flush()
            print(f"[progress] {min(start + args.batch_size, len(test_rows))}/{len(test_rows)} | parse_failures={failures_count}")

    finally:
        pred_f.close()
        fail_f.close()

    # Reload preds from disk
    preds = []
    with open(pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                preds.append(json.loads(line))
            except Exception:
                pass

    # Save CSV
    try:
        import pandas as pd
        pd.DataFrame(preds).to_csv(pred_csv, index=False)
        print("Wrote:", pred_csv)
    except Exception as e:
        print("[warn] CSV write failed:", str(e))

    # Metrics only on test50 ids
    test50_set = set(test_ids)
    y_true, y_pred = [], []
    for o in preds:
        cid = str(o.get("clip_id", ""))
        if cid in test50_set and cid in labels_map:
            y_true.append(labels_map[cid])
            y_pred.append(float(o.get("predicted_cohesion", 4)))

    n_fail = int(sum(1 for o in preds if o.get("failure_flag")))
    print(f"[Metrics] test50 matched labels: {len(y_true)} | total preds file: {len(preds)} | parse_failures: {n_fail}")

    if len(y_true) > 0:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        llm_m = regression_metrics(y_true, y_pred)
        base_m = regression_metrics(y_true, mean_baseline(y_true))

        metrics = {
            "split": "50/50 stratified",
            "stratify_used": strat_used,
            "n_train50": int(len(train_rows)),
            "n_test50": int(len(test_rows)),
            "n_test50_with_gt": int(len(y_true)),
            "n_failures_parse": int(n_fail),
            "llm_metrics": llm_m,
            "mean_baseline_metrics": base_m,
            "config": {
                "model": args.model,
                "k": args.k,
                "seed": args.seed,
                "device": args.device,
                "dtype": args.dtype,
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "resume": bool(args.resume),
            }
        }

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print("Wrote:", metrics_path)
        print("\n=== LLM Metrics (Retrieval few-shot, split50) ===")
        print(json.dumps(llm_m, indent=2))
        print("\n=== Mean Baseline Metrics ===")
        print(json.dumps(base_m, indent=2))
    else:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"error": "No test50 ground-truth labels matched clip_ids."}, f, indent=2)
        print("Wrote:", metrics_path)


if __name__ == "__main__":
    main()