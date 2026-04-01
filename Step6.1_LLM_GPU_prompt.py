# ------------------------------------------------------------
# NTU Capstone Project: Group Cohesion Estimation using LLM and Deep Learning
# Maryam Ali Aljanobi
# April 2026

# File: Step6.1_LLM_GPU_prompt.py
#
# GPU run for the full-dataset LLM evaluation on structured JSON evidence
# this code was used to test various prompting styles to see how the LLM perform
# the system_prompt snippet was replaced with every run 
#
# Example GPU run:
#   python Step6.1_LLM_GPU_prompt.py \
#     --jsonl /home/$USER/GCE_Project/results/llm_inputs_v1_all_modalities.jsonl \
#     --labels /home/$USER/GCE_Project/results/llm_eval_labels.csv \
#     --outdir /home/$USER/GCE_Project/results/llm_runs/qwen3b_gpu \
#     --model Qwen/Qwen2.5-3B-Instruct \
#     --split test \
#     --device cuda \
#     --dtype fp16 \
#     --batch_size 16 \
#     --max_new_tokens 160
#

import os

# -------- HuggingFace download/cache stability 
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
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


# =========================
# 1) Prompt (strict + short)
# =========================
SYSTEM_PROMPT = SYSTEM_PROMPT = """
YYou are an annotation model for a research study on GROUP COHESION.

You will receive ONE clip described ONLY by a JSON object.
Your task is to predict group cohesion on a 1–7 scale.

IMPORTANT:
- Start from a neutral baseline of 4 (moderate cohesion).
- Adjust upward or downward ONLY if the evidence clearly supports it.
- Make small adjustments unless multiple indicators align strongly.

Rules:
- Use ONLY the fields in the JSON.
- Do NOT invent or assume information.
- Do NOT describe seeing or hearing the clip.
- Output ONE valid JSON object on ONE line.
- Keep rationale to at most 2 short sentences.
- Select at most 6 evidence fields.

Scale guidance:
- 1–2: clearly low cohesion
- 3–5: moderate / mixed cohesion
- 6–7: clearly high cohesion

Return EXACT schema:
{"clip_id":"...","predicted_cohesion":1,"confidence":"low|medium|high","evidence_used":["..."],"rationale":"...","failure_flag":false}
"""


# =========================
# 2) Helpers
# =========================
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_rows(
    rows: List[Dict[str, Any]],
    n_samples: Optional[int],
    split: Optional[str],
    seed: int,
) -> List[Dict[str, Any]]:
    pool = rows
    if split is not None:
        pool = [r for r in rows if str(r.get("split", "")).lower() == split.lower()]
    if not pool:
        raise ValueError(f"No rows found for split={split}")

    if n_samples is None or n_samples <= 0 or n_samples >= len(pool):
        return pool

    rng = random.Random(seed)
    return rng.sample(pool, n_samples)


def build_chat_prompt(tokenizer, clip_json: Dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(clip_json, ensure_ascii=False)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    # Try raw load
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

    # Regex fallback
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


def load_labels_map(path: str) -> Dict[str, float]:
    import pandas as pd
    df = pd.read_csv(path)

    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("vid", cols.get("clip_id", cols.get("title", df.columns[0])))
    y_col = cols.get("y_true", cols.get("cohesion", df.columns[-1]))

    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        out[str(r[id_col])] = float(r[y_col])
    return out


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
# 3) Model load
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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        dtype=dtype,
    )
    model.eval()
    return tokenizer, model


# =========================
# 4) Resume support
# =========================
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
# 5) Batch inference
# =========================
@torch.no_grad()
def run_batch(
    tokenizer,
    model,
    clips: List[Dict[str, Any]],
    max_new_tokens: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prompts = [build_chat_prompt(tokenizer, c) for c in clips]

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
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

    outputs: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for i, clip in enumerate(clips):
        gen_part = out_ids[i, prompt_len:]
        raw = tokenizer.decode(gen_part, skip_special_tokens=True).strip()

        parsed = extract_json_object(raw)
        if parsed is None:
            out = {
                "clip_id": clip.get("clip_id", "UNKNOWN"),
                "split": clip.get("split", "UNKNOWN"),
                "predicted_cohesion": 4,
                "confidence": "low",
                "evidence_used": [],
                "rationale": "Parsing failed; model did not return valid JSON.",
                "failure_flag": True,
            }
            outputs.append(out)
            failures.append({"clip_id": out["clip_id"], "raw_text": raw})
            continue

        out = {
            "clip_id": parsed.get("clip_id", clip.get("clip_id", "UNKNOWN")),
            "split": clip.get("split", "UNKNOWN"),
            "predicted_cohesion": clamp_int_1_7(parsed.get("predicted_cohesion", 4)),
            "confidence": normalize_confidence(parsed.get("confidence", "low")),
            "evidence_used": (parsed.get("evidence_used", []) or [])[:6],
            "rationale": str(parsed.get("rationale", ""))[:600],
            "failure_flag": bool(parsed.get("failure_flag", False)),
        }
        outputs.append(out)

    return outputs, failures


def run_all_streaming(
    tokenizer,
    model,
    clips: List[Dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    pred_jsonl_path: str,
    fail_jsonl_path: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns full outputs/failures lists
    """
    all_out: List[Dict[str, Any]] = []
    all_fail: List[Dict[str, Any]] = []

    # append mode for resume
    pred_f = open(pred_jsonl_path, "a", encoding="utf-8")
    fail_f = open(fail_jsonl_path, "a", encoding="utf-8")

    try:
        for start in range(0, len(clips), batch_size):
            chunk = clips[start:start + batch_size]
            outs, fails = run_batch(tokenizer, model, chunk, max_new_tokens=max_new_tokens)

            # stream write
            for o in outs:
                pred_f.write(json.dumps(o, ensure_ascii=False) + "\n")
            pred_f.flush()

            for x in fails:
                fail_f.write(json.dumps(x, ensure_ascii=False) + "\n")
            fail_f.flush()

            all_out.extend(outs)
            all_fail.extend(fails)

            print(f"[progress] {min(start + batch_size, len(clips))}/{len(clips)} done | failures so far: {len(all_fail)}")
    finally:
        pred_f.close()
        fail_f.close()

    return all_out, all_fail


# =========================
# 6) Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to llm_inputs_v1_all_modalities.jsonl")
    ap.add_argument("--labels", required=True, help="Path to llm_eval_labels.csv")
    ap.add_argument("--outdir", required=True, help="Output directory for predictions/metrics")

    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--split", default="test", help="Split filter (train/valid/test) or 'all'")
    ap.add_argument("--n_samples", type=int, default=0, help="If >0, sample N clips instead of full split")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--dtype", default="fp16", help="fp16|bf16|fp32")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=160)

    ap.add_argument("--resume", action="store_true", help="If set, skip clip_ids already in predictions.jsonl")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    split = None if args.split.lower() == "all" else args.split

    # Files
    pred_jsonl = os.path.join(args.outdir, "predictions.jsonl")
    fail_jsonl = os.path.join(args.outdir, "failures_raw.jsonl")
    metrics_path = os.path.join(args.outdir, "metrics.json")
    pred_csv = os.path.join(args.outdir, "predictions.csv")

    # Load data
    rows = read_jsonl(args.jsonl)
    clips = sample_rows(rows, args.n_samples if args.n_samples > 0 else None, split, args.seed)

    # Resume skip
    if args.resume:
        done = load_done_clip_ids(pred_jsonl)
        if done:
            clips = [c for c in clips if str(c.get("clip_id")) not in done]
            print(f"[resume] Found {len(done)} completed clip_ids in {pred_jsonl}. Remaining: {len(clips)}")
        else:
            print(f"[resume] No existing predictions.jsonl found. Starting fresh.")

    print(f"Loaded {len(clips)} clips to run (split={args.split}, n_samples={args.n_samples}).")
    print("First clip_ids:", [c.get("clip_id") for c in clips[:5]])

    # Load labels
    labels_map = load_labels_map(args.labels)

    # Model
    dtype = parse_dtype(args.dtype)
    if args.device.lower() == "cpu":
        dtype = torch.float32

    print(f"Loading model={args.model} | device={args.device} | dtype={dtype} | batch_size={args.batch_size}")
    tokenizer, model = load_llm(args.model, device=args.device, dtype=dtype)

    # Inference (streaming)
    if len(clips) > 0:
        outputs, failures = run_all_streaming(
            tokenizer, model, clips,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            pred_jsonl_path=pred_jsonl,
            fail_jsonl_path=fail_jsonl,
        )
    else:
        outputs, failures = [], []
        print("Nothing to run (all clips already completed).")

    # Reload all predictions from disk (so metrics include resumed + new)
    all_preds = []
    if os.path.exists(pred_jsonl):
        with open(pred_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_preds.append(json.loads(line))
                    except Exception:
                        pass

    # Save CSV
    try:
        import pandas as pd
        df = pd.DataFrame(all_preds)
        df.to_csv(pred_csv, index=False)
        print("Wrote:", pred_csv)
    except Exception as e:
        print("[warn] Could not write CSV:", str(e))

    # Metrics (only where GT exists)
    y_true, y_pred = [], []
    for o in all_preds:
        cid = str(o.get("clip_id", ""))
        if cid in labels_map:
            y_true.append(labels_map[cid])
            y_pred.append(float(o.get("predicted_cohesion", 4)))

    n_fail = int(sum(1 for o in all_preds if o.get("failure_flag")))
    print(f"[Metrics] matched labels: {len(y_true)} / total preds: {len(all_preds)} | parse_failures: {n_fail}")

    if len(y_true) > 0:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        llm_m = regression_metrics(y_true, y_pred)
        base_m = regression_metrics(y_true, mean_baseline(y_true))

        metrics = {
            "n_total_predictions": int(len(all_preds)),
            "n_with_gt": int(len(y_true)),
            "n_failures_parse": n_fail,
            "llm_metrics": llm_m,
            "mean_baseline_metrics": base_m,
            "config": {
                "model": args.model,
                "split": args.split,
                "n_samples": args.n_samples,
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
        print("\n=== LLM Metrics ===")
        print(json.dumps(llm_m, indent=2))
        print("\n=== Mean Baseline Metrics ===")
        print(json.dumps(base_m, indent=2))
    else:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"error": "No ground-truth labels matched clip_ids.", "n_total_predictions": len(all_preds)}, f, indent=2)
        print("Wrote:", metrics_path)


if __name__ == "__main__":
    main()
