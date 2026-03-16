"""
DMCA decision-layer pipeline for 3-class fashion infringement classification.

Notebook/Kaggle-friendly:
- Default RUN_MODE is "notebook" and works without CLI args.
- Auto-discovers dataset .pt under /kaggle/input if DATASET_PT is not set.
- Still supports CLI mode for local runs.

Expected dataset format:
- top-level keys: train, val, test
- each split has: embeddings, labels
- top-level label mapping: label_to_idx
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# Kaggle / Notebook Config
# =========================
RUN_MODE = "notebook"  # "notebook" or "cli"
DATASET_PT = None  # e.g. "/kaggle/input/.../dataset_500_with_original.pt"
SEED = 42
MLP_HIDDEN = [256, 64]
MLP_MAX_ITER = 350
AUTO_FLAG_CONF = 0.80
REVIEW_CONF = 0.55
OUT_PREFIX = "/kaggle/working/dmca_decision_layer"
BOOTSTRAP_N = 500
POLICY_GRID_STEP = 0.05


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return l2_normalize(a) @ l2_normalize(b).T


def resolve_classes(label_to_idx: Dict[str, int]) -> Tuple[int, int, int, Dict[int, str]]:
    lower_map = {k.lower(): int(v) for k, v in label_to_idx.items()}
    if "original" not in lower_map:
        raise ValueError("label_to_idx must include 'original'.")

    middle_name = None
    for name in ("inspired", "similar"):
        if name in lower_map:
            middle_name = name
            break
    if middle_name is None:
        raise ValueError("label_to_idx must include 'inspired' or 'similar'.")

    if "knockoff" in lower_map:
        far_name = "knockoff"
    else:
        remaining = [k for k in lower_map if k not in {"original", middle_name}]
        if len(remaining) != 1:
            raise ValueError("Could not infer knockoff/far class.")
        far_name = remaining[0]

    idx_to_name = {v: k for k, v in lower_map.items()}
    return lower_map["original"], lower_map[middle_name], lower_map[far_name], idx_to_name


def build_pairwise_features(
    embeddings: np.ndarray,
    ref_embeddings: np.ndarray,
    exclude_self_indices: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    sims = cosine_similarity_matrix(embeddings, ref_embeddings)
    if exclude_self_indices is not None:
        for i, ref_idx in enumerate(exclude_self_indices):
            if ref_idx >= 0:
                sims[i, ref_idx] = -1.0

    nearest_idx = sims.argmax(axis=1)
    nearest = ref_embeddings[nearest_idx]
    cos_sim = sims.max(axis=1)
    cos_dist = 1.0 - cos_sim
    l2_dist = np.linalg.norm(embeddings - nearest, axis=1)

    abs_diff = np.abs(embeddings - nearest)
    mean_abs = abs_diff.mean(axis=1)
    var_abs = abs_diff.var(axis=1)
    max_abs = abs_diff.max(axis=1)
    p95_abs = np.percentile(abs_diff, 95, axis=1)

    # Add top-2 similarity gap as a stability feature.
    top2 = np.partition(sims, kth=-2, axis=1)[:, -2:]
    top_gap = top2[:, 1] - top2[:, 0]

    scalar = np.stack(
        [cos_sim, cos_dist, l2_dist, mean_abs, var_abs, max_abs, p95_abs, top_gap],
        axis=1,
    )
    features = np.concatenate([abs_diff, scalar], axis=1)
    return features.astype(np.float32), cos_dist.astype(np.float32)


def threshold_search_3class(
    y_true: np.ndarray,
    distances: np.ndarray,
    original_idx: int,
    middle_idx: int,
    knockoff_idx: int,
) -> Tuple[float, float, float]:
    candidates = np.unique(np.quantile(distances, np.linspace(0.0, 1.0, 121)))
    best_t1, best_t2, best_f1 = 0.0, 0.0, -1.0
    for i in range(len(candidates) - 1):
        for j in range(i + 1, len(candidates)):
            t1 = float(candidates[i])
            t2 = float(candidates[j])
            preds = np.where(
                distances < t1,
                original_idx,
                np.where(distances < t2, middle_idx, knockoff_idx),
            )
            score = f1_score(y_true, preds, average="macro")
            if score > best_f1:
                best_t1, best_t2, best_f1 = t1, t2, float(score)
    return best_t1, best_t2, best_f1


def predict_threshold(
    distances: np.ndarray,
    t1: float,
    t2: float,
    original_idx: int,
    middle_idx: int,
    knockoff_idx: int,
) -> np.ndarray:
    return np.where(distances < t1, original_idx, np.where(distances < t2, middle_idx, knockoff_idx))


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def tune_temperature(probs: np.ndarray, y_true: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    probs = np.clip(probs, 1e-12, 1.0)
    logp = np.log(probs)
    best_t = 1.0
    best_nll = float("inf")
    for t in np.linspace(0.5, 5.0, 91):
        scaled = np.exp(logp / t)
        scaled = scaled / np.clip(scaled.sum(axis=1, keepdims=True), 1e-12, None)
        nll = float(log_loss(y_true, scaled, labels=labels))
        if nll < best_nll:
            best_t = float(t)
            best_nll = nll
    return best_t, best_nll


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    probs = np.clip(probs, 1e-12, 1.0)
    logp = np.log(probs)
    scaled = np.exp(logp / temperature)
    return scaled / np.clip(scaled.sum(axis=1, keepdims=True), 1e-12, None)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf < hi)
        if not np.any(m):
            continue
        bin_conf = float(conf[m].mean())
        bin_acc = float(acc[m].mean())
        ece += float(m.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def calibration_bins(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> List[Dict[str, float]]:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: List[Dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = float(bins[i]), float(bins[i + 1])
        if i == n_bins - 1:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)
        count = int(np.sum(m))
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "count": count,
                "mean_confidence": float(conf[m].mean()) if count else float("nan"),
                "accuracy": float(acc[m].mean()) if count else float("nan"),
            }
        )
    return rows


def bootstrap_macro_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def dmca_policy_actions(
    probs: np.ndarray,
    pred: np.ndarray,
    knockoff_idx: int,
    auto_flag_conf: float,
    review_conf: float,
) -> Dict[str, int]:
    conf = probs.max(axis=1)
    auto_flag = int(np.sum((pred == knockoff_idx) & (conf >= auto_flag_conf)))
    review = int(np.sum((conf >= review_conf) & ~((pred == knockoff_idx) & (conf >= auto_flag_conf))))
    no_action = int(len(pred) - auto_flag - review)
    return {"auto_flag": auto_flag, "review": review, "no_action": no_action}


def dmca_policy_metrics(
    probs: np.ndarray,
    pred: np.ndarray,
    y_true: np.ndarray,
    knockoff_idx: int,
    auto_flag_conf: float,
    review_conf: float,
) -> Dict[str, float]:
    conf = probs.max(axis=1)
    auto_mask = (pred == knockoff_idx) & (conf >= auto_flag_conf)
    review_mask = (conf >= review_conf) & ~auto_mask
    no_action_mask = ~(auto_mask | review_mask)
    true_knockoff = y_true == knockoff_idx

    auto_count = int(np.sum(auto_mask))
    review_count = int(np.sum(review_mask))
    no_action_count = int(np.sum(no_action_mask))
    total = len(y_true)
    total_knockoff = int(np.sum(true_knockoff))

    auto_true_knockoff = int(np.sum(auto_mask & true_knockoff))
    review_true_knockoff = int(np.sum(review_mask & true_knockoff))
    no_action_true_knockoff = int(np.sum(no_action_mask & true_knockoff))
    auto_false_positive = int(np.sum(auto_mask & ~true_knockoff))

    return {
        "auto_flag_conf": float(auto_flag_conf),
        "review_conf": float(review_conf),
        "auto_flag_count": auto_count,
        "review_count": review_count,
        "no_action_count": no_action_count,
        "auto_flag_rate": float(auto_count / total),
        "review_rate": float(review_count / total),
        "no_action_rate": float(no_action_count / total),
        "auto_flag_precision_knockoff": float(auto_true_knockoff / auto_count) if auto_count else float("nan"),
        "auto_flag_knockoff_recall": float(auto_true_knockoff / total_knockoff) if total_knockoff else float("nan"),
        "review_knockoff_recall": float(review_true_knockoff / total_knockoff) if total_knockoff else float("nan"),
        "missed_knockoffs_no_action": no_action_true_knockoff,
        "missed_knockoff_rate_no_action": float(no_action_true_knockoff / total_knockoff) if total_knockoff else float("nan"),
        "auto_flag_false_positives": auto_false_positive,
    }


def policy_sweep_rows(
    probs: np.ndarray,
    pred: np.ndarray,
    y_true: np.ndarray,
    knockoff_idx: int,
    step: float = 0.05,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    thresholds = np.round(np.arange(step, 1.0, step), 2)
    for review_conf in thresholds:
        for auto_flag_conf in thresholds:
            if auto_flag_conf < review_conf:
                continue
            rows.append(
                dmca_policy_metrics(
                    probs=probs,
                    pred=pred,
                    y_true=y_true,
                    knockoff_idx=knockoff_idx,
                    auto_flag_conf=float(auto_flag_conf),
                    review_conf=float(review_conf),
                )
            )
    return rows


def select_recommended_policy(
    rows: List[Dict[str, float]],
    min_precision: float = 0.90,
) -> Dict[str, float]:
    valid = [r for r in rows if not np.isnan(r["auto_flag_precision_knockoff"])]
    if not valid:
        return rows[0]

    preferred = [r for r in valid if r["auto_flag_precision_knockoff"] >= min_precision]
    if not preferred:
        preferred = [r for r in valid if r["auto_flag_precision_knockoff"] >= 0.80]
    if not preferred:
        preferred = valid

    preferred.sort(
        key=lambda r: (
            r["auto_flag_knockoff_recall"],
            -r["review_rate"],
            r["auto_flag_precision_knockoff"],
        ),
        reverse=True,
    )
    return preferred[0]


def load_embeddings_dataset(path: Path):
    data = torch.load(path, map_location="cpu")
    if "label_to_idx" not in data:
        raise ValueError("Dataset must include top-level 'label_to_idx'.")

    out = {}
    for split in ("train", "val", "test"):
        if split not in data:
            raise ValueError(f"Dataset missing split: {split}")
        split_obj = data[split]
        if "embeddings" not in split_obj or "labels" not in split_obj:
            raise ValueError(f"Split '{split}' missing embeddings/labels.")
        out[split] = {
            "embeddings": as_numpy(split_obj["embeddings"]).astype(np.float32),
            "labels": as_numpy(split_obj["labels"]).astype(np.int64),
        }
    out["label_to_idx"] = {str(k): int(v) for k, v in data["label_to_idx"].items()}
    return out


def discover_dataset_pt() -> Path:
    candidates = [
        Path("/kaggle/working/dataset_500_with_original.pt"),
        Path("/kaggle/working/dataset_clean_real_test.pt"),
    ]
    for p in candidates:
        if p.exists():
            return p

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for p in input_root.rglob("*.pt"):
            n = p.name.lower()
            if "dataset" in n and ("500" in n or "original" in n or "clean" in n):
                return p

    raise FileNotFoundError(
        "Could not auto-discover dataset .pt file. Set DATASET_PT in the config block."
    )


def build_args():
    parser = argparse.ArgumentParser(description="Stable DMCA decision-layer pipeline.")
    parser.add_argument("--dataset", default=None, help="Path to .pt dataset with train/val/test embeddings.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlp-hidden", type=int, nargs="+", default=[256, 64])
    parser.add_argument("--mlp-max-iter", type=int, default=350)
    parser.add_argument("--auto-flag-conf", type=float, default=0.80)
    parser.add_argument("--review-conf", type=float, default=0.55)
    parser.add_argument("--out-prefix", default="dmca_decision_layer")
    parser.add_argument("--bootstrap-n", type=int, default=500)
    parser.add_argument("--policy-grid-step", type=float, default=0.05)
    args, _unknown = parser.parse_known_args()

    if RUN_MODE == "notebook":
        dataset_path = Path(DATASET_PT).expanduser() if DATASET_PT else discover_dataset_pt()
        args.dataset = str(dataset_path)
        args.seed = SEED
        args.mlp_hidden = list(MLP_HIDDEN)
        args.mlp_max_iter = MLP_MAX_ITER
        args.auto_flag_conf = AUTO_FLAG_CONF
        args.review_conf = REVIEW_CONF
        args.out_prefix = OUT_PREFIX
        args.bootstrap_n = BOOTSTRAP_N
        args.policy_grid_step = POLICY_GRID_STEP
    elif args.dataset is None:
        raise ValueError("--dataset is required in cli mode.")

    return args


def main():
    args = build_args()
    np.random.seed(args.seed)

    data = load_embeddings_dataset(Path(args.dataset))
    label_to_idx = data["label_to_idx"]
    original_idx, middle_idx, knockoff_idx, idx_to_name = resolve_classes(label_to_idx)
    label_order = sorted(idx_to_name.keys())

    train_emb = data["train"]["embeddings"]
    val_emb = data["val"]["embeddings"]
    test_emb = data["test"]["embeddings"]
    train_lbl = data["train"]["labels"]
    val_lbl = data["val"]["labels"]
    test_lbl = data["test"]["labels"]

    ref_mask = train_lbl == original_idx
    ref_emb = train_emb[ref_mask]
    if ref_emb.shape[0] < 2:
        raise ValueError("Need at least 2 original samples in train split for nearest-reference features.")

    # Exclude self-neighbor for train originals.
    train_self_ref = np.full((len(train_emb),), -1, dtype=int)
    train_orig_indices = np.where(ref_mask)[0]
    ref_map = {orig_idx: j for j, orig_idx in enumerate(train_orig_indices)}
    for i in range(len(train_emb)):
        if i in ref_map:
            train_self_ref[i] = ref_map[i]

    x_train, _d_train = build_pairwise_features(train_emb, ref_emb, train_self_ref)
    x_val, d_val = build_pairwise_features(val_emb, ref_emb)
    x_test, d_test = build_pairwise_features(test_emb, ref_emb)

    # Track A: threshold model.
    t1, t2, val_t_f1 = threshold_search_3class(val_lbl, d_val, original_idx, middle_idx, knockoff_idx)
    val_pred_t = predict_threshold(d_val, t1, t2, original_idx, middle_idx, knockoff_idx)
    test_pred_t = predict_threshold(d_test, t1, t2, original_idx, middle_idx, knockoff_idx)
    threshold_val = eval_metrics(val_lbl, val_pred_t, label_order)
    threshold_test = eval_metrics(test_lbl, test_pred_t, label_order)
    t_ci_lo, t_ci_hi = bootstrap_macro_f1_ci(test_lbl, test_pred_t, args.bootstrap_n, args.seed)

    # Track B: MLP + temperature calibration.
    mlp = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=tuple(args.mlp_hidden),
                    max_iter=args.mlp_max_iter,
                    random_state=args.seed,
                    early_stopping=True,
                ),
            ),
        ]
    )
    mlp.fit(x_train, train_lbl)

    val_probs_raw = mlp.predict_proba(x_val)
    test_probs_raw = mlp.predict_proba(x_test)
    all_labels = np.array(sorted(np.unique(np.concatenate([train_lbl, val_lbl, test_lbl]))))

    temp, val_nll = tune_temperature(val_probs_raw, val_lbl, labels=all_labels)
    val_probs_cal = apply_temperature(val_probs_raw, temp)
    test_probs_cal = apply_temperature(test_probs_raw, temp)

    val_pred_m = val_probs_cal.argmax(axis=1)
    test_pred_m = test_probs_cal.argmax(axis=1)
    mlp_val = eval_metrics(val_lbl, val_pred_m, label_order)
    mlp_test = eval_metrics(test_lbl, test_pred_m, label_order)
    m_ci_lo, m_ci_hi = bootstrap_macro_f1_ci(test_lbl, test_pred_m, args.bootstrap_n, args.seed)

    ece_raw = expected_calibration_error(val_probs_raw, val_lbl)
    ece_cal = expected_calibration_error(val_probs_cal, val_lbl)
    val_calibration_raw_bins = calibration_bins(val_probs_raw, val_lbl)
    val_calibration_cal_bins = calibration_bins(val_probs_cal, val_lbl)

    val_policy_rows = policy_sweep_rows(
        probs=val_probs_cal,
        pred=val_pred_m,
        y_true=val_lbl,
        knockoff_idx=knockoff_idx,
        step=args.policy_grid_step,
    )
    test_policy_rows = policy_sweep_rows(
        probs=test_probs_cal,
        pred=test_pred_m,
        y_true=test_lbl,
        knockoff_idx=knockoff_idx,
        step=args.policy_grid_step,
    )
    recommended_val_policy = select_recommended_policy(val_policy_rows)
    recommended_test_policy = dmca_policy_metrics(
        probs=test_probs_cal,
        pred=test_pred_m,
        y_true=test_lbl,
        knockoff_idx=knockoff_idx,
        auto_flag_conf=float(recommended_val_policy["auto_flag_conf"]),
        review_conf=float(recommended_val_policy["review_conf"]),
    )
    test_policy = dmca_policy_actions(
        test_probs_cal,
        test_pred_m,
        knockoff_idx=knockoff_idx,
        auto_flag_conf=args.auto_flag_conf,
        review_conf=args.review_conf,
    )

    report = {
        "dataset": str(args.dataset),
        "label_to_idx": label_to_idx,
        "idx_to_label": {str(k): idx_to_name[k] for k in idx_to_name},
        "class_indices": {
            "original": original_idx,
            "middle": middle_idx,
            "knockoff": knockoff_idx,
        },
        "track_threshold": {
            "thresholds": {"t1": t1, "t2": t2},
            "val_search_macro_f1": val_t_f1,
            "val_metrics": threshold_val,
            "test_metrics": threshold_test,
            "test_macro_f1_ci95": [t_ci_lo, t_ci_hi],
        },
        "track_mlp": {
            "mlp_hidden": list(args.mlp_hidden),
            "mlp_max_iter": args.mlp_max_iter,
            "temperature": temp,
            "val_nll_after_temp_tuning": val_nll,
            "val_ece_raw": ece_raw,
            "val_ece_calibrated": ece_cal,
            "val_metrics": mlp_val,
            "test_metrics": mlp_test,
            "test_macro_f1_ci95": [m_ci_lo, m_ci_hi],
            "calibration_bins_val_raw": val_calibration_raw_bins,
            "calibration_bins_val_calibrated": val_calibration_cal_bins,
            "policy": {
                "auto_flag_conf": args.auto_flag_conf,
                "review_conf": args.review_conf,
                "test_action_counts": test_policy,
                "test_policy_metrics_requested_thresholds": dmca_policy_metrics(
                    probs=test_probs_cal,
                    pred=test_pred_m,
                    y_true=test_lbl,
                    knockoff_idx=knockoff_idx,
                    auto_flag_conf=args.auto_flag_conf,
                    review_conf=args.review_conf,
                ),
                "recommended_from_validation": {
                    "selection_rule": "max auto-flag knockoff recall subject to auto-flag precision >= 0.90; fallback to >= 0.80 if needed",
                    "validation_metrics": recommended_val_policy,
                    "test_metrics": recommended_test_policy,
                },
            },
        },
    }

    out_json = Path(f"{args.out_prefix}_report.json")
    out_csv = Path(f"{args.out_prefix}_summary.csv")
    out_cm_threshold = Path(f"{args.out_prefix}_cm_threshold_test.csv")
    out_cm_mlp = Path(f"{args.out_prefix}_cm_mlp_test.csv")
    out_policy_val_csv = Path(f"{args.out_prefix}_policy_sweep_val.csv")
    out_policy_test_csv = Path(f"{args.out_prefix}_policy_sweep_test.csv")
    out_calibration_raw_csv = Path(f"{args.out_prefix}_calibration_val_raw.csv")
    out_calibration_cal_csv = Path(f"{args.out_prefix}_calibration_val_calibrated.csv")

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "val_accuracy",
                "val_macro_f1",
                "test_accuracy",
                "test_macro_f1",
                "test_macro_f1_ci95_lo",
                "test_macro_f1_ci95_hi",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "threshold",
                "val_accuracy": threshold_val["accuracy"],
                "val_macro_f1": threshold_val["macro_f1"],
                "test_accuracy": threshold_test["accuracy"],
                "test_macro_f1": threshold_test["macro_f1"],
                "test_macro_f1_ci95_lo": t_ci_lo,
                "test_macro_f1_ci95_hi": t_ci_hi,
            }
        )
        writer.writerow(
            {
                "model": "mlp_calibrated",
                "val_accuracy": mlp_val["accuracy"],
                "val_macro_f1": mlp_val["macro_f1"],
                "test_accuracy": mlp_test["accuracy"],
                "test_macro_f1": mlp_test["macro_f1"],
                "test_macro_f1_ci95_lo": m_ci_lo,
                "test_macro_f1_ci95_hi": m_ci_hi,
            }
        )

    np.savetxt(out_cm_threshold, np.array(threshold_test["confusion_matrix"], dtype=int), fmt="%d", delimiter=",")
    np.savetxt(out_cm_mlp, np.array(mlp_test["confusion_matrix"], dtype=int), fmt="%d", delimiter=",")

    with out_policy_val_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(val_policy_rows[0].keys()))
        writer.writeheader()
        writer.writerows(val_policy_rows)

    with out_policy_test_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_policy_rows[0].keys()))
        writer.writeheader()
        writer.writerows(test_policy_rows)

    with out_calibration_raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(val_calibration_raw_bins[0].keys()))
        writer.writeheader()
        writer.writerows(val_calibration_raw_bins)

    with out_calibration_cal_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(val_calibration_cal_bins[0].keys()))
        writer.writeheader()
        writer.writerows(val_calibration_cal_bins)

    print(f"Dataset: {args.dataset}")
    print(f"Saved report JSON: {out_json}")
    print(f"Saved summary CSV: {out_csv}")
    print(f"Saved threshold test confusion matrix CSV: {out_cm_threshold}")
    print(f"Saved MLP test confusion matrix CSV: {out_cm_mlp}")
    print(f"Saved validation policy sweep CSV: {out_policy_val_csv}")
    print(f"Saved test policy sweep CSV: {out_policy_test_csv}")
    print(f"Saved validation raw calibration CSV: {out_calibration_raw_csv}")
    print(f"Saved validation calibrated calibration CSV: {out_calibration_cal_csv}")
    print(
        "Recommended policy from validation: "
        f"auto_flag_conf={recommended_val_policy['auto_flag_conf']:.2f}, "
        f"review_conf={recommended_val_policy['review_conf']:.2f}"
    )
    print(
        "Best-by-test-macroF1: "
        + ("mlp_calibrated" if mlp_test["macro_f1"] >= threshold_test["macro_f1"] else "threshold")
    )


if __name__ == "__main__":
    main()
