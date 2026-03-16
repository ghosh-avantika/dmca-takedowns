import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return l2_normalize(a) @ l2_normalize(b).T


def build_pairwise_features(embeddings: np.ndarray, ref_embeddings: np.ndarray, exclude_self_indices=None):
    sims = cosine_similarity_matrix(embeddings, ref_embeddings)
    if exclude_self_indices is not None:
        for i, ref_idx in enumerate(exclude_self_indices):
            if ref_idx >= 0:
                sims[i, ref_idx] = -1.0

    nearest_idx = sims.argmax(axis=1)
    nearest = ref_embeddings[nearest_idx]
    abs_diff = np.abs(embeddings - nearest)
    mean_abs = abs_diff.mean(axis=1)
    var_abs = abs_diff.var(axis=1)
    cos_sim = sims.max(axis=1)
    cos_dist = 1.0 - cos_sim
    l2_dist = np.linalg.norm(embeddings - nearest, axis=1)
    scalar = np.stack([cos_sim, l2_dist, mean_abs, var_abs], axis=1)
    return scalar, abs_diff, cos_dist


def threshold_search_3class(y_true, dists, label_original, label_middle, label_far):
    candidates = np.unique(np.quantile(dists, np.linspace(0.0, 1.0, 121)))
    best_t1, best_t2, best_f1 = 0.0, 0.0, -1.0
    for i in range(len(candidates) - 1):
        for j in range(i + 1, len(candidates)):
            t1, t2 = float(candidates[i]), float(candidates[j])
            preds = np.where(dists < t1, label_original, np.where(dists < t2, label_middle, label_far))
            score = f1_score(y_true, preds, average="macro")
            if score > best_f1:
                best_t1, best_t2, best_f1 = t1, t2, float(score)
    return best_t1, best_t2, best_f1


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def label_indices(label_to_idx):
    lower_map = {k.lower(): int(v) for k, v in label_to_idx.items()}
    if "original" not in lower_map:
        raise ValueError("label_to_idx must contain 'original'.")
    middle_name = "similar" if "similar" in lower_map else "inspired"
    if middle_name not in lower_map:
        raise ValueError("label_to_idx must contain 'similar' or 'inspired'.")
    far_name = "knockoff" if "knockoff" in lower_map else [k for k in lower_map if k not in {"original", middle_name}][0]
    return lower_map["original"], lower_map[middle_name], lower_map[far_name]


def reason_from_dist(true_name: str, pred_name: str, dist: float, t1: float, t2: float) -> str:
    if true_name == "knockoff" and pred_name in {"similar", "original"}:
        return "Strong visual overlap with nearest original; model underestimates infringement severity."
    if true_name == "original" and pred_name in {"similar", "knockoff"}:
        return "Likely false positive from shared category features (silhouette/color/layout)."
    margin = min(abs(dist - t1), abs(dist - t2))
    if margin < 0.01:
        return "Boundary case near decision threshold; slight embedding shifts can flip class."
    return "Ambiguous design cues; likely needs richer texture/detail features or more hard negatives."


def main():
    parser = argparse.ArgumentParser(description="Build final submission metrics and error-case pack.")
    parser.add_argument("--dataset", default="dataset_500_with_original.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--error-count", type=int, default=15)
    parser.add_argument("--out-prefix", default="final_submission_pack")
    args = parser.parse_args()

    np.random.seed(args.seed)
    data = torch.load(Path(args.dataset), map_location="cpu")
    train_emb = np.asarray(data["train"]["embeddings"], dtype=np.float32)
    val_emb = np.asarray(data["val"]["embeddings"], dtype=np.float32)
    test_emb = np.asarray(data["test"]["embeddings"], dtype=np.float32)
    train_lbl = np.asarray(data["train"]["labels"], dtype=np.int64)
    val_lbl = np.asarray(data["val"]["labels"], dtype=np.int64)
    test_lbl = np.asarray(data["test"]["labels"], dtype=np.int64)

    label_to_idx = {str(k): int(v) for k, v in data["label_to_idx"].items()}
    idx_to_label = {int(v): str(k).lower() for k, v in label_to_idx.items()}
    original_idx, middle_idx, far_idx = label_indices(label_to_idx)

    ref_mask = train_lbl == original_idx
    ref_emb = train_emb[ref_mask]
    train_self_ref = np.full((len(train_emb),), -1, dtype=int)
    train_orig_indices = np.where(ref_mask)[0]
    ref_index_map = {orig_idx: j for j, orig_idx in enumerate(train_orig_indices)}
    for i in range(len(train_emb)):
        if i in ref_index_map:
            train_self_ref[i] = ref_index_map[i]

    train_scalar, train_abs, train_dist = build_pairwise_features(train_emb, ref_emb, train_self_ref)
    val_scalar, val_abs, val_dist = build_pairwise_features(val_emb, ref_emb)
    test_scalar, test_abs, test_dist = build_pairwise_features(test_emb, ref_emb)

    t1, t2, val_threshold_f1 = threshold_search_3class(val_lbl, val_dist, original_idx, middle_idx, far_idx)
    threshold_pred = np.where(test_dist < t1, original_idx, np.where(test_dist < t2, middle_idx, far_idx))

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=args.seed)),
    ])
    lr.fit(train_scalar, train_lbl)
    lr_pred = lr.predict(test_scalar)

    mlp_x_train = np.concatenate([train_abs, train_scalar], axis=1)
    mlp_x_test = np.concatenate([test_abs, test_scalar], axis=1)
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=400, random_state=args.seed)),
    ])
    mlp.fit(mlp_x_train, train_lbl)
    mlp_pred = mlp.predict(mlp_x_test)
    mlp_prob = mlp.predict_proba(mlp_x_test)

    rows = [
        {"model": "Cosine Thresholds", **metrics(test_lbl, threshold_pred), "val_threshold_macro_f1": float(val_threshold_f1)},
        {"model": "Logistic Regression (pairwise features)", **metrics(test_lbl, lr_pred), "val_threshold_macro_f1": ""},
        {"model": "MLP (abs diff + pairwise features)", **metrics(test_lbl, mlp_pred), "val_threshold_macro_f1": ""},
    ]
    best = max(rows, key=lambda r: r["macro_f1"])["model"]

    out_prefix = Path(args.out_prefix)
    metrics_csv = out_prefix.with_name(f"{out_prefix.name}_metrics.csv")
    summary_json = out_prefix.with_name(f"{out_prefix.name}_summary.json")
    errors_csv = out_prefix.with_name(f"{out_prefix.name}_error_cases.csv")

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "macro_f1", "val_threshold_macro_f1"])
        writer.writeheader()
        writer.writerows(rows)

    best_pred = mlp_pred if best.startswith("MLP") else threshold_pred if best.startswith("Cosine") else lr_pred
    errors = []
    for i in range(len(test_lbl)):
        if int(best_pred[i]) == int(test_lbl[i]):
            continue
        true_name = idx_to_label[int(test_lbl[i])]
        pred_name = idx_to_label[int(best_pred[i])]
        conf = float(mlp_prob[i].max()) if best.startswith("MLP") else float("nan")
        errors.append(
            {
                "test_index": i,
                "true_label": true_name,
                "pred_label": pred_name,
                "cosine_distance_to_nearest_original": float(test_dist[i]),
                "nearest_original_cosine_similarity": float(1.0 - test_dist[i]),
                "mlp_max_confidence": conf,
                "distance_to_t1": float(abs(test_dist[i] - t1)),
                "distance_to_t2": float(abs(test_dist[i] - t2)),
                "likely_failure_reason": reason_from_dist(true_name, pred_name, float(test_dist[i]), float(t1), float(t2)),
            }
        )

    if best.startswith("MLP"):
        errors.sort(key=lambda r: (r["mlp_max_confidence"], -r["distance_to_t1"]), reverse=True)
    else:
        errors.sort(key=lambda r: min(r["distance_to_t1"], r["distance_to_t2"]))
    errors = errors[: args.error_count]

    with errors_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test_index",
                "true_label",
                "pred_label",
                "cosine_distance_to_nearest_original",
                "nearest_original_cosine_similarity",
                "mlp_max_confidence",
                "distance_to_t1",
                "distance_to_t2",
                "likely_failure_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(errors)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": str(Path(args.dataset)),
                "seed": args.seed,
                "thresholds": {"t1": float(t1), "t2": float(t2)},
                "best_model_by_test_macro_f1": best,
                "num_total_test_errors_best_model": int(np.sum(best_pred != test_lbl)),
                "exported_error_cases": len(errors),
                "outputs": {
                    "metrics_csv": str(metrics_csv),
                    "error_cases_csv": str(errors_csv),
                },
            },
            f,
            indent=2,
        )

    print(f"Saved: {metrics_csv}")
    print(f"Saved: {errors_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
