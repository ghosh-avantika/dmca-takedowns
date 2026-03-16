"""
Baselines for 3-class decision layer using fixed embeddings.

Approach:
- Use "original" embeddings from the train split as reference set.
- For each sample, find nearest original (by cosine similarity).
- Compute pairwise features vs that nearest original.
- Baselines:
  1) Cosine-distance thresholds (t1, t2) learned on validation set.
  2) Logistic Regression on pairwise features.
  3) Small MLP (1-2 layers) on pairwise features.
"""

import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return a_n @ b_n.T


def build_pairwise_features(
    embeddings: np.ndarray,
    ref_embeddings: np.ndarray,
    exclude_self_indices: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each embedding, find nearest original by cosine similarity.
    Returns:
      features: [cos_sim, l2_dist, mean_abs_diff, var_abs_diff]
      abs_diff_vec: element-wise abs diff (for MLP)
      cos_dist: 1 - cos_sim
    """
    sims = cosine_similarity_matrix(embeddings, ref_embeddings)
    if exclude_self_indices is not None:
        for i, ref_idx in enumerate(exclude_self_indices):
            if ref_idx >= 0:
                sims[i, ref_idx] = -1.0

    nearest_idx = sims.argmax(axis=1)
    nearest = ref_embeddings[nearest_idx]

    diffs = np.abs(embeddings - nearest)
    mean_abs = diffs.mean(axis=1)
    var_abs = diffs.var(axis=1)
    cos_sim = sims.max(axis=1)
    cos_dist = 1.0 - cos_sim
    l2 = np.linalg.norm(embeddings - nearest, axis=1)

    features = np.stack([cos_sim, l2, mean_abs, var_abs], axis=1)
    return features, diffs, cos_dist


def threshold_search(y_true: np.ndarray, dists: np.ndarray) -> tuple[float, float, float]:
    # Grid search on candidate thresholds from val distances
    candidates = np.unique(np.quantile(dists, np.linspace(0.0, 1.0, 101)))
    best = (0.0, 0.0, -1.0)
    for i in range(len(candidates) - 1):
        for j in range(i + 1, len(candidates)):
            t1, t2 = candidates[i], candidates[j]
            preds = np.where(dists < t1, 0, np.where(dists < t2, 1, 2))
            f1 = f1_score(y_true, preds, average="macro")
            if f1 > best[2]:
                best = (t1, t2, f1)
    return best


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compute_centroids(embeddings: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    centroids = []
    for cls in range(num_classes):
        cls_emb = embeddings[labels == cls]
        if len(cls_emb) == 0:
            raise ValueError(f"No samples for class {cls} in train split.")
        centroids.append(cls_emb.mean(axis=0))
    return np.stack(centroids, axis=0)


def centroid_features(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    cos = cosine_similarity_matrix(embeddings, centroids)
    l2 = np.linalg.norm(embeddings[:, None, :] - centroids[None, :, :], axis=2)
    return np.concatenate([cos, l2], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Pairwise baselines for 3-class decision layer.")
    parser.add_argument("--dataset", default="dataset_500_with_original.pt")
    parser.add_argument("--out-json", default="pairwise_baselines_log.json")
    parser.add_argument("--out-csv", default="pairwise_baselines_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", default="distance_by_class.png")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    data = torch.load(Path(args.dataset), map_location="cpu")
    label_to_idx = data["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    original_idx = label_to_idx.get("original")
    if original_idx is None:
        raise ValueError("Dataset must include 'original' label.")

    def get_split(split):
        emb = data[split]["embeddings"].numpy()
        lbl = data[split]["labels"].numpy()
        return emb, lbl

    train_emb, train_lbl = get_split("train")
    val_emb, val_lbl = get_split("val")
    test_emb, test_lbl = get_split("test")

    # Reference originals from train split only
    ref_mask = train_lbl == original_idx
    ref_emb = train_emb[ref_mask]
    if len(ref_emb) < 2:
        raise ValueError("Need at least 2 'original' samples in train split.")

    # For train originals, exclude self when nearest original is selected
    train_self_idx = np.full((len(train_emb),), -1, dtype=int)
    train_orig_indices = np.where(ref_mask)[0]
    # Map each train original sample to its index within ref_emb
    ref_index_map = {orig_idx: i for i, orig_idx in enumerate(train_orig_indices)}
    for i in range(len(train_emb)):
        if i in ref_index_map:
            train_self_idx[i] = ref_index_map[i]

    train_feat, train_diff, train_dist = build_pairwise_features(train_emb, ref_emb, train_self_idx)
    val_feat, val_diff, val_dist = build_pairwise_features(val_emb, ref_emb)
    test_feat, test_diff, test_dist = build_pairwise_features(test_emb, ref_emb)

    results = []

    # 1) Threshold baseline on cosine distance
    t1, t2, best_val_f1 = threshold_search(val_lbl, val_dist)
    test_preds = np.where(test_dist < t1, 0, np.where(test_dist < t2, 1, 2))
    res = {"model": "Cosine Thresholds", "t1": float(t1), "t2": float(t2)}
    res.update(eval_metrics(test_lbl, test_preds))
    results.append(res)

    # Plot distance distributions by class (test)
    plt.figure(figsize=(7, 4))
    for cls_idx, cls_name in idx_to_label.items():
        cls_mask = test_lbl == cls_idx
        if cls_mask.sum() == 0:
            continue
        plt.hist(test_dist[cls_mask], bins=30, alpha=0.5, label=cls_name)
    plt.axvline(t1, color="k", linestyle="--", linewidth=1, label=f"t1={t1:.3f}")
    plt.axvline(t2, color="k", linestyle=":", linewidth=1, label=f"t2={t2:.3f}")
    plt.title("Cosine Distance to Nearest Original (Test)")
    plt.xlabel("Cosine distance (1 - cosine similarity)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot, dpi=200)
    plt.close()

    # 1b) Centroid features baselines (train-only centroids)
    num_classes = len(label_to_idx)
    centroids = compute_centroids(train_emb, train_lbl, num_classes)
    train_cent = centroid_features(train_emb, centroids)
    test_cent = centroid_features(test_emb, centroids)

    lr_cent = LogisticRegression(max_iter=1000, random_state=args.seed)
    lr_cent.fit(train_cent, train_lbl)
    lr_cent_preds = lr_cent.predict(test_cent)
    res = {"model": "Logistic Regression (centroid features)"}
    res.update(eval_metrics(test_lbl, lr_cent_preds))
    results.append(res)

    mlp_cent = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=args.seed)
    mlp_cent.fit(train_cent, train_lbl)
    mlp_cent_preds = mlp_cent.predict(test_cent)
    res = {"model": "MLP (centroid features)"}
    res.update(eval_metrics(test_lbl, mlp_cent_preds))
    results.append(res)

    # 2) Logistic Regression on pairwise features
    lr = LogisticRegression(max_iter=1000, random_state=args.seed)
    lr.fit(train_feat, train_lbl)
    lr_preds = lr.predict(test_feat)
    res = {"model": "Logistic Regression (pairwise features)"}
    res.update(eval_metrics(test_lbl, lr_preds))
    results.append(res)

    # 3) Small MLP on abs-diff + scalar features
    mlp_X_train = np.concatenate([train_diff, train_feat], axis=1)
    mlp_X_test = np.concatenate([test_diff, test_feat], axis=1)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300, random_state=args.seed)
    mlp.fit(mlp_X_train, train_lbl)
    mlp_preds = mlp.predict(mlp_X_test)
    res = {"model": "MLP (abs diff + pairwise features)"}
    res.update(eval_metrics(test_lbl, mlp_preds))
    results.append(res)

    # Save results
    with open(args.out_json, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "label_to_idx": label_to_idx,
                "results": results,
            },
            f,
            indent=2,
        )

    import csv
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "macro_f1", "confusion_matrix", "t1", "t2"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Saved: {args.out_json}")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
