"""
Visualize DMCA decision boundaries for the graded fashion infringement task.

Outputs:
- <out-prefix>_threshold_distance.png
- <out-prefix>_mlp_pca_boundary.png
- <out-prefix>_pairwise_feature_scatter.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


COLORS = {
    "original": "#1f77b4",
    "similar": "#ff7f0e",
    "inspired": "#ff7f0e",
    "knockoff": "#d62728",
}


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return l2_normalize(a) @ l2_normalize(b).T


def resolve_classes(label_to_idx):
    lower_map = {str(k).lower(): int(v) for k, v in label_to_idx.items()}
    if "original" not in lower_map:
        raise ValueError("label_to_idx must include 'original'.")
    middle_name = "similar" if "similar" in lower_map else "inspired"
    if middle_name not in lower_map:
        raise ValueError("label_to_idx must include 'similar' or 'inspired'.")
    far_name = "knockoff" if "knockoff" in lower_map else [k for k in lower_map if k not in {"original", middle_name}][0]
    idx_to_name = {v: k for k, v in lower_map.items()}
    return lower_map["original"], lower_map[middle_name], lower_map[far_name], idx_to_name


def build_pairwise_features(embeddings, ref_embeddings, exclude_self_indices=None):
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

    top2 = np.partition(sims, kth=-2, axis=1)[:, -2:]
    top_gap = top2[:, 1] - top2[:, 0]

    scalar = np.stack(
        [cos_sim, cos_dist, l2_dist, mean_abs, var_abs, max_abs, p95_abs, top_gap],
        axis=1,
    )
    features = np.concatenate([abs_diff, scalar], axis=1)
    return features.astype(np.float32), cos_dist.astype(np.float32), scalar.astype(np.float32)


def threshold_search_3class(y_true, distances, original_idx, middle_idx, knockoff_idx):
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
            macro_f1 = _macro_f1(y_true, preds)
            if macro_f1 > best_f1:
                best_t1, best_t2, best_f1 = t1, t2, macro_f1
    return best_t1, best_t2, best_f1


def _macro_f1(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1_scores = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1_scores))


def load_dataset(dataset_path: Path):
    data = torch.load(dataset_path, map_location="cpu")
    splits = {}
    for split in ("train", "val", "test"):
        splits[split] = {
            "embeddings": as_numpy(data[split]["embeddings"]).astype(np.float32),
            "labels": as_numpy(data[split]["labels"]).astype(np.int64),
        }
    label_to_idx = {str(k): int(v) for k, v in data["label_to_idx"].items()}
    return splits, label_to_idx


def plot_threshold_distance(out_path, labels, idx_to_name, distances, t1, t2, split_names):
    name_to_y = {"original": 0, "similar": 1, "inspired": 1, "knockoff": 2}
    split_markers = {"train": "o", "val": "s", "test": "^"}

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for split in ("train", "val", "test"):
        mask = split_names == split
        if not np.any(mask):
            continue
        for label_idx in np.unique(labels[mask]):
            class_mask = mask & (labels == label_idx)
            label_name = idx_to_name[int(label_idx)]
            ax.scatter(
                distances[class_mask],
                np.full(np.sum(class_mask), name_to_y[label_name], dtype=float),
                c=COLORS[label_name],
                marker=split_markers[split],
                s=34,
                alpha=0.72,
                edgecolors="none",
            )

    ax.axvspan(distances.min() - 0.02, t1, color=COLORS["original"], alpha=0.08)
    ax.axvspan(t1, t2, color=COLORS["similar"], alpha=0.08)
    ax.axvspan(t2, distances.max() + 0.02, color=COLORS["knockoff"], alpha=0.08)
    ax.axvline(t1, color="#333333", linestyle="--", linewidth=1.2, label=f"t1={t1:.3f}")
    ax.axvline(t2, color="#111111", linestyle="--", linewidth=1.2, label=f"t2={t2:.3f}")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["original", "similar", "knockoff"])
    ax.set_xlabel("Cosine distance to nearest original reference")
    ax.set_ylabel("True class")
    ax.set_title("Threshold Decision Bands on Gold Cases")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_mlp_pca_boundary(out_path, pca, mlp, x_all, labels_all, idx_to_name, split_names):
    x_2d = pca.transform(x_all)
    x_min, x_max = x_2d[:, 0].min() - 1.0, x_2d[:, 0].max() + 1.0
    y_min, y_max = x_2d[:, 1].min() - 1.0, x_2d[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 360), np.linspace(y_min, y_max, 360))
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_full = pca.inverse_transform(grid_2d)
    grid_pred = mlp.predict(grid_full).reshape(xx.shape)

    ordered_labels = sorted(idx_to_name.keys())
    ordered_names = [idx_to_name[i] for i in ordered_labels]
    bg_cmap = ListedColormap([COLORS[name] for name in ordered_names])
    label_to_pos = {label: i for i, label in enumerate(ordered_labels)}
    grid_pos = np.vectorize(label_to_pos.get)(grid_pred)

    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    ax.contourf(xx, yy, grid_pos, levels=np.arange(len(ordered_labels) + 1) - 0.5, cmap=bg_cmap, alpha=0.18)

    split_markers = {"train": "o", "val": "s", "test": "^"}
    for split in ("train", "val", "test"):
        mask = split_names == split
        for label_idx in ordered_labels:
            class_mask = mask & (labels_all == label_idx)
            if not np.any(class_mask):
                continue
            class_name = idx_to_name[int(label_idx)]
            ax.scatter(
                x_2d[class_mask, 0],
                x_2d[class_mask, 1],
                c=COLORS[class_name],
                marker=split_markers[split],
                s=36,
                alpha=0.82,
                edgecolors="black" if split == "test" else "none",
                linewidths=0.4,
                label=f"{split} {class_name}",
            )

    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title("MLP Decision Regions in PCA-Projected Pairwise Feature Space")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_pairwise_feature_scatter(out_path, scalar_features, labels_all, idx_to_name, split_names):
    cos_dist = scalar_features[:, 1]
    mean_abs = scalar_features[:, 3]
    split_markers = {"train": "o", "val": "s", "test": "^"}

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    for split in ("train", "val", "test"):
        mask = split_names == split
        for label_idx in sorted(idx_to_name.keys()):
            class_mask = mask & (labels_all == label_idx)
            if not np.any(class_mask):
                continue
            class_name = idx_to_name[int(label_idx)]
            ax.scatter(
                cos_dist[class_mask],
                mean_abs[class_mask],
                c=COLORS[class_name],
                marker=split_markers[split],
                s=34,
                alpha=0.76,
                edgecolors="black" if split == "test" else "none",
                linewidths=0.35,
                label=f"{split} {class_name}",
            )

    ax.set_xlabel("Cosine distance to nearest original")
    ax.set_ylabel("Mean absolute embedding difference")
    ax.set_title("Gold Cases in an Interpretable Pairwise Feature Plane")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot DMCA decision-boundary visualizations.")
    parser.add_argument("--dataset", default="dataset_500_with_original_cases.pt")
    parser.add_argument("--out-prefix", default="dmca_boundary")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlp-hidden", type=int, nargs="+", default=[256, 64])
    parser.add_argument("--mlp-max-iter", type=int, default=350)
    args = parser.parse_args()

    np.random.seed(args.seed)
    dataset_path = Path(args.dataset)
    splits, label_to_idx = load_dataset(dataset_path)
    original_idx, middle_idx, knockoff_idx, idx_to_name = resolve_classes(label_to_idx)

    train_emb = splits["train"]["embeddings"]
    val_emb = splits["val"]["embeddings"]
    test_emb = splits["test"]["embeddings"]
    train_lbl = splits["train"]["labels"]
    val_lbl = splits["val"]["labels"]
    test_lbl = splits["test"]["labels"]

    ref_mask = train_lbl == original_idx
    ref_emb = train_emb[ref_mask]
    if ref_emb.shape[0] < 2:
        raise ValueError("Need at least 2 original samples in train split.")

    train_self_ref = np.full((len(train_emb),), -1, dtype=int)
    train_orig_indices = np.where(ref_mask)[0]
    ref_map = {orig_idx: j for j, orig_idx in enumerate(train_orig_indices)}
    for i in range(len(train_emb)):
        if i in ref_map:
            train_self_ref[i] = ref_map[i]

    x_train, d_train, scalar_train = build_pairwise_features(train_emb, ref_emb, train_self_ref)
    x_val, d_val, scalar_val = build_pairwise_features(val_emb, ref_emb)
    x_test, d_test, scalar_test = build_pairwise_features(test_emb, ref_emb)

    t1, t2, _ = threshold_search_3class(val_lbl, d_val, original_idx, middle_idx, knockoff_idx)

    mlp = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=tuple(args.mlp_hidden), max_iter=args.mlp_max_iter, random_state=args.seed, early_stopping=True)),
        ]
    )
    mlp.fit(x_train, train_lbl)

    x_all = np.concatenate([x_train, x_val, x_test], axis=0)
    labels_all = np.concatenate([train_lbl, val_lbl, test_lbl], axis=0)
    d_all = np.concatenate([d_train, d_val, d_test], axis=0)
    scalar_all = np.concatenate([scalar_train, scalar_val, scalar_test], axis=0)
    split_names = np.array(
        ["train"] * len(train_lbl) + ["val"] * len(val_lbl) + ["test"] * len(test_lbl),
        dtype=object,
    )

    pca = PCA(n_components=2, random_state=args.seed)
    pca.fit(x_all)

    out_prefix = Path(args.out_prefix)
    plot_threshold_distance(
        out_prefix.with_name(f"{out_prefix.name}_threshold_distance.png"),
        labels_all,
        idx_to_name,
        d_all,
        t1,
        t2,
        split_names,
    )
    plot_mlp_pca_boundary(
        out_prefix.with_name(f"{out_prefix.name}_mlp_pca_boundary.png"),
        pca,
        mlp,
        x_all,
        labels_all,
        idx_to_name,
        split_names,
    )
    plot_pairwise_feature_scatter(
        out_prefix.with_name(f"{out_prefix.name}_pairwise_feature_scatter.png"),
        scalar_all,
        labels_all,
        idx_to_name,
        split_names,
    )

    print(f"Saved: {out_prefix.with_name(f'{out_prefix.name}_threshold_distance.png')}")
    print(f"Saved: {out_prefix.with_name(f'{out_prefix.name}_mlp_pca_boundary.png')}")
    print(f"Saved: {out_prefix.with_name(f'{out_prefix.name}_pairwise_feature_scatter.png')}")


if __name__ == "__main__":
    main()
