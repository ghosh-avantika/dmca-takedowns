"""
Run Stage-2 decision-layer experiments for graded infringement classification.

This script supports two data modes:
1) Embedding mode: load train/val/test embeddings directly from a .pt file.
2) Checkpoint mode: if split objects contain image paths, compute embeddings using
   a triplet checkpoint first, then run the same decision experiments.

Outputs:
- <out-prefix>_results.json
- <out-prefix>_results.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return l2_normalize(a) @ l2_normalize(b).T


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_order: List[int]) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_order).tolist(),
    }


def build_pairwise_features(
    embeddings: np.ndarray,
    ref_embeddings: np.ndarray,
    exclude_self_indices: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each sample, find nearest "original" reference by cosine similarity.
    Returns:
      scalar_features: [cosine_similarity, l2_distance, mean_abs_diff, var_abs_diff]
      abs_diff: element-wise absolute difference vector (N, D)
      cosine_distance: 1 - cosine_similarity (N,)
    """
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

    scalar_features = np.stack([cos_sim, l2_dist, mean_abs, var_abs], axis=1)
    return scalar_features, abs_diff, cos_dist


def threshold_search_3class(
    y_true: np.ndarray,
    dists: np.ndarray,
    label_original: int,
    label_middle: int,
    label_far: int,
) -> Tuple[float, float, float]:
    """
    Search thresholds t1 < t2:
      dist < t1         -> original
      t1 <= dist < t2   -> middle (inspired/similar)
      dist >= t2        -> far class (knockoff)
    """
    candidates = np.unique(np.quantile(dists, np.linspace(0.0, 1.0, 121)))
    best_t1, best_t2, best_f1 = 0.0, 0.0, -1.0
    for i in range(len(candidates) - 1):
        for j in range(i + 1, len(candidates)):
            t1, t2 = float(candidates[i]), float(candidates[j])
            preds = np.where(
                dists < t1,
                label_original,
                np.where(dists < t2, label_middle, label_far),
            )
            score = f1_score(y_true, preds, average="macro")
            if score > best_f1:
                best_t1, best_t2, best_f1 = t1, t2, float(score)
    return best_t1, best_t2, best_f1


def build_model(backbone: str, embed_dim: int) -> nn.Module:
    if backbone == "resnet18":
        model = models.resnet18(weights=None)
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    model.fc = nn.Linear(model.fc.in_features, embed_dim)
    return model


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[str], tfm):
        self.paths = [Path(p).expanduser() for p in paths]
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.tfm(img)


@torch.no_grad()
def embed_paths_with_checkpoint(
    model: nn.Module,
    paths: List[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    tfm = build_eval_transform()
    ds = ImagePathDataset(paths, tfm)

    def run_loader(worker_count: int):
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=worker_count)
        all_vecs: List[torch.Tensor] = []
        for images in dl:
            images = images.to(device)
            z = model(images)
            z = torch.nn.functional.normalize(z, p=2, dim=1)
            all_vecs.append(z.cpu())
        return torch.cat(all_vecs, dim=0).numpy()

    try:
        return run_loader(num_workers)
    except RuntimeError as exc:
        msg = str(exc)
        if num_workers > 0 and ("torch_shm_manager" in msg or "Operation not permitted" in msg):
            print("Falling back to num_workers=0 for embedding extraction.")
            return run_loader(0)
        raise


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(backbone=ckpt["backbone"], embed_dim=int(ckpt["embed_dim"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def split_to_numpy(split_obj: Dict, split_name: str) -> Tuple[np.ndarray, np.ndarray]:
    if "embeddings" not in split_obj or "labels" not in split_obj:
        raise ValueError(f"Split '{split_name}' missing 'embeddings' or 'labels'.")
    emb = split_obj["embeddings"]
    lbl = split_obj["labels"]
    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()
    else:
        emb = np.asarray(emb)
    if isinstance(lbl, torch.Tensor):
        lbl = lbl.cpu().numpy()
    else:
        lbl = np.asarray(lbl)
    return emb.astype(np.float32), lbl.astype(np.int64)


def resolve_label_indices(label_to_idx: Dict[str, int]) -> Tuple[int, int, int, str]:
    lower_map = {k.lower(): v for k, v in label_to_idx.items()}
    if "original" not in lower_map:
        raise ValueError("label_to_idx must contain an 'original' class.")

    middle_name = None
    for candidate in ["inspired", "similar"]:
        if candidate in lower_map:
            middle_name = candidate
            break
    if middle_name is None:
        raise ValueError("label_to_idx must contain 'inspired' or 'similar' class.")

    far_name = "knockoff" if "knockoff" in lower_map else None
    if far_name is None:
        remaining = [k for k in lower_map.keys() if k not in {"original", middle_name}]
        if len(remaining) != 1:
            raise ValueError("Could not infer far class; expected 'knockoff' or exactly one remaining class.")
        far_name = remaining[0]

    return lower_map["original"], lower_map[middle_name], lower_map[far_name], middle_name


def run_one_experiment(
    train_emb: np.ndarray,
    train_lbl: np.ndarray,
    val_emb: np.ndarray,
    val_lbl: np.ndarray,
    test_emb: np.ndarray,
    test_lbl: np.ndarray,
    label_to_idx: Dict[str, int],
    seed: int,
) -> List[Dict[str, object]]:
    original_idx, middle_idx, far_idx, middle_name = resolve_label_indices(label_to_idx)

    ref_mask = train_lbl == original_idx
    ref_emb = train_emb[ref_mask]
    if ref_emb.shape[0] < 2:
        raise ValueError("Need at least 2 'original' samples in train split for nearest-original features.")

    # Exclude self when train sample belongs to reference set.
    train_self_ref = np.full((len(train_emb),), -1, dtype=int)
    train_orig_indices = np.where(ref_mask)[0]
    ref_index_map = {orig_idx: j for j, orig_idx in enumerate(train_orig_indices)}
    for i in range(len(train_emb)):
        if i in ref_index_map:
            train_self_ref[i] = ref_index_map[i]

    train_scalar, train_abs, train_dist = build_pairwise_features(train_emb, ref_emb, train_self_ref)
    val_scalar, val_abs, val_dist = build_pairwise_features(val_emb, ref_emb)
    test_scalar, test_abs, test_dist = build_pairwise_features(test_emb, ref_emb)

    label_order = [far_idx, middle_idx, original_idx]
    results: List[Dict[str, object]] = []

    # 1) Threshold baseline
    t1, t2, best_val_f1 = threshold_search_3class(
        y_true=val_lbl,
        dists=val_dist,
        label_original=original_idx,
        label_middle=middle_idx,
        label_far=far_idx,
    )
    threshold_preds = np.where(
        test_dist < t1,
        original_idx,
        np.where(test_dist < t2, middle_idx, far_idx),
    )
    row = {
        "model": "Cosine Thresholds",
        "middle_class": middle_name,
        "t1": float(t1),
        "t2": float(t2),
        "val_macro_f1_for_threshold_search": float(best_val_f1),
    }
    row.update(eval_metrics(test_lbl, threshold_preds, label_order=label_order))
    results.append(row)

    # 2) Logistic Regression
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
    ])
    lr.fit(train_scalar, train_lbl)
    lr_preds = lr.predict(test_scalar)
    row = {"model": "Logistic Regression (pairwise features)", "middle_class": middle_name}
    row.update(eval_metrics(test_lbl, lr_preds, label_order=label_order))
    results.append(row)

    # 3) MLP decision layer
    mlp_x_train = np.concatenate([train_abs, train_scalar], axis=1)
    mlp_x_test = np.concatenate([test_abs, test_scalar], axis=1)
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=400, random_state=seed)),
    ])
    mlp.fit(mlp_x_train, train_lbl)
    mlp_preds = mlp.predict(mlp_x_test)
    row = {"model": "MLP (abs diff + pairwise features)", "middle_class": middle_name}
    row.update(eval_metrics(test_lbl, mlp_preds, label_order=label_order))
    results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description="Stage-2 decision-layer experiments for 3-class fashion infringement.")
    parser.add_argument(
        "--dataset-pt",
        nargs="+",
        default=["dataset_500_with_original.pt"],
        help="One or more .pt files containing train/val/test splits.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Optional triplet checkpoint; used only when splits contain image paths instead of embeddings.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for checkpoint embedding mode.")
    parser.add_argument("--num-workers", type=int, default=2, help="Num workers for checkpoint embedding mode.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", default="stage2_decision_results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_model = None
    if args.checkpoint:
        checkpoint_model = load_checkpoint_model(Path(args.checkpoint).expanduser(), device=device)
        print(f"Loaded checkpoint for embedding mode: {args.checkpoint}")

    all_results: List[Dict[str, object]] = []
    run_logs: List[Dict[str, object]] = []

    for dataset_raw in args.dataset_pt:
        dataset_path = Path(dataset_raw).expanduser()
        data = torch.load(dataset_path, map_location="cpu")
        if not isinstance(data, dict):
            raise ValueError(f"Unsupported dataset format: {dataset_path}")

        for split in ["train", "val", "test"]:
            if split not in data:
                raise ValueError(f"{dataset_path} missing split '{split}'.")

        label_to_idx = data.get("label_to_idx")
        if label_to_idx is None:
            raise ValueError(f"{dataset_path} missing 'label_to_idx'.")

        def read_split(split_name: str) -> Tuple[np.ndarray, np.ndarray]:
            split_obj = data[split_name]
            if "embeddings" in split_obj:
                return split_to_numpy(split_obj, split_name)

            if "paths" in split_obj and "labels" in split_obj:
                if checkpoint_model is None:
                    raise ValueError(
                        f"{dataset_path}:{split_name} contains paths but no embeddings. "
                        "Provide --checkpoint to compute embeddings."
                    )
                labels = split_obj["labels"]
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy().astype(np.int64)
                else:
                    labels = np.asarray(labels, dtype=np.int64)
                embeddings = embed_paths_with_checkpoint(
                    model=checkpoint_model,
                    paths=[str(p) for p in split_obj["paths"]],
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                return embeddings.astype(np.float32), labels

            raise ValueError(
                f"{dataset_path}:{split_name} has unsupported split keys {list(split_obj.keys())}. "
                "Expected embeddings+labels or paths+labels."
            )

        train_emb, train_lbl = read_split("train")
        val_emb, val_lbl = read_split("val")
        test_emb, test_lbl = read_split("test")

        results = run_one_experiment(
            train_emb=train_emb,
            train_lbl=train_lbl,
            val_emb=val_emb,
            val_lbl=val_lbl,
            test_emb=test_emb,
            test_lbl=test_lbl,
            label_to_idx=label_to_idx,
            seed=args.seed,
        )
        for row in results:
            row["dataset"] = str(dataset_path)
            all_results.append(row)

        run_logs.append(
            {
                "dataset": str(dataset_path),
                "train_size": int(train_emb.shape[0]),
                "val_size": int(val_emb.shape[0]),
                "test_size": int(test_emb.shape[0]),
                "feature_dim": int(train_emb.shape[1]),
                "label_to_idx": label_to_idx,
            }
        )
        print(f"Finished dataset: {dataset_path}")

    out_prefix = Path(args.out_prefix)
    out_json = out_prefix.with_name(f"{out_prefix.name}_results.json")
    out_csv = out_prefix.with_name(f"{out_prefix.name}_results.csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "runs": run_logs,
                "results": all_results,
                "notes": {
                    "confusion_label_order": "[far_class, middle_class, original]",
                    "models": [
                        "Cosine Thresholds",
                        "Logistic Regression (pairwise features)",
                        "MLP (abs diff + pairwise features)",
                    ],
                },
            },
            f,
            indent=2,
        )

    fields = [
        "dataset",
        "model",
        "middle_class",
        "accuracy",
        "macro_f1",
        "confusion_matrix",
        "t1",
        "t2",
        "val_macro_f1_for_threshold_search",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
