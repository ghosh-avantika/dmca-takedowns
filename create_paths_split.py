"""
Create a paths-based split file for Stage-2 checkpoint experiments.

Expected input CSV columns:
- path: absolute or relative image path
- label: one of {original, similar|inspired, knockoff}

Output format:
{
  "train": {"paths": [...], "labels": Tensor[int64]},
  "val":   {"paths": [...], "labels": Tensor[int64]},
  "test":  {"paths": [...], "labels": Tensor[int64]},
  "label_to_idx": {...},
  "idx_to_label": {...},
  "metadata": {...}
}
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


ALLOWED_LABELS = {"original", "similar", "inspired", "knockoff"}


def normalize_label(raw: str) -> str:
    value = str(raw).strip().lower()
    if value == "inspired":
        return "similar"
    if value not in ALLOWED_LABELS:
        raise ValueError(
            f"Unsupported label '{raw}'. Expected one of: original, similar/inspired, knockoff."
        )
    return value


def stratified_split_indices(labels: np.ndarray, seed: int, train_ratio: float, val_ratio: float):
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(labels))
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for cls in np.unique(labels):
        cls_idx = all_indices[labels == cls]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        n_train = min(n_train, n - n_val - 1) if n > 1 else 1
        n_train = max(1, n_train) if n >= 3 else max(0, n - 2)
        n_val = max(1, n_val) if n >= 3 else max(0, n - n_train - 1)
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)

        train_idx.extend(cls_idx[:n_train].tolist())
        val_idx.extend(cls_idx[n_train:n_train + n_val].tolist())
        test_idx.extend(cls_idx[n_train + n_val:].tolist())

    return np.array(train_idx, dtype=int), np.array(val_idx, dtype=int), np.array(test_idx, dtype=int)


def main():
    parser = argparse.ArgumentParser(description="Create paths_split.pt from a labeled image manifest CSV.")
    parser.add_argument("--manifest-csv", required=True, help="CSV with columns: path,label")
    parser.add_argument("--out", default="paths_split.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--strict-exists",
        action="store_true",
        help="If set, fail when any path does not exist on disk.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_csv).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Use ratios like train=0.7 val=0.15 so test gets remaining fraction.")

    df = pd.read_csv(manifest_path)
    required_cols = {"path", "label"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_cols)}")

    rows = []
    dropped = 0
    for _, row in df.iterrows():
        path_val = row["path"]
        label_val = row["label"]
        if pd.isna(path_val) or pd.isna(label_val):
            dropped += 1
            continue
        path = Path(str(path_val)).expanduser()
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        if args.strict_exists and not path.exists():
            raise FileNotFoundError(f"Path from manifest does not exist: {path}")
        try:
            label = normalize_label(str(label_val))
        except ValueError:
            dropped += 1
            continue
        rows.append((str(path), label))

    if len(rows) < 30:
        raise ValueError(f"Need at least 30 valid rows, found {len(rows)}.")

    paths = np.array([r[0] for r in rows], dtype=object)
    labels_str = np.array([r[1] for r in rows], dtype=object)

    unique_labels = sorted(set(labels_str.tolist()))
    if not {"original", "knockoff"}.issubset(unique_labels):
        raise ValueError(
            "Manifest must include both 'original' and 'knockoff' labels."
        )
    if "similar" not in unique_labels:
        raise ValueError("Manifest must include 'similar' (or 'inspired') labels.")

    label_to_idx: Dict[str, int] = {name: i for i, name in enumerate(unique_labels)}
    idx_to_label: Dict[int, str] = {i: name for name, i in label_to_idx.items()}
    labels = np.array([label_to_idx[x] for x in labels_str], dtype=np.int64)

    train_idx, val_idx, test_idx = stratified_split_indices(
        labels, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    def build_split(idxs: np.ndarray):
        return {
            "paths": paths[idxs].tolist(),
            "labels": torch.from_numpy(labels[idxs]),
        }

    out_obj = {
        "train": build_split(train_idx),
        "val": build_split(val_idx),
        "test": build_split(test_idx),
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "metadata": {
            "manifest_csv": str(manifest_path),
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "dropped_rows": int(dropped),
            "num_rows": int(len(rows)),
        },
    }

    out_path = Path(args.out).expanduser()
    torch.save(out_obj, out_path)

    print(f"Saved: {out_path}")
    print(f"Labels: {label_to_idx}")
    print(
        f"Split sizes: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)} | dropped={dropped}"
    )


if __name__ == "__main__":
    main()
