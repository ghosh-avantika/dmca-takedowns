"""
Encode negative_cases_originals.csv into CLIP text embeddings, add as 'original',
and resplit into train/val/test.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import clip


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_texts(texts, device, batch_size=16):
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(device)
            vecs = model.encode_text(tokens).float()
            vecs = vecs / vecs.norm(dim=1, keepdim=True)
            all_vecs.append(vecs.cpu())
    return torch.cat(all_vecs, dim=0)


def stratified_split(labels, split, seed):
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    indices = np.arange(len(labels))
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(labels):
        cls_idx = indices[labels == cls]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(n * split[0])
        n_val = int(n * split[1])
        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train:n_train + n_val])
        test_idx.extend(cls_idx[n_train + n_val:])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def main():
    parser = argparse.ArgumentParser(description="Add 'original' class to dataset_500_samples.pt")
    parser.add_argument("--dataset-pt", default="dataset_500_samples.pt")
    parser.add_argument("--original-csv", default="negative_cases_originals.csv")
    parser.add_argument("--out", default="dataset_500_with_original.pt")
    parser.add_argument("--split", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_path = Path(args.dataset_pt)
    csv_path = Path(args.original_csv)

    data = torch.load(dataset_path, map_location="cpu")
    # Merge existing splits into a single pool
    all_emb = torch.cat([data["train"]["embeddings"], data["val"]["embeddings"], data["test"]["embeddings"]], dim=0)
    all_lbl = torch.cat([data["train"]["labels"], data["val"]["labels"], data["test"]["labels"]], dim=0)

    label_to_idx = dict(data.get("label_to_idx", {"knockoff": 0, "similar": 1}))
    if "original" not in label_to_idx:
        label_to_idx["original"] = max(label_to_idx.values()) + 1
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Encode originals
    df = pd.read_csv(csv_path)
    texts = df["original_design_elements"].fillna("").astype(str).tolist()
    texts = [t for t in texts if t.strip() != ""]
    if not texts:
        raise ValueError("No valid original_design_elements found in CSV.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    orig_emb = encode_texts(texts, device=device, batch_size=args.batch_size)
    orig_lbl = torch.full((orig_emb.shape[0],), label_to_idx["original"], dtype=torch.long)

    # Combine
    combined_emb = torch.cat([all_emb, orig_emb], dim=0)
    combined_lbl = torch.cat([all_lbl, orig_lbl], dim=0)

    # Resplit
    train_idx, val_idx, test_idx = stratified_split(combined_lbl.numpy(), args.split, args.seed)

    def subset(idxs):
        return {"embeddings": combined_emb[idxs], "labels": combined_lbl[idxs]}

    out = {
        "train": subset(train_idx),
        "val": subset(val_idx),
        "test": subset(test_idx),
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "metadata": {
            "source_dataset": str(dataset_path),
            "original_csv": str(csv_path),
            "split": args.split,
            "seed": args.seed,
            "original_count": len(texts),
            "total_samples": int(combined_emb.shape[0]),
        },
    }

    torch.save(out, args.out)
    print(f"Saved: {args.out}")
    print(f"Labels: {label_to_idx}")
    print(f"Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")


if __name__ == "__main__":
    main()
