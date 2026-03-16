"""
Metric learning baselines for DeepFashion-style datasets.

Expected input format (torch .pt):
1) Split dict:
   {
     "train": {"embeddings": Tensor[N,D], "item_ids": list, "keywords": list or "text": list},
     "val":   {...},
     "test":  {...}
   }
2) Flat dict with "split":
   {
     "embeddings": Tensor[N,D], "item_ids": list, "keywords" or "text": list, "split": list[str]
   }

Keywords can be:
- list[list[str]]: pre-tokenized keywords
- list[str]: raw text, tokenized by this script
"""

import argparse
import json
import os
import random
import re
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def _normalize_keywords(raw_keywords: List[Any]) -> List[List[str]]:
    norm = []
    for k in raw_keywords:
        if isinstance(k, list):
            tokens = []
            for t in k:
                tokens.extend(_tokenize(str(t)))
            norm.append(tokens)
        else:
            norm.append(_tokenize(str(k)))
    return norm


def _load_pt(path: str) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict) and "train" in data and "val" in data and "test" in data:
        return data

    if not isinstance(data, dict) or "embeddings" not in data:
        raise ValueError("Unsupported .pt format. Expected dict with embeddings or train/val/test splits.")

    if "split" not in data:
        raise ValueError("Flat .pt must include a 'split' list with values train/val/test.")

    split = data["split"]
    splits = {"train": [], "val": [], "test": []}
    for i, s in enumerate(split):
        if s not in splits:
            continue
        splits[s].append(i)

    def _subset(idxs: List[int]) -> Dict[str, Any]:
        emb = data["embeddings"][idxs]
        item_ids = [data["item_ids"][i] for i in idxs]
        if "keywords" in data:
            keywords = [data["keywords"][i] for i in idxs]
        elif "text" in data:
            keywords = [data["text"][i] for i in idxs]
        else:
            raise ValueError("Need 'keywords' or 'text' in data.")
        return {"embeddings": emb, "item_ids": item_ids, "keywords": keywords}

    return {k: _subset(v) for k, v in splits.items()}


def load_dataset(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.endswith(".pt"):
        data = _load_pt(path)
    else:
        raise ValueError("Only .pt dataset files are supported for now.")

    for split in ("train", "val", "test"):
        if split not in data:
            raise ValueError(f"Missing split: {split}")
        if "embeddings" not in data[split] or "item_ids" not in data[split]:
            raise ValueError(f"Split {split} missing embeddings or item_ids.")
        if "keywords" not in data[split] and "text" not in data[split]:
            raise ValueError(f"Split {split} missing keywords or text.")
    return data


def prepare_split(split: Dict[str, Any]) -> Dict[str, Any]:
    embeddings = split["embeddings"]
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    embeddings = embeddings.float()
    item_ids = split["item_ids"]

    if "keywords" in split:
        raw_keywords = split["keywords"]
    else:
        raw_keywords = split["text"]
    keywords = _normalize_keywords(raw_keywords)
    keyword_sets = [set(k) for k in keywords]

    return {"embeddings": embeddings, "item_ids": item_ids, "keyword_sets": keyword_sets}


class TripletDataset(Dataset):
    def __init__(
        self,
        embeddings: torch.Tensor,
        item_ids: List[Any],
        keyword_sets: List[set],
        min_keyword_overlap: int = 1,
        max_tries: int = 50,
    ):
        self.embeddings = embeddings
        self.item_ids = item_ids
        self.keyword_sets = keyword_sets
        self.min_overlap = min_keyword_overlap
        self.max_tries = max_tries
        self.num_samples = embeddings.shape[0]

        self.item_to_indices: Dict[Any, List[int]] = {}
        for idx, item in enumerate(item_ids):
            self.item_to_indices.setdefault(item, []).append(idx)

        self.valid_indices = [i for i in range(self.num_samples) if self._has_positive(i)]
        if not self.valid_indices:
            raise ValueError("No valid anchors found (no positives by item_id or keywords).")

    def _overlap(self, i: int, j: int) -> int:
        return len(self.keyword_sets[i].intersection(self.keyword_sets[j]))

    def _has_positive(self, idx: int) -> bool:
        item = self.item_ids[idx]
        if len(self.item_to_indices[item]) > 1:
            return True
        for j in range(self.num_samples):
            if j == idx:
                continue
            if self._overlap(idx, j) >= self.min_overlap:
                return True
        return False

    def _sample_positive(self, idx: int) -> int:
        item = self.item_ids[idx]
        candidates = self.item_to_indices[item]
        if len(candidates) > 1:
            pos = idx
            while pos == idx:
                pos = random.choice(candidates)
            return pos

        for _ in range(self.max_tries):
            j = random.randrange(self.num_samples)
            if j != idx and self._overlap(idx, j) >= self.min_overlap:
                return j
        # Fallback: deterministic scan (should be rare)
        for j in range(self.num_samples):
            if j != idx and self._overlap(idx, j) >= self.min_overlap:
                return j
        raise RuntimeError("Unable to sample positive.")

    def _sample_negative(self, idx: int) -> int:
        for _ in range(self.max_tries):
            j = random.randrange(self.num_samples)
            if self.item_ids[j] != self.item_ids[idx] and self._overlap(idx, j) < self.min_overlap:
                return j
        # Fallback: any different item_id
        for j in range(self.num_samples):
            if self.item_ids[j] != self.item_ids[idx]:
                return j
        raise RuntimeError("Unable to sample negative.")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_idx = self.valid_indices[idx]
        pos_idx = self._sample_positive(anchor_idx)
        neg_idx = self._sample_negative(anchor_idx)
        return (
            self.embeddings[anchor_idx],
            self.embeddings[pos_idx],
            self.embeddings[neg_idx],
        )


def dataset_diagnostics(data: Dict[str, Any], min_overlap: int) -> Dict[str, float]:
    item_ids = data["item_ids"]
    keyword_sets = data["keyword_sets"]
    n = len(item_ids)

    item_counts = {}
    for item in item_ids:
        item_counts[item] = item_counts.get(item, 0) + 1
    multi_view_items = sum(1 for c in item_counts.values() if c > 1)

    total_pairs = 0
    pos_item_pairs = 0
    pos_keyword_pairs = 0
    pos_union_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            same_item = item_ids[i] == item_ids[j]
            overlap = len(keyword_sets[i].intersection(keyword_sets[j])) >= min_overlap
            if same_item:
                pos_item_pairs += 1
            if overlap:
                pos_keyword_pairs += 1
            if same_item or overlap:
                pos_union_pairs += 1

    return {
        "samples": n,
        "unique_items": len(item_counts),
        "multi_view_items": multi_view_items,
        "pos_item_pair_rate": pos_item_pairs / total_pairs if total_pairs else 0.0,
        "pos_keyword_pair_rate": pos_keyword_pairs / total_pairs if total_pairs else 0.0,
        "pos_union_pair_rate": pos_union_pairs / total_pairs if total_pairs else 0.0,
    }


class ShallowProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=1)


def triplet_loss_fn(anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, margin: float) -> torch.Tensor:
    # Cosine distance = 1 - cosine similarity
    cos = nn.CosineSimilarity(dim=1)
    d_ap = 1 - cos(anchor, pos)
    d_an = 1 - cos(anchor, neg)
    loss = torch.relu(d_ap - d_an + margin).mean()
    return loss


def train_projection(
    model: nn.Module,
    train_loader: DataLoader,
    val_data: Dict[str, Any],
    device: torch.device,
    epochs: int,
    lr: float,
    margin: float,
) -> Dict[str, Any]:
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_r1 = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for anchor, pos, neg in train_loader:
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            optimizer.zero_grad()
            a = l2_normalize(model(anchor))
            p = l2_normalize(model(pos))
            n = l2_normalize(model(neg))
            loss = triplet_loss_fn(a, p, n, margin)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate_retrieval(
            model, val_data, device, mode="item_or_keyword", min_overlap=1, k_list=(1, 5, 10)
        )
        val_r1 = val_metrics["recall@1"]

        if val_r1 > best_val_r1:
            best_val_r1 = val_r1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {total_loss / len(train_loader):.4f} | Val R@1: {val_r1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_recall@1": best_val_r1}


def _positive_mask(
    item_ids: List[Any],
    keyword_sets: List[set],
    i: int,
    min_overlap: int,
    mode: str,
) -> List[bool]:
    mask = []
    for j in range(len(item_ids)):
        if i == j:
            mask.append(False)
            continue
        same_item = item_ids[i] == item_ids[j]
        overlap = len(keyword_sets[i].intersection(keyword_sets[j])) >= min_overlap
        if mode == "item_id":
            mask.append(same_item)
        else:
            mask.append(same_item or overlap)
    return mask


def evaluate_retrieval(
    model: nn.Module,
    data: Dict[str, Any],
    device: torch.device,
    mode: str,
    min_overlap: int,
    k_list: Tuple[int, ...],
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        embeddings = data["embeddings"].to(device)
        projected = l2_normalize(model(embeddings)).cpu()

    item_ids = data["item_ids"]
    keyword_sets = data["keyword_sets"]
    n = projected.shape[0]

    recalls = {k: 0 for k in k_list}
    ap_sum = 0.0
    valid_queries = 0

    for i in range(n):
        mask = _positive_mask(item_ids, keyword_sets, i, min_overlap, mode)
        if not any(mask):
            continue
        valid_queries += 1
        sims = torch.mv(projected, projected[i])
        sims[i] = -1e9
        ranked = torch.argsort(sims, descending=True).tolist()

        hits = [mask[j] for j in ranked]
        for k in k_list:
            if any(hits[:k]):
                recalls[k] += 1

        # Average precision
        num_pos = 0
        precision_sum = 0.0
        for rank, is_pos in enumerate(hits, start=1):
            if is_pos:
                num_pos += 1
                precision_sum += num_pos / rank
        ap = precision_sum / num_pos if num_pos > 0 else 0.0
        ap_sum += ap

    if valid_queries == 0:
        raise ValueError("No valid queries for retrieval evaluation.")

    metrics = {f"recall@{k}": recalls[k] / valid_queries for k in k_list}
    metrics["mAP"] = ap_sum / valid_queries
    return metrics


class IdentityModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def main() -> None:
    parser = argparse.ArgumentParser(description="Metric learning baselines for DeepFashion.")
    parser.add_argument("--data", required=True, help="Path to DeepFashion .pt dataset")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--min-keyword-overlap", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", default="metric_baselines_log.json")
    parser.add_argument("--out-csv", default="metric_baselines_results.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("METRIC LEARNING BASELINES (DEEPFASHION)")
    print("=" * 80)
    print(f"Data: {args.data}")
    print(f"Device: {device}")

    data = load_dataset(args.data)
    train = prepare_split(data["train"])
    val = prepare_split(data["val"])
    test = prepare_split(data["test"])

    input_dim = train["embeddings"].shape[1]
    print(f"Embeddings: {input_dim} dims")
    print(f"Train samples: {train['embeddings'].shape[0]}")
    print(f"Val samples: {val['embeddings'].shape[0]}")
    print(f"Test samples: {test['embeddings'].shape[0]}")

    print("\nDataset diagnostics (pair rates):")
    for name, split in (("train", train), ("val", val), ("test", test)):
        stats = dataset_diagnostics(split, args.min_keyword_overlap)
        print(
            f"  {name}: samples={stats['samples']} unique_items={stats['unique_items']} "
            f"multi_view_items={stats['multi_view_items']} "
            f"pos_item_pair_rate={stats['pos_item_pair_rate']:.6f} "
            f"pos_keyword_pair_rate={stats['pos_keyword_pair_rate']:.6f} "
            f"pos_union_pair_rate={stats['pos_union_pair_rate']:.6f}"
        )

    train_dataset = TripletDataset(
        train["embeddings"],
        train["item_ids"],
        train["keyword_sets"],
        min_keyword_overlap=args.min_keyword_overlap,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    results = []

    # Baseline 1: Raw embeddings (no training)
    identity = IdentityModel().to(device)
    print("\nBaseline: Raw embeddings")
    metrics_raw = evaluate_retrieval(
        identity, test, device, mode="item_or_keyword", min_overlap=args.min_keyword_overlap, k_list=(1, 5, 10)
    )
    results.append({"model": "Raw Embeddings", **metrics_raw})

    # Baseline 2: Shallow projection + triplet loss
    print("\nBaseline: Shallow projection (triplet)")
    shallow = ShallowProjection(input_dim, args.output_dim).to(device)
    train_projection(shallow, train_loader, val, device, args.epochs, args.lr, args.margin)
    metrics_shallow = evaluate_retrieval(
        shallow, test, device, mode="item_or_keyword", min_overlap=args.min_keyword_overlap, k_list=(1, 5, 10)
    )
    results.append({"model": "Shallow Projection", **metrics_shallow})

    # Baseline 3: Deep projection + triplet loss
    print("\nBaseline: Deep projection (triplet)")
    deep = DeepProjection(input_dim, args.output_dim).to(device)
    train_projection(deep, train_loader, val, device, args.epochs, args.lr, args.margin)
    metrics_deep = evaluate_retrieval(
        deep, test, device, mode="item_or_keyword", min_overlap=args.min_keyword_overlap, k_list=(1, 5, 10)
    )
    results.append({"model": "Deep Projection", **metrics_deep})

    # Save results
    with open(args.out_json, "w") as f:
        json.dump(
            {
                "data": args.data,
                "input_dim": input_dim,
                "output_dim": args.output_dim,
                "min_keyword_overlap": args.min_keyword_overlap,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved log: {args.out_json}")

    import csv

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "recall@1", "recall@5", "recall@10", "mAP"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Saved results: {args.out_csv}")


if __name__ == "__main__":
    main()
