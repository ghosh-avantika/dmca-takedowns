"""
Fine-tune a ResNet backbone with triplet loss on DeepFashion-style images.

Positives: same parsed item_id from filename (id_00012345-01).
Negatives: different item_id.
"""

import argparse
import os
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_item_id(filename: str, mode: str) -> str:
    name = filename.lower()
    if mode == "base_id":
        match = re.search(r"id_\d+", name, flags=re.IGNORECASE)
        return match.group(0) if match else Path(filename).stem
    if mode == "parsed_id":
        match = re.search(r"id_\d+(?:-\d+)?", name, flags=re.IGNORECASE)
        return match.group(0) if match else Path(filename).stem
    if mode == "filename":
        return Path(filename).stem
    raise ValueError(f"Unknown item-id mode: {mode}")


def list_images(root: str) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in exts]


def split_by_item_ids(
    paths: List[Path],
    split: Tuple[float, float, float],
    seed: int,
    item_id_mode: str,
) -> Tuple[List[int], List[int], List[int]]:
    item_to_indices: Dict[str, List[int]] = {}
    for i, p in enumerate(paths):
        item_id = parse_item_id(p.name, item_id_mode)
        item_to_indices.setdefault(item_id, []).append(i)

    items = list(item_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(items)

    n_items = len(items)
    n_train = int(n_items * split[0])
    n_val = int(n_items * split[1])
    train_items = set(items[:n_train])
    val_items = set(items[n_train:n_train + n_val])
    test_items = set(items[n_train + n_val:])

    train_idx, val_idx, test_idx = [], [], []
    for item, idxs in item_to_indices.items():
        if item in train_items:
            train_idx.extend(idxs)
        elif item in val_items:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)
    return train_idx, val_idx, test_idx


class TripletImageDataset(Dataset):
    def __init__(self, paths: List[Path], transform, item_id_mode: str):
        self.paths = paths
        self.transform = transform
        self.item_ids = [parse_item_id(p.name, item_id_mode) for p in paths]
        self.item_to_indices: Dict[str, List[int]] = {}
        for i, item in enumerate(self.item_ids):
            self.item_to_indices.setdefault(item, []).append(i)
        self.valid_indices = [i for i in range(len(paths)) if len(self.item_to_indices[self.item_ids[i]]) > 1]
        if not self.valid_indices:
            raise ValueError("No items with multiple images to form positives.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        anchor_idx = self.valid_indices[idx]
        anchor_path = self.paths[anchor_idx]
        item_id = self.item_ids[anchor_idx]

        pos_idx = anchor_idx
        while pos_idx == anchor_idx:
            pos_idx = random.choice(self.item_to_indices[item_id])

        neg_item = item_id
        while neg_item == item_id:
            neg_item = random.choice(list(self.item_to_indices.keys()))
        neg_idx = random.choice(self.item_to_indices[neg_item])

        anchor = self.transform(Image.open(anchor_path).convert("RGB"))
        positive = self.transform(Image.open(self.paths[pos_idx]).convert("RGB"))
        negative = self.transform(Image.open(self.paths[neg_idx]).convert("RGB"))
        return anchor, positive, negative


def build_model(backbone: str, pretrained: bool, embed_dim: int) -> nn.Module:
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_dim = model.fc.in_features
        model.fc = nn.Linear(in_dim, embed_dim)
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        in_dim = model.fc.in_features
        model.fc = nn.Linear(in_dim, embed_dim)
    else:
        raise ValueError("Unsupported backbone")
    return model


def freeze_all_but_last_block(model: nn.Module, backbone: str) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if backbone.startswith("resnet"):
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=1)


def triplet_loss(anchor, pos, neg, margin: float) -> torch.Tensor:
    cos = nn.CosineSimilarity(dim=1)
    d_ap = 1 - cos(anchor, pos)
    d_an = 1 - cos(anchor, neg)
    return torch.relu(d_ap - d_an + margin).mean()


def main(cli_args=None):
    parser = argparse.ArgumentParser(description="Fine-tune ResNet with triplet loss.")
    parser.add_argument("--image-dir", help="Directory containing training images")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument("--out-model", default="finetuned_triplet_resnet.pt")
    parser.add_argument("--item-id-mode", choices=["parsed_id", "base_id", "filename"], default="base_id")
    args, unknown = parser.parse_known_args(cli_args)
    if unknown:
        print(f"Ignoring unknown args: {unknown}")
    if not args.image_dir:
        parser.error(
            "--image-dir is required. Kaggle example:\n"
            "main(['--image-dir', '/kaggle/input/<dataset-folder>'])"
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = list_images(args.image_dir)
    if not paths:
        raise ValueError("No images found.")

    train_idx, val_idx, test_idx = split_by_item_ids(paths, tuple(args.split), args.seed, args.item_id_mode)
    train_paths = [paths[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    test_paths = [paths[i] for i in test_idx]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Diagnostics: ensure positives exist in train split
    train_item_ids = [parse_item_id(p.name, args.item_id_mode) for p in train_paths]
    counts = {}
    for item in train_item_ids:
        counts[item] = counts.get(item, 0) + 1
    multi_view = sum(1 for c in counts.values() if c > 1)
    print(f"Train items: {len(counts)} | multi-view items: {multi_view}")
    if multi_view == 0:
        raise ValueError("No items with multiple images to form positives. Try --item-id-mode base_id.")

    train_ds = TripletImageDataset(train_paths, transform, args.item_id_mode)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = build_model(args.backbone, args.pretrained, args.embed_dim)
    freeze_all_but_last_block(model, args.backbone)
    model.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        num_batches = len(train_loader)
        for step, (anchor, pos, neg) in enumerate(train_loader, start=1):
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            optimizer.zero_grad()
            a = l2_normalize(model(anchor))
            p = l2_normalize(model(pos))
            n = l2_normalize(model(neg))
            loss = triplet_loss(a, p, n, args.margin)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / step
            print(
                f"\rEpoch {epoch:03d}/{args.epochs:03d} | "
                f"Batch {step:04d}/{num_batches:04d} | "
                f"Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}",
                end="",
                flush=True,
            )
        print()
        print(f"Epoch {epoch:03d} | Train loss: {total_loss / len(train_loader):.4f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone": args.backbone,
            "embed_dim": args.embed_dim,
            "item_id_mode": args.item_id_mode,
            "split": args.split,
            "image_dir": args.image_dir,
        },
        args.out_model,
    )
    print(f"Saved model: {args.out_model}")


if __name__ == "__main__":
    main()
