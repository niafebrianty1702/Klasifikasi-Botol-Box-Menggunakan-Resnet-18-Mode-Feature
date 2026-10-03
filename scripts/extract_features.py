#!/usr/bin/env python3
"""
ResNet-18 FEATURE MODE
- Uses ImageNet-pretrained ResNet-18.
- Removes the final classification layer.
- Extracts a 512-dimensional feature vector per image.
- Saves features/features.csv and features/features.npz.
"""
from pathlib import Path
import argparse, time, csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "dataset"
OUT = ROOT / "features"
OUT.mkdir(exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    rows = []
    all_features = []
    all_labels = []

    for split in ["train", "val"]:
        ds = datasets.ImageFolder(Path(args.data) / split, transform=transform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type=="cuda"))
        for images, labels in loader:
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            with torch.inference_mode():
                feats = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            all_features.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())

            # ImageFolder order is deterministic.
        # Map file paths after feature extraction in exactly the same order.
        for path, label in zip(ds.samples, ds.targets):
            rows.append([split, path, ds.classes[label]])

    X = np.concatenate(all_features, axis=0).astype(np.float32)
    y = np.asarray(all_labels, dtype=np.int64)

    # Save feature matrix and labels.
    np.savez_compressed(OUT / "features.npz", X=X, y=y)
    with open(OUT / "features.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split","path","class","label"] + [f"f{i}" for i in range(X.shape[1])])
        for i, r in enumerate(rows):
            w.writerow(r + [int(y[i])] + X[i].tolist())

    print(f"Device: {device}")
    print(f"Feature shape: {X.shape}")
    print(f"Saved: {OUT/'features.npz'}")
    print(f"Saved: {OUT/'features.csv'}")

if __name__ == "__main__":
    main()
