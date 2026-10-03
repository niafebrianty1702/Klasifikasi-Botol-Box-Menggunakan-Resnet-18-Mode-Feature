#!/usr/bin/env python3
"""
Benchmark ResNet-18 FEATURE MODE latency.
Reports warm-up and repeated inference latency.
"""
from pathlib import Path
import argparse, csv, time
import numpy as np
import torch
from PIL import Image
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.ResNet18_Weights.DEFAULT
    tfm = weights.transforms()
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    image_path = Path(args.image) if args.image else next((ROOT/"dataset"/"val"/"botol").glob("*"))
    x = tfm(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        for _ in range(10):
            _ = model(x)
        sync(device)

        times = []
        for _ in range(args.runs):
            sync(device)
            t0 = time.perf_counter()
            _ = model(x)
            sync(device)
            times.append((time.perf_counter()-t0)*1000)

    arr = np.asarray(times)
    result = {
        "device": str(device),
        "image": str(image_path),
        "runs": args.runs,
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr,95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "fps_equivalent": float(1000.0/arr.mean())
    }

    with open(RES/"latency_resnet18_feature.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(result.keys()); w.writerow(result.values())
    print(result)

if __name__ == "__main__":
    main()
