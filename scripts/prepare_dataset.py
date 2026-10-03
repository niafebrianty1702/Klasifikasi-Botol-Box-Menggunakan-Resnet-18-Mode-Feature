#!/usr/bin/env python3
"""
Prepare a reproducible dataset from the original ZIP/folder.

Expected source:
  dataset_raw/
    botol/
    box/

Output:
  dataset/
    train/<class>/
    val/<class>/

The provided project has already been prepared. This script is included
so the process can be reproduced if dataset_raw is replaced.
"""
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
import random, shutil

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset_raw"
OUT = ROOT / "dataset"
CLASSES = ["botol", "box"]

random.seed(42)
for split in ["train", "val"]:
    for cls in CLASSES:
        (OUT / split / cls).mkdir(parents=True, exist_ok=True)

for cls in CLASSES:
    imgs = sorted([p for p in (RAW / cls).glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}])
    random.shuffle(imgs)
    train, val = imgs[:40], imgs[40:]

    for p in val:
        shutil.copy2(p, OUT / "val" / cls / p.name)

    for p in train:
        shutil.copy2(p, OUT / "train" / cls / p.name)
        with Image.open(p).convert("RGB") as im:
            a1 = ImageEnhance.Contrast(ImageOps.mirror(im)).enhance(1.12)
            a1.save(OUT / "train" / cls / f"{p.stem}_augflip.jpg", quality=95)

            a2 = ImageEnhance.Sharpness(ImageEnhance.Brightness(im).enhance(1.10)).enhance(1.25)
            a2.save(OUT / "train" / cls / f"{p.stem}_augbright.jpg", quality=95)

print("Dataset preparation selesai.")
