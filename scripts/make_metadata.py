#!/usr/bin/env python3
"""Generate metadata.csv for dataset/."""
from pathlib import Path
import csv, hashlib
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"dataset"
OUT = ROOT/"metadata.csv"

rows=[]
for split in ["train","val"]:
    for cls_dir in sorted((DATA/split).iterdir()):
        if not cls_dir.is_dir(): continue
        for p in sorted(cls_dir.glob("*")):
            if p.suffix.lower() not in {".jpg",".jpeg",".png",".bmp",".webp"}: continue
            with Image.open(p) as im:
                b=p.read_bytes()
                rows.append({
                    "split":split, "class":cls_dir.name, "filename":p.name,
                    "relative_path":str(p.relative_to(ROOT)),
                    "width":im.width, "height":im.height, "mode":im.mode,
                    "augmentation":"original" if "_aug" not in p.stem else
                                  ("flip_contrast" if "_augflip" in p.stem else "brightness_sharpness"),
                    "md5":hashlib.md5(b).hexdigest()
                })
with open(OUT,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"metadata.csv dibuat: {len(rows)} baris")
