#!/usr/bin/env python3
"""
Train a small classifier on frozen ResNet-18 features.
This is FEATURE MODE: the ResNet-18 backbone is not fine-tuned.
Outputs:
  results/accuracy_per_epoch.csv
  results/accuracy_per_epoch.png
  results/model.pt
"""
from pathlib import Path
import csv, time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "features" / "features.npz"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

EPOCHS = 30
LR = 1e-3
BATCH = 32
SEED = 42

def main():
    torch.manual_seed(SEED)
    z = np.load(FEAT)
    X, y = z["X"], z["y"]

    # Features are ordered train then val by extraction script.
    # Reconstruct split sizes from dataset.
    train_n = sum(1 for _ in (ROOT/"dataset"/"train"/"botol").glob("*")) + \
              sum(1 for _ in (ROOT/"dataset"/"train"/"box").glob("*"))
    Xtr, ytr = X[:train_n], y[:train_n]
    Xva, yva = X[train_n:], y[train_n:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(X.shape[1], 2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                              batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(yva)),
                            batch_size=BATCH, shuffle=False)

    history = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        correct = total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
        train_acc = correct / total

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                pred = logits.argmax(1).cpu()
                correct += (pred == yb).sum().item()
                total += yb.numel()
        val_acc = correct / total
        history.append((epoch, train_acc, val_acc))
        print(f"Epoch {epoch:02d}/{EPOCHS} - train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    with open(RES/"accuracy_per_epoch.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["epoch","train_accuracy","val_accuracy"]); w.writerows(history)

    plt.figure(figsize=(8,5))
    plt.plot([h[0] for h in history], [h[1] for h in history], label="Train accuracy")
    plt.plot([h[0] for h in history], [h[2] for h in history], label="Validation accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.ylim(0,1.05)
    plt.title("ResNet-18 Feature Mode - Accuracy per Epoch")
    plt.grid(True, alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(RES/"accuracy_per_epoch.png", dpi=160)
    plt.close()

    torch.save({"classifier": model.state_dict(), "classes": ["botol","box"]},
               RES/"classifier.pt")
    print("Hasil tersimpan di folder results/")

if __name__ == "__main__":
    main()
