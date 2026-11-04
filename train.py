"""Train the U-Net.

Supported dataset layouts (pass the folder with --data):
  1) Kaggle "LGG MRI Segmentation" (mateuszbuda/lgg-mri-segmentation):
       kaggle_3m/<patient>/<patient>_1.tif  +  <patient>_1_mask.tif
  2) Generic:  <data>/images/<name>.png|jpg|tif  +  <data>/masks/<name>.png|jpg|tif
       (works for Data Science Bowl 2018 nuclei after merging each image's masks into one)

Example:
  python train.py --data /path/to/kaggle_3m --epochs 30 --out models/unet_best.pt
"""
import argparse, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import UNet

EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def find_pairs(root: Path):
    """Return list of (image_path, mask_path, group). Group is used to split without leakage."""
    pairs = []
    if (root / "images").is_dir() and (root / "masks").is_dir():
        masks = {p.stem: p for p in (root / "masks").iterdir() if p.suffix.lower() in EXTS}
        for p in sorted((root / "images").iterdir()):
            if p.suffix.lower() in EXTS and p.stem in masks:
                pairs.append((p, masks[p.stem], p.stem))
    else:
        for m in sorted(root.rglob("*_mask.*")):
            if m.suffix.lower() not in EXTS:
                continue
            img = m.with_name(m.name.replace("_mask", ""))
            if img.exists():
                pairs.append((img, m, m.parent.name))  # group = patient folder
    return pairs


class SegDataset(Dataset):
    def __init__(self, pairs, size, augment):
        self.pairs, self.size, self.augment = pairs, size, augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp, _ = self.pairs[i]
        img = Image.open(ip).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        msk = Image.open(mp).convert("L").resize((self.size, self.size), Image.NEAREST)
        x = np.asarray(img, dtype=np.float32) / 255.0
        y = (np.asarray(msk) > 0).astype(np.float32)
        if self.augment:
            if random.random() < 0.5: x, y = x[:, ::-1], y[:, ::-1]
            if random.random() < 0.5: x, y = x[::-1], y[::-1]
            k = random.randint(0, 3)
            x, y = np.rot90(x, k), np.rot90(y, k)
            x = np.clip(x * random.uniform(0.9, 1.1), 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(y))[None]
        return x, y


def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps)).mean()


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    d_sum = i_sum = n = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        p = (torch.sigmoid(model(x)) > 0.5).float()
        inter = (p * y).sum((1, 2, 3))
        union = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
        d_sum += ((2 * inter + 1) / (union + 1)).sum().item()
        i_sum += ((inter + 1) / (union - inter + 1)).sum().item()
        n += x.size(0)
    return d_sum / n, i_sum / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/unet_best.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = find_pairs(Path(a.data))
    if not pairs:
        raise SystemExit("No image/mask pairs found. Check --data and the folder layout in this file's docstring.")

    groups = sorted({g for _, _, g in pairs})
    random.shuffle(groups)
    n_val = max(1, int(0.15 * len(groups)))
    val_g = set(groups[:n_val])
    train_p = [p for p in pairs if p[2] not in val_g]
    val_p = [p for p in pairs if p[2] in val_g]
    print(f"{len(train_p)} train / {len(val_p)} val slices on {dev}")

    tl = DataLoader(SegDataset(train_p, a.img_size, True), a.batch, shuffle=True, num_workers=a.workers, pin_memory=dev == "cuda")
    vl = DataLoader(SegDataset(val_p, a.img_size, False), a.batch, num_workers=a.workers)

    model = UNet(3, 1, a.base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    best = -1.0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); total = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev, enabled=dev == "cuda"):
                logits = model(x)
            logits = logits.float()
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += loss.item() * x.size(0)
        vd, vi = evaluate(model, vl, dev)
        sched.step(vd)
        print(f"epoch {ep:02d}  loss {total/len(train_p):.4f}  val dice {vd:.4f}  val IoU {vi:.4f}  ({time.time()-t0:.0f}s)")
        if vd > best:
            best = vd
            torch.save({"state_dict": model.state_dict(), "img_size": a.img_size, "in_ch": 3,
                        "base": a.base, "val_dice": vd, "val_iou": vi}, a.out)
            print(f"  saved {a.out}")
    print(f"best val dice: {best:.4f}")


if __name__ == "__main__":
    main()
