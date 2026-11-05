import os
from pathlib import Path

import cv2
import numpy as np


class Segmenter:
    """Loads the trained U-Net if the checkpoint exists. Otherwise runs in DEMO mode
    (a crude brightness threshold) so the app can be tested end to end."""

    def __init__(self, ckpt_path):
        self.path = Path(ckpt_path)
        self.model = None
        self.demo = True
        self.info = {}
        if os.getenv("MODEL_SOURCE", "").lower() == "hub":
            try:
                self._load_hub()
            except Exception as e:
                print(f"[segmenter] could not load the pretrained hub model: {e}\n[segmenter] falling back to DEMO mode")
        elif self.path.exists():
            try:
                self._load()
            except Exception as e:  # missing torch, corrupt file, etc.
                print(f"[segmenter] could not load {self.path}: {e}\n[segmenter] falling back to DEMO mode")

    def _load(self):
        import torch
        from model import UNet
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(self.path, map_location=self.device)
        self.size = int(ck.get("img_size", 256))
        m = UNet(ck.get("in_ch", 3), 1, ck.get("base", 32))
        m.load_state_dict(ck["state_dict"])
        self.model = m.to(self.device).eval()
        self.info = {"val_dice": ck.get("val_dice"), "val_iou": ck.get("val_iou")}
        self.demo = False
        print(f"[segmenter] loaded {self.path} on {self.device} (val dice {self.info['val_dice']})")

    def _load_hub(self):
        # Public U-Net pretrained on the Kaggle LGG brain MRI dataset (downloads once, needs internet).
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        m = torch.hub.load("mateuszbuda/brain-segmentation-pytorch", "unet",
                           in_channels=3, out_channels=1, init_features=32, pretrained=True, trust_repo=True)
        self.model = m.to(self.device).eval()
        self.size = 256
        self.hub = True
        self.info = {"val_dice": None, "val_iou": None, "source": "torch.hub mateuszbuda/brain-segmentation-pytorch"}
        self.demo = False
        print("[segmenter] loaded pretrained hub U-Net")

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """rgb uint8 HxWx3 -> float probability map HxW in [0,1] at the original size."""
        return self._demo(rgb) if self.demo else self._infer(rgb)

    def _infer(self, rgb):
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if getattr(self, "hub", False):  # that model expects per-image standardised input
            x = (x - x.mean()) / (x.std() + 1e-8)
        t = self.torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)
        with self.torch.no_grad():
            out = self.model(t)
        p = (out if getattr(self, "hub", False) else self.torch.sigmoid(out))[0, 0].cpu().numpy()
        return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _demo(rgb):
        g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
        t1, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = g[g > t1]
        if bright.size < 50:
            return np.zeros(g.shape, np.float32)
        t2, _ = cv2.threshold(bright.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = (g > max(t2, t1 + 1)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            return np.zeros(g.shape, np.float32)
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (lab == keep).astype(np.float32)
