"""Runs during the Render build so the pretrained model is baked into the deploy
(Render's free disk is wiped on restart, so downloading at runtime would repeat on every wake-up)."""
import os

if os.getenv("MODEL_SOURCE", "").lower() == "hub":
    import torch
    print("Downloading pretrained U-Net to", torch.hub.get_dir())
    torch.hub.load("mateuszbuda/brain-segmentation-pytorch", "unet", in_channels=3, out_channels=1,
                   init_features=32, pretrained=True, trust_repo=True)
    print("done")
else:
    print("MODEL_SOURCE is not 'hub'; skipping download (using models/unet_best.pt if present).")
