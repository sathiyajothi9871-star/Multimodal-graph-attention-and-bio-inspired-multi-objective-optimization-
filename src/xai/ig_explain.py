"""Integrated Gradients for 3D CT imaging branch.

Run:
    python -m src.xai.ig_explain
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

from src import config
from src.data.dataset import NSCLCRadiomicsDataset
from src.models.full_model import MultimodalNSCLCModel


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model = MultimodalNSCLCModel().to(device)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ds = NSCLCRadiomicsDataset("test")
    sample = ds[0]  # take one patient for visualization
    ct = sample["ct"].unsqueeze(0).to(device)  # (1,1,D,H,W)
    clinical = sample["clinical_features"].unsqueeze(0).to(device)

    def forward_imaging_only(ct_input):
        return model(ct_input, clinical)

    ig = IntegratedGradients(forward_imaging_only)
    attributions, _ = ig.attribute(ct, target=1, return_convergence_delta=True)
    attrs = attributions.squeeze().detach().cpu().numpy()  # (D,H,W)
    ct_np = ct.squeeze().detach().cpu().numpy()

    # Overlay on a middle slice
    mid_z = attrs.shape[0] // 2
    slice_attr = attrs[mid_z]
    slice_ct = ct_np[mid_z]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(slice_ct, cmap="gray")
    ax[0].set_title("CT slice")
    ax[0].axis("off")

    ax[1].imshow(slice_ct, cmap="gray")
    ax[1].imshow(slice_attr, alpha=0.5)
    ax[1].set_title("Integrated Gradients overlay")
    ax[1].axis("off")

    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(config.FIGURES_DIR, "ig_ct_overlay.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved IG overlay to {out_path}")


if __name__ == "__main__"]:
    main()
