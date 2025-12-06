"""SHAP-based explainability for clinical + radiomic features.

Run:
    python -m src.xai.shap_explain
"""
import os
import numpy as np
import pandas as pd
import shap
import torch

from src import config
from src.data.dataset import NSCLCRadiomicsDataset
from src.models.full_model import MultimodalNSCLCModel


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model = MultimodalNSCLCModel().to(device)
    # Load your best checkpoint
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Here we approximate tabular contributions using only clinical features and
    # a simple surrogate model (last fusion layer). A more accurate treatment
    # would expose a dedicated clinical+graph branch.
    ds = NSCLCRadiomicsDataset("test")
    X = []
    y = []
    for i in range(len(ds)):
        sample = ds[i]
        X.append(sample["clinical_features"].numpy())
        y.append(int(sample["label"])))

    X = np.stack(X, axis=0)
    y = np.array(y)

    background = X[np.random.choice(X.shape[0], size=min(50, X.shape[0]), replace=False)]

    def model_fn(x_np: np.ndarray):
        x_t = torch.from_numpy(x_np).float().to(device)
        # Dummy CT input: zeros; here we focus on clinical branch contribution
        ct_dummy = torch.zeros(x_t.size(0), 1, config.CT_DEPTH, config.CT_HEIGHT, config.CT_WIDTH, device=device)
        with torch.no_grad():
            logits = model(ct_dummy, x_t)
            probs = torch.softmax(logits, dim=-1)[:, 1]
        return probs.cpu().numpy()

    explainer = shap.KernelExplainer(model_fn, background)
    shap_values = explainer.shap_values(X, nsamples=100)

    clinical_df = pd.read_csv(config.PROCESSED_CLINICAL_CSV)
    feature_names = [c for c in clinical_df.columns if c not in ["patient_id", "label"]]

    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    import matplotlib.pyplot as plt
    fig_path = os.path.join(config.FIGURES_DIR, "shap_summary_clinical.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved SHAP summary plot to {fig_path}")


if __name__ == "__main__"]:
    main()
