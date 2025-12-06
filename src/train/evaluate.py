"""Evaluate a trained model on the held-out test set.

Assumes a checkpoint `best_model.pt` exists under CHECKPOINT_DIR,
or you can load the latest one.
"""
import os
import torch
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import NSCLCRadiomicsDataset
from src.models.full_model import MultimodalNSCLCModel
from src.utils.metrics import classification_metrics
from src.utils.logging_utils import save_metrics


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # You may change this path to your actual checkpoint
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = MultimodalNSCLCModel().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_ds = NSCLCRadiomicsDataset("test")
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ct = batch["ct"].to(device)
            clinical = batch["clinical_features"].to(device)
            labels = batch["label"].to(device)
            logits = model(ct, clinical)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = (probs >= 0.5).long()

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    metrics = classification_metrics(y_true, y_pred, y_prob)
    metrics_path = save_metrics(metrics, name="test_metrics_gat_mogwo")
    print("Test metrics:", metrics)
    print(f"Saved to: {metrics_path}")


if __name__ == "__main__"]:
    main()
