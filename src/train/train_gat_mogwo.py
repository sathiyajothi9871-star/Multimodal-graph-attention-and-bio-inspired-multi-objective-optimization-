"""Training script for the multimodal GAT + 3D CNN model.

Run:
    python -m src.train.train_gat_mogwo
"""
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from src import config
from src.data.dataset import NSCLCRadiomicsDataset
from src.models.full_model import MultimodalNSCLCModel
from src.optimization.mo_gwo import MOGreyWolfOptimizer
from src.utils.seed import set_global_seed
from src.utils.metrics import classification_metrics
from src.utils.logging_utils import save_metrics


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    for batch in loader:
        ct = batch["ct"].to(device)
        clinical = batch["clinical_features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(ct, clinical)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * ct.size(0)
    return total_loss / len(loader.dataset)


def evaluate_model(model, loader, device) -> Dict:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            ct = batch["ct"].to(device)
            clinical = batch["clinical_features"].to(device)
            labels = batch["label"].to(device)
            logits = model(ct, clinical)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = (probs >= 0.5).long()

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
    return classification_metrics(y_true, y_pred, y_prob)


def train_with_config(hparams: Dict) -> Dict:
    set_global_seed(config.SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    train_ds = NSCLCRadiomicsDataset("train")
    val_ds = NSCLCRadiomicsDataset("val")
    train_loader = DataLoader(train_ds, batch_size=hparams["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hparams["batch_size"], shuffle=False)

    model = MultimodalNSCLCModel(
        img_emb_dim=hparams["img_emb_dim"],
        graph_emb_dim=hparams["graph_emb_dim"],
        fusion_emb_dim=hparams["fusion_emb_dim"],
        num_classes=config.NUM_CLASSES,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])

    best_val_auc = 0.0
    best_state = None
    for epoch in range(hparams["num_epochs"]):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device)
        val_auc = val_metrics.get("auc", 0.0)
        print(f"Epoch {epoch+1}/{hparams['num_epochs']}  loss={loss:.4f}  val_auc={val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation metrics
    val_metrics = evaluate_model(model, val_loader, device)
    metrics_path = save_metrics(val_metrics, name="val_metrics_gat_mogwo")
    print(f"Validation metrics saved to {metrics_path}")
    return val_metrics


def main():
    if config.USE_MO_GWO:
        # Define a small discrete search space
        search_space = {
            "batch_size": [2, 4, 6],
            "lr": [1e-3, 5e-4, 1e-4],
            "weight_decay": [1e-4, 1e-5],
            "img_emb_dim": [128, 256],
            "graph_emb_dim": [64, 128],
            "fusion_emb_dim": [64, 128],
            "num_epochs": [50, 80],
        }

        def objective_fn(hparams):
            # We want to minimize (1 - AUC, calibration_error, complexity_penalty)
            val_metrics = train_with_config(hparams)
            auc = val_metrics.get("auc", 0.0)
            # Placeholder: you can compute calibration error explicitly
            calibration_error = 0.0
            complexity_penalty = (hparams["img_emb_dim"] + hparams["graph_emb_dim"]) / 512.0
            return (1.0 - auc, calibration_error, complexity_penalty)

        opt = MOGreyWolfOptimizer(
            search_space=search_space,
            objective_fn=objective_fn,
            num_wolves=config.MO_GWO_NUM_WOLVES,
            num_iters=config.MO_GWO_NUM_ITERS,
            seed=config.SEED,
        )
        pareto = opt.optimize()
        print("Pareto front solutions:")
        for hparams, objs in pareto:
            print(hparams, "->", objs)

    else:
        default_hparams = {
            "batch_size": config.BATCH_SIZE,
            "lr": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY,
            "img_emb_dim": config.IMG_EMB_DIM,
            "graph_emb_dim": config.GRAPH_EMB_DIM,
            "fusion_emb_dim": config.FUSION_EMB_DIM,
            "num_epochs": config.NUM_EPOCHS,
        }
        train_with_config(default_hparams)


if __name__ == "__main__":
    main()
