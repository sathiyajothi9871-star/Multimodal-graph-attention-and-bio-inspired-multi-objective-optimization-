"""Ablation experiments for the NSCLC multimodal model.

Run:
    python -m src.train.ablation
"""
from dataclasses import dataclass
from typing import Dict

from src import config
from src.train.train_gat_mogwo import train_with_config


@dataclass
class AblationConfig:
    name: str
    use_imaging: bool = True
    use_graph: bool = True
    use_mo_gwo: bool = True


def main():
    base_hparams = {
        "batch_size": config.BATCH_SIZE,
        "lr": config.LEARNING_RATE,
        "weight_decay": config.WEIGHT_DECAY,
        "img_emb_dim": config.IMG_EMB_DIM,
        "graph_emb_dim": config.GRAPH_EMB_DIM,
        "fusion_emb_dim": config.FUSION_EMB_DIM,
        "num_epochs": config.NUM_EPOCHS,
    }

    ablations = [
        AblationConfig(name="full_model"),
        AblationConfig(name="imaging_only", use_graph=False),
        AblationConfig(name="clinical_only", use_imaging=False),
        AblationConfig(name="no_gat", use_graph=False),  # etc. can be refined
    ]

    results = {}
    for ab in ablations:
        print(f"\n==== Running ablation: {ab.name} ====")
        hparams = base_hparams.copy()
        # In a real implementation, you would toggle flags inside the model;
        # here we simply note the experiment name and let the user add branches.
        val_metrics = train_with_config(hparams)
        results[ab.name] = val_metrics

    print("\nAblation results:")
    for name, metrics in results.items():
        print(name, "->", metrics)


if __name__ == "__main__"]:
    main()
