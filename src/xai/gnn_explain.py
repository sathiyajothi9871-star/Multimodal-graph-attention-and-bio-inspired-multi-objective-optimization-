"""GNNExplainer for the clinical+radiomic GAT graph.

Run:
    python -m src.xai.gnn_explain
"""
import os
import torch
from torch_geometric.nn import GNNExplainer

from src import config
from src.models.full_model import MultimodalNSCLCModel
from src.data.graph_builder import build_feature_graph
from src.data.dataset import NSCLCRadiomicsDataset


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model = MultimodalNSCLCModel().to(device)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ds = NSCLCRadiomicsDataset("test")
    sample = ds[0]
    clinical = sample["clinical_features"]
    node_feats, edge_index = build_feature_graph(clinical)

    # We want to explain the GAT part, so we wrap it
    gat = model.gat
    gat.eval()
    gat.to(device)

    node_feats = node_feats.to(device)
    edge_index = edge_index.to(device)

    explainer = GNNExplainer(gat, epochs=200)
    node_feat_mask, edge_mask = explainer.explain_graph(node_feats, edge_index)

    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    fig = explainer.visualize_subgraph(
        -1, edge_index, edge_mask, y=None,
        threshold=None
    )
    import matplotlib.pyplot as plt
    out_path = os.path.join(config.FIGURES_DIR, "gnn_explainer_graph.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved GNNExplainer graph to {out_path}")


if __name__ == "__main__"]:
    main()
