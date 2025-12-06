"""Graph construction utilities for clinical + radiomic features.

The paper models each patient as a small graph whose nodes correspond to
clinical and radiomic features, with edges encoding correlations or
domain knowledge. Here we provide a simple implementation:

- One graph *per patient*
- Nodes: all clinical + radiomic features
- Edges: fully connected graph (or thresholded correlation)
"""
import torch


def build_feature_graph(clinical_features, radiomic_features=None):
    """Build a feature graph for a single patient.

    Args:
        clinical_features: 1D tensor of size (C,)
        radiomic_features: 1D tensor of size (R,) or None

    Returns:
        node_feats: (N, F) tensor of node features
        edge_index: (2, E) tensor of edge indices
    """
    if radiomic_features is None:
        feats = clinical_features
    else:
        feats = torch.cat([clinical_features, radiomic_features], dim=0)

    # Node features: treat each scalar feature as a node with a single value
    node_feats = feats.view(-1, 1)  # (N,1)

    N = node_feats.shape[0]
    src = []
    dst = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return node_feats, edge_index
