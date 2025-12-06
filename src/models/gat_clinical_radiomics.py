import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class ClinicalRadiomicsGAT(nn.Module):
    """Graph Attention Network over clinical + radiomic features.

    Input: node_feats (N,F_in), edge_index (2,E)
    Output: patient-level embedding (graph embedding)
    """

    def __init__(self, in_dim=1, hidden_dim=128, heads=4, out_dim=128):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, node_feats, edge_index):
        x = self.gat1(node_feats, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        x = torch.relu(x)
        # Global mean pooling over nodes
        x = x.mean(dim=0, keepdim=True)  # (1, hidden_dim)
        emb = self.fc(x)  # (1, out_dim)
        return emb.squeeze(0)
