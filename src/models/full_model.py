import torch
import torch.nn as nn

from src import config
from src.models.dense3d import SimpleDense3D
from src.models.gat_clinical_radiomics import ClinicalRadiomicsGAT
from src.data.graph_builder import build_feature_graph


class MultimodalNSCLCModel(nn.Module):
    """End-to-end multimodal model: 3D CNN + GAT fusion."""

    def __init__(
        self,
        img_emb_dim=config.IMG_EMB_DIM,
        graph_emb_dim=config.GRAPH_EMB_DIM,
        fusion_emb_dim=config.FUSION_EMB_DIM,
        num_classes=config.NUM_CLASSES,
    ):
        super().__init__()
        self.img_branch = SimpleDense3D(in_channels=1, out_dim=img_emb_dim)
        self.gat = ClinicalRadiomicsGAT(in_dim=1, hidden_dim=config.GAT_HIDDEN_DIM,
                                       heads=config.GAT_HEADS, out_dim=graph_emb_dim)

        self.fusion = nn.Sequential(
            nn.Linear(img_emb_dim + graph_emb_dim, fusion_emb_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_emb_dim, num_classes),
        )

    def forward(self, ct, clinical_features, radiomic_features=None):
        # Imaging branch
        img_emb = self.img_branch(ct)

        # Graph branch
        node_feats, edge_index = build_feature_graph(clinical_features, radiomic_features)
        graph_emb = self.gat(node_feats, edge_index)

        # Fusion
        fused = torch.cat([img_emb, graph_emb], dim=-1)
        logits = self.fusion(fused)
        return logits
