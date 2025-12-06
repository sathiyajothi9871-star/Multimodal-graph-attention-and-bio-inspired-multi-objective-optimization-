import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDense3D(nn.Module):
    """A lightweight 3D CNN with dense-style skip connections.

    This is not a full DenseNet-121 3D replication, but a compact architecture
    that preserves the key idea of aggregated skip connections and can be
    trained on limited GPU memory.
    """

    def __init__(self, in_channels=1, out_dim=256):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)

        self.conv3 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)

        self.conv4 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)

        self.pool = nn.MaxPool3d(2)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        # x: (B, 1, D, H, W)
        x1 = F.relu(self.bn1(self.conv1(x)))   # (B,32,...)
        x2 = F.relu(self.bn2(self.conv2(x1)))  # (B,32,...)
        x_cat1 = torch.cat([x1, x2], dim=1)    # (B,64,...)
        x_cat1 = self.pool(x_cat1)

        x3 = F.relu(self.bn3(self.conv3(x_cat1)))  # (B,64,...)
        x_cat2 = torch.cat([x_cat1, x3], dim=1)    # (B,128,...)
        x_cat2 = self.pool(x_cat2)

        x4 = F.relu(self.bn4(self.conv4(x_cat2)))  # (B,128,...)
        x4 = self.global_pool(x4)                  # (B,128,1,1,1)
        x4 = x4.view(x4.size(0), -1)
        emb = self.fc(x4)
        return emb
