import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

from src import config


class NSCLCRadiomicsDataset(Dataset):
    """PyTorch Dataset for NSCLC-Radiomics CT + clinical features.

    Expects:
    - CT volumes preprocessed as .npy files under PROCESSED_CT_DIR,
      named as `{patient_id}_ct.npy`
    - Clinical CSV with a `patient_id` column and encoded feature columns
    - A splits JSON listing which patient_ids belong to this split
    """

    def __init__(self, split: str, transform=None):
        assert split in {"train", "val", "test"}
        self.split = split
        self.transform = transform

        # Load splits
        import json
        with open(config.SPLITS_JSON, "r", encoding="utf-8") as f:
            splits = json.load(f)
        self.patient_ids = splits[split]

        # Load clinical dataframe
        self.clinical_df = pd.read_csv(config.PROCESSED_CLINICAL_CSV)
        self.clinical_df = self.clinical_df.set_index("patient_id")

        # Identify feature and label columns
        self.label_col = "label"
        self.feature_cols = [c for c in self.clinical_df.columns if c != self.label_col]

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        # Load CT volume (1 x D x H x W)
        ct_path = os.path.join(config.PROCESSED_CT_DIR, f"{patient_id}_ct.npy")
        ct = np.load(ct_path).astype("float32")  # (D,H,W)
        if ct.ndim == 3:
            ct = ct[None, ...]  # add channel dim

        # Clinical features
        row = self.clinical_df.loc[patient_id]
        features = row[self.feature_cols].values.astype("float32")
        label = int(row[self.label_col])

        ct_tensor = torch.from_numpy(ct)
        feat_tensor = torch.from_numpy(features)
        label_tensor = torch.tensor(label, dtype=torch.long)

        sample = {
            "patient_id": patient_id,
            "ct": ct_tensor,
            "clinical_features": feat_tensor,
            "label": label_tensor,
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
