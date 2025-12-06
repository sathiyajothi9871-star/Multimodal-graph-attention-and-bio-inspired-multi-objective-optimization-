"""Preprocessing pipeline for NSCLC-Radiomics CT volumes and clinical data.

Run as:
    python -m src.data.preprocessing

Steps (high-level):
- Load NIfTI/NRRD CT and mask files
- Resample to 1x1x1 mm spacing
- Extract tumor ROI using mask bounding box
- Resize/crop to (CT_DEPTH, CT_HEIGHT, CT_WIDTH)
- Normalize intensities (e.g., lung window, z-score)
- Save as .npy
- Clean and encode clinical.csv
- Generate deterministic stratified train/val/test splits
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src import config
from src.utils.seed import set_global_seed


def _load_ct_and_mask(ct_path, mask_path):
    try:
        import SimpleITK as sitk
    except ImportError as e:
        raise ImportError(
            "SimpleITK is required for preprocessing NIfTI/NRRD CT volumes. "
            "Install it with `pip install simpleitk`.\n"
            f"Missing import error: {e}"
        )

    ct_img = sitk.ReadImage(ct_path)
    mask_img = sitk.ReadImage(mask_path)

    return ct_img, mask_img


def _resample_to_isotropic(ct_img, mask_img, spacing=(1.0, 1.0, 1.0)):
    import SimpleITK as sitk

    original_spacing = ct_img.GetSpacing()
    original_size = ct_img.GetSize()

    out_spacing = spacing
    out_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, out_spacing)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(ct_img.GetDirection())
    resample.SetOutputOrigin(ct_img.GetOrigin())

    ct_resampled = resample.Execute(ct_img)

    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_resampled = resample.Execute(mask_img)

    return ct_resampled, mask_resampled


def _extract_roi(ct_resampled, mask_resampled):
    import SimpleITK as sitk

    mask_arr = sitk.GetArrayFromImage(mask_resampled)  # (Z,Y,X)
    ct_arr = sitk.GetArrayFromImage(ct_resampled)

    coords = np.where(mask_arr > 0)
    z_min, y_min, x_min = np.min(coords[0]), np.min(coords[1]), np.min(coords[2])
    z_max, y_max, x_max = np.max(coords[0]), np.max(coords[1]), np.max(coords[2])

    ct_crop = ct_arr[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    return ct_crop


def _resize_to_fixed_shape(volume, shape):
    from scipy.ndimage import zoom
    z, y, x = volume.shape
    dz, dy, dx = shape
    zoom_factors = (dz / z, dy / y, dx / x)
    vol_resized = zoom(volume, zoom_factors, order=1)
    return vol_resized


def preprocess_ct_and_save():
    os.makedirs(config.PROCESSED_CT_DIR, exist_ok=True)
    set_global_seed(config.SEED)

    ct_files = [
        f for f in os.listdir(config.RAW_CT_DIR)
        if f.endswith("_ct.nii") or f.endswith("_ct.nii.gz") or f.endswith("_ct.mha")
    ]

    for f in ct_files:
        patient_id = f.split("_ct")[0]
        ct_path = os.path.join(config.RAW_CT_DIR, f)
        mask_candidates = [
            f"{patient_id}_mask.nii.gz",
            f"{patient_id}_mask.nii",
            f"{patient_id}_mask.mha",
        ]
        mask_path = None
        for cand in mask_candidates:
            p = os.path.join(config.RAW_CT_DIR, cand)
            if os.path.exists(p):
                mask_path = p
                break
        if mask_path is None:
            print(f"[WARN] No mask found for {patient_id}, skipping.")
            continue

        print(f"Processing {patient_id} ...")
        ct_img, mask_img = _load_ct_and_mask(ct_path, mask_path)
        ct_res, mask_res = _resample_to_isotropic(ct_img, mask_img)
        ct_roi = _extract_roi(ct_res, mask_res)

        # Intensity normalization (lung window [-1000, 400])
        ct_roi = np.clip(ct_roi, -1000, 400)
        ct_roi = (ct_roi - ct_roi.mean()) / (ct_roi.std() + 1e-8)

        # Resize to fixed depth/height/width
        ct_fixed = _resize_to_fixed_shape(
            ct_roi,
            (config.CT_DEPTH, config.CT_HEIGHT, config.CT_WIDTH)
        ).astype("float32"))

        out_path = os.path.join(config.PROCESSED_CT_DIR, f"{patient_id}_ct.npy")
        np.save(out_path, ct_fixed)


def preprocess_clinical_and_splits():
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df = pd.read_csv(config.RAW_CLINICAL_CSV)

    # Basic cleaning: drop rows without CT or mask, etc.
    df = df.dropna(subset=["patient_id", "label"])  # label: 0/1 or similar

    # Example encoding: you should adapt to your real columns.
    categorical_cols = [c for c in df.columns if df[c].dtype == "object" and c not in ["patient_id"]]
    df = pd.get_dummies(df, columns=categorical_cols)

    # Save processed clinical CSV
    df.to_csv(config.PROCESSED_CLINICAL_CSV, index=False)

    # Stratified split by label
    set_global_seed(config.SEED)
    y = df["label"].values
    patient_ids = df["patient_id"].values

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=config.SEED)
    trainval_idx, test_idx = next(sss1.split(patient_ids, y))

    trainval_ids = patient_ids[trainval_idx]
    test_ids = patient_ids[test_idx]
    y_trainval = y[trainval_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1765, random_state=config.SEED)  # 0.1765 ~ 0.15 / 0.85
    train_idx, val_idx = next(sss2.split(trainval_ids, y_trainval))

    train_ids = trainval_ids[train_idx].tolist()
    val_ids = trainval_ids[val_idx].tolist()
    test_ids = test_ids.tolist()

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    with open(config.SPLITS_JSON, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


def main():
    preprocess_ct_and_save()
    preprocess_clinical_and_splits()


if __name__ == "__main__":
    main()
