# Trustworthy Multimodal GAT + 3D CNN for NSCLC Radiomics

This repository contains an end-to-end **reproducible implementation** of the framework described in the paper:

> Multimodal graph-attention and bio-inspired multi-objective optimization for explainable NSCLC outcome prediction (title placeholder).

It includes:

- **Preprocessing** of NSCLC-Radiomics CT scans and clinical data
- **3D CNN (Dense-like) imaging branch**
- **Graph Attention Network (GAT) for clinical + radiomic features**
- **Multimodal fusion** of imaging and graph embeddings
- **Multi-objective Grey Wolf Optimizer (MO-GWO)** for hyperparameter search
- **Training / validation / testing** with stratified splits and leakage prevention
- **Ablation studies** (imaging-only, clinical-only, no-GAT, no-MO-GWO, etc.)
- **Explainability** visualizations:
  - SHAP global and local importance (clinical + radiomics)
  - Integrated Gradients for 3D CT
  - GNNExplainer for the GAT graph
  - Calibration, uncertainty and stratified ROC/PR plots

> **Important:** This code is structured to be *GitHub-ready* and fully reproducible, but you must provide the actual NSCLC-Radiomics data
> (CT volumes, masks, and clinical CSV) and adjust the paths in `src/config.py` and the notebooks.

---

## 1. Repository structure

```text
nsclc_gat_mogwo_repo/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── graph_builder.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dense3d.py
│   │   ├── gat_clinical_radiomics.py
│   │   └── full_model.py
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── mo_gwo.py
│   ├── train/
│   │   ├── train_gat_mogwo.py
│   │   ├── evaluate.py
│   │   └── ablation.py
│   ├── xai/
│   │   ├── __init__.py
│   │   ├── shap_explain.py
│   │   ├── ig_explain.py
│   │   └── gnn_explain.py
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       ├── metrics.py
│       ├── logging_utils.py
│       └── splits.py
└── notebooks/
    ├── 01_preprocessing.ipynb
    ├── 02_train_test_main_model.ipynb
    ├── 03_ablation_studies.ipynb
    └── 04_xai_visualizations.ipynb
```

---

## 2. Installation

Create a fresh environment (recommended: Python 3.10+):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Note: `torch` and `torch-geometric` versions are GPU/OS specific; if the pinned versions
> do not work on your system, please install them following the official installation guides
> and adjust `requirements.txt` accordingly.

---

## 3. Data preparation

Expected layout (you may adapt this in `src/config.py`):

```text
data/
├── raw/
│   ├── ct/
│   │   ├── PATIENT_0001_ct.nii.gz
│   │   ├── PATIENT_0001_mask.nii.gz
│   │   └── ...
│   └── clinical.csv
└── processed/
    ├── ct_npy/          # 3D numpy arrays after preprocessing
    ├── clinical.csv     # cleaned and encoded
    └── splits.json      # train/val/test patient IDs
```

Run preprocessing with:

- Notebook: `notebooks/01_preprocessing.ipynb`  
  or
- Script: `python -m src.data.preprocessing`

This will:
- Load raw CT & masks
- Resample to isotropic spacing
- Extract tumor ROIs
- Normalize and save as `.npy`
- Clean and encode clinical variables
- Generate stratified train/val/test splits

---

## 4. Training and evaluation

Use the main notebook:

- `notebooks/02_train_test_main_model.ipynb`  

or the CLI script:

```bash
python -m src.train.train_gat_mogwo
```

This will:

- Load processed data and splits
- Instantiate the 3D CNN + GAT fusion model
- Optionally run MO-GWO for hyperparameter search
- Train on train set, monitor validation
- Evaluate on the held-out test set
- Save trained weights and metrics JSON files

---

## 5. Ablation studies

Use `notebooks/03_ablation_studies.ipynb` or:

```bash
python -m src.train.ablation
```

Ablations include:

- Imaging-only (3D CNN without clinical/radiomics)
- Clinical-only (MLP / GAT without imaging)
- No GAT (simple concatenation)
- No MO-GWO (fixed hyperparameters)
- No harmonization / resampling (if you want to simulate this)  

Results are stored under `outputs/ablation/` as CSV or JSON.

---

## 6. Explainability & visualization

Use `notebooks/04_xai_visualizations.ipynb` or:

```bash
python -m src.xai.shap_explain
python -m src.xai.ig_explain
python -m src.xai.gnn_explain
```

This will generate:

- **Global SHAP bar & beeswarm plots**
- **Local SHAP/force plots for selected patients**
- **Integrated Gradients heatmaps overlayed on CT slices**
- **GNNExplainer subgraphs for the clinical+radiomic graph**
- **Calibration curve, probability histograms, and accuracy-vs-confidence plots**

Figures are saved under `outputs/figures/` and ready to be included in the manuscript.

---

## 7. Reproducibility

- We fix random seeds (NumPy, PyTorch, Python) via `src/utils/seed.py`
- Preprocessing and splits are deterministic and stored in `data/processed/splits.json`
- All main experiments can be reproduced by running the four notebooks in order.

