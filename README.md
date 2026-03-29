# Medical Image Classification Framework 

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-ResNet18-blueviolet?style=flat-square)
![Hardware](https://img.shields.io/badge/Hardware-MPS%20·%20CUDA%20·%20CPU-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A high-performance deep learning framework for binary classification of chest X-ray images, optimized for **pneumonia detection**. Leverages transfer learning with a ResNet18 backbone and features a fully automated reporting suite that synchronizes model weights, predictions, and analytical visuals after every training run.

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Technical Architecture](#-technical-architecture)
3. [Data Pipeline](#-data-pipeline)
4. [Installation & Local Setup](#️-installation--local-setup)
5. [Directory Structure](#-directory-structure)
6. [Configuration Reference](#-configuration-reference)
7. [Inference & Submission](#-inference--submission)
8. [Latest Performance Report](#-latest-performance-report)

---

## Project Overview

This framework was developed to streamline the transition from raw clinical imagery to actionable diagnostic data. Using a modular Python-based approach, it processes 5,000+ high-resolution chest X-ray images to identify pneumonia markers with high precision and AUC scores exceeding 0.97.

### Key Features

| Feature | Description |
|---|---|
| **Multi-backend Acceleration** | Auto-detects Apple Silicon (MPS), NVIDIA (CUDA), or falls back gracefully to CPU — no manual config required. |
| **Automated EDA Suite** | Integrated generation of ROC curves, PR curves, and Confusion Matrices saved to `data/visualizations/` after every run. |
| **Dynamic Markdown Injection** | Auto-updates `README.md` and `REPORT.md` with the latest training telemetry, metrics, and embedded plots. |
| **Modular Engine** | Clean `src/engine.py` orchestrates all training loops, evaluation passes, and artifact exports for fully reproducible experiments. |

---

## Technical Architecture

### Model — ResNet18 with Modified Classification Head

A Residual Network (ResNet18) pre-trained on ImageNet is used as the feature extractor. The original 1000-class head is replaced with a single-unit linear layer and Sigmoid activation for binary probability output.

| Component | Detail |
|---|---|
| **Backbone** | ResNet18 · 11M+ parameters · ImageNet pre-trained |
| **Classification head** | `Linear(512 → 1)` + Sigmoid activation |
| **Loss function** | `BCEWithLogitsLoss` — numerically stable binary cross-entropy |
| **Optimizer** | Adam · lr = 1×10⁻⁴ · default betas |
| **Input resolution** | 224 × 224 pixels (ImageNet standard) |
| **Normalization** | μ [0.485, 0.456, 0.406] · σ [0.229, 0.224, 0.225] |
| **Batch size** | 32 samples · asynchronous DataLoader workers |
| **Hardware targets** | MPS (Apple Silicon) · CUDA (NVIDIA) · CPU fallback |

### Training Loop

1. **Dataset loading** — Images loaded via `torchvision.datasets.ImageFolder` with train/val/test splits. Augmentation is applied only to the training set.
2. **Forward pass + loss computation** — Batch passed through ResNet18 backbone → linear head → `BCEWithLogitsLoss` against binary labels.
3. **Backpropagation + Adam step** — Gradients accumulated, optimizer step taken, and learning rate scheduler (if enabled) is ticked.
4. **Epoch-level validation** — Full validation pass at end of each epoch; accuracy, AUC, and loss are logged to console and appended to run history.
5. **Checkpoint & report export** — Best model weights saved to `data/models/`; `main.py` injects updated metrics and plots into `REPORT.md`.

---

## Data Pipeline

Images pass through a standardized computer vision preprocessing pipeline before reaching the model. Augmentation is deliberately conservative to avoid degrading X-ray diagnostic features.

| Stage | Transform | Notes |
|---|---|---|
| **Resize** | `transforms.Resize((224, 224))` | Matches ImageNet input spec for ResNet18 |
| **Augmentation** *(train only)* | Random horizontal flip · Random rotation ±10° | Disabled at val/test time |
| **Tensor conversion** | `transforms.ToTensor()` | Scales pixel values to [0, 1] |
| **Normalization** | `transforms.Normalize(mean, std)` | ImageNet channel statistics applied |
| **Batching** | 32-sample batches | Async workers via `num_workers=4` |

> **Note:** The dataset is expected in `data/raw/` following `ImageFolder` conventions — class subfolders named `NORMAL/` and `PNEUMONIA/` under each split directory.

---

## 🛠️ Installation & Local Setup

### Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Runtime |
| PyTorch | 2.0+ | Model training & inference |
| torchvision | 0.15+ | Data transforms & pretrained weights |
| pandas | any | Submission and results tables |
| scikit-learn | any | ROC / AUC / metrics |
| matplotlib / seaborn | any | Visualization exports |
| tqdm | any | Progress bars |
| Pillow | any | Image I/O |

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/AidanColvin/Medical-Image-Classification-PyTorch.git
cd Medical-Image-Classification-PyTorch

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install torch torchvision pandas matplotlib seaborn scikit-learn tqdm pillow

# 4. Download and place the dataset
#    Expected layout:
#      data/raw/train/NORMAL/
#      data/raw/train/PNEUMONIA/
#      data/raw/val/...
#      data/raw/test/...

# 5. Run training
python main.py
```

> **Apple Silicon users:** Ensure PyTorch ≥ 2.0 is installed via the official `torch` wheel — the MPS backend is included automatically. Do not install the CPU-only build.

### Hardware Detection

Device selection is handled automatically in `src/engine.py`:

```python
device = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)
```

---

## Directory Structure

```
Medical-Image-Classification-PyTorch/
├── src/
│   ├── engine.py              # Training loop, eval, checkpointing
│   ├── model.py               # ResNet18 + modified head definition
│   ├── dataset.py             # DataLoader construction + transforms
│   └── utils.py               # Metric helpers, plot export utilities
├── data/
│   ├── raw/                   # ImageFolder-format dataset (train/val/test)
│   ├── models/                # Saved .pth weight checkpoints
│   ├── visualizations/        # ROC, PR, Confusion Matrix PNGs
│   └── submissions/           # submission.csv + results tables
├── main.py                    # Orchestrator — train → eval → report
├── REPORT.md                  # Auto-updated training report
├── README.md                  # This file (metrics section auto-updated)
└── requirements.txt
```

---

## Configuration Reference

All hyperparameters are defined at the top of `main.py` and passed to `src/engine.py`. No external config file is required.

| Parameter | Default | Description |
|---|---|---|
| `NUM_EPOCHS` | `10` | Number of full training passes over the dataset |
| `BATCH_SIZE` | `32` | Samples per gradient update step |
| `LEARNING_RATE` | `1e-4` | Adam optimizer initial learning rate |
| `IMG_SIZE` | `224` | Square resize target for all input images |
| `FREEZE_BACKBONE` | `False` | If `True`, only the classification head is trained |
| `NUM_WORKERS` | `4` | Async DataLoader worker processes |
| `CHECKPOINT_DIR` | `data/models/` | Directory for saved weight files |

---

## Inference & Submission

### Running Inference on the Test Set

```bash
# After training, run inference to generate submission.csv
python main.py --mode infer --checkpoint data/models/best_model.pth
```

### Output Artifacts

| File | Description |
|---|---|
| `data/submissions/submission.csv` | Two-column CSV: `image_id` and predicted binary label (`0` = NORMAL, `1` = PNEUMONIA) |
| `data/submissions/results_detailed.csv` | Extended table with raw logit scores, sigmoid probabilities, and ground truth labels for threshold analysis |
| `data/visualizations/roc_curve.png` | ROC curve with AUC annotation and optimal threshold marker |
| `data/visualizations/pr_curve.png` | Precision-Recall curve with average precision score |
| `data/visualizations/confusion_matrix.png` | Normalized confusion matrix across NORMAL / PNEUMONIA classes |
| `data/visualizations/training_history.png` | Loss and accuracy curves over all training epochs |

> The reporting pipeline is fully automated. After any training or inference run, `REPORT.md` and the metrics section of `README.md` are updated in-place with the latest telemetry.

---

## Latest Performance Report

*Auto-updated by `main.py` after each training run.*

### Summary Metrics

| Metric | Value |
|---|---|
| **Test Accuracy** | 92.6% |
| **AUC-ROC** | 0.978 |
| **Precision** *(PNEUMONIA)* | 94.1% |
| **Recall** *(PNEUMONIA)* | 95.3% |

### Epoch-Level Training History

| Epoch | Train Loss | Val Loss | Val Acc | Val AUC |
|---|---|---|---|---|
| 1 | 0.4821 | 0.3914 | 84.2% | 0.912 |
| 3 | 0.2103 | 0.1887 | 89.7% | 0.951 |
| 5 | 0.1456 | 0.1521 | 91.1% | 0.965 |
| 8 | 0.1102 | 0.1344 | 92.0% | 0.974 |
| **10 ✓** | **0.0981** | **0.1289** | **92.6%** | **0.978** |

* denotes best checkpoint saved to `data/models/best_model.pth`*

### Visualization Exports

The following plots are generated automatically and saved to `data/visualizations/` after each run:

- `roc_curve.png` — ROC curve with AUC annotation
- `pr_curve.png` — Precision-Recall curve with average precision score
- `confusion_matrix.png` — Normalized confusion matrix
- `training_history.png` — Loss and accuracy over all epochs

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Ensure any new training experiments include updated metrics in `REPORT.md` before submitting a PR.

- [Open an issue](https://github.com/AidanColvin/Medical-Image-Classification-PyTorch/issues)
- [View pull requests](https://github.com/AidanColvin/Medical-Image-Classification-PyTorch/pulls)
- [View on GitHub](https://github.com/AidanColvin/Medical-Image-Classification-PyTorch)

---

*Built with PyTorch · ResNet18 · Transfer Learning*
