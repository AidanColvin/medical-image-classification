# Chest X-Ray Phobia Classification

A binary image classification pipeline for detecting **phobia** (vs. normal) from chest X-ray images using a fine-tuned ResNet-50 in PyTorch.

---

## Overview

This project trains a deep learning model on labeled chest X-ray images to distinguish between:

- **Label 1 — Phobia**: X-rays showing the condition of interest
- **Label 0 — Normal**: Healthy / negative X-rays

The model uses a ResNet-50 backbone (>20M parameters) with a single output neuron, suited for binary classification.

---

## Project Structure

```
├── main.py                  # Model definition, dataset class, transforms, config
├── train_label.csv          # Image manifest with Filename and Label columns
├── train/                   # Training images (≥224×224 px)
├── test/                    # Test images
└── tests/
    └── test_pipeline.py     # Pytest suite (11 tests)
```

---

## Data Format

`train_label.csv` must have exactly two columns:

| Column | Description |
|---|---|
| `Filename` | Image filename (e.g., `img_001.jpg`) |
| `Label` | Binary integer — `1` = phobia, `0` = normal |

Images must be at least **224×224 pixels** and live inside the `train/` or `test/` folders.

---

## Setup

```bash
pip install torch torchvision pillow pandas pytest
```

Requires **PyTorch 2.x or 3.x**.

---

## Configuration

Key settings are defined in `CONFIG` inside `main.py`, including:

- `device`: `"cuda"`, `"cpu"`, or `"mps"`

---

## Running Tests

```bash
pytest tests/test_pipeline.py -v
```

The test suite validates:

- PyTorch version compatibility
- Model parameter count (>20M)
- CSV structure and label integrity (strictly binary)
- Dataset output types (tensor image, int/tensor label, filename)
- Image resolution (≥224px wide)
- Transform normalization bounds (tensor values in [−3.0, 3.0])
- Device config validity
- Folder presence (`train/`, `test/`)
- Forward pass output shape `(batch_size, 1)`
- Non-empty CSV manifest

---

## Notes

- Labels are strictly `0` or `1` — no multi-class or soft labels
- Validation transforms are applied separately from training augmentations via `get_transforms()`
- The dataset class is `ChestXrayDataset`, returning `(image_tensor, label, filename)` per sample