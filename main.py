import os
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "seed": 42,
    "img_size": 224,
    "batch_size": 128,
    "epochs": 10,
    "lr": 1e-3,
    "val_size": 0.20,
    "num_workers": 2,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "data_zip": "pneumonia.zip",
    "train_dir": "train",
    "test_dir": "test",
    "label_csv": "train_label.csv",
    "output_dir": "output",
    "model_path": "output/final_model.pt"
}

# ==========================================
# 2. UTILITIES & SETUP
# ==========================================
def set_seed(seed: int):
    """
    Sets the random seed across all possible libraries for reproducibility.
    - Takes: An integer seed value.
    - Process: Configures random, numpy, and torch (including CUDA backends).
    - Returns: None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_filesystem():
    """
    Handles data extraction and directory setup.
    - Takes: Nothing (reads from global CONFIG).
    - Process: Unzips data if the train folder is missing and creates the output directory.
    - Returns: None.
    """
    if not os.path.exists(CONFIG["train_dir"]):
        os.system(f"unzip -q {CONFIG['data_zip']}")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==========================================
# 3. DATA PREPROCESSING
# ==========================================
def load_and_split_data():
    """
    Prepares the training and validation dataframes.
    - Takes: Nothing (reads label CSV from CONFIG).
    - Process: Loads CSV, ensures correct data types, and performs a stratified split.
    - Returns: Two DataFrames (train_df, val_df).
    """
    df = pd.read_csv(CONFIG["label_csv"])
    df["Filename"] = df["Filename"].astype(str)
    df["Label"] = df["Label"].astype(int)
    
    train, val = train_test_split(
        df, 
        test_size=CONFIG["val_size"], 
        random_state=CONFIG["seed"], 
        stratify=df["Label"]
    )
    return train, val

def get_test_filenames():
    """
    Scans the test directory for available images.
    - Takes: Nothing (reads test_dir from CONFIG).
    - Process: Filters for .png files and sorts them to maintain order.
    - Returns: A DataFrame containing a list of test filenames.
    """
    files = sorted([f for f in os.listdir(CONFIG["test_dir"]) if f.lower().endswith(".png")])
    return pd.DataFrame({"Filename": files})

class ChestXrayDataset(Dataset):
    """
    A custom Dataset class to load medical images for PyTorch.
    - Takes: DataFrame with filenames/labels, image directory path, and transforms.
    - Process: Opens images, converts to RGB, and applies augmentation.
    - Returns: A single data sample (Image, Label, Filename).
    """
    def __init__(self, dataframe, image_dir, transform=None, has_labels=True):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.image_dir / row["Filename"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        if self.has_labels:
            label = torch.tensor(float(row["Label"]), dtype=torch.float32)
            return img, label, row["Filename"]
        return img, row["Filename"]

def get_transforms():
    """
    Constructs the image augmentation pipelines.
    - Takes: Nothing (reads img_size from CONFIG).
    - Process: Creates a 'noisy' pipeline for training and a 'clean' one for validation.
    - Returns: A tuple of two Compose objects (train_transform, val_transform).
    """
    train_tf = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.RandomRotation(7),
        transforms.RandomAffine(0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf

# ==========================================
# 4. MODEL ARCHITECTURE
# ==========================================
def build_resnet50(num_classes: int = 1):
    """
    Initializes a ResNet-50 model with a custom classification head.
    - Takes: The number of output classes (default is 1 for binary).
    - Process: Loads pre-trained weights and replaces the final fully connected layer.
    - Returns: A PyTorch model object moved to the configured device.
    """
    model = models.resnet50(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(CONFIG["device"])

def calculate_pos_weight(df: pd.DataFrame):
    """
    Calculates the weight for the positive class to combat dataset imbalance.
    - Takes: The training DataFrame containing labels.
    - Process: Divides the number of negative samples by the number of positive samples.
    - Returns: A single-element Tensor containing the weight.
    """
    n_pos = (df["Label"] == 1).sum()
    n_neg = (df["Label"] == 0).sum()
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(CONFIG["device"])

# ==========================================
# 5. TRAINING & EVALUATION ENGINE
# ==========================================
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Computes statistical performance metrics.
    - Takes: Array of true labels and array of predicted probabilities.
    - Process: Thresholds probabilities at 0.5 and calculates accuracy and F1 score.
    - Returns: A dictionary containing 'accuracy' and 'f1' scores.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }

def run_epoch(model, loader, criterion, optimizer=None):
    """
    Executes a single pass (training or validation) over the dataset.
    - Takes: Model, DataLoader, Loss Function, and an optional Optimizer.
    - Process: Iterates through batches, moves data to GPU, computes loss, and backprops if training.
    - Returns: A tuple (average_loss, metrics_dict).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss, all_probs, all_labels = 0.0, [], []
    
    for images, labels, _ in loader:
        images = images.to(CONFIG["device"], non_blocking=True)
        labels = labels.to(CONFIG["device"], non_blocking=True).unsqueeze(1)
        
        if is_train:
            optimizer.zero_grad()
        
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().ravel())
        all_labels.extend(labels.detach().cpu().numpy().ravel())
        
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    return avg_loss, metrics

# ==========================================
# 6. PREDICTION & POST-PROCESSING
# ==========================================
@torch.no_grad()
def generate_predictions(model, loader):
    """
    Generates model predictions for unlabelled test data.
    - Takes: The trained model and a DataLoader for the test set.
    - Process: Passes images through the model and formats the filenames for competition submission.
    - Returns: A pandas DataFrame with columns ['id', 'label'].
    """
    model.eval()
    filenames, probs = [], []
    
    for images, batch_filenames in loader:
        images = images.to(CONFIG["device"])
        logits = model(images)
        probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
        filenames.extend(batch_filenames)
        
    return pd.DataFrame({
        "id": [f.replace('test_', '').replace('.png', '') for f in filenames],
        "label": (np.array(probs) >= 0.5).astype(int)
    })

# ==========================================
# 7. MAIN EXECUTION FLOW
# ==========================================
def main():
    """
    Orchestrates the entire machine learning pipeline.
    - Takes: Nothing.
    - Process: Sets seeds, prepares data, trains the model, saves weights, and generates a submission file.
    - Returns: None.
    """
    set_seed(CONFIG["seed"])
    prepare_filesystem()
    
    train_df, val_df = load_and_split_data()
    test_df = get_test_filenames()
    train_tf, val_tf = get_transforms()
    
    train_loader = DataLoader(ChestXrayDataset(train_df, CONFIG["train_dir"], train_tf), 
                              batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(ChestXrayDataset(val_df, CONFIG["train_dir"], val_tf), 
                            batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"])
    test_loader = DataLoader(ChestXrayDataset(test_df, CONFIG["test_dir"], val_tf, has_labels=False), 
                             batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"])
    
    model = build_resnet50()
    criterion = nn.BCEWithLogitsLoss(pos_weight=calculate_pos_weight(train_df))
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    
    print(f"Starting training on {CONFIG['device']}...")
    for epoch in range(CONFIG["epochs"]):
        t_loss, t_met = run_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_met = run_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: Train Acc {t_met['accuracy']:.4f} | Val Acc {v_met['accuracy']:.4f}")
    
    torch.save(model.state_dict(), CONFIG["model_path"])
    submission = generate_predictions(model, test_loader)
    submission.to_csv("submission.csv", index=False)
    print("Workflow complete.")

if __name__ == "__main__":
    main()
