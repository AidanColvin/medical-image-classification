import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from PIL import Image
from pathlib import Path

ROOT = Path("/Users/aidancolvin/Medical-Image-Classification-PyTorch")
sys.path.insert(0, str(ROOT))

from main import build_resnet50, CONFIG, get_transforms

CSV_PATH  = ROOT / "data" / "train" / "train_label.csv"
IMG_ROOT  = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "data" / "submission_visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output folder ready: {OUTPUT_DIR}")
print(f"CSV: {CSV_PATH}")
print(f"Images: {IMG_ROOT}")

class ChestXrayDataset(Dataset):
    def __init__(self, df, image_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["Label"])
        fname = row["Filename"]
        img = Image.open(self.image_root / str(label) / fname).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, fname

def get_label_counts():
    df = pd.read_csv(CSV_PATH)
    return int((df["Label"]==0).sum()), int((df["Label"]==1).sum())

def load_splits(val_frac=0.2):
    df = pd.read_csv(CSV_PATH)
    train_tf, val_tf = get_transforms()
    val_size   = int(len(df) * val_frac)
    train_size = len(df) - val_size
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df   = df.iloc[train_size:].reset_index(drop=True)
    return (ChestXrayDataset(train_df, IMG_ROOT, transform=train_tf),
            ChestXrayDataset(val_df,   IMG_ROOT, transform=val_tf))

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(out) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss/total, correct/total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs, all_fnames = [], [], []
    with torch.no_grad():
        for imgs, labels, fnames in loader:
            imgs = imgs.to(device)
            labels_t = labels.float().unsqueeze(1).to(device)
            out = model(imgs)
            loss = criterion(out, labels_t)
            probs = torch.sigmoid(out).squeeze(1)
            total_loss += loss.item() * imgs.size(0)
            preds = (probs > 0.5).float()
            correct += (preds == labels.to(device).float()).sum().item()
            total += imgs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_fnames.extend(fnames)
    return total_loss/total, correct/total, np.array(all_labels), np.array(all_probs), all_fnames

def run_training(model, train_loader, val_loader, epochs, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    for epoch in range(epochs):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, va, _, _, _ = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(ta);  history["val_acc"].append(va)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss:{tl:.4f} Acc:{ta:.4f} | Val Loss:{vl:.4f} Acc:{va:.4f}")
    return history, model

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        model.layer4[-1].register_forward_hook(lambda m,i,o: setattr(self,'activations',o.detach()))
        model.layer4[-1].register_full_backward_hook(lambda m,i,o: setattr(self,'gradients',o[0].detach()))
    def generate(self, img_tensor):
        self.model.eval()
        t = img_tensor.clone().requires_grad_(True)
        self.model(t).backward()
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

def save(name): 
    path = OUTPUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

def plot_label_distribution(normal, phobia):
    fig, ax = plt.subplots(figsize=(6,5))
    bars = ax.bar(["Normal (0)","Phobia (1)"], [normal,phobia], color=["#4C9BE8","#E8654C"])
    for bar, count in zip(bars,[normal,phobia]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, str(count), ha="center", fontsize=12, fontweight="bold")
    ax.set_title("Label Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); save("label_distribution.png")

def plot_training_curves(history):
    epochs = range(1, len(history["train_loss"])+1)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
    ax1.plot(epochs, history["train_loss"], label="Train", color="#4C9BE8", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="#E8654C", linewidth=2, linestyle="--")
    ax1.set_title("Loss over Epochs", fontsize=13, fontweight="bold"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss"); ax1.legend(); ax1.spines[["top","right"]].set_visible(False)
    ax2.plot(epochs, history["train_acc"], label="Train", color="#4C9BE8", linewidth=2)
    ax2.plot(epochs, history["val_acc"],   label="Val",   color="#E8654C", linewidth=2, linestyle="--")
    ax2.set_title("Accuracy over Epochs", fontsize=13, fontweight="bold"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_ylim(0,1.05); ax2.legend(); ax2.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); save("training_curves.png")

def plot_roc(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(fpr, tpr, color="#4C9BE8", linewidth=2.5, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0,1],[0,1], color="gray", linestyle="--")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=12); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); save("roc_curve.png")

def plot_pr(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(recall, precision, color="#6DBE6D", linewidth=2.5, label=f"AP = {pr_auc:.3f}")
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(loc="upper right", fontsize=12); ax.set_xlim([0,1]); ax.set_ylim([0,1.05]); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); save("precision_recall_curve.png")

def plot_confusion(y_true, y_probs):
    y_preds = (y_probs > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_preds)
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues"); plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Normal","Phobia"], fontsize=12)
    ax.set_yticks([0,1]); ax.set_yticklabels(["Normal","Phobia"], fontsize=12)
    thresh = cm.max()/2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", color="white" if cm[i,j]>thresh else "black", fontsize=14)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold"); ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.tight_layout(); save("confusion_matrix.png")

def plot_samples(y_probs, y_true, fnames, n=8):
    y_preds = (y_probs > 0.5).astype(int)
    correct_idx = np.where(y_preds == y_true)[0][:n//2]
    wrong_idx   = np.where(y_preds != y_true)[0][:n//2]
    fig, axes = plt.subplots(2, n//2, figsize=(14,6))
    fig.suptitle("Sample Predictions  |  Green=Correct  Red=Incorrect", fontsize=13, fontweight="bold")
    labels_map = {0:"Normal", 1:"Phobia"}
    for row, (group, title) in enumerate([(correct_idx,"Correct"),(wrong_idx,"Incorrect")]):
        for col, idx in enumerate(group):
            ax = axes[row][col]
            try:
                label = int(y_true[idx])
                ax.imshow(Image.open(IMG_ROOT / str(label) / fnames[idx]).convert("RGB"))
            except Exception:
                ax.text(0.5,0.5,"N/A",ha="center",va="center")
            color = "#2ECC71" if row==0 else "#E74C3C"
            ax.set_title(f"True:{labels_map[int(y_true[idx])]}\nPred:{labels_map[int(y_preds[idx])]}", fontsize=8, color=color)
            ax.axis("off")
        axes[row][0].set_ylabel(title, fontsize=11, fontweight="bold")
    plt.tight_layout(); save("sample_predictions.png")

def plot_gradcam(model, y_true, fnames, device, n=4):
    cam_extractor = GradCAM(model)
    _, val_tf = get_transforms()
    fig, axes = plt.subplots(2, n, figsize=(14,6))
    plt.suptitle("Grad-CAM — Model Attention on Chest X-Rays", fontsize=13, fontweight="bold")
    plotted = 0
    for idx in range(len(fnames)):
        if plotted >= n: break
        try:
            label = int(y_true[idx])
            raw_img = Image.open(IMG_ROOT / str(label) / fnames[idx]).convert("RGB")
        except Exception:
            continue
        tensor = val_tf(raw_img).unsqueeze(0).to(device)
        heatmap = cam_extractor.generate(tensor)
        heatmap_r = np.array(Image.fromarray((heatmap*255).astype(np.uint8)).resize(raw_img.size, Image.BILINEAR))/255.0
        axes[0][plotted].imshow(raw_img); axes[0][plotted].set_title(f"Original\n({'Phobia' if label==1 else 'Normal'})", fontsize=9); axes[0][plotted].axis("off")
        axes[1][plotted].imshow(raw_img); axes[1][plotted].imshow(heatmap_r, cmap="jet", alpha=0.45); axes[1][plotted].set_title("Grad-CAM", fontsize=9); axes[1][plotted].axis("off")
        plotted += 1
    plt.tight_layout(); save("gradcam.png")

def main():
    device = "mps"
    epochs = CONFIG.get("epochs", 10)
    batch_size = CONFIG.get("batch_size", 32)
    print(f"\nDevice: {device} | Epochs: {epochs} | Batch: {batch_size}\n")

    normal_count, phobia_count = get_label_counts()
    plot_label_distribution(normal_count, phobia_count)

    train_ds, val_ds = load_splits()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_resnet50().to(device)
    print(f"Training for {epochs} epoch(s)...\n")
    history, model = run_training(model, train_loader, val_loader, epochs, device)

    criterion = nn.BCEWithLogitsLoss()
    _, _, y_true, y_probs, fnames = evaluate(model, val_loader, criterion, device)

    print("\nGenerating all visualizations...\n")
    plot_training_curves(history)
    plot_roc(y_true, y_probs)
    plot_pr(y_true, y_probs)
    plot_confusion(y_true, y_probs)
    plot_samples(y_probs, y_true, fnames)
    plot_gradcam(model, y_true, fnames, device)
    print(f"\nAll done! Check: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
