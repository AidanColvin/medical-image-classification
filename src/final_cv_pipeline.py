import os
import shutil
import glob
import torch
import torch.nn as nn
import re
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from tqdm import tqdm
from PIL import Image

def setup_workspace():
    print("\n--- [1/5] Cleaning & Organizing Workspace ---")
    # Folders to maintain
    for d in ['data/visualizations', 'data/tables', 'submissions', 'src']:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    
    # Clean root of stray CSVs/old reports
    for f in glob.glob("*.csv"): 
        if f not in ['sample_submission.csv', 'train_label.csv']: shutil.move(f, f"data/{f}")
    for old in glob.glob("*REPORT*.md") + ['Report', 'Visualizations']:
        if os.path.exists(old): os.remove(old)
    print("✓ Workspace organized.")

class TestDataset(Dataset):
    def __init__(self, directory, transform):
        self.filepaths = sorted(glob.glob(os.path.join(directory, '*.png')))
        self.transform = transform
    def __len__(self): return len(self.filepaths)
    def __getitem__(self, idx):
        path = self.filepaths[idx]
        filename = os.path.basename(path)
        # Extract integer ID from test_<id>.png
        img_id = re.findall(r'\d+', filename)[0]
        return self.transform(Image.open(path).convert('RGB')), int(img_id)

def build_model(device):
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2), # Layer 0
        nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2), # Layer 3
        nn.Flatten(),
        nn.Linear(32*14*14, 128), nn.ReLU(),            # Layer 7
        nn.Linear(128, 2)                               # Layer 9
    ).to(device)
    return model

def train_5_fold():
    print("\n--- [2/5] Running 5-Fold Cross Validation ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_data = datasets.ImageFolder('data/train', transform=transform)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    criterion = nn.CrossEntropyLoss()
    
    all_losses, all_true, all_probs = [], [], []
    final_model = build_model(device)
    
    fold_pbar = tqdm(total=5, desc="5-Fold Progress")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
        train_loader = DataLoader(train_data, batch_size=32, sampler=SubsetRandomSampler(train_idx))
        val_loader = DataLoader(train_data, batch_size=32, sampler=SubsetRandomSampler(val_idx))
        optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
        
        for epoch in range(2):
            final_model.train()
            epoch_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(final_model(images), labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            all_losses.append(epoch_loss / len(train_loader))
        
        final_model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                probs = torch.softmax(final_model(images.to(device)), dim=1)[:, 1]
                all_true.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        fold_pbar.update(1)
    fold_pbar.close()
    return final_model, device, transform, all_losses, all_true, all_probs

def create_enhanced_visuals(model, losses, true_labels, pred_probs):
    print("\n--- [3/5] Replacing Visuals & Generating Tables ---")
    
    # 1. AUC Curve
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Area = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title('Receiver Operating Characteristic (5-Fold CV)')
    plt.legend(loc="lower right")
    plt.savefig('data/visualizations/auc_curve.png')
    plt.close()

    # 2. Fixed Parameter Impact (Readable Labels)
    layer_map = {
        '0.weight': 'Initial Features (Conv1)',
        '3.weight': 'Mid-Level Patterns (Conv2)',
        '7.weight': 'Complex Associations (Dense1)',
        '9.weight': 'Final Classifier (Dense2)'
    }
    impact = []
    for n, p in model.named_parameters():
        if n in layer_map:
            impact.append({'Feature/Layer': layer_map[n], 'Impact Score': p.abs().mean().item()})
    
    impact_df = pd.DataFrame(impact).sort_values(by='Impact Score', ascending=False)
    plt.figure(figsize=(10,6))
    plt.barh(impact_df['Feature/Layer'], impact_df['Impact Score'], color='#008080')
    plt.title('Most Impactful Model Components')
    plt.xlabel('Mean Absolute Weight (Importance)')
    plt.tight_layout()
    plt.savefig('data/visualizations/parameter_impact.png')
    plt.close()
    impact_df.to_csv('data/tables/impact_table.csv', index=False)
    
    loss_df = pd.DataFrame({'Epoch': range(1, len(losses)+1), 'Loss': losses})
    loss_df.to_csv('data/tables/loss_table.csv', index=False)
    return loss_df, impact_df

def generate_submission(model, device, transform):
    print("\n--- [4/5] Generating Root Submission File ---")
    test_dir = 'data/test' if os.path.exists('data/test') else 'test'
    test_loader = DataLoader(TestDataset(test_dir, transform), batch_size=32, shuffle=False)
    
    model.eval()
    results = []
    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="Predicting Test Set"):
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            for img_id, pred in zip(ids, predicted):
                results.append({'id': img_id, 'label': pred.item()})
    
    df = pd.DataFrame(results).sort_values(by='id')
    df.to_csv('submission.csv', index=False)
    print("✓ Created submission.csv in root.")

def write_report(loss_df, impact_df):
    print("\n--- [5/5] Finalizing Report ---")
    report = f"""# Medical Image Classification: 5-Fold CV Report
**Status:** Full Re-run Complete

## Parameter Impact (Fixed)
The visualization below replaces the raw layer indices with descriptive component names.
![Impact](data/visualizations/parameter_impact.png)

## Model Performance
![AUC](data/visualizations/auc_curve.png)

### Impact Data Table
{impact_df.to_markdown(index=False)}
"""
    with open('PROJECT_REPORT.md', 'w') as f:
        f.write(report)
    shutil.move('final_cv_pipeline.py', 'src/final_cv_pipeline.py')

if __name__ == "__main__":
    setup_workspace()
    model, device, transform, losses, true, probs = train_5_fold()
    l_df, i_df = create_enhanced_visuals(model, losses, true, probs)
    generate_submission(model, device, transform)
    write_report(l_df, i_df)
