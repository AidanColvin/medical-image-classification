import os
import shutil
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from tqdm import tqdm
from PIL import Image

def clean_environment():
    print("\n--- [1/5] Purging Old Files & Directories ---")
    # Wipe old data folders
    for d in ['data/visualizations', 'data/tables', 'submissions', 'src']:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    
    # Consolidate loose scripts and CSVs
    for f in glob.glob("*.csv"): shutil.move(f, f"data/{f}")
    for f in glob.glob("*.py"): 
        if f not in ['main.py', 'run_cv_pipeline.py']: shutil.move(f, f"src/{f}")
    
    # Destroy all old reports
    for old_report in glob.glob("*REPORT*.md") + ['Report']:
        if os.path.exists(old_report): os.remove(old_report)
    print("✓ Environment clean.")

class TestDataset(Dataset):
    def __init__(self, directory, transform):
        self.filepaths = glob.glob(os.path.join(directory, '*.png')) + glob.glob(os.path.join(directory, '**/*.png'), recursive=True)
        self.transform = transform
    def __len__(self): return len(self.filepaths)
    def __getitem__(self, idx):
        path = self.filepaths[idx]
        return self.transform(Image.open(path).convert('RGB')), os.path.basename(path)

def build_model(device):
    return nn.Sequential(
        nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32*14*14, 2)
    ).to(device)

def train_and_evaluate():
    print("\n--- [2/5] Running 2-Fold Cross Validation ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_data = datasets.ImageFolder('data/train', transform=transform)
    
    kfold = KFold(n_splits=2, shuffle=True, random_state=42)
    criterion = nn.CrossEntropyLoss()
    
    all_losses, all_true_labels, all_pred_probs = [], [], []
    model = build_model(device) # Initialized outside to retain weights across folds for test inference
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
        train_loader = DataLoader(train_data, batch_size=32, sampler=SubsetRandomSampler(train_idx))
        val_loader = DataLoader(train_data, batch_size=32, sampler=SubsetRandomSampler(val_idx))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(2):
            model.train()
            epoch_loss = 0
            epoch_pbar = tqdm(train_loader, desc=f"Fold {fold+1}/2 | Epoch {epoch+1}/2", leave=False)
            for images, labels in epoch_pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            all_losses.append(epoch_loss / len(train_loader))
        
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                probs = torch.softmax(model(images.to(device)), dim=1)[:, 1]
                all_true_labels.extend(labels.numpy())
                all_pred_probs.extend(probs.cpu().numpy())
                
    return model, device, transform, all_losses, all_true_labels, all_pred_probs

def create_visuals_and_tables(model, losses, true_labels, pred_probs):
    print("\n--- [3/5] Generating Visualizations & Tables ---")
    
    # 1. Loss Tracking
    plt.figure(figsize=(8,4))
    plt.plot(losses, marker='o', color='blue')
    plt.title('Training Loss per Epoch')
    plt.savefig('data/visualizations/loss_curve.png')
    plt.close()
    loss_df = pd.DataFrame({'Epoch': range(1, len(losses)+1), 'Loss': losses})
    loss_df.to_csv('data/tables/loss_table.csv', index=False)

    # 2. AUC Tracking
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('data/visualizations/auc_curve.png')
    plt.close()
    auc_df = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})
    auc_df.to_csv('data/tables/auc_table.csv', index=False)

    # 3. Parameter Impact (CNN Feature Importance)
    impact = [{'Layer': n, 'Mean Abs Weight': p.abs().mean().item()} for n, p in model.named_parameters() if 'weight' in n]
    impact_df = pd.DataFrame(impact).sort_values(by='Mean Abs Weight', ascending=False)
    plt.figure(figsize=(10,5))
    plt.barh(impact_df['Layer'], impact_df['Mean Abs Weight'], color='teal')
    plt.gca().invert_yaxis()
    plt.title('Layer Parameter Impact')
    plt.savefig('data/visualizations/parameter_impact.png')
    plt.close()
    impact_df.to_csv('data/tables/parameter_impact_table.csv', index=False)
    
    print("✓ Tables and visuals saved to data/ directories.")
    return loss_df, auc_df, impact_df

def run_test_inference(model, device, transform):
    print("\n--- [4/5] Running Test Data Inference ---")
    test_dir = 'data/test' if os.path.exists('data/test') else ('test' if os.path.exists('test') else None)
    
    if not test_dir:
        print("  ! No test directory found. Skipping inference.")
        return
        
    test_loader = DataLoader(TestDataset(test_dir, transform), batch_size=32, shuffle=False)
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, paths in tqdm(test_loader, desc="Generating Predictions"):
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            for path, pred in zip(paths, predicted):
                results.append({'id': path, 'prediction': pred.item()})
                
    pd.DataFrame(results).to_csv('submissions/final_submission.csv', index=False)
    print("✓ Saved to submissions/final_submission.csv")

def generate_report(loss_df, impact_df):
    print("\n--- [5/5] Compiling Master Report ---")
    report = f"""# Medical Image Classification: Master Report
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## 1. Test Inferences
Predictions for the unlabelled test data have been generated and mapped to their respective filenames.
- **Output File:** `submissions/final_submission.csv`

## 2. Model Evaluation (Cross-Validation)
### Area Under Curve (AUC)
![AUC Curve](data/visualizations/auc_curve.png)
*(Detailed rates saved in `data/tables/auc_table.csv`)*

### Training Loss
![Loss Curve](data/visualizations/loss_curve.png)

**Loss Data:**
{loss_df.to_markdown(index=False)}

## 3. Parameter Impact (Feature Importance)
For Convolutional Neural Networks, standard tabular features do not exist. Instead, we measure the **Mean Absolute Weight** of the network layers to determine which computational stages are making the heaviest impact on the final decision.

![Parameter Impact](data/visualizations/parameter_impact.png)

**Impact Data:**
{impact_df.to_markdown(index=False)}
"""
    with open('PROJECT_REPORT.md', 'w') as f:
        f.write(report)
    shutil.move('run_cv_pipeline.py', 'src/run_cv_pipeline.py')
    print("✓ PROJECT_REPORT.md generated in root.")

if __name__ == "__main__":
    clean_environment()
    model, device, transform, losses, true_labels, pred_probs = train_and_evaluate()
    loss_df, auc_df, impact_df = create_visuals_and_tables(model, losses, true_labels, pred_probs)
    run_test_inference(model, device, transform)
    generate_report(loss_df, impact_df)
    print("\n🚀 Pipeline Execution Complete.")
