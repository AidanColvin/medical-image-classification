import os
import shutil
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets, transforms

def clean_and_prep():
    print("\n--- [Phase 1/4] Purging Old Files & Folders ---")
    folders_to_clear = ['results', 'data/visualizations', 'data/tables', 'submission_visualizations']
    for folder in folders_to_clear:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    
    os.makedirs('data/visualizations', exist_ok=True)
    os.makedirs('data/tables', exist_ok=True)
    if os.path.exists('PROJECT_REPORT.md'): os.remove('PROJECT_REPORT.md')

def run_analysis():
    print("--- [Phase 2/4] Re-Running Training & Generating Data ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    
    # Ensure data is organized for the run
    train_dataset = datasets.ImageFolder('data/train', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU(), nn.Flatten(), nn.Linear(16*62*62, 2)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    # Training Loop with Progress Bar
    for epoch in range(2):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        epoch_loss = 0
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(model(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss/len(train_loader))

    # Save Fresh Visuals
    plt.figure(figsize=(8, 4))
    plt.plot(losses, marker='o', color='blue')
    plt.title('Training Loss Trend (Updated)')
    plt.savefig('data/visualizations/loss_curve.png')
    
    # Save Fresh Tables (CSV)
    stats = pd.DataFrame({'Epoch': [1, 2], 'Loss': losses})
    stats.to_csv('data/tables/training_stats.csv', index=False)
    print("✓ New Visuals and Tables saved to data/")

def generate_report():
    print("--- [Phase 3/4] Compiling PROJECT_REPORT.md ---")
    stats = pd.read_csv('data/tables/training_stats.csv')
    
    report_content = f"""# Medical Image Classification: Final Report
**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**Status:** All Files Re-Generated

## 1. Training Performance
Below is the loss curve from the most recent training run.

![Loss Curve](data/visualizations/loss_curve.png)

### Training Statistics Table
{stats.to_markdown(index=False)}

---
## 2. Directory Structure Update
The repository has been consolidated. All scripts are now in `src/` and data outputs are in `data/`.
"""
    with open('PROJECT_REPORT.md', 'w') as f:
        f.write(report_content)
    print("✓ PROJECT_REPORT.md created in root.")

def final_organization():
    print("--- [Phase 4/4] Moving Scripts to src/ ---")
    os.makedirs('src', exist_ok=True)
    scripts = ['run_full_analysis.py', 'deep_clean.py', 'total_pipeline.py']
    for s in scripts:
        if os.path.exists(s) and s != 'total_pipeline.py':
            shutil.move(s, f'src/{s}')

if __name__ == "__main__":
    clean_and_prep()
    run_analysis()
    generate_report()
    final_organization()
