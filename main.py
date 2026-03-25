import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from PIL import Image

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    # 1. Path & Directory Management
    # Ensuring logical structure: outputs/ for results, outputs/visuals/ for plots
    os.makedirs('outputs/visuals', exist_ok=True)
    
    base_data = 'data' if os.path.isdir('data') else '.'
    train_path = os.path.join(base_data, 'train')
    test_path = os.path.join(base_data, 'test')

    if not os.path.isdir(train_path):
        print(f"❌ Error: {train_path} not found.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Dataset & Training Logic
    full_dataset = datasets.ImageFolder(train_path, transform=transform)
    train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
    
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print("\n📊 Training and Calculating Accuracy...")
    all_preds, all_labels = [], []
    
    model.train()
    pbar = tqdm(train_loader, desc="Training Epoch")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device).float().view(-1, 1)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Track accuracy
        preds = torch.sigmoid(outputs) > 0.5
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    train_acc = accuracy_score(all_labels, all_preds)

    # 3. Generate Visualizations (Replace Old Ones)
    print("\n📈 Generating Visuals...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix (Train Acc: {train_acc:.2%})")
    plt.savefig('outputs/visuals/confusion_matrix.png')
    plt.close()

    # 4. Predict Test Set
    print("\n📝 Predicting Test Set...")
    test_results = []
    if os.path.isdir(test_path):
        test_files = sorted([f for f in os.listdir(test_path) if f.endswith('.png')])
        model.eval()
        with torch.no_grad():
            for filename in tqdm(test_files, desc="Processing"):
                try:
                    img_id = int(filename.split('_')[1].split('.')[0])
                    img = Image.open(os.path.join(test_path, filename)).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(device)
                    output = torch.sigmoid(model(img_t))
                    test_results.append({'id': img_id, 'label': 1 if output.item() > 0.5 else 0})
                except: continue
    
    # 5. Save Outputs & Final Report
    df_sub = pd.DataFrame(test_results)
    df_sub.to_csv('submission.csv', index=False)
    
    with open('REPORT.md', 'w') as f:
        f.write("# Medical Image Classification Final Report\n\n")
        f.write("## Performance Metrics\n")
        f.write(f"- **Final Training Accuracy**: {train_acc:.2%}\n")
        f.write(f"- **Total Test Predictions**: {len(test_results)}\n\n")
        f.write("## Visualizations\n")
        f.write("![Confusion Matrix](outputs/visuals/confusion_matrix.png)\n")

    print(f"\n✅ Pipeline Complete. Accuracy: {train_acc:.2%}")

if __name__ == "__main__":
    main()
