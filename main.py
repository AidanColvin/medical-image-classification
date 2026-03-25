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

def find_dir(name):
    for root, dirs, files in os.walk('.'):
        if name in dirs:
            # Verify it has class folders 0 and 1
            full_path = os.path.join(root, name)
            if all(d in os.listdir(full_path) for d in ['0', '1']):
                return full_path
    return None

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    train_path = find_dir('train')
    test_path = find_dir('test') or './test'

    if not train_path:
        print(f"❌ Error: Could not find 'train' folder with '0' and '1' subfolders.")
        print(f"Current structure: {os.listdir('.')}")
        return

    print(f"📂 Found Train: {train_path} | Test: {test_path}")
    os.makedirs('outputs/visuals', exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. Load & Filter Dataset
    full_dataset = datasets.ImageFolder(train_path, transform=transform)
    valid_indices = [i for i, (path, label) in enumerate(full_dataset.imgs) 
                     if os.path.basename(os.path.dirname(path)) in ['0', '1']]
    dataset = Subset(full_dataset, valid_indices)

    # 2. Train (5 Epochs)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(5):
        model.train()
        all_probs, all_labels = [], []
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/5")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            all_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 3. Metrics & Visuals
    all_preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Blues')
    plt.savefig('outputs/visuals/confusion_matrix.png')
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.savefig('outputs/visuals/roc_curve.png')

    # 4. Predict Test
    results = []
    if os.path.isdir(test_path):
        model.eval()
        for f in tqdm(os.listdir(test_path), desc="Final Testing"):
            if f.endswith('.png'):
                try:
                    img_id = int(f.split('_')[1].split('.')[0])
                    img = Image.open(os.path.join(test_path, f)).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        prob = torch.sigmoid(model(img_t)).item()
                    results.append({'id': img_id, 'label': 1 if prob > 0.5 else 0})
                except: continue

    pd.DataFrame(results).sort_values('id').to_csv('submission.csv', index=False)
    with open('REPORT.md', 'w') as f:
        f.write(f"# Final Medical Classification Report\n\n")
        f.write(f"- **Training Accuracy**: {acc:.2%}\n- **AUC**: {roc_auc:.2f}\n\n")
        f.write("![Confusion Matrix](outputs/visuals/confusion_matrix.png)\n")
        f.write("![ROC Curve](outputs/visuals/roc_curve.png)\n")
    
    print(f"\n✅ Finished. Acc: {acc:.2%}")

if __name__ == "__main__":
    main()
