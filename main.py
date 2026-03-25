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
    print(f"🚀 Running on: {device}")
    
    # 1. FIND DATA (Bulletproof search)
    possible_train = ['data/train', 'data', 'train']
    train_path = next((p for p in possible_train if os.path.isdir(p) and any(os.path.isdir(os.path.join(p, d)) for d in os.listdir(p))), None)
    
    if not train_path:
        print(f"❌ Error: Could not find a folder with class subdirectories. Dirs: {os.listdir('.')}")
        return
    
    test_path = train_path.replace('train', 'test') if 'train' in train_path else 'data/test'
    os.makedirs('outputs/visuals', exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. LOAD & TRAIN
    dataset = datasets.ImageFolder(train_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print(f"📊 Training on {train_path}...")
    all_labels, all_probs = [], []
    
    model.train()
    for _ in range(1): # 1 Epoch for speed
        for imgs, labels in tqdm(loader, desc="Training"):
            imgs, labels = imgs.to(device), labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            all_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 3. STATS & VISUALS
    all_preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, all_preds)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig('outputs/visuals/confusion_matrix.png')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.savefig('outputs/visuals/roc_curve.png')

    # 4. PREDICT TEST SET
    print("📝 Predicting Test Data...")
    results = []
    if os.path.isdir(test_path):
        for f in tqdm(os.listdir(test_path), desc="Testing"):
            if f.endswith('.png'):
                try:
                    img_id = int(f.split('_')[1].split('.')[0])
                    img = Image.open(os.path.join(test_path, f)).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(device)
                    model.eval()
                    with torch.no_grad():
                        prob = torch.sigmoid(model(img_t)).item()
                    results.append({'id': img_id, 'label': 1 if prob > 0.5 else 0})
                except: continue

    # 5. SAVE & REPORT
    pd.DataFrame(results).sort_values('id').to_csv('submission.csv', index=False)
    with open('REPORT.md', 'w') as f:
        f.write(f"# Medical Classification Report\n\n- **Accuracy**: {acc:.2%}\n- **AUC**: {roc_auc:.2f}\n\n")
        f.write("![Confusion Matrix](outputs/visuals/confusion_matrix.png)\n")
        f.write("![ROC Curve](outputs/visuals/roc_curve.png)\n")
    
    print(f"✅ Finished! Acc: {acc:.2%}. Files saved to root.")

if __name__ == "__main__":
    main()
