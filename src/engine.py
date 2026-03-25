import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from PIL import Image

def run_pipeline(train_path, test_path, device):
    vis_dir = 'data/visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs('data/submissions', exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Data Loading logic
    full_dataset = datasets.ImageFolder(train_path, transform=transform)
    dataset = Subset(full_dataset, range(len(full_dataset)))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    print(f"Training for 5 Epochs...")
    history = {'loss': [], 'acc': []}
    for epoch in range(5):
        model.train()
        all_probs, all_labels = [], []
        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/5"):
            imgs, labels = imgs.to(device), labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            all_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_probs])
        history['acc'].append(epoch_acc)
        history['loss'].append(loss.item())

    # Visualizations
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
    plt.savefig(f'{vis_dir}/roc_curve.png')
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(range(1, 6), history['loss'], label='Loss')
    plt.plot(range(1, 6), history['acc'], label='Accuracy')
    plt.legend()
    plt.savefig(f'{vis_dir}/metrics_curve.png')
    plt.close()

    # Inference (Fixed to ensure results are captured)
    results = []
    print(f"Running inference on: {test_path}")
    if os.path.isdir(test_path):
        model.eval()
        files = [f for f in os.listdir(test_path) if f.endswith('.png')]
        for f in tqdm(files, desc="Inference"):
            try:
                # Expecting format test_<id>.png
                img_id = int(f.split('_')[1].split('.')[0])
                img = Image.open(os.path.join(test_path, f)).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = torch.sigmoid(model(img_t)).item()
                results.append({'id': img_id, 'label': 1 if prob > 0.5 else 0})
            except Exception as e:
                continue

    # Final Save Logic
    df_sub = pd.DataFrame(results).sort_values('id') if results else pd.DataFrame(columns=['id', 'label'])
    df_sub.to_csv('submission.csv', index=False)
    df_sub.to_csv('data/submissions/final_results_table.csv', index=False)

    return history['acc'][-1], auc(fpr, tpr), len(results)
