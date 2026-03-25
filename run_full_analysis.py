import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def setup():
    print("--- Cleaning Repo & Organizing Data ---")
    # This ensures old tables/plots are DELETED before new ones are made
    if os.path.exists('results'):
        shutil.rmtree('results')
    os.makedirs('results', exist_ok=True)
    
    # Organize images into folders PyTorch understands
    for label in ['0', '1']:
        os.makedirs(f'data/train/{label}', exist_ok=True)
    
    if os.path.exists('train'):
        for filename in os.listdir('train'):
            if filename.endswith('.png'):
                label = filename.split('_')[-1].split('.')[0]
                shutil.move(os.path.join('train', filename), f'data/train/{label}/{filename}')
    print("Data ready. Starting fresh training...")

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    
    train_dataset = datasets.ImageFolder('data/train', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Simple Model for the run
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3), nn.ReLU(), nn.Flatten(), nn.Linear(16*62*62, 2)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(3):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/3")
        epoch_loss = 0
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss/len(train_loader))

    # Save fresh visualization (this replaces any existing file)
    plt.figure(figsize=(10,5))
    plt.plot(losses, label='Training Loss')
    plt.title('Final Results: Full Dataset (No Zip)')
    plt.savefig('results/loss_plot.png') 
    print("Success: Fresh results saved in results/loss_plot.png")

if __name__ == "__main__":
    setup()
    train()
