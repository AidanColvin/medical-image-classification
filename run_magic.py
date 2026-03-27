import torch
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from src.engine import get_model  # Assumes your model loader is here

def main():
    """
    Runs FAANG-level inference on 624 test images.
    Uses trained model to generate actual predictions.
    Saves submission_v14.csv to root.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing pipeline on: {device}")

    # 1. Load Model
    model = get_model()
    ckpt = 'data/models/best_model.pth'
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        print("Error: Model weights not found.")
        return
    model.to(device).eval()

    # 2. Setup Transform (Standard ImageNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Process Test Set
    test_dir = "./data/raw/test/test"
    images = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    results = []
    with torch.no_grad():
        for i, img_name in enumerate(images):
            img_path = os.path.join(test_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            
            # Generate actual prediction
            output = model(tensor)
            prob = torch.sigmoid(output).item()
            label = 1 if prob >= 0.5 else 0
            
            results.append({"id": i, "label": label})

    # 4. Save EXACT format
    df = pd.DataFrame(results)
    df[['id', 'label']].to_csv('submission_v14.csv', index=False)
    print("Success: submission_v14.csv saved to root.")

if __name__ == "__main__":
    main()
