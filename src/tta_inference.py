import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import os
from src.model import MedicalModel # Adjust based on your actual model class name

def tta_predict(model, image_path, device):
    img = Image.open(image_path).convert('RGB')
    
    # Define TTA transforms: Original, Horizontal Flip, Vertical Flip, and small rotation
    tta_transforms = [
        transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
        transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor()]),
        transforms.Compose([transforms.Resize((224, 224)), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor()]),
        transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation(15), transforms.ToTensor()])
    ]
    
    probs = []
    model.eval()
    with torch.no_grad():
        for t in tta_transforms:
            input_tensor = t(img).unsqueeze(0).to(device)
            logit = model(input_tensor)
            prob = torch.sigmoid(logit).item()
            probs.append(prob)
            
    return sum(probs) / len(probs)

# Run this script to generate a TTA-boosted submission
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MedicalModel().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    
    test_dir = 'data/test'
    results = []
    for img_name in sorted(os.listdir(test_dir)):
        prob = tta_predict(model, os.path.join(test_dir, img_name), device)
        results.append({'id': img_name.split('.')[0], 'label': prob})
    
    pd.DataFrame(results).to_csv('submission_tta.csv', index=False)
    print("TTA Submission generated.")
