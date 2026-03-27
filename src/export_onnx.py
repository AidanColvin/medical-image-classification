import torch
import os
from src.train_and_predict import get_resnet_model

device = torch.device("cpu") # Export requires CPU
model = get_resnet_model(device)
model_path = "data/models/best_model.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create a dummy input that matches ResNet18 expected size
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # Export to the docs folder so GitHub Pages can serve it
    os.makedirs("docs", exist_ok=True)
    torch.onnx.export(
        model, 
        dummy_input, 
        "docs/model.onnx", 
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print("[SUCCESS] Model exported to docs/model.onnx")
else:
    print("[ERROR] best_model.pth not found.")
