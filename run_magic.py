import torch
import pandas as pd
import os

def main():
    """
    Automated Inference Pipeline v13.
    Saves formatted results to root and data folders.
    """
    # Force exact filename for root as requested
    root_filename = "submission_v13.csv"
    data_folder_path = "data/submissions/submission_v13.csv"
    os.makedirs("data/submissions", exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing v13 pipeline on: {device}")

    test_dir = "./data/raw/test/test"
    images = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    results = []
    # Placeholder for your model.forward logic
    for i, _ in enumerate(images):
        results.append({"id": i, "label": 1}) 

    df = pd.DataFrame(results)
    
    # Save to both locations for redundancy and organization
    df[['id', 'label']].to_csv(root_filename, index=False)
    df[['id', 'label']].to_csv(data_folder_path, index=False)
    
    print(f"[SUCCESS] Saved to root: {root_filename}")
    print(f"[SUCCESS] Saved to folder: {data_folder_path}")

if __name__ == "__main__":
    main()
