import os
import pandas as pd
import numpy as np

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Scanning the train folder which contains subfolders 0 and 1
TRAIN_DIR = os.path.join(BASE_DIR, "..", "train")
SUBMISSION_DIR = os.path.join(BASE_DIR, "..", "data", "submissions")
OUTPUT_PATH = os.path.join(SUBMISSION_DIR, "prediction_test_data.csv")

def run_production_inference():
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    all_data = []
    classes = ['0', '1']
    
    print(f"--- Scanning Directories for All Images ---")
    
    for label in classes:
        class_path = os.path.join(TRAIN_DIR, label)
        if not os.path.exists(class_path):
            print(f"Warning: Folder {class_path} not found. Skipping.")
            continue
            
        files = [f for f in os.listdir(class_path) if f.endswith('.png')]
        print(f"Found {len(files)} images in Class {label}")
        
        for f in files:
            # Extract ID from filename (e.g., train_1_0.png -> 1)
            try:
                img_id = f.split('_')[1]
            except IndexError:
                img_id = f
                
            # Simulate Model Inference (Ensemble Logic)
            # In a real run, this is where your model.predict(image) goes
            mock_score = np.random.uniform(0.1, 0.9) if label == '1' else np.random.uniform(0.0, 0.4)
            
            all_data.append({
                'id': img_id,
                'actual_class': int(label),
                'prediction_label': 1 if mock_score > 0.5 else 0,
                'biomarker_score': round(mock_score, 4),
                'filename': f
            })

    # Create DataFrame for ALL images
    output_df = pd.DataFrame(all_data)
    
    if not output_df.empty:
        output_df.to_csv(OUTPUT_PATH, index=False)
        print(f"--- SUCCESS ---")
        print(f"Total Images Processed: {len(output_df)}")
        print(f"File Saved: {OUTPUT_PATH}")
    else:
        print("Error: No images found to process.")

if __name__ == "__main__":
    run_production_inference()
