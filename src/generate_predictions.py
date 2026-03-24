import os
import pandas as pd
import torch
import numpy as np

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data source
DATA_PATH = os.path.join(BASE_DIR, "train_label.csv") 

# Output destination: data/submissions
SUBMISSION_DIR = os.path.join(BASE_DIR, "..", "data", "submissions")
OUTPUT_NAME = "prediction_test_data.csv"
OUTPUT_PATH = os.path.join(SUBMISSION_DIR, OUTPUT_NAME)

def run_full_inference():
    # Ensure the directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data not found at {DATA_PATH}")
        return
    
    # Load entire dataset
    df = pd.read_csv(DATA_PATH)
    num_samples = len(df)
    print(f"--- Processing {num_samples} images ---")
    
    # Placeholder for Ensemble Logic (XGBoost + Clinical Biomarkers)
    # Applying to the full column
    mock_probabilities = np.clip(df['biomarker_value'] + np.random.normal(0, 0.05, num_samples), 0, 1)
    
    # Final Output Formatting
    output_df = pd.DataFrame({
        'id': df['id'],
        'prediction_label': (mock_probabilities > 0.5).astype(int),
        'biomarker_score': mock_probabilities.round(4)
    })
    
    # Export to the submissions folder
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_full_inference()
