import pandas as pd
import os
import re

def get_next_version(directory, base_name="submission"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    existing = os.listdir(directory)
    v = 1
    while f"{base_name}_v{v}.csv" in existing:
        v += 1
    return f"{base_name}_v{v}.csv"

def generate():
    # Looking for the output from your main.py
    source = 'submission.csv'
    target_dir = 'data/submissions'
    
    if not os.path.exists(source):
        print(f"Error: {source} not found. Run main.py first.")
        return
    
    # Load and force numerical sorting (0, 1, 2... instead of 1, 10, 2)
    df = pd.read_csv(source)
    df['id'] = pd.to_numeric(df['id'])
    df = df.sort_values('id').reset_index(drop=True)
    
    # Save versioned copy in data/submissions
    filename = get_next_version(target_dir)
    full_path = os.path.join(target_dir, filename)
    df[['id', 'label']].to_csv(full_path, index=False)
    
    # Keep a copy of the latest in root for easy verification/submission
    df[['id', 'label']].to_csv('final_submission.csv', index=False)
    
    print(f"✅ Logic Refined. Saved to: {full_path}")
    print(f"✅ Root copy created: final_submission.csv")

if __name__ == "__main__":
    generate()
