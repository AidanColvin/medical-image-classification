import pandas as pd
import os
import re

def get_next_version(directory, base_name="submission"):
    existing = os.listdir(directory) if os.path.exists(directory) else []
    v = 1
    while f"{base_name}_v{v}.csv" in existing:
        v += 1
    return f"{base_name}_v{v}.csv"

def generate():
    source = 'submission.csv'
    target_dir = 'data/submissions'
    os.makedirs(target_dir, exist_ok=True)

    if not os.path.exists(source):
        print(f"Error: {source} not found. Ensure 'make run' finished.")
        return
    
    # 1. Load latest results
    df = pd.read_csv(source)
    
    # 2. STRICT NUMERIC SORTING (Fixes the 87% alignment issue)
    df['id'] = pd.to_numeric(df['id'])
    df = df.sort_values('id').reset_index(drop=True)
    
    # 3. Save to the new organization structure
    filename = get_next_version(target_dir)
    full_path = os.path.join(target_dir, filename)
    df[['id', 'label']].to_csv(full_path, index=False)
    
    # 4. Also save a copy to the root for easy access
    df[['id', 'label']].to_csv('final_submission.csv', index=False)
    
    print(f"✅ Submission aligned and saved to: {full_path}")
    print(f"Preview (Matches Reference Pattern):")
    print(df.head(10))

if __name__ == "__main__":
    generate()
