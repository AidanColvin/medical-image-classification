import pandas as pd
import glob
import os
from src.utils import get_versioned_path

def ensemble_csvs(file_pattern, output_path=None):
    files = sorted(glob.glob(file_pattern))
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return

    dfs = [pd.read_csv(f) for f in files]
    ensemble_df = dfs[0].copy()
    ensemble_df['label'] = sum(df['label'] for df in dfs) / len(dfs)
    ensemble_df['label'] = (ensemble_df['label'] >= 0.5).astype(int)
    
    final_path = output_path if output_path else get_versioned_path("submission.csv")
    ensemble_df[['id', 'label']].to_csv(final_path, index=False)
    print(f"Success: Saved to {final_path}")

if __name__ == "__main__":
    ensemble_csvs('submission_fold_*.csv')
