import pandas as pd
import glob
import os

def get_next_versioned_filename(base_name="submission", ext="csv"):
    """Returns the next available versioned filename."""
    # Check if the base file exists
    first_file = f"{base_name}.{ext}"
    if not os.path.exists(first_file):
        return first_file
    
    # Check for versioned files
    i = 2
    while os.path.exists(f"{base_name}_v{i}.{ext}"):
        i += 1
    return f"{base_name}_v{i}.{ext}"

def ensemble_csvs(file_pattern):
    files = sorted(glob.glob(file_pattern))
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return

    print(f"Ensembling {len(files)} files...")
    dfs = [pd.read_csv(f) for f in files]
    
    ensemble_df = dfs[0].copy()
    # Average the probabilities across all folds
    ensemble_df['label'] = sum(df['label'] for df in dfs) / len(dfs)
    
    # Binary classification threshold
    ensemble_df['label'] = (ensemble_df['label'] >= 0.5).astype(int)
    
    output_path = get_next_versioned_filename()
    ensemble_df.to_csv(output_path, index=False)
    print(f"Success: New submission saved as {output_path}")

if __name__ == "__main__":
    # Aggregates all fold-specific CSVs
    ensemble_csvs('submission_fold_*.csv')
