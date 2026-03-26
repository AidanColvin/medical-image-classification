import pandas as pd
import glob

def ensemble_csvs(file_pattern, output_file='ensembled_submission.csv'):
    files = glob.glob(file_pattern)
    if not files:
        print("No files found matching pattern.")
        return

    # Load all submission dataframes
    dfs = [pd.read_csv(f) for f in files]
    
    # Average the raw probabilities
    ensemble_df = dfs[0].copy()
    ensemble_df['label'] = sum(df['label'] for df in dfs) / len(dfs)
    
    # Apply standard threshold (or update to use optimized threshold)
    ensemble_df['label'] = (ensemble_df['label'] >= 0.5).astype(int)
    
    ensemble_df.to_csv(output_file, index=False)
    print(f"Ensemble saved to {output_file}")

if __name__ == "__main__":
    # Ensure your fold predictions are saved as probabilities, not hard integers, before ensembling
    ensemble_csvs('submission_fold_*.csv')
