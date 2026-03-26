import pandas as pd
import glob
from src.utils import get_versioned_path

def rank_blend(file_pattern):
    files = glob.glob(file_pattern)
    dfs = [pd.read_csv(f) for f in files]
    
    # Convert labels to ranks
    for df in dfs:
        df['rank'] = df['label'].rank()
        
    # Average the ranks
    final_df = dfs[0].copy()
    final_df['rank_avg'] = sum(df['rank'] for df in dfs) / len(dfs)
    
    # Normalize back to 0-1 for submission
    final_df['label'] = final_df['rank_avg'] / final_df['rank_avg'].max()
    final_df['label'] = (final_df['label'] >= 0.5).astype(int)
    
    out = get_versioned_path('submission_ranked.csv')
    final_df[['id', 'label']].to_csv(out, index=False)
    print(f"Rank-blended submission saved to {out}")

if __name__ == "__main__":
    rank_blend('submission_fold_*.csv')
