import os
import torch
import pandas as pd
from src.engine import run_pipeline

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # PATH FIX: Pointing to where the images actually are
    train_path = './data/raw/test/train'
    test_path = './data/raw/test/test' 
    
    print(f"Starting pipeline on device: {device}")
    acc, auc_score, count = run_pipeline(train_path, test_path, device)

    with open('REPORT.md', 'w') as f:
        f.write("# Medical Image Classification Report\n\n")
        f.write("## Performance Summary\n")
        f.write(f"| Metric | Result |\n| :--- | :--- |\n| Accuracy | **{acc:.2%}** |\n| AUC | **{auc_score:.2f}** |\n| Test Images Processed | **{count}** |\n\n")

        f.write("## Visualizations\n")
        f.write("### Training Progress\n")
        f.write("![Loss Curve](data/visualizations/loss_curve.png)\n")
        f.write("![Metrics Curve](data/visualizations/metrics_curve.png)\n\n")

        f.write("### Model Performance Analysis\n")
        f.write("![Confusion Matrix](data/visualizations/confusion_matrix.png)\n")
        f.write("![ROC Curve](data/visualizations/roc_curve.png)\n")
        f.write("![PR Curve](data/visualizations/pr_curve.png)\n\n")

        f.write("## Data Tables\n")
        f.write("### Final Results Sample\n")
        
        table_path = 'data/submissions/final_results_table.csv'
        if os.path.exists(table_path) and os.path.getsize(table_path) > 0:
            sample_df = pd.read_csv(table_path).head(5)
            f.write(sample_df.to_markdown(index=False))
        else:
            f.write("_No prediction data available yet._")

    print(f"Re-run complete. Accuracy: {acc:.2%}")
    print(f"Submission count: {count}")

if __name__ == "__main__":
    main()
