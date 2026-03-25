import os
import torch
from src.engine import run_pipeline

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_path = './data/raw/test/train'
    test_path = './test'
    
    print(f"Starting pipeline on device: {device}")
    acc, auc_score, count = run_pipeline(train_path, test_path, device)

    with open('REPORT.md', 'w') as f:
        f.write("# Medical Image Classification Report\n\n")
        f.write("## Performance Metrics\n")
        f.write(f"| Metric | Result |\n| :--- | :--- |\n| Accuracy | **{acc:.2%}** |\n| AUC | **{auc_score:.2f}** |\n| Test Images Processed | **{count}** |\n\n")
        f.write("## Visualizations\n")
        f.write("### Confusion Matrix\n![CM](data/visualizations/confusion_matrix.png)\n\n")
        f.write("### ROC Curve\n![ROC](data/visualizations/roc_curve.png)\n\n")
        f.write("### Training Metrics\n![Metrics](data/visualizations/metrics_curve.png)\n")

    print(f"Re-run complete. Accuracy: {acc:.2%}")
    print(f"Submission count: {count}")

if __name__ == "__main__":
    main()
