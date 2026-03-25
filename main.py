import os
import torch
from src.engine import run_pipeline

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Path Resolution - using your specific path found in logs
    train_path = './data/raw/test/train' 
    test_path = './test'
    
    if not os.path.isdir(train_path):
        print(f"❌ Error: {train_path} not found. Ensure data is extracted correctly.")
        return

    acc, auc_score, test_count = run_pipeline(train_path, test_path, device)

    # Generate Comprehensive REPORT.md
    with open('REPORT.md', 'w') as f:
        f.write("# 🔬 Medical Image Classification Performance Report\n\n")
        f.write("## 📈 Key Metrics\n")
        f.write(f"| Metric | Value |\n| :--- | :--- |\n| Training Accuracy | **{acc:.2%}** |\n| ROC AUC Score | **{auc_score:.2f}** |\n| Test Samples Predicted | {test_count} |\n\n")
        f.write("## 🖼️ Model Visualizations\n")
        f.write("### Confusion Matrix\n![CM](visuals/confusion_matrix.png)\n\n")
        f.write("### ROC Curve\n![ROC](visuals/roc_curve.png)\n")

    print(f"\n✅ Pipeline Complete.")
    print(f"📊 Accuracy: {acc:.2%}")
    print(f"📂 Visuals: /visuals | Submission: /data/submissions/submission.csv")

if __name__ == "__main__":
    main()
