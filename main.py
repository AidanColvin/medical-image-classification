import os
import torch
from src.engine import run_pipeline

"""
Medical Image Classification Pipeline
------------------------------------
Main entry point for training the ResNet18 model on chest X-ray data 
and generating a comprehensive performance report.
"""

def main():
    # Set device to MPS for Mac M-series chips, otherwise CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Define dataset paths
    # Note: Using the specific sub-path provided in previous training logs
    train_path = './data/raw/test/train'
    test_path = './test'
    
    print(f"🚀 Starting pipeline on device: {device}")

    # Execute the training and inference pipeline
    # Returns final metrics and count of processed test images
    acc, auc_score, count = run_pipeline(train_path, test_path, device)

    # Generate the Markdown Report for GitHub UI
    with open('REPORT.md', 'w') as f:
        f.write("# 🔬 Medical Image Classification Report\n\n")
        f.write("## 📊 Performance Metrics\n")
        f.write(f"| Metric | Result |\n| :--- | :--- |\n| Accuracy | **{acc:.2%}** |\n| AUC | **{auc_score:.2f}** |\n| Test Images Processed | **{count}** |\n\n")
        
        f.write("## 🖼️ Visualizations\n")
        f.write("### Confusion Matrix\n")
        f.write("![Confusion Matrix](data/visualizations/confusion_matrix.png)\n\n")
        f.write("### ROC Curve\n")
        f.write("![ROC Curve](data/visualizations/roc_curve.png)\n\n")
        f.write("### Training Metrics\n")
        f.write("![Metrics Curve](data/visualizations/metrics_curve.png)\n")

    print(f"✅ Re-run complete. Accuracy: {acc:.2%}")
    print(f"📝 Report generated: REPORT.md")

if __name__ == "__main__":
    main()
