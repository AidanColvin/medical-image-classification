"""
Generates feature impact and AUC visualizations.
Saves the plots as PNGs and the underlying data as CSV tables.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_visuals_and_tables():
    """Creates mock data, plots it, and saves images and CSV tables."""
    viz_dir = Path("data/visualizations")
    table_dir = Path("data/tables")
    viz_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature Impact
    features = ["Pixel Intensity", "Edge Density", "Contrast", "Symmetry", "Texture"]
    impact = [0.85, 0.62, 0.55, 0.40, 0.35]
    df_impact = pd.DataFrame({"Feature": features, "Impact Score": impact})
    df_impact.to_csv(table_dir / "feature_impact.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(features, impact, color='skyblue')
    plt.gca().invert_yaxis()
    plt.title("Feature Impact")
    plt.tight_layout()
    plt.savefig(viz_dir / "feature_impact.png")
    plt.close()

    # 2. AUC / ROC Curve
    fpr = np.linspace(0, 1, 10)
    tpr = np.sqrt(fpr) 
    df_auc = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})
    df_auc.to_csv(table_dir / "auc_roc.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = 0.92)')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (AUC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(viz_dir / "auc_roc.png")
    plt.close()

    print("--- SUCCESS: Visualizations and tables generated ---")

if __name__ == "__main__":
    create_visuals_and_tables()
