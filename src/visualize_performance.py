import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

# Pathing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_PATH = os.path.join(BASE_DIR, "..", "data", "submissions", "prediction_test_data.csv")
PLOT_PATH = os.path.join(BASE_DIR, "..", "data", "submissions", "roc_curve.png")

def plot_roc():
    if not os.path.exists(PRED_PATH):
        print("Error: Prediction file not found.")
        return

    df = pd.read_csv(PRED_PATH)
    
    # Calculate ROC metrics
    fpr, tpr, _ = roc_curve(df['actual_class'], df['biomarker_score'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Ensemble Model)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.savefig(PLOT_PATH)
    print(f"ROC Curve saved to: {PLOT_PATH}")

if __name__ == "__main__":
    plot_roc()
