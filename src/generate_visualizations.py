import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from PIL import Image

# Pathing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_PATH = os.path.join(BASE_DIR, "..", "data", "submissions", "prediction_test_data.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "..", "train")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "submissions")

def generate_all_plots():
    if not os.path.exists(PRED_PATH):
        print("Error: Run generate_predictions.py first.")
        return

    df = pd.read_csv(PRED_PATH)
    y_true = df['actual_class']
    y_score = df['biomarker_score']
    y_pred = df['prediction_label']
    
    # 1. ROC CURVE
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#3498DB', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='#BDC3C7', linestyle='--')
    plt.title("Model Performance: ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plt.close()

    # 2. CONFUSION MATRIX
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    # 3. SAMPLE PREDICTIONS GRID
    # Picking 4 correct and 4 incorrect samples to show
    labels_map = {0: "Healthy", 1: "Pneumonia"}
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    
    # Simple logic to find samples in your local folders
    for row, title in enumerate(["Top Correct Predictions", "Potential Misclassifications"]):
        # Filter for logic
        if row == 0:
            subset = df[df['actual_class'] == df['prediction_label']].head(4)
        else:
            subset = df[df['actual_class'] != df['prediction_label']].head(4)
            
        for col, (_, data) in enumerate(subset.iterrows()):
            ax = axes[row][col]
            # Search for the image in its class folder
            img_name = f"train_{data['id']}_{data['actual_class']}.png"
            img_path = os.path.join(TRAIN_DIR, str(data['actual_class']), img_name)
            
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img, cmap="gray")
            except:
                ax.text(0.5, 0.5, "Image Not Found", ha='center')

            color = "#2ECC71" if row == 0 else "#E74C3C"
            ax.set_title(f"ID: {data['id']}\nTrue: {labels_map[data['actual_class']]}\nPred: {labels_map[data['prediction_label']]}", 
                         fontsize=9, color=color)
            ax.axis("off")
            
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_predictions.png"), dpi=150)
    plt.close()
    
    print(f"--- Visualizations Generated in {OUTPUT_DIR} ---")
    print(f"1. roc_curve.png")
    print(f"2. confusion_matrix.png")
    print(f"3. sample_predictions.png")

if __name__ == "__main__":
    generate_all_plots()
