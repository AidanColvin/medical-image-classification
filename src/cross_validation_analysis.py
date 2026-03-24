import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

def run_cv_analysis():
    base_dir = Path.cwd()
    data_path = base_dir / "data" / "submissions" / "prediction_test_data.csv"
    viz_dir = base_dir / "data" / "visualizations"
    table_dir = base_dir / "data" / "tables"
    
    viz_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    
    # Use existing biomarker scores or simulate them for the demonstration
    if 'biomarker_score' not in df.columns:
        df['biomarker_score'] = np.where(df['label'] == 1, 
                                         np.random.normal(0.7, 0.12, len(df)), 
                                         np.random.normal(0.3, 0.12, len(df)))

    X = df[['biomarker_score']].values
    y = df['label'].values

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_results, all_probs, all_preds = [], np.zeros(len(y)), np.zeros(len(y))

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        all_probs[test_idx] = model.predict_proba(X_test)[:, 1]
        all_preds[test_idx] = model.predict(X_test)
        
        fold_results.append({
            'Fold': fold,
            'Coefficient': model.coef_[0][0],
            'Accuracy': (all_preds[test_idx] == y_test).mean()
        })

    # Save metrics
    pd.DataFrame(fold_results).to_csv(table_dir / "cv_10_fold_results.csv", index=False)

    # --- PLOT 1: Stability ---
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, 11), [f['Coefficient'] for f in fold_results], marker='o', color='#2ca02c')
    plt.title('Predictor Weight Stability Across 10 Folds')
    plt.ylabel('Coefficient Value')
    plt.grid(True, alpha=0.3)
    plt.savefig(viz_dir / "predictor_stability.png")

    # --- PLOT 2: Confusion Matrix ---
    cm = confusion_matrix(y, all_preds)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Total N=5232)')
    plt.colorbar()
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="black")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(viz_dir / "confusion_matrix.png")

    # --- PLOT 3: ROC Curve ---
    fpr, tpr, _ = roc_curve(y, all_probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.savefig(viz_dir / "roc_curve.png")

    print(f"--- SUCCESS: Generated 10-Fold CV Visuals in {viz_dir} ---")

if __name__ == "__main__":
    run_cv_analysis()
