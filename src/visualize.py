import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.inspection import permutation_importance

def save_visuals(model, X_test, y_test, feature_names):
    # 1. Feature Impact (Permutation Importance)
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    sorted_idx = result.importances_mean.argsort()
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title("Feature Impact (Permutation Importance)")
    plt.tight_layout()
    plt.savefig('feature_impact.png')
    
    # Save table data for report
    impact_df = pd.DataFrame({'Feature': feature_names, 'Importance': result.importances_mean})
    impact_df.sort_values(by='Importance', ascending=False).to_csv('impact_table.csv', index=False)

    # 2. AUC Visualization
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (AUC)')
    plt.legend(loc="lower right")
    plt.savefig('auc_curve.png')
    
    # Save AUC metrics for report table
    pd.DataFrame({'Metric': ['AUC Score'], 'Value': [roc_auc]}).to_csv('auc_table.csv', index=False)

if __name__ == "__main__":
    print("Visualizations and data tables generated.")
