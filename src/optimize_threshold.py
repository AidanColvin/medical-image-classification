import numpy as np
from sklearn.metrics import f1_score

def find_optimal_threshold(y_true, y_prob):
    """Finds the best classification threshold based on validation probabilities."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0.0

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = thresh

    print(f"Optimal Threshold: {best_threshold:.4f} (F1 Score: {best_score:.4f})")
    return best_threshold
