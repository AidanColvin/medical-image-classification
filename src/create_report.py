import pandas as pd

def generate_markdown():
    impact_data = pd.read_csv('impact_table.csv').to_markdown(index=False)
    auc_data = pd.read_csv('auc_table.csv').to_markdown(index=False)

    report_content = f"""# Project Analysis Report
    
## Executive Summary
I developed this model to classify status based on the provided clinical dataset. The following sections detail the performance and feature dependencies I identified during testing.

## Feature Impact
I replaced the standard Predictor Strength metrics with Permutation Importance to better capture how each feature affects model error.

![Feature Impact](feature_impact.png)

### Feature Importance Data
{impact_data}

---

## Model Performance (AUC)
To validate the classification thresholds, I generated a Receiver Operating Characteristic curve. This visualizes the trade-off between sensitivity and specificity.

![AUC Curve](auc_curve.png)

### Performance Metrics
{auc_data}
"""
    with open('REPORT.md', 'w') as f:
        f.write(report_content)

if __name__ == "__main__":
    generate_markdown()
    print("REPORT.md has been created with 'I' terminology.")
