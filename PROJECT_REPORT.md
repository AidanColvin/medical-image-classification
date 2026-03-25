# Medical Image Classification: 5-Fold CV Report
**Status:** Full Re-run Complete

## Parameter Impact (Fixed)
The visualization below replaces the raw layer indices with descriptive component names.
![Impact](data/visualizations/parameter_impact.png)

## Model Performance
![AUC](data/visualizations/auc_curve.png)

### Impact Data Table
| Feature/Layer                 |   Impact Score |
|:------------------------------|---------------:|
| Initial Features (Conv1)      |      0.100333  |
| Final Classifier (Dense2)     |      0.0590203 |
| Mid-Level Patterns (Conv2)    |      0.0531388 |
| Complex Associations (Dense1) |      0.0157678 |
