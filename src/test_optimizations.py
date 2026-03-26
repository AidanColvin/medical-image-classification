import numpy as np
import pandas as pd
from optimize_threshold import find_optimal_threshold
from ensemble_submissions import ensemble_csvs

def test_find_optimal_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    
    optimal = find_optimal_threshold(y_true, y_prob)
    
    # Any threshold between 0.41 and 0.60 yields a perfect F1 score here
    assert 0.4 < optimal <= 0.61

def test_ensemble_csvs(tmp_path):
    # Setup mock fold probabilities
    df1 = pd.DataFrame({'id': [0, 1], 'label': [0.2, 0.8]})
    df2 = pd.DataFrame({'id': [0, 1], 'label': [0.4, 0.6]})
    
    df1.to_csv(tmp_path / 'submission_fold_1.csv', index=False)
    df2.to_csv(tmp_path / 'submission_fold_2.csv', index=False)
    
    out_file = tmp_path / 'ensembled.csv'
    
    # Run ensembling logic
    ensemble_csvs(str(tmp_path / 'submission_fold_*.csv'), str(out_file))
    res = pd.read_csv(out_file)
    
    # Assert id 0 average is 0.3 -> class 0
    assert res['label'].iloc[0] == 0
    # Assert id 1 average is 0.7 -> class 1
    assert res['label'].iloc[1] == 1
