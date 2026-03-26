import pytest
import torch
import numpy as np
import pandas as pd
import os
from optimize_threshold import find_optimal_threshold
from ensemble_submissions import ensemble_csvs

# --- 1-5: Threshold Optimization Tests ---
def test_threshold_perfect_split():
    y_true, y_prob = np.array([0, 1]), np.array([0.1, 0.9])
    assert 0.1 < find_optimal_threshold(y_true, y_prob) <= 0.9

def test_threshold_all_zeros():
    y_true, y_prob = np.array([0, 0]), np.array([0.1, 0.2])
    assert find_optimal_threshold(y_true, y_prob) >= 0.1

def test_threshold_all_ones():
    y_true, y_prob = np.array([1, 1]), np.array([0.8, 0.9])
    assert find_optimal_threshold(y_true, y_prob) <= 0.8

def test_threshold_boundary_low():
    y_true, y_prob = np.array([0, 1]), np.array([0.05, 0.15])
    assert find_optimal_threshold(y_true, y_prob) <= 0.15

def test_threshold_output_type():
    assert isinstance(find_optimal_threshold(np.array([0, 1]), np.array([0.1, 0.9])), float)

# --- 6-10: Ensembling Tests ---
def test_ensemble_math(tmp_path):
    df1 = pd.DataFrame({'id': [0], 'label': [0.9]})
    df2 = pd.DataFrame({'id': [0], 'label': [0.1]})
    p1, p2 = tmp_path / "f1.csv", tmp_path / "f2.csv"
    df1.to_csv(p1, index=False); df2.to_csv(p2, index=False)
    out = tmp_path / "out.csv"
    ensemble_csvs(str(tmp_path / "f*.csv"), str(out))
    assert pd.read_csv(out)['label'].iloc[0] == 1 # (0.9+0.1)/2 = 0.5, which is >= 0.5

def test_ensemble_no_files(capsys):
    ensemble_csvs("non_existent_*.csv")
    captured = capsys.readouterr()
    assert "No files found" in captured.out

def test_ensemble_column_consistency(tmp_path):
    df1 = pd.DataFrame({'id': [101], 'label': [0.8]})
    p1 = tmp_path / "f1.csv"
    df1.to_csv(p1, index=False)
    out = tmp_path / "out.csv"
    ensemble_csvs(str(p1), str(out))
    assert 'id' in pd.read_csv(out).columns

def test_ensemble_multi_row(tmp_path):
    df = pd.DataFrame({'id': [1, 2, 3], 'label': [0.1, 0.8, 0.5]})
    p = tmp_path / "f1.csv"
    df.to_csv(p, index=False)
    ensemble_csvs(str(p), str(tmp_path / "out.csv"))
    assert len(pd.read_csv(tmp_path / "out.csv")) == 3

def test_ensemble_averaging_logic(tmp_path):
    df1 = pd.DataFrame({'id': [0], 'label': [0.6]})
    df2 = pd.DataFrame({'id': [0], 'label': [0.2]})
    df3 = pd.DataFrame({'id': [0], 'label': [0.2]}) # Mean = 0.33
    for i, d in enumerate([df1, df2, df3]): d.to_csv(tmp_path/f"{i}.csv", index=False)
    out = tmp_path / "out.csv"
    ensemble_csvs(str(tmp_path / "*.csv"), str(out))
    assert pd.read_csv(out)['label'].iloc[0] == 0

# --- 11-15: Mock Data & Tensor Tests ---
def test_tensor_shapes():
    img = torch.randn(1, 3, 224, 224)
    assert img.shape == (1, 3, 224, 224)

def test_model_output_range():
    logits = torch.tensor([-5.0, 0.0, 5.0])
    probs = torch.sigmoid(logits)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)

def test_binary_cross_entropy_logic():
    loss_fn = torch.nn.BCEWithLogitsLoss()
    input = torch.tensor([10.0]) # High logit
    target = torch.tensor([1.0]) # Positive label
    loss = loss_fn(input, target)
    assert loss < 0.1 # Should be very low

def test_mps_availability():
    # Since you are on a Mac, checking if the device logic handles MPS
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    assert device.type in ["mps", "cpu"]

def test_dummy_dataloader_iteration():
    data = [(torch.randn(3, 224, 224), torch.tensor(1)) for _ in range(5)]
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    assert len(loader) == 3

# --- 16-20: File System & Environment ---
def test_directory_structure():
    assert os.path.exists('src') or os.path.exists('main.py')

def test_csv_submission_format():
    # Test if the existing submission matches requirements
    if os.path.exists('submission.csv'):
        df = pd.read_csv('submission.csv')
        assert 'id' in df.columns and 'label' in df.columns

def test_numpy_to_torch_conversion():
    arr = np.array([1, 2, 3])
    tensor = torch.from_numpy(arr)
    assert tensor.sum() == 6

def test_f1_score_calculation():
    from sklearn.metrics import f1_score
    assert f1_score([1, 0, 1], [1, 0, 0]) == pytest.approx(0.666, 0.01)

def test_random_seed_consistency():
    torch.manual_seed(42)
    val1 = torch.randn(1)
    torch.manual_seed(42)
    val2 = torch.randn(1)
    assert torch.equal(val1, val2)

def test_versioning_logic(tmp_path):
    import os
    from ensemble_submissions import get_next_versioned_filename
    
    # Change directory to tmp_path for the test
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Create base file
        open("submission.csv", "w").close()
        v2 = get_next_versioned_filename()
        assert v2 == "submission_v2.csv"
        
        # Create v2 file
        open("submission_v2.csv", "w").close()
        v3 = get_next_versioned_filename()
        assert v3 == "submission_v3.csv"
    finally:
        os.chdir(original_cwd)

def test_versioning_logic(tmp_path):
    import os
    from ensemble_submissions import get_next_versioned_filename
    
    # Change directory to tmp_path for the test
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Create base file
        open("submission.csv", "w").close()
        v2 = get_next_versioned_filename()
        assert v2 == "submission_v2.csv"
        
        # Create v2 file
        open("submission_v2.csv", "w").close()
        v3 = get_next_versioned_filename()
        assert v3 == "submission_v3.csv"
    finally:
        os.chdir(original_cwd)
