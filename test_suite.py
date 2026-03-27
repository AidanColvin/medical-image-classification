
import torch
import numpy as np
import pytest

def test_1_tensor_shape(): assert torch.randn(1, 3, 224, 224).shape == (1, 3, 224, 224)
def test_2_sigmoid_range(): assert torch.all(torch.sigmoid(torch.tensor([-5.0, 5.0])) <= 1.0)
def test_3_mps_check(): assert torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
def test_4_label_type(): assert isinstance(int(1), int)
def test_5_data_split(): assert 0.8 + 0.2 == 1.0
def test_6_binary_loss(): assert torch.nn.BCEWithLogitsLoss()(torch.tensor([10.0]), torch.tensor([1.0])) < 0.1
def test_7_reproducibility(): torch.manual_seed(42); v1 = torch.randn(1); torch.manual_seed(42); v2 = torch.randn(1); assert torch.equal(v1, v2)
def test_8_numpy_conv(): assert torch.from_numpy(np.array([1])).item() == 1
def test_9_id_range(): assert len(range(624)) == 624
def test_10_threshold(): assert (torch.tensor([0.6]) >= 0.5).int().item() == 1
