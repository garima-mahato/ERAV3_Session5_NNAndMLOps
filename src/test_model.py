import torch
import pytest
from model import MNISTNet

def test_model_parameters_less_than_25000():
    model = MNISTNet()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_output_shape():
    model = MNISTNet()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_training_accuracy_greater_than_95():
    from train import train
    accuracy, _ = train()
    assert accuracy > 95, f"Training accuracy is {accuracy}%, should be > 95%"

def test_input_shape():
    model = MNISTNet()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Model does not work with 28x28x1 input"

def test_training_accuracy_greater_than_95_data_augmentation():
    from train import train
    accuracy, _ = train(is_aug=True)
    assert accuracy > 95, f"Training accuracy is {accuracy}%, should be > 95%"

if __name__ == "__main__":
    pytest.main([__file__]) 