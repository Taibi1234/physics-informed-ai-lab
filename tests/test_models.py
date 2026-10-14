import torch

from physics_ai.models import MLP


def test_mlp_returns_one_prediction_per_time():
    model = MLP(hidden_width=16, hidden_layers=2)
    inputs = torch.linspace(0, 1, 12).reshape(-1, 1)
    predictions = model(inputs)
    assert predictions.shape == (12, 1)
