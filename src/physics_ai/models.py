"""Small multilayer perceptron used by all experimental variants."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """MLP mapping a scalar time input to a scalar position prediction."""

    def __init__(self, hidden_width: int = 64, hidden_layers: int = 3) -> None:
        super().__init__()
        if hidden_width < 1 or hidden_layers < 1:
            raise ValueError("hidden_width and hidden_layers must be positive")
        layers: list[nn.Module] = [nn.Linear(1, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.Tanh()])
        layers.append(nn.Linear(hidden_width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.network(t)
