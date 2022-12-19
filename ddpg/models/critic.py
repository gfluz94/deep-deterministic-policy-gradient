from typing import Dict, List

import torch
import torch.nn as nn


class Critic(nn.Module):
    """Critic Neural Network, which outputs Q-values, based on state and action as inputs.

    Parameters:
        input_dim (int): State space dimension + Action space dimension - NN's input
        hidden_layers (List[int]): List of hidden layers' units
    """

    def __init__(self, input_dim: int, hidden_layers: List[int]) -> None:
        """Constructor method of Critic object.

        Args:
            input_dim (int): State space dimension - NN's input
            hidden_layers (List[int]): List of hidden layers' units

        Returns:
           (None)
        """
        super(Critic, self).__init__()
        self._input_dim = input_dim
        self._hidden_layers = hidden_layers

        hidden_layers = [self._input_dim] + hidden_layers
        self._fc = []

        self._fc = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(in_, out_), nn.ReLU())
                for in_, out_ in zip(hidden_layers, hidden_layers[1:])
            ]
        )
        self._out = nn.Sequential(nn.Linear(hidden_layers[-1], 1))

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def hidden_layers(self) -> int:
        return self._hidden_layers

    def architecture(self) -> Dict[str, str]:
        return {
            "input_dim": self._input_dim,
            "hidden_layers": " | ".join(map(str, self._hidden_layers)),
        }

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass on NN. Computes Q(state, action) based on current state and action

        Args:
            state (torch.Tensor): Current state tensor.
            action (torch.Tensor): Action tensor.

        Returns:
            torch.Tensor: Q(state, action) tensor (N, 1).
        """
        inputs = torch.cat([state, action], axis=-1)
        return self._out(self._fc(inputs))
