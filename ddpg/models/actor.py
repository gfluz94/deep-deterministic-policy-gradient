from typing import List

import torch
import torch.nn as nn


class Actor(nn.Module):
    """Actor Neural Network, which outputs actions, based on state as input.

    Parameters:
        input_dim (int): State space dimension - NN's input
        output_dim (int): Action space dimension - NN's output
        hidden_layers (List[int]): List of hidden layers' units
        max_output (float): Maximum value allowed for action - range(-max_output, max_output)
    """

    def __init__(
        self, input_dim: int, hidden_layers: List[int], output_dim: int, max_output: float
    ) -> None:
        """Constructor method of Actor object.

        Args:
            input_dim (int): State space dimension - NN's input
            output_dim (int): Action space dimension - NN's output
            hidden_layers (List[int]): List of hidden layers' units
            max_output (float): Maximum value allowed for action - range(-max_output, max_output)

        Returns:
           (None)
        """
        super(Actor, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_layers = hidden_layers
        self._max_output = max_output

        hidden_layers = [self._input_dim] + hidden_layers
        self._fc = []
        
        self._fc = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(in_, out_), nn.ReLU()
            )
            for in_, out_ in zip(hidden_layers, hidden_layers[1:])
        ])
        self._out = nn.Sequential(
                nn.Linear(hidden_layers[-1], self._output_dim), nn.Tanh()
            )

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def hidden_layers(self) -> int:
        return self._hidden_layers

    @property
    def max_output(self) -> float:
        return self._max_output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass on NN. Computes actions (output) based on current state (inputs)

        Args:
            inputs (torch.Tensor): Current state tensor.

        Returns:
            torch.Tensor: Action tensor.
        """
        return self._max_output*self._out(
            self._fc(inputs)
        )