"""
MP-LSTM architecture used for six-pollutant reconstruction.

This module intentionally contains only the model definition and manuscript-level
configuration. Dataset-specific paths, filenames, and preprocessing details are
kept outside the publication-facing model code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


# Computational task order used by the original training implementation.
POLLUTANTS: Tuple[str, ...] = (
    "PM2.5",
    "PM10",
    "CO",
    "NO2",
    "SO2",
    "O3",
)


@dataclass(frozen=True)
class ModelConfig:
    """Model settings reported in the manuscript and Supplementary Methods."""

    input_size: int = 24
    hidden_size: int = 256
    num_layers: int = 4
    dropout: float = 0.01
    num_tasks: int = 6
    sequence_length: int = 48


class MultiTaskLSTMRegressor(nn.Module):
    """
    Shared four-layer LSTM backbone with six pollutant-specific regression heads.

    Input
    -----
    x : torch.Tensor
        Shape [batch, sequence_length, input_size].

    Output
    ------
    torch.Tensor
        Shape [batch, 6], ordered according to ``POLLUTANTS``.
    """

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.task_heads = nn.ModuleDict(
            {
                f"task_{i}": nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size, 1),
                )
                for i in range(config.num_tasks)
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape [batch, time, features], got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.config.input_size:
            raise ValueError(
                f"Expected {self.config.input_size} predictor variables, "
                f"got {x.shape[-1]}"
            )

        sequence, _ = self.lstm(x)
        shared = sequence[:, -1, :]

        outputs = [
            self.task_heads[f"task_{i}"](shared)
            for i in range(self.config.num_tasks)
        ]
        return torch.cat(outputs, dim=1)