"""Abstract base model interface for Polymarket BTC RL trading agent.

All model variants (LSTM-DQN, Stacked DQN, etc.) implement this interface
so that agents and trainers can swap models without code changes.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract base class for Q-network models.

    Interface contract:
      - forward(static_features, dynamic_features, hidden_state=None)
            -> (q_values, new_hidden_state)
      - get_initial_hidden(batch_size) -> initial hidden state (zeros)
      - Properties: hidden_size, static_dim, dynamic_dim, num_actions
    """

    def __init__(
        self,
        static_dim: int = 37,
        dynamic_dim: int = 11,
        num_actions: int = 9,
    ) -> None:
        super().__init__()
        self._static_dim = static_dim
        self._dynamic_dim = dynamic_dim
        self._num_actions = num_actions

    @property
    def static_dim(self) -> int:
        """Dimensionality of the static (episode-level) feature vector."""
        return self._static_dim

    @property
    def dynamic_dim(self) -> int:
        """Dimensionality of each dynamic (per-timestep) feature vector."""
        return self._dynamic_dim

    @property
    def num_actions(self) -> int:
        """Number of discrete actions the model outputs Q-values for."""
        return self._num_actions

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Size of the recurrent hidden state (0 for non-recurrent models)."""
        ...

    @abstractmethod
    def forward(
        self,
        static_features: torch.Tensor,
        dynamic_features: torch.Tensor,
        hidden_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Compute Q-values for the given observation.

        Args:
            static_features: (batch, static_dim) episode-level features.
            dynamic_features: (batch, seq_len, dynamic_dim) per-timestep features.
                For recurrent models, seq_len can be 1 (online) or >1 (sequence).
                For feedforward models, all timesteps are stacked internally.
            hidden_state: Optional recurrent hidden state from a previous call.
                Tuple of (h, c) for LSTM-based models, None for feedforward.

        Returns:
            q_values: (batch, num_actions) Q-value estimates for each action.
                For recurrent models processing seq_len > 1, returns Q-values
                for the LAST timestep only.
            new_hidden_state: Updated hidden state (or None for feedforward).
        """
        ...

    @abstractmethod
    def get_initial_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return the initial hidden state (zeros) for a batch.

        Args:
            batch_size: Number of sequences in the batch.
            device: Device for the tensors (defaults to model's device).

        Returns:
            Tuple of (h_0, c_0) each shaped (num_layers, batch, hidden_size),
            or None for non-recurrent models.
        """
        ...
