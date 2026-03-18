"""Stacked DQN model for Polymarket BTC RL trading agent.

Baseline feedforward model that stacks the last K dynamic observations
instead of using recurrence.

Architecture:
  Input: concat of static features (37) + last K dynamic features (K * 11)
  Linear(37 + K*11, 64) -> ReLU
  Linear(64, 32) -> LayerNorm -> ReLU -> Dropout(0.15)
  Linear(32, 9) -> Q-values

No hidden state -- returns None for hidden_state in forward().
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.base import BaseModel


class StackedDQN(BaseModel):
    """Feedforward Q-network using stacked recent observations.

    Args:
        static_dim: Dimensionality of static (episode-level) features.
        dynamic_dim: Dimensionality of dynamic (per-timestep) features.
        num_actions: Number of discrete actions.
        stack_size: Number of recent observations to stack (K).
        hidden_dim: Width of hidden layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        static_dim: int = 37,
        dynamic_dim: int = 11,
        num_actions: int = 9,
        stack_size: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__(
            static_dim=static_dim,
            dynamic_dim=dynamic_dim,
            num_actions=num_actions,
        )
        self._stack_size = stack_size

        input_dim = static_dim + stack_size * dynamic_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_actions),
        )

    @property
    def hidden_size(self) -> int:
        """No recurrent hidden state."""
        return 0

    @property
    def stack_size(self) -> int:
        """Number of recent observations this model expects."""
        return self._stack_size

    def forward(
        self,
        static_features: torch.Tensor,
        dynamic_features: torch.Tensor,
        hidden_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, None]:
        """Compute Q-values from stacked observations.

        Args:
            static_features: (batch, static_dim).
            dynamic_features: (batch, seq_len, dynamic_dim).
                The last `stack_size` timesteps are used. If seq_len < stack_size,
                earlier positions are zero-padded.
            hidden_state: Ignored (feedforward model).

        Returns:
            q_values: (batch, num_actions).
            None: No hidden state.
        """
        batch_size = static_features.shape[0]
        seq_len = dynamic_features.shape[1]

        # Pad or slice to get exactly stack_size timesteps
        if seq_len >= self._stack_size:
            # Take the last stack_size timesteps
            stacked = dynamic_features[:, -self._stack_size:, :]
        else:
            # Zero-pad on the left
            pad_len = self._stack_size - seq_len
            padding = torch.zeros(
                batch_size, pad_len, self._dynamic_dim,
                device=dynamic_features.device,
                dtype=dynamic_features.dtype,
            )
            stacked = torch.cat([padding, dynamic_features], dim=1)

        # Flatten the stacked timesteps: (batch, stack_size * dynamic_dim)
        flat_dynamic = stacked.reshape(batch_size, -1)

        # Concat with static features
        combined = torch.cat([static_features, flat_dynamic], dim=-1)

        # MLP
        q_values = self.mlp(combined)

        return q_values, None

    def get_initial_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """No hidden state for feedforward model."""
        return None
