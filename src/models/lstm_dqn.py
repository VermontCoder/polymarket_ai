"""LSTM-DQN model for Polymarket BTC RL trading agent.

Primary architecture (~8,070 parameters):

  Static Encoder:
    Linear(37, 16) -> ReLU -> 16-dim static embedding

  Dynamic Encoder (LSTM):
    LSTM(input_size=11, hidden_size=32, num_layers=1)
    Output per timestep: 32-dim hidden state

  Combiner + Q-Head:
    Concat: [32-dim LSTM output, 16-dim static embedding] = 48 dims
    Linear(48, 32) -> LayerNorm -> ReLU -> Dropout(0.15)
    Linear(32, 9) -> Q-values for 9 actions

Key design choices:
  - 1-layer LSTM, hidden=32: Minimal recurrent capacity
  - LayerNorm (not BatchNorm): Stable with RL's non-stationary targets
  - Dropout 0.15: Light regularization
  - Static features encoded separately from LSTM
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.base import BaseModel


class LSTMDQN(BaseModel):
    """LSTM-DQN Q-network with separate static and dynamic encoders.

    Args:
        static_dim: Dimensionality of static (episode-level) features.
        dynamic_dim: Dimensionality of dynamic (per-timestep) features.
        num_actions: Number of discrete actions.
        lstm_hidden_size: LSTM hidden state size.
        lstm_num_layers: Number of LSTM layers.
        static_embed_dim: Output dimension of static encoder.
        dropout: Dropout rate in the Q-head.
    """

    def __init__(
        self,
        static_dim: int = 37,
        dynamic_dim: int = 11,
        num_actions: int = 9,
        lstm_hidden_size: int = 32,
        lstm_num_layers: int = 1,
        static_embed_dim: int = 16,
        dropout: float = 0.15,
    ) -> None:
        super().__init__(
            static_dim=static_dim,
            dynamic_dim=dynamic_dim,
            num_actions=num_actions,
        )
        self._lstm_hidden_size = lstm_hidden_size
        self._lstm_num_layers = lstm_num_layers

        # Static encoder: Linear -> ReLU
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, static_embed_dim),
            nn.ReLU(),
        )

        # Dynamic encoder: LSTM
        self.lstm = nn.LSTM(
            input_size=dynamic_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        # Combiner + Q-head
        combined_dim = lstm_hidden_size + static_embed_dim
        self.q_head = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_actions),
        )

    @property
    def hidden_size(self) -> int:
        """Size of the LSTM hidden state."""
        return self._lstm_hidden_size

    def forward(
        self,
        static_features: torch.Tensor,
        dynamic_features: torch.Tensor,
        hidden_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute Q-values from static and dynamic features.

        Args:
            static_features: (batch, static_dim).
            dynamic_features: (batch, seq_len, dynamic_dim).
            hidden_state: Optional (h, c) each (num_layers, batch, hidden_size).

        Returns:
            q_values: (batch, num_actions) for the last timestep.
            new_hidden_state: Updated (h, c) tuple.
        """
        # Static encoder
        static_embed = self.static_encoder(static_features)  # (batch, 16)

        # Dynamic encoder (LSTM)
        if hidden_state is not None:
            lstm_out, new_hidden = self.lstm(dynamic_features, hidden_state)
        else:
            lstm_out, new_hidden = self.lstm(dynamic_features)
        # lstm_out: (batch, seq_len, hidden_size)

        # Take the last timestep's output
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Combine static embedding with LSTM output
        combined = torch.cat([last_hidden, static_embed], dim=-1)  # (batch, 48)

        # Q-head
        q_values = self.q_head(combined)  # (batch, num_actions)

        return q_values, new_hidden

    def get_initial_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized LSTM hidden state.

        Args:
            batch_size: Number of sequences in the batch.
            device: Device for the tensors. Defaults to the model's device.

        Returns:
            Tuple of (h_0, c_0) each shaped (num_layers, batch, hidden_size).
        """
        if device is None:
            device = next(self.parameters()).device
        h_0 = torch.zeros(
            self._lstm_num_layers, batch_size, self._lstm_hidden_size,
            device=device,
        )
        c_0 = torch.zeros(
            self._lstm_num_layers, batch_size, self._lstm_hidden_size,
            device=device,
        )
        return (h_0, c_0)
