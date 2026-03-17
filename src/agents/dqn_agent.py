"""DQN agent for Polymarket BTC 5-minute RL trading agent.

Loads a trained model checkpoint and runs greedy inference (epsilon=0).
Handles LSTM hidden state management across timesteps within an episode.
"""

from typing import Optional

import numpy as np
import torch

from src.models.base import BaseModel


class DQNAgent:
    """Trained model agent with greedy action selection.

    Args:
        model: A trained BaseModel instance.
        device: Torch device for inference.
    """

    def __init__(
        self,
        model: BaseModel,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self._hidden_state = None

    def reset(self) -> None:
        """Reset hidden state for a new episode."""
        self._hidden_state = self.model.get_initial_hidden(
            batch_size=1, device=self.device
        )

    @torch.no_grad()
    def select_action(
        self,
        static_features: np.ndarray,
        dynamic_features: np.ndarray,
        action_mask: np.ndarray,
    ) -> int:
        """Select the greedy action (highest Q-value among valid actions).

        Args:
            static_features: Shape (35,) static episode features.
            dynamic_features: Shape (11,) single-timestep dynamic features.
            action_mask: Boolean array of shape (9,). True = allowed.

        Returns:
            Action index (0-8).
        """
        # Convert to tensors: add batch and sequence dims
        static_t = torch.tensor(
            static_features, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, 35)
        dynamic_t = torch.tensor(
            dynamic_features, dtype=torch.float32, device=self.device
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 11)

        q_values, self._hidden_state = self.model(
            static_t, dynamic_t, self._hidden_state
        )
        q_values = q_values.squeeze(0).cpu().numpy()  # (9,)

        # Apply action mask: set invalid actions to -inf
        q_values[~action_mask] = -np.inf

        return int(np.argmax(q_values))

    @classmethod
    def from_checkpoint(
        cls,
        model: BaseModel,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
    ) -> "DQNAgent":
        """Load model weights from checkpoint and create agent.

        Args:
            model: Model instance (architecture must match checkpoint).
            checkpoint_path: Path to saved state dict.
            device: Torch device.

        Returns:
            DQNAgent with loaded weights.
        """
        device = device or torch.device("cpu")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        return cls(model=model, device=device)
