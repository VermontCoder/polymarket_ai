"""Random agent for Polymarket BTC 5-minute RL trading agent.

Selects uniformly at random from unmasked (valid) actions. Used for
initial environment verification and as a baseline.
"""

import numpy as np


class RandomAgent:
    """Uniform random action selection from valid actions.

    The agent makes at most one trade per episode. After acting,
    the action mask restricts it to action 0 (do nothing).
    """

    def select_action(self, action_mask: np.ndarray) -> int:
        """Select a random valid action.

        Args:
            action_mask: Boolean array of shape (9,). True = allowed.

        Returns:
            Action index (0-8).
        """
        valid_actions = np.where(action_mask)[0]
        return int(np.random.choice(valid_actions))
