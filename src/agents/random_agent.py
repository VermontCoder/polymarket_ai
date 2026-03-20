"""Random agent for Polymarket BTC 5-minute RL trading agent.

Selects uniformly at random from unmasked (valid) actions. Used for
initial environment verification and as a baseline.
"""

import numpy as np


class RandomAgent:
    """Uniform random action selection from valid actions.

    Selects independently on every row. The environment's action mask
    handles buy/sell mode restrictions (e.g. no buying while holding shares).
    """

    def select_action(self, action_mask: np.ndarray) -> int:
        """Select a random valid action.

        95% of the time does nothing (action 0). The remaining 5%
        selects uniformly from the valid non-zero actions.

        Args:
            action_mask: Boolean array of shape (9,). True = allowed.

        Returns:
            Action index (0-8).
        """
        valid_nonzero = np.where(action_mask[1:])[0] + 1
        if len(valid_nonzero) == 0 or np.random.random() < 0.95:
            return 0
        return int(np.random.choice(valid_nonzero))
