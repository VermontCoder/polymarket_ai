"""Rollout worker for parallel single-run training.

Top-level function required for multiprocessing pickle compatibility.
"""
from __future__ import annotations

import numpy as np
import torch

from src.models.lstm_dqn import LSTMDQN
from src.normalizer import Normalizer
from src.trainer import Trainer


def run_rollout_worker(
    state_dict: dict,
    episodes: list[dict],
    normalizer: Normalizer,
    config: dict,
    episode_count: int,
    device_str: str,
) -> list[tuple[float, np.ndarray, list[dict]]]:
    """Run a list of episodes and return collected transitions.

    Does NOT add transitions to a replay buffer. The coordinator merges
    returned transitions into its own buffer.

    Args:
        state_dict: Online network weights (CPU tensors).
        episodes: Episodes to run (subset of train_eps for this round).
        normalizer: Fitted feature normalizer.
        config: Training config dict (same as Trainer config).
        episode_count: Current total episode count (used for epsilon calc).
        device_str: Device string, e.g. "cuda:0" or "cpu".

    Returns:
        List of (reward, action_counts, transitions) tuples, one per episode.
    """
    device = torch.device(device_str)
    model = LSTMDQN(lstm_hidden_size=config.get("lstm_hidden", 32))
    trainer = Trainer(model=model, normalizer=normalizer, config=config, device=device)
    trainer.online_net.load_state_dict(
        {k: v.to(device) for k, v in state_dict.items()}
    )
    trainer._episode_count = episode_count  # sets correct epsilon

    results = []
    for ep in episodes:
        reward, action_counts, transitions = trainer.collect_episode(ep)
        results.append((reward, action_counts, transitions))

    return results
