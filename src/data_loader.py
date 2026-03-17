"""Load and split Polymarket BTC 5-minute episode data."""

import json
import random
from typing import Any


def load_episodes(path: str) -> list[dict[str, Any]]:
    """Load JSON file and return list of episode dicts."""
    with open(path, "r") as f:
        episodes = json.load(f)
    return episodes


def split_episodes(
    episodes: list[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split episodes into train/val/test sets with random shuffle.

    Args:
        episodes: List of episode dicts.
        train_ratio: Fraction for training set (default 0.8).
        val_ratio: Fraction for validation set (default 0.1).
        test_ratio: Fraction for test set (default 0.1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_episodes, val_episodes, test_episodes).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, (
        "Ratios must sum to 1.0"
    )

    shuffled = list(episodes)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    return train, val, test
