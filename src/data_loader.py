"""Load and split Polymarket BTC 5-minute episode data."""

import json
import random
from typing import Any


def _reassign_stale_leading_rows(
    episodes: list[dict[str, Any]],
    threshold_ms: float = 5000,
) -> list[dict[str, Any]]:
    """Move stale leading rows to the previous episode.

    Some episodes start with a row whose time_to_close is near zero —
    a data snapshot that arrived just after the prior 5-minute session
    closed. These rows belong to the previous episode and are moved there.

    Episodes must be in chronological order.
    """
    from datetime import datetime

    result = [dict(ep) for ep in episodes]
    for r in result:
        r["rows"] = list(r["rows"])

    for i in range(len(result)):
        rows = result[i]["rows"]
        stale_count = 0
        for row in rows:
            ttc = row.get("time_to_close")
            if ttc is not None and ttc < threshold_ms:
                stale_count += 1
            else:
                break

        if stale_count == 0:
            continue

        if i == 0:
            # No previous episode — drop the stale rows
            result[i]["rows"] = rows[stale_count:]
            continue

        # Verify previous episode is consecutive (5-minute gap)
        t_prev = datetime.fromisoformat(result[i - 1]["session_id"].replace("Z", ""))
        t_curr = datetime.fromisoformat(result[i]["session_id"].replace("Z", ""))
        gap_min = (t_curr - t_prev).total_seconds() / 60
        if gap_min > 5:
            # No adjacent previous episode — drop the stale rows
            result[i]["rows"] = rows[stale_count:]
            continue

        # Move stale rows to previous episode
        result[i - 1]["rows"].extend(rows[:stale_count])
        result[i]["rows"] = rows[stale_count:]

    return result


def load_episodes(path: str) -> list[dict[str, Any]]:
    """Load JSON file and return list of episode dicts.

    Reassigns stale leading rows (time_to_close < 5s) to the previous
    episode, since they are data snapshots from the prior session that
    arrived slightly late.
    """
    with open(path, "r") as f:
        episodes = json.load(f)
    episodes = _reassign_stale_leading_rows(episodes)
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
