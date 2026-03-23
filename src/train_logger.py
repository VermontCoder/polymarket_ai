"""JSON Lines log writer for single-run training sessions."""
from __future__ import annotations

import json
import os
from datetime import datetime


class TrainLogger:
    """Appends one JSON entry per validation checkpoint to a .jsonl file.

    Safe to construct multiple times on the same path (entries accumulate).
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def append(
        self,
        checkpoint: int,
        episode: int,
        elapsed_seconds: float,
        val_profit_cents: float,
        best_profit_cents: float,
        median_profit_cents: float,
        epoch_median_cents: float,
        epsilon: float,
        action_distribution: dict[str, float],
    ) -> None:
        """Append one checkpoint entry to the log file."""
        entry = {
            "checkpoint": checkpoint,
            "episode": episode,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed_seconds),
            "val_profit_cents": round(val_profit_cents, 2),
            "best_profit_cents": round(best_profit_cents, 2),
            "median_profit_cents": round(median_profit_cents, 2),
            "epoch_median_cents": round(epoch_median_cents, 2),
            "epsilon": round(epsilon, 4),
            "action_distribution": {
                k: round(v, 4) for k, v in action_distribution.items()
            },
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
