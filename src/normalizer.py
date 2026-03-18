"""Feature normalization pipeline for Polymarket BTC 5-minute episodes.

Static features (per episode): 37 dims
  [0-23]  hour one-hot (24 dims)
  [24-30] day one-hot, Mon=24..Sun=30 (7 dims)
  [31]    diff_pct_prev_session / std (0 if null)
  [32]    diff_pct_prev_session is_null flag
  [33]    diff_pct_hour / std (0 if null)
  [34]    diff_pct_hour is_null flag
  [35]    avg_pct_variance_hour / std (0 if null)
  [36]    avg_pct_variance_hour is_null flag

Dynamic features (per row): 11 dims
  [0]  up_bid / 100 (0 if null)
  [1]  up_bid is_null flag
  [2]  up_ask / 100 (0 if null)
  [3]  up_ask is_null flag
  [4]  down_bid / 100 (0 if null)
  [5]  down_bid is_null flag
  [6]  down_ask / 100 (0 if null)
  [7]  down_ask is_null flag
  [8]  diff_pct / std (0 if null)
  [9]  diff_pct is_null flag
  [10] time_to_close / 300000, clamped [0, 1]
"""

import math
from typing import Any

import numpy as np


class Normalizer:
    """Computes training-set statistics and encodes episode features."""

    STATIC_DIM = 37
    DYNAMIC_DIM = 11

    def __init__(self) -> None:
        self.std_diff_pct_prev_session: float = 1.0
        self.std_diff_pct_hour: float = 1.0
        self.std_avg_pct_variance_hour: float = 1.0
        self.std_diff_pct: float = 1.0
        self._fitted = False

    def fit(self, train_episodes: list[dict[str, Any]]) -> None:
        """Compute normalization statistics from training episodes only.

        Computes standard deviations for:
          - diff_pct_prev_session (episode-level)
          - diff_pct_hour (episode-level)
          - diff_pct (row-level)

        Null values are excluded from the std computation.
        """
        prev_session_vals: list[float] = []
        hour_vals: list[float] = []
        avg_var_hour_vals: list[float] = []
        diff_pct_vals: list[float] = []

        for ep in train_episodes:
            if ep.get("diff_pct_prev_session") is not None:
                prev_session_vals.append(ep["diff_pct_prev_session"])
            if ep.get("diff_pct_hour") is not None:
                hour_vals.append(ep["diff_pct_hour"])
            if ep.get("avg_pct_variance_hour") is not None:
                avg_var_hour_vals.append(ep["avg_pct_variance_hour"])
            for row in ep["rows"]:
                if row.get("diff_pct") is not None:
                    diff_pct_vals.append(row["diff_pct"])

        self.std_diff_pct_prev_session = _std(prev_session_vals)
        self.std_diff_pct_hour = _std(hour_vals)
        self.std_avg_pct_variance_hour = _std(avg_var_hour_vals)
        self.std_diff_pct = _std(diff_pct_vals)
        self._fitted = True

    def encode_static(self, episode: dict[str, Any]) -> np.ndarray:
        """Encode episode-level static features into a 37-dim vector.

        Args:
            episode: An episode dict with keys hour, day,
                     diff_pct_prev_session, diff_pct_hour,
                     avg_pct_variance_hour.

        Returns:
            numpy array of shape (37,) with float32 dtype.
        """
        assert self._fitted, "Must call fit() before encoding"

        vec = np.zeros(self.STATIC_DIM, dtype=np.float32)

        # Hour one-hot: indices 0-23
        hour = episode["hour"]
        vec[hour] = 1.0

        # Day one-hot: indices 24-30 (Monday=0 -> index 24, Sunday=6 -> index 30)
        day = episode["day"]
        vec[24 + day] = 1.0

        # diff_pct_prev_session: index 31 (value), 32 (is_null)
        val = episode.get("diff_pct_prev_session")
        if val is None:
            vec[31] = 0.0
            vec[32] = 1.0
        else:
            vec[31] = val / self.std_diff_pct_prev_session
            vec[32] = 0.0

        # diff_pct_hour: index 33 (value), 34 (is_null)
        val = episode.get("diff_pct_hour")
        if val is None:
            vec[33] = 0.0
            vec[34] = 1.0
        else:
            vec[33] = val / self.std_diff_pct_hour
            vec[34] = 0.0

        # avg_pct_variance_hour: index 35 (value), 36 (is_null)
        val = episode.get("avg_pct_variance_hour")
        if val is None:
            vec[35] = 0.0
            vec[36] = 1.0
        else:
            vec[35] = val / self.std_avg_pct_variance_hour
            vec[36] = 0.0

        return vec

    def encode_dynamic(self, row: dict[str, Any]) -> np.ndarray:
        """Encode a single row's dynamic features into an 11-dim vector.

        Args:
            row: A row dict from an episode's 'rows' list.

        Returns:
            numpy array of shape (11,) with float32 dtype.
        """
        assert self._fitted, "Must call fit() before encoding"

        vec = np.zeros(self.DYNAMIC_DIM, dtype=np.float32)

        # up_bid: index 0 (value/100), 1 (is_null)
        vec[0], vec[1] = _encode_bid_ask(row.get("up_bid"))

        # up_ask: index 2 (value/100), 3 (is_null)
        vec[2], vec[3] = _encode_bid_ask(row.get("up_ask"))

        # down_bid: index 4 (value/100), 5 (is_null)
        vec[4], vec[5] = _encode_bid_ask(row.get("down_bid"))

        # down_ask: index 6 (value/100), 7 (is_null)
        vec[6], vec[7] = _encode_bid_ask(row.get("down_ask"))

        # diff_pct: index 8 (value/std), 9 (is_null)
        diff = row.get("diff_pct")
        if diff is None:
            vec[8] = 0.0
            vec[9] = 1.0
        else:
            vec[8] = diff / self.std_diff_pct
            vec[9] = 0.0

        # time_to_close: index 10, clamped to [0, 1]
        ttc = row.get("time_to_close")
        if ttc is None:
            vec[10] = 0.0
        else:
            vec[10] = max(0.0, min(1.0, ttc / 300000.0))

        return vec

    def encode_episode_dynamic(
        self, episode: dict[str, Any]
    ) -> np.ndarray:
        """Encode all rows in an episode into a (T, 11) array.

        Args:
            episode: An episode dict.

        Returns:
            numpy array of shape (num_rows, 11) with float32 dtype.
        """
        rows = episode["rows"]
        result = np.zeros((len(rows), self.DYNAMIC_DIM), dtype=np.float32)
        for i, row in enumerate(rows):
            result[i] = self.encode_dynamic(row)
        return result


def _encode_bid_ask(value: float | None) -> tuple[float, float]:
    """Encode a bid/ask value: (value/100, is_null).

    Returns (0.0, 1.0) if null, otherwise (value/100, 0.0).
    """
    if value is None:
        return 0.0, 1.0
    return value / 100.0, 0.0


def _std(values: list[float]) -> float:
    """Compute population standard deviation, returning 1.0 if empty or zero.

    Uses population std (ddof=0) to avoid division by zero issues with
    single-element lists. Returns 1.0 as a safe default if the result
    would be zero or the list is empty.
    """
    if len(values) == 0:
        return 1.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    s = math.sqrt(variance)
    if s == 0.0:
        return 1.0
    return s
