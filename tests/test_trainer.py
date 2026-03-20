"""Tests for Trainer multi-trade episode processing.

Verifies that _run_episode() and evaluate() process ALL rows of an episode
(no early termination). With multi-trade mechanics, the agent may buy and sell
multiple times per episode, so every row must be visited.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.base import BaseModel
from src.normalizer import Normalizer
from src.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(
    up_bid=55.0, up_ask=56.0, down_bid=44.0, down_ask=45.0,
    diff_pct=0.01, time_to_close=150000,
):
    return {
        "timestamp": "2026-03-14T17:23:00Z",
        "up_bid": up_bid, "up_ask": up_ask,
        "down_bid": down_bid, "down_ask": down_ask,
        "current_price": 70000.0, "diff_pct": diff_pct,
        "diff_usd": 5.0, "time_to_close": time_to_close,
    }


def _make_episode(outcome="UP", num_rows=5, rows=None):
    if rows is None:
        rows = [_make_row() for _ in range(num_rows)]
    return {
        "session_id": "test", "outcome": outcome,
        "hour": 12, "day": 2,
        "start_price": 70000.0, "end_price": 70100.0,
        "diff_pct_prev_session": 0.05, "diff_pct_hour": 0.02,
        "rows": rows,
    }


class CountingModel(BaseModel):
    """Minimal model that counts forward() calls and always picks one action."""

    def __init__(self, forced_action: int = 0):
        super().__init__()
        self._forced_action = forced_action
        self.call_count = 0
        # nn.Parameter required so .to(device) / deepcopy work correctly
        self._dummy = nn.Parameter(torch.zeros(1))

    @property
    def hidden_size(self) -> int:
        return 0

    def forward(self, static_features, dynamic_features, hidden_state=None):
        self.call_count += 1
        B = static_features.shape[0]
        q = torch.full((B, 9), -10.0)
        q[:, self._forced_action] = 10.0
        return q, None

    def get_initial_hidden(self, batch_size, device=None):
        return None


def _make_trainer(model, episodes, config=None):
    """Create a trainer fitted on the given episodes."""
    normalizer = Normalizer()
    normalizer.fit(episodes)
    return Trainer(model=model, normalizer=normalizer, config=config)


def _greedy_trainer(model, episodes):
    """Trainer with epsilon pinned to 0 (always greedy, no random actions)."""
    return _make_trainer(
        model, episodes,
        config={"epsilon_start": 0.0, "epsilon_end": 0.0},
    )


# ---------------------------------------------------------------------------
# Tests: _run_episode() processes all rows
# ---------------------------------------------------------------------------

class TestRunEpisodeAllRows:
    """_run_episode() must visit every row — no early termination."""

    def test_model_called_for_every_row(self):
        """Model is called exactly once per row when always doing nothing."""
        rows = [_make_row() for _ in range(10)]
        ep = _make_episode(outcome="UP", rows=rows)
        model = CountingModel(forced_action=0)  # always do nothing
        trainer = _greedy_trainer(model, [ep])

        trainer._run_episode(ep)

        assert model.call_count == 10

    def test_reward_is_zero_when_no_trades(self):
        """No trades produces zero reward."""
        rows = [_make_row() for _ in range(5)]
        ep = _make_episode(outcome="UP", rows=rows)
        model = CountingModel(forced_action=0)
        trainer = _greedy_trainer(model, [ep])

        reward, _ = trainer._run_episode(ep)

        assert reward == pytest.approx(0.0)

    def test_action_counts_sum_to_num_rows(self):
        """Action counts must sum to the number of rows (one action per row)."""
        rows = [_make_row() for _ in range(7)]
        ep = _make_episode(outcome="UP", rows=rows)
        model = CountingModel(forced_action=0)
        trainer = _greedy_trainer(model, [ep])

        _, action_counts = trainer._run_episode(ep)

        assert action_counts.sum() == 7

    def test_reward_matches_evaluate(self):
        """_run_episode reward (normalized) equals evaluate() result / 100."""
        rows = [_make_row() for _ in range(5)]
        ep = _make_episode(outcome="UP", rows=rows)

        model_run = CountingModel(forced_action=0)
        trainer_run = _greedy_trainer(model_run, [ep])
        episode_reward, _ = trainer_run._run_episode(ep)

        model_eval = CountingModel(forced_action=0)
        trainer_eval = _greedy_trainer(model_eval, [ep])
        evaluate_profit = trainer_eval.evaluate([ep])

        assert episode_reward * 100.0 == pytest.approx(evaluate_profit)


# ---------------------------------------------------------------------------
# Tests: evaluate() processes all rows
# ---------------------------------------------------------------------------

class TestEvaluateAllRows:
    """evaluate() must visit every row across all episodes."""

    def test_model_called_for_every_row(self):
        """Model is called exactly once per row."""
        rows = [_make_row() for _ in range(10)]
        ep = _make_episode(outcome="UP", rows=rows)
        model = CountingModel(forced_action=0)
        trainer = _make_trainer(model, [ep])

        trainer.evaluate([ep])

        assert model.call_count == 10

    def test_profit_zero_when_no_trades(self):
        """No trades → zero total profit."""
        rows = [_make_row() for _ in range(5)]
        ep = _make_episode(outcome="UP", rows=rows)
        model = CountingModel(forced_action=0)
        trainer = _make_trainer(model, [ep])

        total_profit = trainer.evaluate([ep])

        assert total_profit == pytest.approx(0.0)

    def test_multiple_episodes_model_call_count(self):
        """Model is called sum(rows) times across all episodes."""
        ep1 = _make_episode(outcome="UP", num_rows=3)
        ep2 = _make_episode(outcome="DOWN", num_rows=4)
        model = CountingModel(forced_action=0)
        trainer = _make_trainer(model, [ep1, ep2])

        trainer.evaluate([ep1, ep2])

        # 3 + 4 = 7 rows total
        assert model.call_count == 7

    def test_multiple_episodes_accumulate_profit(self):
        """evaluate() sums profit correctly across multiple episodes."""
        ep1 = _make_episode(outcome="UP", num_rows=3)
        ep2 = _make_episode(outcome="DOWN", num_rows=3)
        model = CountingModel(forced_action=0)  # no trades → zero each
        trainer = _make_trainer(model, [ep1, ep2])

        total_profit = trainer.evaluate([ep1, ep2])

        assert total_profit == pytest.approx(0.0)
