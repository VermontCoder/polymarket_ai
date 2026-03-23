"""Tests for the rollout worker function."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.base import BaseModel
from src.normalizer import Normalizer
from src.rollout_worker import run_rollout_worker


def _make_row():
    return {
        "timestamp": "2026-03-14T17:23:00Z",
        "up_bid": 55.0, "up_ask": 56.0,
        "down_bid": 44.0, "down_ask": 45.0,
        "current_price": 70000.0, "diff_pct": 0.01,
        "diff_usd": 5.0, "time_to_close": 150000,
    }


def _make_episode(num_rows=5):
    return {
        "session_id": "test", "outcome": "UP",
        "hour": 12, "day": 2,
        "start_price": 70000.0, "end_price": 70100.0,
        "diff_pct_prev_session": 0.05, "diff_pct_hour": 0.02,
        "rows": [_make_row() for _ in range(num_rows)],
    }


class TestRunRolloutWorker:
    def _make_state_dict(self):
        from src.models.lstm_dqn import LSTMDQN
        model = LSTMDQN(lstm_hidden_size=32)
        return {k: v.cpu() for k, v in model.state_dict().items()}

    def _make_normalizer(self, episodes):
        n = Normalizer()
        n.fit(episodes)
        return n

    def test_returns_one_result_per_episode(self):
        episodes = [_make_episode() for _ in range(3)]
        normalizer = self._make_normalizer(episodes)
        state_dict = self._make_state_dict()
        config = {"lstm_hidden": 32, "epsilon_start": 1.0,
                  "epsilon_end": 0.05, "epsilon_decay_episodes": 300}

        results = run_rollout_worker(
            state_dict=state_dict, episodes=episodes,
            normalizer=normalizer, config=config,
            episode_count=0, device_str="cpu",
        )

        assert len(results) == 3

    def test_each_result_has_transitions(self):
        episodes = [_make_episode(num_rows=4)]
        normalizer = self._make_normalizer(episodes)
        state_dict = self._make_state_dict()
        config = {"lstm_hidden": 32, "epsilon_start": 1.0,
                  "epsilon_end": 0.05, "epsilon_decay_episodes": 300}

        results = run_rollout_worker(
            state_dict=state_dict, episodes=episodes,
            normalizer=normalizer, config=config,
            episode_count=0, device_str="cpu",
        )

        reward, action_counts, transitions = results[0]
        assert isinstance(reward, float)
        assert action_counts.shape == (9,)
        assert len(transitions) == 4
