"""Tests for parallel grid search worker function."""
import pytest
from unittest.mock import patch, MagicMock


def make_fake_episodes(n=10):
    """Create minimal fake episode dicts for testing."""
    episodes = []
    for i in range(n):
        episodes.append({
            "hour": i % 24,
            "day": i % 7,
            "diff_pct_prev_session": 0.01,
            "diff_pct_hour": 0.02,
            "avg_pct_variance_hour": 0.005,
            "outcome": "up",
            "rows": [
                {
                    "up_bid": 48,
                    "up_ask": 52,
                    "down_bid": 48,
                    "down_ask": 52,
                    "diff_pct": 0.001,
                    "time_to_close": 270000,
                }
            ],
        })
    return episodes


def test_run_config_worker_returns_tuple():
    """run_config_worker returns (config_key, seed_profits, median)."""
    from train import run_config_worker

    eps = make_fake_episodes(20)
    config = {"lr": 1e-4, "epsilon_decay": 150, "seq_len": 10, "lstm_hidden": 16, "epochs": 1}
    seeds = [42]

    key, seed_profits, median = run_config_worker(config, seeds, eps, eps, eps)

    assert isinstance(key, str)
    assert isinstance(seed_profits, list)
    assert len(seed_profits) == 1
    assert isinstance(median, float)
    assert isinstance(seed_profits[0], float)


def test_run_config_worker_key_matches_config():
    """Worker returns the correct config key."""
    from train import run_config_worker, _config_key

    eps = make_fake_episodes(20)
    config = {"lr": 1e-4, "epsilon_decay": 300, "seq_len": 20, "lstm_hidden": 48, "epochs": 1}
    seeds = [42]

    key, _, _ = run_config_worker(config, seeds, eps, eps, eps)

    assert key == _config_key(config)


def test_run_config_worker_multiple_seeds():
    """Worker runs all seeds and returns one profit per seed."""
    from train import run_config_worker

    eps = make_fake_episodes(20)
    config = {"lr": 1e-4, "epsilon_decay": 150, "seq_len": 10, "lstm_hidden": 16, "epochs": 1}
    seeds = [42, 123]

    key, seed_profits, median = run_config_worker(config, seeds, eps, eps, eps)

    assert len(seed_profits) == 2
    import numpy as np
    assert median == float(np.median(seed_profits))


def test_parse_args_num_workers_default():
    """--num-workers defaults to None (meaning use cpu_count)."""
    import sys
    from unittest.mock import patch
    with patch.object(sys, 'argv', ['train.py', '--data', 'x.json']):
        from train import parse_args
        args = parse_args()
    assert args.num_workers is None


def test_parse_args_num_workers_explicit():
    """--num-workers can be set explicitly."""
    import sys
    from unittest.mock import patch
    with patch.object(sys, 'argv', ['train.py', '--data', 'x.json', '--num-workers', '4']):
        from train import parse_args
        args = parse_args()
    assert args.num_workers == 4
