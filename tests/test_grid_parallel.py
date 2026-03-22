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


def test_run_config_worker_gpu_assignment():
    """Worker assigns GPUs round-robin based on worker_id."""
    from train import run_config_worker
    from unittest.mock import patch

    eps = make_fake_episodes(20)
    config = {"lr": 1e-4, "epsilon_decay": 150, "seq_len": 10, "lstm_hidden": 16, "epochs": 1}
    seeds = [42]

    # Mock torch.cuda.device_count to return 4 GPUs
    with patch('torch.cuda.device_count', return_value=4):
        # Test worker_id 0 -> GPU 0
        with patch('train.train_single') as mock_train:
            mock_train.return_value = 100.0
            run_config_worker(config, seeds, eps, eps, eps, worker_id=0)
            # Check that train_single was called with cuda:0
            call_args = mock_train.call_args
            device_arg = call_args.kwargs.get('device')
            assert str(device_arg) == 'cuda:0'

        # Test worker_id 3 -> GPU 3
        with patch('train.train_single') as mock_train:
            mock_train.return_value = 100.0
            run_config_worker(config, seeds, eps, eps, eps, worker_id=3)
            call_args = mock_train.call_args
            device_arg = call_args.kwargs.get('device')
            assert str(device_arg) == 'cuda:3'

        # Test worker_id 4 -> GPU 0 (wrap around)
        with patch('train.train_single') as mock_train:
            mock_train.return_value = 100.0
            run_config_worker(config, seeds, eps, eps, eps, worker_id=4)
            call_args = mock_train.call_args
            device_arg = call_args.kwargs.get('device')
            assert str(device_arg) == 'cuda:0'

    # Test no GPUs available -> CPU
    with patch('torch.cuda.device_count', return_value=0):
        with patch('train.train_single') as mock_train:
            mock_train.return_value = 100.0
            run_config_worker(config, seeds, eps, eps, eps, worker_id=0)
            call_args = mock_train.call_args
            device_arg = call_args.kwargs.get('device')
            assert str(device_arg) == 'cpu'
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


def test_grid_search_uses_multiple_workers(tmp_path):
    """grid_search dispatches work to ProcessPoolExecutor with num_workers."""
    import concurrent.futures
    from unittest.mock import patch, MagicMock, call
    from train import grid_search

    eps = make_fake_episodes(30)

    captured_max_workers = []

    original_executor = concurrent.futures.ProcessPoolExecutor

    class CapturingExecutor:
        def __init__(self, max_workers=None):
            captured_max_workers.append(max_workers)
            self._inner = original_executor(max_workers=1)  # use 1 for test speed

        def submit(self, fn, *args, **kwargs):
            return self._inner.submit(fn, *args, **kwargs)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

    results_path = str(tmp_path / "grid_results.json")
    save_path = str(tmp_path / "model.pt")

    with patch("train.GRID_RESULTS_PATH", results_path), \
         patch("concurrent.futures.ProcessPoolExecutor", CapturingExecutor):
        # Run with a tiny param grid — patch param_grid inside grid_search
        tiny_grid = {
            "lr": [1e-4],
            "epsilon_decay": [150],
            "seq_len": [10],
            "lstm_hidden": [16],
        }
        with patch("train.PARAM_GRID", tiny_grid):
            grid_search(eps, eps, eps, save_path, seeds=[42], num_workers=3)

    assert captured_max_workers == [3]


def test_grid_search_resumes_skips_completed(tmp_path):
    """grid_search skips configs already present in grid_results.json."""
    import json
    import concurrent.futures
    from unittest.mock import patch
    from train import grid_search, _config_key, run_config_worker

    eps = make_fake_episodes(30)
    results_path = tmp_path / "grid_results.json"
    save_path = str(tmp_path / "model.pt")

    # Pre-populate results with one config
    completed_config = {"lr": 1e-4, "epsilon_decay": 150, "seq_len": 10, "lstm_hidden": 16}
    key = _config_key(completed_config)
    existing = {
        key: {
            "config": completed_config,
            "seed_profits": [100.0],
            "median_val_profit": 100.0,
        }
    }
    results_path.write_text(json.dumps(existing))

    submitted_keys = []

    def capturing_worker(config, seeds, *args):
        submitted_keys.append(_config_key(config))
        return run_config_worker(config, seeds, *args)

    tiny_grid = {
        "lr": [1e-4],
        "epsilon_decay": [150],
        "seq_len": [10],
        "lstm_hidden": [16],
    }

    # IMPORTANT: Use ThreadPoolExecutor instead of ProcessPoolExecutor so that
    # unittest.mock patches applied in the parent process are visible to
    # workers. ProcessPoolExecutor spawns subprocesses on Windows (spawn start
    # method), which import 'train' fresh — bypassing any in-process patches.
    with patch("train.GRID_RESULTS_PATH", str(results_path)), \
         patch("train.PARAM_GRID", tiny_grid), \
         patch("train.run_config_worker", side_effect=capturing_worker), \
         patch("concurrent.futures.ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor):
        grid_search(eps, eps, eps, save_path, seeds=[42], num_workers=1)

    # The one already-completed config should NOT have been submitted
    assert key not in submitted_keys


def test_run_config_worker_pushes_events_to_queue():
    """When status_queue is provided, worker pushes seed_start, val, and seed_done events."""
    import queue
    from train import run_config_worker, _config_key

    # Need enough episodes for validation to fire: val_every_episodes default is 50,
    # so use 60 training eps. Pass val_every_episodes=50 explicitly via config to be
    # sure the callback path is exercised regardless of DEFAULT_CONFIG changes.
    eps = make_fake_episodes(60)
    config = {
        "lr": 1e-4, "epsilon_decay": 150, "seq_len": 10, "lstm_hidden": 16, "epochs": 1,
        "val_every_episodes": 50,
    }

    # Use a plain queue.Queue — call worker in-process to avoid spawn overhead
    q = queue.Queue()

    key, seed_profits, median = run_config_worker(config, [42], eps, eps, eps, status_queue=q)

    events = []
    while not q.empty():
        events.append(q.get_nowait())

    event_types = [e["event"] for e in events]
    assert "seed_start" in event_types, f"Expected seed_start in {event_types}"
    assert "seed_done" in event_types, f"Expected seed_done in {event_types}"
    assert "val" in event_types, (
        f"Expected val event (on_validation callback path) in {event_types}"
    )
    # All events should carry the correct key
    for e in events:
        assert e["key"] == key
