"""Tests for GridDisplay and Trainer on_validation callback."""


def test_trainer_on_validation_callback_is_called():
    """Trainer calls on_validation at each validation checkpoint."""
    from unittest.mock import MagicMock
    from src.trainer import Trainer
    from src.models.lstm_dqn import LSTMDQN
    from src.normalizer import Normalizer

    callback = MagicMock()

    eps = [
        {
            "hour": 9, "day": 0,
            "diff_pct_prev_session": 0.01,
            "diff_pct_hour": 0.02,
            "avg_pct_variance_hour": 0.005,
            "outcome": "up",
            "rows": [
                {"up_bid": 48, "up_ask": 52, "down_bid": 48, "down_ask": 52,
                 "diff_pct": 0.001, "time_to_close": 150000}
            ] * 5,
        }
    ] * 60  # 60 episodes so val_every_episodes=50 triggers once

    normalizer = Normalizer()
    normalizer.fit(eps)
    model = LSTMDQN(lstm_hidden_size=16)
    trainer = Trainer(
        model=model,
        normalizer=normalizer,
        config={"val_every_episodes": 50, "epsilon_decay_episodes": 100},
        on_validation=callback,
    )
    trainer.train(train_episodes=eps[:50], val_episodes=eps[50:], num_epochs=1)
    trainer.close()

    assert callback.called, "on_validation callback was never called"
    # Callback receives (episode_count, val_profit, epsilon)
    args = callback.call_args[0]
    assert len(args) == 3
    episode_count, val_profit, epsilon = args
    assert isinstance(episode_count, int)
    assert isinstance(val_profit, float)
    assert isinstance(epsilon, float)


# ──────────────────────────────────────────────────────────────────────────────
# GridDisplay tests
# ──────────────────────────────────────────────────────────────────────────────

def _make_configs():
    return [
        {"lr": 1e-4, "epsilon_decay": 150, "seq_len": 10, "lstm_hidden": 48, "epochs": 1},
        {"lr": 3e-4, "epsilon_decay": 300, "seq_len": 20, "lstm_hidden": 48, "epochs": 1},
    ]


def test_grid_display_initial_state_is_pending():
    """All configs start as Pending."""
    from src.grid_display import GridDisplay
    configs = _make_configs()
    display = GridDisplay(configs, total=10, completed=5)
    for key, state in display._states.items():
        assert state["status"] == "Pending"


def test_grid_display_update_seed_start():
    """seed_start event moves config to Running."""
    from src.grid_display import GridDisplay
    from src.grid_utils import config_key
    configs = _make_configs()
    display = GridDisplay(configs, total=10, completed=5)
    key = config_key(configs[0])
    display.update({"key": key, "event": "seed_start", "seed": 42, "total_seeds": 3})
    assert display._states[key]["status"] == "Running"
    assert display._states[key]["total_seeds"] == 3


def test_grid_display_update_val():
    """val event stores val_profit and updates best_val."""
    from src.grid_display import GridDisplay
    from src.grid_utils import config_key
    configs = _make_configs()
    display = GridDisplay(configs, total=10, completed=5)
    key = config_key(configs[0])
    display.update({"key": key, "event": "seed_start", "seed": 42, "total_seeds": 3})
    display.update({"key": key, "event": "val", "seed": 42,
                    "episode": 50, "val_profit": 294.5, "epsilon": 0.8})
    assert display._states[key]["val_profit"] == 294.5
    assert display._states[key]["best_val"] == 294.5
    assert display._states[key]["episode"] == 50

    # Second val — best_val should track the maximum; episode advances
    display.update({"key": key, "event": "val", "seed": 42,
                    "episode": 100, "val_profit": 412.0, "epsilon": 0.6})
    assert display._states[key]["best_val"] == 412.0
    assert display._states[key]["episode"] == 100


def test_grid_display_update_config_done():
    """config_done event sets status to Done and stores median."""
    from src.grid_display import GridDisplay
    from src.grid_utils import config_key
    configs = _make_configs()
    display = GridDisplay(configs, total=10, completed=5)
    key = config_key(configs[0])
    display.update({"key": key, "event": "config_done",
                    "median": 388.0, "seed_profits": [294.5, 387.9, 485.0]})
    assert display._states[key]["status"] == "Done [OK]"
    assert display._states[key]["median"] == 388.0
    assert display._done_count == 6  # was 5, incremented by 1


def test_grid_display_unknown_key_is_ignored():
    """Messages with unknown keys do not raise."""
    from src.grid_display import GridDisplay
    configs = _make_configs()
    display = GridDisplay(configs, total=10, completed=5)
    display.update({"key": "nonexistent_key", "event": "val", "val_profit": 100.0})
    # Should not raise


def test_grid_display_seed_done_resets_val_profit():
    """seed_done event clears val_profit so it doesn't bleed across seeds."""
    from src.grid_display import GridDisplay
    from src.grid_utils import config_key
    configs = _make_configs()
    display = GridDisplay(configs, total=10, completed=5)
    key = config_key(configs[0])
    # Simulate: seed_start -> val -> seed_done
    display.update({"key": key, "event": "seed_start", "seed": 42, "total_seeds": 3})
    display.update({"key": key, "event": "val", "seed": 42,
                    "episode": 50, "val_profit": 294.5, "epsilon": 0.8})
    assert display._states[key]["val_profit"] == 294.5
    display.update({"key": key, "event": "seed_done", "seed": 42, "seeds_done": 1})
    assert display._states[key]["val_profit"] is None  # reset between seeds
    assert display._states[key]["episode"] is None     # reset between seeds


def test_grid_display_start_polling_flushes_after_stop():
    """start_polling processes messages that arrive after stop_event is set."""
    import queue
    import threading
    from src.grid_display import GridDisplay
    from src.grid_utils import config_key

    configs = _make_configs()
    display = GridDisplay(configs, total=10, completed=5)
    key = config_key(configs[0])

    # Use a plain queue.Queue for same-process testing
    q = queue.Queue()
    stop_event = threading.Event()

    # Start polling thread
    t = threading.Thread(target=display.start_polling, args=(q, stop_event), daemon=True)
    t.start()

    # Set stop before putting the message — flush loop must still pick it up
    stop_event.set()
    q.put({"key": key, "event": "config_done", "median": 200.0, "seed_profits": [200.0]})

    t.join(timeout=2.0)
    assert not t.is_alive(), "Polling thread did not exit"
    # Message put after stop_event.set() must have been processed
    assert display._states[key]["median"] == 200.0
