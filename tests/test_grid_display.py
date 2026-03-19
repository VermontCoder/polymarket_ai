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
