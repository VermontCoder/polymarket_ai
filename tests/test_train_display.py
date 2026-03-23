"""Tests for TrainDisplay rendering (no Live, just panel output)."""
import pytest
from src.train_display import TrainDisplay, _format_profit, _action_bar


class TestFormatProfit:
    def test_positive(self):
        assert _format_profit(12.5) == "+$0.13"

    def test_negative(self):
        assert _format_profit(-50.0) == "-$0.50"

    def test_zero(self):
        assert _format_profit(0.0) == "+$0.00"


class TestActionBar:
    def test_full_bar(self):
        bar = _action_bar(1.0, width=4)
        assert bar == "████"

    def test_empty_bar(self):
        bar = _action_bar(0.0, width=4)
        assert bar == "░░░░"

    def test_half_bar(self):
        bar = _action_bar(0.5, width=4)
        assert bar == "██░░"


class TestTrainDisplay:
    _DIST = {
        "do_nothing": 0.5, "buy_up_taker": 0.1, "sell_up_taker": 0.05,
        "buy_down_taker": 0.1, "sell_down_taker": 0.05,
        "limit_buy_up": 0.05, "limit_sell_up": 0.05,
        "limit_buy_down": 0.05, "limit_sell_down": 0.05,
    }
    _CONFIG = {"lr": 1e-4, "lstm_hidden": 64, "seq_len": 20,
               "epsilon_decay": 150, "num_gpus": 1}

    def test_update_adds_history_entry(self):
        display = TrainDisplay(config=self._CONFIG, max_hours=12.0)
        display.update(
            episode=50, val_profit=10.0, best_profit=10.0,
            median_profit=10.0, epoch_median=10.0, epoch=0, epsilon=0.9,
            action_distribution=self._DIST, checkpoint_num=1,
            is_new_best=True,
        )
        assert len(display._history) == 1

    def test_history_capped_at_ten(self):
        display = TrainDisplay(config=self._CONFIG, max_hours=12.0)
        for i in range(15):
            display.update(
                episode=(i+1)*50, val_profit=float(i), best_profit=float(i),
                median_profit=float(i), epoch_median=float(i), epoch=0, epsilon=0.9,
                action_distribution=self._DIST, checkpoint_num=i+1,
                is_new_best=False,
            )
        assert len(display._history) == 10

    def test_render_returns_renderable(self):
        from rich.console import ConsoleRenderable
        display = TrainDisplay(config=self._CONFIG, max_hours=12.0)
        renderable = display._render()
        # Should not raise; Rich Group is renderable
        assert renderable is not None
