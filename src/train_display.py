"""Rich Live terminal display for single-run training progress."""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

_ACTION_DISPLAY_NAMES = [
    "Do nothing",
    "Buy UP (taker)",
    "Sell UP (taker)",
    "Buy DOWN (taker)",
    "Sell DOWN (taker)",
    "Limit Buy UP",
    "Limit Sell UP",
    "Limit Buy DOWN",
    "Limit Sell DOWN",
]
_ACTION_KEYS = [
    "do_nothing", "buy_up_taker", "sell_up_taker",
    "buy_down_taker", "sell_down_taker",
    "limit_buy_up", "limit_sell_up",
    "limit_buy_down", "limit_sell_down",
]
_MAX_HISTORY_ROWS = 10
_BAR_WIDTH = 20


def _format_profit(cents: float) -> str:
    """Format profit in cents as a dollar string (e.g. '+$0.13')."""
    dollars = cents / 100.0
    sign = "+" if dollars >= 0 else "-"
    abs_rounded = Decimal(str(abs(dollars))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{sign}${abs_rounded}"


def _action_bar(fraction: float, width: int = _BAR_WIDTH) -> str:
    """Render a text progress bar for an action fraction."""
    filled = int(round(fraction * width))
    filled = max(0, min(filled, width))
    return "\u2588" * filled + "\u2591" * (width - filled)


class TrainDisplay:
    """Three-panel Rich Live display for single-run training.

    Use as a context manager to start/stop the Live display.
    Call update() after each validation checkpoint.
    """

    def __init__(
        self,
        config: dict,
        max_hours: float,
        elapsed_offset: float = 0.0,
    ) -> None:
        self._config = config
        self._max_hours = max_hours
        self._start_wall = datetime.now()
        self._elapsed_offset = elapsed_offset  # seconds already spent before this session
        self._history: list[dict] = []
        self._latest_dist: Optional[dict] = None
        self._episode_count = 0
        self._epoch = 0
        self._epsilon = 1.0
        self._last_ep_count: int = 0
        self._last_update_time: datetime = datetime.now()
        self._speed_eps_per_sec: float = 0.0
        self._live = Live(
            self._render(), refresh_per_second=2, screen=False
        )

    def __enter__(self) -> "TrainDisplay":
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self._live.__exit__(*args)

    def update(
        self,
        episode: int,
        val_profit: float,
        best_profit: float,
        median_profit: float,
        epoch_median: float,
        epoch: int,
        epsilon: float,
        action_distribution: dict[str, float],
        checkpoint_num: int,
        is_new_best: bool = False,
    ) -> None:
        """Refresh display with latest validation results.

        Args:
            is_new_best: True if this checkpoint set a new best profit.
                         Passed explicitly from the coordinator to avoid
                         ambiguity (best_profit is already updated by the
                         time update() is called).
        """
        import statistics as _stats

        # Compute speed (eps/sec) since last update
        now = datetime.now()
        dt = (now - self._last_update_time).total_seconds()
        if dt > 0 and episode > self._last_ep_count:
            self._speed_eps_per_sec = (episode - self._last_ep_count) / dt
        self._last_ep_count = episode
        self._last_update_time = now

        self._episode_count = episode
        self._epoch = epoch
        self._epsilon = epsilon
        self._latest_dist = action_distribution

        # Recent median: median of the last _MAX_HISTORY_ROWS val profits
        recent_vals = [r["val_profit"] for r in self._history[-((_MAX_HISTORY_ROWS - 1)):]] + [val_profit]
        recent_median = _stats.median(recent_vals)

        self._history.append({
            "checkpoint": checkpoint_num,
            "episode": episode,
            "val_profit": val_profit,
            "best_profit": best_profit,
            "median_profit": median_profit,
            "recent_median": recent_median,
            "epoch_median": epoch_median,
            "epsilon": epsilon,
            "is_best": is_new_best,
        })
        # Cap history to _MAX_HISTORY_ROWS
        if len(self._history) > _MAX_HISTORY_ROWS:
            self._history = self._history[-_MAX_HISTORY_ROWS:]
        self._live.update(self._render())

    def _elapsed_total(self) -> float:
        """Total elapsed seconds including offset from prior sessions."""
        wall = (datetime.now() - self._start_wall).total_seconds()
        return self._elapsed_offset + wall

    def _render(self) -> Group:
        return Group(
            self._status_panel(),
            self._history_panel(),
            self._action_panel(),
        )

    def _status_panel(self) -> Panel:
        elapsed_secs = self._elapsed_total()
        elapsed_td = timedelta(seconds=int(elapsed_secs))
        remaining_secs = self._max_hours * 3600 - elapsed_secs
        if remaining_secs > 0:
            remaining_str = str(timedelta(seconds=int(remaining_secs)))
        else:
            remaining_str = "EXPIRED"

        cfg = self._config
        content = (
            f"lr={cfg.get('lr')}  hidden={cfg.get('lstm_hidden')}  "
            f"seq={cfg.get('seq_len')}  \u03b5-decay={cfg.get('epsilon_decay')}  "
            f"gpus={cfg.get('num_gpus', 1)}\n"
            f"Started: {self._start_wall.strftime('%Y-%m-%d %H:%M')}  \u2502  "
            f"Elapsed: {elapsed_td}  \u2502  Remaining: {remaining_str}\n"
            f"Epoch: {self._epoch}  \u2502  Episodes: {self._episode_count:,}  \u2502  Speed: {self._speed_eps_per_sec:.1f} eps/sec  \u2502  \u03b5: {self._epsilon:.3f}"
        )
        return Panel(content, title="Single Run Training")

    def _history_panel(self) -> Panel:
        table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        table.add_column("#", style="dim", width=5)
        table.add_column("Episode", width=9)
        table.add_column("Val Profit", width=12)
        table.add_column("Best", width=10)
        table.add_column("All-time", width=10)
        table.add_column("Recent", width=10)
        table.add_column("Epoch", width=10)
        table.add_column("\u03b5", width=7)

        rows = self._history[-_MAX_HISTORY_ROWS:]
        for r in rows:
            star = " \u2605" if r["is_best"] else ""
            style = "bold green" if r["is_best"] else ""
            table.add_row(
                str(r["checkpoint"]),
                str(r["episode"]),
                f"{_format_profit(r['val_profit'])}{star}",
                _format_profit(r["best_profit"]),
                _format_profit(r["median_profit"]),
                _format_profit(r["recent_median"]),
                _format_profit(r["epoch_median"]),
                f"{r['epsilon']:.3f}",
                style=style,
            )
        return Panel(table, title="Validation History")

    def _action_panel(self) -> Panel:
        checkpoint = self._history[-1]["checkpoint"] if self._history else 0
        if not self._latest_dist:
            return Panel("(waiting for first checkpoint...)",
                         title=f"Action Distribution (checkpoint #{checkpoint})")

        lines = []
        for key, name in zip(_ACTION_KEYS, _ACTION_DISPLAY_NAMES):
            pct = self._latest_dist.get(key, 0.0)
            bar = _action_bar(pct)
            lines.append(f"  {name:<18} {bar}  {pct*100:5.1f}%")

        return Panel(
            "\n".join(lines),
            title=f"Action Distribution (checkpoint #{checkpoint})",
        )
