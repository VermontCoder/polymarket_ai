"""Rich live display for parallel grid search progress."""

import threading

from rich.live import Live
from rich.table import Table
from rich.text import Text

from src.grid_utils import config_key as _config_key


class GridDisplay:
    """Live-updating Rich table showing status of all grid search workers.

    Usage:
        with GridDisplay(pending_configs, total=54, completed=10) as display:
            stop = threading.Event()
            t = threading.Thread(
                target=display.start_polling, args=(queue, stop), daemon=True
            )
            t.start()
            # ... run executor ...
            stop.set()
            t.join()
    """

    def __init__(self, pending_configs: list, total: int, completed: int) -> None:
        self._lock = threading.Lock()
        self._total = total
        self._done_count = completed
        self._states: dict = {}

        for config in pending_configs:
            key = _config_key(config)
            self._states[key] = {
                "config": config,
                "status": "Pending",
                "current_seed": None,
                "seeds_done": 0,
                "total_seeds": None,
                "episode": None,
                "val_profit": None,
                "best_val": None,
                "median": None,
                "epsilon": None,
            }

        self._live = Live(self._make_table(), refresh_per_second=4, transient=False)

    def _make_table(self) -> Table:
        """Build a Rich Table from current state."""
        table = Table(
            title=f"Grid Search  {self._done_count}/{self._total} complete",
            show_lines=False,
        )
        table.add_column("LR", style="cyan", no_wrap=True)
        table.add_column("ε-decay", justify="right")
        table.add_column("seq", justify="right")
        table.add_column("hidden", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Seeds", justify="center")
        table.add_column("Episode", justify="right")
        table.add_column("Val Profit", justify="right")
        table.add_column("Best Val", justify="right")
        table.add_column("Median", justify="right")

        for key, s in self._states.items():
            config = s["config"]
            status_text = self._status_text(s["status"])
            seeds_text = (
                f"{s['seeds_done']}/{s['total_seeds']}"
                if s["total_seeds"] is not None
                else "—"
            )
            episode_text = str(s["episode"]) if s["episode"] is not None else "—"
            val_text = (
                f"{s['val_profit']:+.1f}c" if s["val_profit"] is not None else "—"
            )
            best_val_text = (
                self._profit_text(s["best_val"]) if s["best_val"] is not None else "—"
            )
            median_text = (
                self._profit_text(s["median"]) if s["median"] is not None else "—"
            )
            table.add_row(
                str(config["lr"]),
                str(config["epsilon_decay"]),
                str(config["seq_len"]),
                str(config["lstm_hidden"]),
                status_text,
                seeds_text,
                episode_text,
                val_text,
                best_val_text,
                median_text,
            )

        return table

    @staticmethod
    def _status_text(status: str) -> Text:
        if status == "Running":
            return Text(status, style="bold yellow")
        if status == "Done ✓":
            return Text(status, style="bold green")
        return Text(status, style="dim")

    @staticmethod
    def _profit_text(value: float) -> Text:
        style = "green" if value >= 0 else "red"
        return Text(f"{value:+.1f}c", style=style)

    def update(self, msg: dict) -> None:
        """Process a status message from a worker. Thread-safe."""
        with self._lock:
            key = msg.get("key")
            if key not in self._states:
                return

            event = msg["event"]
            state = self._states[key]

            if event == "seed_start":
                state["status"] = "Running"
                state["current_seed"] = msg["seed"]
                state["total_seeds"] = msg["total_seeds"]
            elif event == "val":
                state["episode"] = msg.get("episode")
                state["val_profit"] = msg["val_profit"]
                state["epsilon"] = msg.get("epsilon")
                prev_best = state["best_val"] if state["best_val"] is not None else -float("inf")
                state["best_val"] = max(prev_best, msg["val_profit"])
            elif event == "seed_done":
                state["seeds_done"] = msg["seeds_done"]
                state["episode"] = None
                state["val_profit"] = None
            elif event == "config_done":
                state["status"] = "Done ✓"
                state["val_profit"] = None
                state["median"] = msg["median"]
                if state["total_seeds"] is not None:
                    state["seeds_done"] = state["total_seeds"]
                self._done_count += 1

            self._live.update(self._make_table())

    def start_polling(self, status_queue, stop_event: threading.Event) -> None:
        """Thread target: drain queue until stop_event is set, then flush remainder."""
        import queue as _queue
        while not stop_event.is_set():
            try:
                msg = status_queue.get(timeout=0.1)
                self.update(msg)
            except _queue.Empty:
                pass
        # Flush any remaining messages after stop is signalled
        while True:
            try:
                msg = status_queue.get_nowait()
                self.update(msg)
            except _queue.Empty:
                break

    def __enter__(self):
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        self._live.__exit__(*args)
