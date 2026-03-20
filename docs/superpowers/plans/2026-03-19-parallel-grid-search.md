# Parallel Grid Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize the hyperparameter grid search in `train.py` so each config runs on its own CPU core simultaneously, and display a live Rich dashboard showing each worker's hyperparameters and training status updated in place.

**Architecture:** A top-level `run_config_worker()` function (required for `multiprocessing` pickling on Windows) runs all seeds for one config sequentially and returns the results. Workers communicate status back to the parent via a `multiprocessing.Manager().Queue()` — the only safe IPC channel across `spawn` processes. The parent runs a daemon polling thread that reads from the queue and updates a `GridDisplay` object (Rich `Live` table). `Trainer` gains an optional `on_validation` callback so workers can push real-time validation events to the queue. Workers redirect their own stdout to devnull when a queue is present so raw prints don't corrupt the Rich display. A `--num-workers` CLI flag controls the pool size.

**Tech Stack:** Python `concurrent.futures.ProcessPoolExecutor` (stdlib), `multiprocessing.Manager`, `threading`, `rich` (Live, Table, Text), existing PyTorch/Normalizer/Trainer infrastructure.

---

## Context & Constraints

- **Windows `spawn` start method** is the default on Windows. This means:
  - Worker functions must be defined at the **top level** of the module (not nested, not lambda), so they can be pickled.
  - The `if __name__ == "__main__":` guard in `main()` is already present — required and must stay.
- **Episode data pickling**: Episodes are plain Python dicts — they pickle fine. Each worker process gets its own copy via `spawn`, which is acceptable for ~1000 episodes.
- **No shared mutable state between workers**: Each worker creates its own model, normalizer, trainer, and replay buffer.
- **JSON checkpoint writes must be safe**: The parent process (main process) owns `grid_results.json`. Workers never write to it — they return results to the parent, which writes after each config completes.
- **TensorBoard log dirs**: Each (config, seed) already uses a unique `log_dir`, so no file conflicts.
- **Temp checkpoint files**: Workers write temp `.pt` files. Each worker must use a unique path (keyed by config+seed) to avoid clobbering.

---

## File Map

| File | Change |
|------|--------|
| `src/grid_utils.py` | **New file** — `config_key()` utility function (moved here from `train.py` to avoid inverted import dependency) |
| `train.py` | Add `run_config_worker()` top-level function; modify `grid_search()` to use `ProcessPoolExecutor` + queue + display; add `--num-workers` CLI arg; add `on_validation` plumbing through `train_single`; import `config_key` from `src.grid_utils` |
| `src/trainer.py` | Add `on_validation` optional callback param; call it in `_log_validation()` |
| `src/grid_display.py` | **New file** — `GridDisplay` class: Rich `Live` table, thread-safe `update()`, `start_polling()` thread target; imports `config_key` from `src.grid_utils` (not from `train`) |
| `requirements.txt` | Add `rich` |
| `tests/test_grid_parallel.py` | New test file — verify parallel grid produces same results as sequential, handles resume, num_workers respected |
| `tests/test_grid_display.py` | **New file** — unit tests for `GridDisplay` state logic (no Live/subprocess dependencies) |

---

## Task 1: Add `run_config_worker()` top-level function to `train.py`

**Files:**
- Modify: `train.py` (add before `grid_search()`, at module top level)

This function is the unit of work dispatched to each process. It must be a **top-level function** (not nested) for `pickle` to serialize it on Windows.

- [ ] **Step 1: Write the failing test**

Create `tests/test_grid_parallel.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_grid_parallel.py -v
```
Expected: `ImportError` or `AttributeError: module 'train' has no attribute 'run_config_worker'`

- [ ] **Step 3: Implement `run_config_worker()` in `train.py`**

Add this function **before** `grid_search()` in `train.py` — it must be at module top level:

```python
def run_config_worker(
    config: dict,
    seeds: list,
    train_eps: list,
    val_eps: list,
    test_eps: list,
) -> tuple:
    """Worker function for parallel grid search.

    Runs all seeds for one config sequentially and returns results.
    Must be a top-level function for multiprocessing pickling on Windows.

    Args:
        config: Hyperparameter dict (including 'epochs').
        seeds: List of random seeds to run.
        train_eps: Training episodes.
        val_eps: Validation episodes.
        test_eps: Test episodes.

    Returns:
        Tuple of (config_key, seed_profits, median_val_profit).
    """
    key = _config_key(config)
    seed_profits = []

    for seed in seeds:
        log_dir = (
            f"runs/grid_lr{config['lr']}_ed{config['epsilon_decay']}"
            f"_sl{config['seq_len']}_h{config['lstm_hidden']}_s{seed}"
        )
        # Unique temp path per worker to avoid file conflicts
        temp_path = (
            f"checkpoints/grid_temp_{key}_s{seed}.pt"
        )
        val_profit = train_single(
            train_eps, val_eps, test_eps, config, seed,
            save_path=temp_path,
            log_dir=log_dir,
        )
        seed_profits.append(val_profit)

    median_profit = float(np.median(seed_profits))
    return key, seed_profits, median_profit
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_grid_parallel.py -v
```
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_grid_parallel.py
git commit -m "feat: add run_config_worker top-level function for multiprocessing"
```

---

## Task 2: Add `--num-workers` CLI argument

**Files:**
- Modify: `train.py` — `parse_args()` function

- [ ] **Step 1: Write the failing test**

Add to `tests/test_grid_parallel.py`:

```python
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
    with patch.object(sys, 'argv', ['train.py', '--data', 'x.json', '--num-workers', '4']):
        from train import parse_args
        args = parse_args()
    assert args.num_workers == 4
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_grid_parallel.py::test_parse_args_num_workers_default tests/test_grid_parallel.py::test_parse_args_num_workers_explicit -v
```
Expected: `AttributeError: Namespace object has no attribute 'num_workers'`

- [ ] **Step 3: Add `--num-workers` to `parse_args()`**

In `train.py`, inside `parse_args()`, add after the existing `--epsilon-decay` argument:

```python
parser.add_argument(
    "--num-workers", type=int, default=None,
    help="Number of parallel worker processes for grid search. "
         "Defaults to os.cpu_count().",
)
```

Also add `import os` at the top of `train.py` if not already present (it is already imported).

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_grid_parallel.py::test_parse_args_num_workers_default tests/test_grid_parallel.py::test_parse_args_num_workers_explicit -v
```
Expected: Both PASS

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_grid_parallel.py
git commit -m "feat: add --num-workers CLI argument for parallel grid search"
```

---

## Task 3: Parallelize `grid_search()` with `ProcessPoolExecutor`

**Files:**
- Modify: `train.py` — `grid_search()` function
- Modify: `train.py` — `main()` to pass `num_workers`

This is the core change. The parent process submits one future per pending config, collects results as they complete (via `as_completed`), and writes the JSON checkpoint after each config finishes — exactly matching the current "save after each config" behavior.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_grid_parallel.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_grid_parallel.py::test_grid_search_uses_multiple_workers tests/test_grid_parallel.py::test_grid_search_resumes_skips_completed -v
```
Expected: FAIL — `grid_search()` doesn't have `num_workers` param; `PARAM_GRID` and `GRID_RESULTS_PATH` not module-level constants yet.

- [ ] **Step 3: Refactor `grid_search()` to use `ProcessPoolExecutor`**

In `train.py`:

**3a.** Add `import concurrent.futures` to the imports at the top.

**3b.** Extract the param grid and results path to module-level constants (makes them patchable in tests and removes magic values from the function):

```python
GRID_RESULTS_PATH = "checkpoints/grid_results.json"

PARAM_GRID = {
    "lr": [5e-5, 1e-4, 3e-4],
    "epsilon_decay": [150, 300],
    "seq_len": [10, 20, 40],
    "lstm_hidden": [16, 32, 48],
}
```

**3c.** Replace the entire `grid_search()` function body with:

```python
def grid_search(train_eps, val_eps, test_eps, save_path, seeds=None, num_workers=None):
    """Run hyperparameter grid search. Configs run in parallel across CPU cores.

    Args:
        train_eps: Training episodes.
        val_eps: Validation episodes.
        test_eps: Test episodes.
        save_path: Where to save the final best model.
        seeds: List of random seeds (default: [42, 123, 456]).
        num_workers: Number of parallel worker processes. Defaults to os.cpu_count().
    """
    if seeds is None:
        seeds = [42, 123, 456]

    results = _load_grid_results(GRID_RESULTS_PATH)

    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    all_combos = list(itertools.product(*values))
    total = len(all_combos)

    # Build list of pending configs (skip already completed)
    pending = []
    for combo in all_combos:
        config = dict(zip(keys, combo))
        config["epochs"] = 1
        if _config_key(config) not in results:
            pending.append(config)

    completed = total - len(pending)
    if completed > 0:
        print(f"Resuming grid search: {completed}/{total} configs already done")
    print(f"Running {len(pending)} remaining configs on up to {num_workers or os.cpu_count()} workers")

    # Restore best from previously completed results
    best_median = -float("inf")
    best_config = None
    for entry in results.values():
        if entry["median_val_profit"] > best_median:
            best_median = entry["median_val_profit"]
            best_config = entry["config"]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all pending configs
        future_to_config = {
            executor.submit(
                run_config_worker, config, seeds, train_eps, val_eps, test_eps
            ): config
            for config in pending
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_config):
            config = future_to_config[future]
            key, seed_profits, median_profit = future.result()

            print(f"\nConfig done: {config} | Median val profit: {median_profit:.2f}c")
            print(f"  Per-seed profits: {seed_profits}")

            results[key] = {
                "config": {k: v for k, v in config.items() if k != "epochs"},
                "seed_profits": seed_profits,
                "median_val_profit": median_profit,
            }
            _save_grid_results(GRID_RESULTS_PATH, results)

            if median_profit > best_median:
                best_median = median_profit
                best_config = config

    print(f"\n{'='*60}")
    print(f"Best config: {best_config}")
    print(f"Best median val profit: {best_median:.2f}c")
    print(f"\nRetraining best config with seed 42...")

    train_single(train_eps, val_eps, test_eps, best_config, seed=42, save_path=save_path)
```

**3d.** Update `main()` to pass `num_workers` through:

```python
if args.grid_search:
    grid_search(train_eps, val_eps, test_eps, args.save_path, num_workers=args.num_workers)
```

- [ ] **Step 4: Run all grid parallel tests**

```
pytest tests/test_grid_parallel.py -v
```
Expected: All tests PASS

- [ ] **Step 5: Run the full test suite to verify nothing broke**

```
pytest tests/ -v
```
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add train.py tests/test_grid_parallel.py
git commit -m "feat: parallelize grid search across CPU cores with ProcessPoolExecutor"
```

---

## Task 4: Verify end-to-end with a dry run

Before running the real grid search on the full dataset, verify the parallel code works correctly with a tiny smoke test.

- [ ] **Step 1: Create smoke test script and run it**

Create `smoke_parallel.py` at the project root. The `if __name__ == "__main__":` guard is **required** — without it, Windows `spawn` workers will re-execute the top-level code when they import the module, causing infinite process spawning.

```python
# smoke_parallel.py
"""Smoke test for parallel grid search. Run from project root."""
import train
from src.data_loader import load_episodes, split_episodes

if __name__ == "__main__":
    # Tiny param grid — just one config
    train.PARAM_GRID = {
        "lr": [1e-4],
        "epsilon_decay": [150],
        "seq_len": [10],
        "lstm_hidden": [16],
    }
    # Use a separate results file so we don't corrupt the real one
    train.GRID_RESULTS_PATH = "checkpoints/smoke_grid_results.json"

    eps = load_episodes("data/episodes.json")
    train_eps, val_eps, test_eps = split_episodes(eps)
    train.grid_search(
        train_eps, val_eps, test_eps,
        save_path="checkpoints/smoke_test.pt",
        seeds=[42],
        num_workers=1,
    )
    print("Smoke test passed.")
```

Run it:

```bash
python smoke_parallel.py
```

Expected: Runs without error, prints "Smoke test passed.", writes `checkpoints/smoke_grid_results.json` with one entry.

- [ ] **Step 2: Verify JSON output**

Open `checkpoints/smoke_grid_results.json` in an editor or run:

```bash
python -c "import json; d=json.load(open('checkpoints/smoke_grid_results.json')); print(list(d.keys()))"
```

Expected: Prints one key like `['lr=0.0001_ed=150_sl=10_h=16']`.

- [ ] **Step 3: Commit smoke test is clean (no changes needed)**

If the smoke test revealed any issues, fix them and re-run. Once clean:

```bash
git add -A
git commit -m "chore: verify parallel grid search end-to-end smoke test passes"
```

---

---

## Task 5a: Extract `config_key()` to `src/grid_utils.py`

**Files:**
- Create: `src/grid_utils.py`
- Modify: `train.py` — replace local `_config_key` with import

`grid_display.py` (in `src/`) must import `config_key`. Having a `src/` module import from the top-level `train.py` script is an inverted dependency — architecturally wrong and fragile. Moving the function to `src/grid_utils.py` gives both `train.py` and `grid_display.py` a neutral shared home to import from.

- [ ] **Step 1: Create `src/grid_utils.py`**

```python
"""Shared utilities for grid search — importable by both train.py and src modules."""


def config_key(config: dict) -> str:
    """Create a stable string key for a config dict (excludes 'epochs')."""
    return (
        f"lr={config['lr']}_ed={config['epsilon_decay']}"
        f"_sl={config['seq_len']}_h={config['lstm_hidden']}"
    )
```

- [ ] **Step 2: Update `train.py` to import from `src.grid_utils`**

Remove the `_config_key` function definition from `train.py`. Add at the top:

```python
from src.grid_utils import config_key as _config_key
```

All existing usages of `_config_key(...)` in `train.py` continue to work unchanged.

- [ ] **Step 3: Run existing tests to verify nothing broke**

```
pytest tests/ -v
```
Expected: All PASS (the function is the same, just moved)

- [ ] **Step 4: Commit**

```bash
git add src/grid_utils.py train.py
git commit -m "refactor: extract config_key to src/grid_utils to avoid inverted import"
```

---

## Task 5: Add `on_validation` callback to `Trainer` and `train_single`

**Files:**
- Modify: `src/trainer.py` — `__init__()` and `_log_validation()`
- Modify: `train.py` — `train_single()` signature

This is a small surgical change. The Trainer already knows when validation runs; we just need to expose a hook so callers (workers) can be notified without knowing about Trainer internals.

- [ ] **Step 1: Write the failing test**

Create `tests/test_grid_display.py` (we'll add display tests here in Task 6; for now just the callback test):

```python
"""Tests for GridDisplay and Trainer on_validation callback."""


def test_trainer_on_validation_callback_is_called():
    """Trainer calls on_validation at each validation checkpoint."""
    import numpy as np
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
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_grid_display.py::test_trainer_on_validation_callback_is_called -v
```
Expected: `TypeError: __init__() got unexpected keyword argument 'on_validation'`

- [ ] **Step 3: Add `on_validation` to `Trainer.__init__()` and `_log_validation()`**

In `src/trainer.py`, add to `__init__` signature and body:

```python
def __init__(
    self,
    model: BaseModel,
    normalizer: Normalizer,
    config: Optional[dict] = None,
    device: Optional[torch.device] = None,
    on_validation: Optional[callable] = None,
) -> None:
    ...
    self._on_validation = on_validation
```

In `_log_validation()`, add after the existing print statement:

```python
if self._on_validation is not None:
    self._on_validation(self._episode_count, val_profit, self.epsilon)
```

Add `on_validation=None` to `train_single()` and thread it through to `Trainer.__init__()`:

```python
def train_single(
    train_eps, val_eps, test_eps, config, seed, save_path, log_dir=None,
    on_validation=None,
):
    ...
    trainer = Trainer(
        model=model,
        normalizer=normalizer,
        config={...},
        device=device,
        on_validation=on_validation,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_grid_display.py::test_trainer_on_validation_callback_is_called -v
```
Expected: PASS

- [ ] **Step 5: Run full suite to verify nothing broke**

```
pytest tests/ -v
```
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/trainer.py train.py tests/test_grid_display.py
git commit -m "feat: add on_validation callback to Trainer for external status hooks"
```

---

## Task 6: Create `src/grid_display.py` with `GridDisplay`

**Files:**
- Create: `src/grid_display.py`
- Modify: `requirements.txt` (add `rich`)
- Modify: `tests/test_grid_display.py` (add display tests)

`GridDisplay` wraps a Rich `Live` table. Workers push status dicts via a queue; the parent's polling thread calls `display.update(msg)`. The display is also a context manager (`__enter__`/`__exit__`) so it integrates cleanly with `grid_search()`.

**Status message format** (pushed by workers to the queue):

```python
# Worker starting a seed:
{"key": "lr=0.0001_ed=150_sl=10_h=48", "event": "seed_start", "seed": 42, "total_seeds": 3}

# Validation checkpoint inside a seed (from on_validation callback):
{"key": "lr=0.0001_ed=150_sl=10_h=48", "event": "val", "seed": 42,
 "episode": 150, "val_profit": 294.5, "epsilon": 0.72}

# Seed finished:
{"key": "lr=0.0001_ed=150_sl=10_h=48", "event": "seed_done", "seed": 42, "seeds_done": 2}

# Config fully done (sent by parent after computing median):
{"key": "lr=0.0001_ed=150_sl=10_h=48", "event": "config_done",
 "median": 388.0, "seed_profits": [294.5, 414.7, 454.8]}
```

- [ ] **Step 1: Add `rich` to `requirements.txt`**

```
torch
tensorboard
pytest
numpy
rich
```

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_grid_display.py`:

```python
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
    assert display._states[key]["status"] == "Done ✓"
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
```

- [ ] **Step 3: Run tests to verify they fail**

```
pytest tests/test_grid_display.py -k "display" -v
```
Expected: `ModuleNotFoundError: No module named 'src.grid_display'`

- [ ] **Step 4: Implement `src/grid_display.py`**

```python
"""Rich live display for parallel grid search progress."""

import threading
from typing import Optional

from rich.live import Live
from rich.table import Table
from rich.text import Text

from src.grid_utils import config_key as _config_key


class GridDisplay:
    """Live-updating Rich table showing status of all grid search workers.

    Usage:
        with GridDisplay(pending_configs, total=54, completed=10) as display:
            # start polling thread
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
        """Build a Rich Table from current state. Caller must hold self._lock (or be in __init__)."""
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
                state["episode"] = None   # reset between seeds
                state["val_profit"] = None  # reset between seeds
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
```

- [ ] **Step 5: Run display tests**

```
pytest tests/test_grid_display.py -k "display" -v
```
Expected: All 5 display tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/grid_display.py requirements.txt tests/test_grid_display.py
git commit -m "feat: add GridDisplay Rich live table for parallel grid search progress"
```

---

## Task 7: Wire queue + display into `run_config_worker` and `grid_search()`

**Files:**
- Modify: `train.py` — `run_config_worker()`, `grid_search()`

This is the integration step. Workers get the queue, suppress their stdout, and push events. The parent creates the Manager + queue + display + polling thread, then wraps the executor.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_grid_parallel.py`:

```python
def test_run_config_worker_pushes_events_to_queue():
    """When status_queue is provided, worker pushes seed_start and seed_done events."""
    import multiprocessing
    from train import run_config_worker, _config_key

    eps = make_fake_episodes(60)
    config = {"lr": 1e-4, "epsilon_decay": 150, "seq_len": 10, "lstm_hidden": 16, "epochs": 1}

    # Use a plain multiprocessing.Queue — run worker in-process via direct call
    # (not via ProcessPoolExecutor) so we can inspect the queue
    q = multiprocessing.Queue()

    # Call worker directly (not via executor) to avoid spawn overhead in tests
    key, seed_profits, median = run_config_worker(config, [42], eps, eps, eps, status_queue=q)

    events = []
    while not q.empty():
        events.append(q.get_nowait())

    event_types = [e["event"] for e in events]
    assert "seed_start" in event_types
    assert "seed_done" in event_types
    # All events should carry the correct key
    for e in events:
        assert e["key"] == key
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_grid_parallel.py::test_run_config_worker_pushes_events_to_queue -v
```
Expected: FAIL — `run_config_worker` doesn't accept `status_queue`

- [ ] **Step 3: Update `run_config_worker` to accept queue and push events**

Replace the existing `run_config_worker` with:

```python
def run_config_worker(
    config: dict,
    seeds: list,
    train_eps: list,
    val_eps: list,
    test_eps: list,
    status_queue=None,
) -> tuple:
    """Worker function for parallel grid search.

    Runs all seeds for one config sequentially and returns results.
    Must be a top-level function for multiprocessing pickling on Windows.

    When status_queue is provided:
    - Redirects stdout to devnull (prevents raw prints corrupting Rich display)
    - Pushes seed_start, val, seed_done events to the queue

    Returns:
        Tuple of (config_key, seed_profits, median_val_profit).
    """
    import sys
    import os as _os

    key = _config_key(config)
    seed_profits = []
    total_seeds = len(seeds)

    # Suppress stdout in workers so Rich display in parent isn't corrupted.
    # stderr is intentionally left alone so tracebacks remain visible.
    # Restore stdout in a finally block to avoid handle leaks.
    _orig_stdout = sys.stdout
    _devnull = None
    if status_queue is not None:
        _devnull = open(_os.devnull, "w")
        sys.stdout = _devnull

    try:

        for seed_idx, seed in enumerate(seeds):
            if status_queue is not None:
                status_queue.put({
                    "key": key,
                    "event": "seed_start",
                    "seed": seed,
                    "total_seeds": total_seeds,
                })

            log_dir = (
                f"runs/grid_lr{config['lr']}_ed{config['epsilon_decay']}"
                f"_sl{config['seq_len']}_h{config['lstm_hidden']}_s{seed}"
            )
            temp_path = f"checkpoints/grid_temp_{key}_s{seed}.pt"

            # Build on_validation callback if queue is present
            on_validation = None
            if status_queue is not None:
                def _make_callback(q, k, s):
                    def callback(episode, val_profit, epsilon):
                        q.put({
                            "key": k,
                            "event": "val",
                            "seed": s,
                            "episode": episode,
                            "val_profit": val_profit,
                            "epsilon": epsilon,
                        })
                    return callback
                on_validation = _make_callback(status_queue, key, seed)

            val_profit = train_single(
                train_eps, val_eps, test_eps, config, seed,
                save_path=temp_path,
                log_dir=log_dir,
                on_validation=on_validation,
            )
            seed_profits.append(val_profit)

            if status_queue is not None:
                status_queue.put({
                    "key": key,
                    "event": "seed_done",
                    "seed": seed,
                    "seeds_done": seed_idx + 1,
                })

    finally:
        # Always restore stdout and close the devnull handle
        if _devnull is not None:
            sys.stdout = _orig_stdout
            _devnull.close()

    median_profit = float(np.median(seed_profits))
    return key, seed_profits, median_profit
```

- [ ] **Step 4: Update `grid_search()` to create queue + display + polling thread**

Replace the `grid_search()` function with:

```python
def grid_search(train_eps, val_eps, test_eps, save_path, seeds=None, num_workers=None):
    """Run hyperparameter grid search with parallel workers and Rich live display."""
    import multiprocessing
    import threading
    from src.grid_display import GridDisplay

    if seeds is None:
        seeds = [42, 123, 456]

    results = _load_grid_results(GRID_RESULTS_PATH)

    keys_list = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    all_combos = list(itertools.product(*values))
    total = len(all_combos)

    pending = []
    for combo in all_combos:
        config = dict(zip(keys_list, combo))
        config["epochs"] = 1
        if _config_key(config) not in results:
            pending.append(config)

    completed_count = total - len(pending)

    # Restore best from previously completed results
    best_median = -float("inf")
    best_config = None
    for entry in results.values():
        if entry["median_val_profit"] > best_median:
            best_median = entry["median_val_profit"]
            best_config = entry["config"]

    # IPC queue for worker -> parent status updates
    manager = multiprocessing.Manager()
    status_queue = manager.Queue()
    stop_event = threading.Event()

    try:
        with GridDisplay(pending, total=total, completed=completed_count) as display:
            # Polling thread reads queue and updates display
            poll_thread = threading.Thread(
                target=display.start_polling,
                args=(status_queue, stop_event),
                daemon=True,
            )
            poll_thread.start()

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_config = {
                    executor.submit(
                        run_config_worker,
                        config, seeds, train_eps, val_eps, test_eps, status_queue,
                    ): config
                    for config in pending
                }

                for future in concurrent.futures.as_completed(future_to_config):
                    config = future_to_config[future]
                    key, seed_profits, median_profit = future.result()

                    results[key] = {
                        "config": {k: v for k, v in config.items() if k != "epochs"},
                        "seed_profits": seed_profits,
                        "median_val_profit": median_profit,
                    }
                    _save_grid_results(GRID_RESULTS_PATH, results)

                    # Push config_done so display marks it complete
                    status_queue.put({
                        "key": key,
                        "event": "config_done",
                        "median": median_profit,
                        "seed_profits": seed_profits,
                    })

                    if median_profit > best_median:
                        best_median = median_profit
                        best_config = config

            stop_event.set()
            poll_thread.join(timeout=2.0)
    finally:
        # Always shut down manager to avoid leaking the manager process
        manager.shutdown()

    print(f"\n{'='*60}")
    print(f"Best config: {best_config}")
    print(f"Best median val profit: {best_median:.2f}c")
    print(f"\nRetraining best config with seed 42...")

    train_single(train_eps, val_eps, test_eps, best_config, seed=42, save_path=save_path)
```

- [ ] **Step 5: Run all tests**

```
pytest tests/test_grid_parallel.py tests/test_grid_display.py -v
```
Expected: All tests PASS

- [ ] **Step 6: Run full suite**

```
pytest tests/ -v
```
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add train.py src/grid_display.py tests/test_grid_parallel.py
git commit -m "feat: wire Rich live display and status queue into parallel grid search"
```

---

## Task 8: Update smoke test to exercise the display

**Files:**
- Modify: `smoke_parallel.py`

Update the smoke test to run with display on (2 configs, 2 seeds) so you can visually confirm the display works.

- [ ] **Step 1: Update `smoke_parallel.py`**

```python
# smoke_parallel.py
"""Smoke test for parallel grid search with Rich display. Run from project root."""
import train
from src.data_loader import load_episodes, split_episodes

if __name__ == "__main__":
    # Two configs so both columns of the display get exercised
    train.PARAM_GRID = {
        "lr": [1e-4, 3e-4],
        "epsilon_decay": [150],
        "seq_len": [10],
        "lstm_hidden": [16],
    }
    train.GRID_RESULTS_PATH = "checkpoints/smoke_grid_results.json"

    eps = load_episodes("data/episodes.json")
    train_eps, val_eps, test_eps = split_episodes(eps)
    train.grid_search(
        train_eps, val_eps, test_eps,
        save_path="checkpoints/smoke_test.pt",
        seeds=[42, 123],     # 2 seeds so seed progress column updates
        num_workers=2,       # 2 workers running simultaneously
    )
    print("Smoke test passed.")
```

- [ ] **Step 2: Run the smoke test**

```bash
python smoke_parallel.py
```

Expected:
- Rich live table appears, showing two rows (one per config)
- Each row updates: Pending → Running (Seed 1/2) → Running (Seed 2/2) → Done ✓
- Val profit column updates as validation checkpoints fire
- Median column fills in green (positive) or red (negative) when config completes
- "Smoke test passed." prints after the display closes

- [ ] **Step 3: Commit**

```bash
git add smoke_parallel.py
git commit -m "chore: update smoke test to exercise Rich display with 2 workers"
```

---

## Notes for Next Grid Run

Once parallelization is working, the recommended next grid parameters (from Phase 1 analysis):

```python
PARAM_GRID = {
    "lr": [1e-4, 2e-4, 3e-4],       # drop 5e-5
    "epsilon_decay": [150, 300],
    "seq_len": [10, 20],              # drop 40
    "lstm_hidden": [48, 64, 96, 128], # expand upward
}
seeds = [42, 123, 456, 789, 999]      # 5 seeds for tighter median
# epochs=3 per config
```

This gives 4×2×2×4 = 64 configs × 5 seeds = 320 training runs. With N cores running in parallel, wall-clock time drops by ~N×.
