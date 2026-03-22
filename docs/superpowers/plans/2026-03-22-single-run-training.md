# Single-Run Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed-epoch single-run training path with an open-ended parallel-rollout coordinator that uses multiple GPUs, displays live Rich progress, and saves resumable checkpoints.

**Architecture:** A coordinator process (GPU 0) owns the model, optimizer, and replay buffer. N rollout worker subprocesses each receive the current model weights, run `val_every_episodes / N` episodes, and return collected transitions. The coordinator merges them, trains, validates every 50 episodes, then updates the Rich display, appends to a JSONL log, and saves checkpoints.

**Tech Stack:** Python 3.11, PyTorch, Rich (`rich.live.Live`), `concurrent.futures.ProcessPoolExecutor`, multiprocessing spawn, JSON Lines

---

## File Map

| File | Change | Responsibility |
|------|--------|---------------|
| `src/trainer.py` | Modify | Remove TensorBoard; add `collect_episode()`, `evaluate_with_actions()`, `_val_profits_history`; add full checkpoint save/load |
| `src/replay_buffer.py` | Modify | Add `state_dict()` / `load_state_dict()` to `SumTree` and `PrioritizedReplayBuffer` |
| `src/train_logger.py` | Create | Appends JSONL checkpoint entries to `train_log.jsonl` |
| `src/train_display.py` | Create | Rich Live three-panel display: status, 10-row validation history, action distribution |
| `src/rollout_worker.py` | Create | Top-level worker function: receives weights + episodes, returns collected transitions |
| `train.py` | Modify | Add `--max-hours`, `--num-gpus`, `--checkpoint-dir`, `--resume` CLI args; add `run_training_session()`; update `main()` |
| `tests/test_trainer.py` | Modify | Tests for new Trainer methods |
| `tests/test_replay_buffer.py` | Modify | Tests for buffer serialization |
| `tests/test_train_logger.py` | Create | Tests for TrainLogger |
| `tests/test_train_display.py` | Create | Tests for TrainDisplay (no Live, just rendering) |
| `tests/test_rollout_worker.py` | Create | Tests for rollout worker function |

**Action index reference** (from `src/environment.py`):
```
0: Do nothing      1: Buy UP (taker)     2: Sell UP (taker)
3: Buy DOWN (taker) 4: Sell DOWN (taker)  5: Limit buy UP
6: Limit sell UP   7: Limit buy DOWN      8: Limit sell DOWN
```

---

## Task 1: Remove TensorBoard from Trainer; add `collect_episode()` and `evaluate_with_actions()`

**Files:**
- Modify: `src/trainer.py`
- Modify: `tests/test_trainer.py`

- [ ] **Step 1: Write failing tests for new Trainer methods**

Add to `tests/test_trainer.py`:

```python
class TestCollectEpisode:
    def test_returns_reward_counts_transitions(self):
        rows = [_make_row() for _ in range(5)]
        ep = _make_episode(outcome="UP", rows=rows)
        model = CountingModel(forced_action=0)
        trainer = _greedy_trainer(model, [ep])

        reward, action_counts, transitions = trainer.collect_episode(ep)

        assert isinstance(reward, float)
        assert action_counts.sum() == 5
        assert len(transitions) == 5

    def test_does_not_add_to_replay_buffer(self):
        rows = [_make_row() for _ in range(5)]
        ep = _make_episode(outcome="UP", rows=rows)
        model = CountingModel(forced_action=0)
        trainer = _greedy_trainer(model, [ep])

        trainer.collect_episode(ep)

        assert len(trainer.replay_buffer) == 0

    def test_transitions_have_required_keys(self):
        rows = [_make_row() for _ in range(3)]
        ep = _make_episode(outcome="UP", rows=rows)
        model = CountingModel(forced_action=0)
        trainer = _greedy_trainer(model, [ep])

        _, _, transitions = trainer.collect_episode(ep)

        required = {"static_features", "dynamic_features", "action",
                    "reward", "next_dynamic_features", "done",
                    "action_mask", "next_action_mask"}
        for t in transitions:
            assert required.issubset(t.keys())


class TestEvaluateWithActions:
    def test_returns_profit_and_dist(self):
        rows = [_make_row() for _ in range(5)]
        ep = _make_episode(outcome="UP", rows=rows)
        model = CountingModel(forced_action=0)
        trainer = _make_trainer(model, [ep])

        profit, dist = trainer.evaluate_with_actions([ep])

        assert isinstance(profit, float)
        assert isinstance(dist, dict)
        assert len(dist) == 9

    def test_dist_sums_to_one(self):
        rows = [_make_row() for _ in range(5)]
        ep = _make_episode(outcome="UP", rows=rows)
        model = CountingModel(forced_action=0)
        trainer = _make_trainer(model, [ep])

        _, dist = trainer.evaluate_with_actions([ep])

        assert sum(dist.values()) == pytest.approx(1.0)

    def test_profit_matches_evaluate(self):
        rows = [_make_row() for _ in range(5)]
        ep = _make_episode(outcome="UP", rows=rows)
        model = CountingModel(forced_action=0)
        trainer = _make_trainer(model, [ep])

        profit_with_actions, _ = trainer.evaluate_with_actions([ep])
        profit_plain = trainer.evaluate([ep])

        assert profit_with_actions == pytest.approx(profit_plain)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_trainer.py::TestCollectEpisode tests/test_trainer.py::TestEvaluateWithActions -v
```
Expected: FAIL — `collect_episode` and `evaluate_with_actions` do not exist.

- [ ] **Step 3: Remove TensorBoard from `src/trainer.py`**

Remove these items from `src/trainer.py`:
- The `_writer = None` line in `__init__`
- The `_get_writer()` method entirely
- The `log_dir` parameter from `train()` and its `if log_dir is not None:` block
- The body of `_log_episode()` — replace with `pass`
- The body of `_log_train_step()` — replace with `pass`
- The `writer = self._get_writer()` and `writer.add_scalar(...)` lines inside `_log_validation()` — keep the `print()` and `self._on_validation` callback call
- The `close()` method entirely
- The `from torch.utils.tensorboard import SummaryWriter` import inside `train()`

Also add `self._val_profits_history: list[float] = []` to `__init__` after `self._best_state_dict`.

Update `_log_validation` to also append to history:
```python
def _log_validation(self, val_profit: float) -> None:
    """Log validation profit."""
    self._val_profits_history.append(val_profit)
    print(
        f"[Episode {self._episode_count}] "
        f"Val profit: {val_profit:.2f}c | "
        f"Best: {self._best_val_profit:.2f}c | "
        f"Epsilon: {self.epsilon:.3f}"
    )
    if self._on_validation is not None:
        self._on_validation(self._episode_count, val_profit, self.epsilon)
```

- [ ] **Step 4: Add `collect_episode()` to `src/trainer.py`**

Add after `_run_episode()`:

```python
def collect_episode(
    self, episode: dict
) -> tuple[float, np.ndarray, list[dict]]:
    """Run one episode and return transitions WITHOUT adding to replay buffer.

    Used by rollout workers in parallel single-run training.

    Returns:
        Tuple of (episode_reward, action_counts, transitions).
    """
    self.env.reset(episode)
    static_features = self.normalizer.encode_static(episode)
    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    transitions: list[dict] = []
    final_reward = 0.0

    for _ in range(self.env.num_rows):
        obs = self.env.get_observation()
        dynamic_features = self.normalizer.encode_dynamic(obs)
        action_mask = self.env.get_action_mask()

        action = self._select_action_train(
            static_features, dynamic_features, action_mask
        )
        action_counts[action] += 1
        done, reward = self.env.step(action)

        next_dynamic = None
        next_mask = None
        if not done:
            next_obs = self.env.get_observation()
            next_dynamic = self.normalizer.encode_dynamic(next_obs)
            next_mask = self.env.get_action_mask()
        else:
            final_reward = reward

        transitions.append({
            "static_features": static_features,
            "dynamic_features": dynamic_features,
            "action": action,
            "reward": reward,
            "next_dynamic_features": next_dynamic,
            "done": done,
            "action_mask": action_mask,
            "next_action_mask": next_mask,
        })

        if done:
            break

    return final_reward, action_counts, transitions
```

- [ ] **Step 5: Add `evaluate_with_actions()` to `src/trainer.py`**

Add after `evaluate()`. Action names must match `src/environment.py` docstring order:

```python
_ACTION_NAMES = [
    "do_nothing", "buy_up_taker", "sell_up_taker",
    "buy_down_taker", "sell_down_taker",
    "limit_buy_up", "limit_sell_up",
    "limit_buy_down", "limit_sell_down",
]

def evaluate_with_actions(
    self, episodes: list[dict]
) -> tuple[float, dict[str, float]]:
    """Evaluate greedily and return total profit plus action distribution.

    Returns:
        Tuple of (total_profit_cents, action_distribution dict).
        action_distribution values sum to 1.0.
    """
    self.online_net.eval()
    total_profit = 0.0
    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)

    for ep in episodes:
        self.env.reset(ep)
        static_features = self.normalizer.encode_static(ep)
        hidden = self.online_net.get_initial_hidden(
            batch_size=1, device=self.device
        )

        for _ in range(self.env.num_rows):
            obs = self.env.get_observation()
            dynamic_features = self.normalizer.encode_dynamic(obs)
            action_mask = self.env.get_action_mask()

            with torch.no_grad():
                static_t = torch.tensor(
                    static_features, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                dynamic_t = torch.tensor(
                    dynamic_features, dtype=torch.float32, device=self.device
                ).unsqueeze(0).unsqueeze(0)
                q_values, hidden = self.online_net(static_t, dynamic_t, hidden)
                q_values = q_values.squeeze(0).cpu().numpy()

            q_values[~action_mask] = -np.inf
            action = int(np.argmax(q_values))
            action_counts[action] += 1

            done, reward = self.env.step(action)
            if done:
                total_profit += reward * 100.0
                break

    total = action_counts.sum()
    dist = {
        name: float(action_counts[i] / total) if total > 0 else 0.0
        for i, name in enumerate(_ACTION_NAMES)
    }
    return total_profit, dist
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_trainer.py -v
```
Expected: ALL PASS (existing tests + new ones).

- [ ] **Step 7: Commit**

```bash
git add src/trainer.py tests/test_trainer.py
git commit -m "refactor: remove TensorBoard from Trainer; add collect_episode and evaluate_with_actions"
```

---

## Task 2: Add replay buffer serialization

**Files:**
- Modify: `src/replay_buffer.py`
- Modify: `tests/test_replay_buffer.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_replay_buffer.py`:

```python
class TestReplayBufferStateDictRoundtrip:
    def _make_transition(self, done=False):
        return {
            "static_features": np.zeros(37, dtype=np.float32),
            "dynamic_features": np.zeros(12, dtype=np.float32),
            "action": 0,
            "reward": 0.0,
            "next_dynamic_features": None if done else np.zeros(12, dtype=np.float32),
            "done": done,
            "action_mask": np.ones(9, dtype=bool),
            "next_action_mask": None if done else np.ones(9, dtype=bool),
        }

    def test_state_dict_restores_size(self):
        buf = PrioritizedReplayBuffer(capacity=100, seq_len=3)
        episode = [self._make_transition() for _ in range(4)]
        episode[-1] = self._make_transition(done=True)
        buf.add_episode(episode)

        state = buf.state_dict()
        buf2 = PrioritizedReplayBuffer(capacity=100, seq_len=3)
        buf2.load_state_dict(state)

        assert len(buf2) == len(buf)

    def test_state_dict_restores_sampling(self, tmp_path):
        import torch
        buf = PrioritizedReplayBuffer(capacity=200, seq_len=3)
        for _ in range(10):
            episode = [self._make_transition() for _ in range(5)]
            episode[-1] = self._make_transition(done=True)
            buf.add_episode(episode)

        state = buf.state_dict()
        path = tmp_path / "buf.pt"
        torch.save(state, path)

        buf2 = PrioritizedReplayBuffer(capacity=200, seq_len=3)
        buf2.load_state_dict(torch.load(path))

        # Both buffers should be sample-able
        batch = buf2.sample(batch_size=4, beta=0.4)
        assert batch["actions"].shape[0] == 4
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_replay_buffer.py::TestReplayBufferStateDictRoundtrip -v
```
Expected: FAIL — `state_dict` / `load_state_dict` do not exist.

- [ ] **Step 3: Add `state_dict()` and `load_state_dict()` to `SumTree`**

Add to `SumTree` class in `src/replay_buffer.py`:

```python
def state_dict(self) -> dict:
    return {
        "capacity": self.capacity,
        "tree": self._tree.copy(),
        "write_idx": self._write_idx,
        "size": self._size,
    }

def load_state_dict(self, state: dict) -> None:
    self._tree = state["tree"].copy()
    self._write_idx = state["write_idx"]
    self._size = state["size"]
```

- [ ] **Step 4: Add `state_dict()` and `load_state_dict()` to `PrioritizedReplayBuffer`**

Add to `PrioritizedReplayBuffer` class:

```python
def state_dict(self) -> dict:
    """Return serializable snapshot of buffer state."""
    return {
        "tree": self._tree.state_dict(),
        "static_features": self._static_features.copy(),
        "dynamic_features": self._dynamic_features.copy(),
        "actions": self._actions.copy(),
        "rewards": self._rewards.copy(),
        "next_dynamic_features": self._next_dynamic_features.copy(),
        "dones": self._dones.copy(),
        "action_masks": self._action_masks.copy(),
        "next_action_masks": self._next_action_masks.copy(),
        "episode_ids": self._episode_ids.copy(),
        "positions": self._positions.copy(),
        "episode_lengths": self._episode_lengths.copy(),
        "write_idx": self._write_idx,
        "size": self._size,
        "max_priority": self._max_priority,
        "episode_counter": self._episode_counter,
    }

def load_state_dict(self, state: dict) -> None:
    """Restore buffer state from a state_dict snapshot."""
    self._tree.load_state_dict(state["tree"])
    self._static_features = state["static_features"].copy()
    self._dynamic_features = state["dynamic_features"].copy()
    self._actions = state["actions"].copy()
    self._rewards = state["rewards"].copy()
    self._next_dynamic_features = state["next_dynamic_features"].copy()
    self._dones = state["dones"].copy()
    self._action_masks = state["action_masks"].copy()
    self._next_action_masks = state["next_action_masks"].copy()
    self._episode_ids = state["episode_ids"].copy()
    self._positions = state["positions"].copy()
    self._episode_lengths = state["episode_lengths"].copy()
    self._write_idx = state["write_idx"]
    self._size = state["size"]
    self._max_priority = state["max_priority"]
    self._episode_counter = state["episode_counter"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_replay_buffer.py -v
```
Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add src/replay_buffer.py tests/test_replay_buffer.py
git commit -m "feat: add state_dict/load_state_dict to replay buffer for checkpoint resumability"
```

---

## Task 3: Add full checkpoint save/load to Trainer

**Files:**
- Modify: `src/trainer.py`
- Modify: `tests/test_trainer.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_trainer.py`:

```python
class TestFullCheckpoint:
    def test_roundtrip_episode_count(self, tmp_path):
        rows = [_make_row() for _ in range(5)]
        ep = _make_episode(outcome="UP", rows=rows)
        model = CountingModel(forced_action=0)
        trainer = _make_trainer(model, [ep])
        trainer._episode_count = 42
        trainer._best_val_profit = 99.0
        trainer._val_profits_history = [10.0, 20.0, 99.0]

        path = str(tmp_path / "full.pt")
        elapsed = trainer.save_full_checkpoint(path, elapsed_seconds=500.0)

        model2 = CountingModel(forced_action=0)
        trainer2 = _make_trainer(model2, [ep])
        returned_elapsed = trainer2.load_full_checkpoint(path)

        assert trainer2._episode_count == 42
        assert trainer2._best_val_profit == pytest.approx(99.0)
        assert trainer2._val_profits_history == [10.0, 20.0, 99.0]
        assert returned_elapsed == pytest.approx(500.0)

    def test_save_checkpoint_still_works(self, tmp_path):
        """Existing save_checkpoint (weights-only) must not break."""
        rows = [_make_row() for _ in range(3)]
        ep = _make_episode(outcome="UP", rows=rows)
        model = CountingModel(forced_action=0)
        trainer = _make_trainer(model, [ep])

        path = str(tmp_path / "model.pt")
        trainer.save_checkpoint(path)

        import os
        assert os.path.exists(path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_trainer.py::TestFullCheckpoint -v
```
Expected: FAIL — `save_full_checkpoint` and `load_full_checkpoint` do not exist.

- [ ] **Step 3: Add `save_full_checkpoint()` and `load_full_checkpoint()` to `src/trainer.py`**

Add after `save_checkpoint()`:

```python
def save_full_checkpoint(self, path: str, elapsed_seconds: float = 0.0) -> float:
    """Save complete training state for resumability.

    Args:
        path: File path for the checkpoint (.pt).
        elapsed_seconds: Accumulated training time to persist.

    Returns:
        elapsed_seconds (passed through, for convenience).
    """
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "online_net": self.online_net.state_dict(),
        "target_net": self.target_net.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "replay_buffer": self.replay_buffer.state_dict(),
        "episode_count": self._episode_count,
        "step_count": self._step_count,
        "best_val_profit": self._best_val_profit,
        "val_profits_history": self._val_profits_history,
        "best_state_dict": self._best_state_dict,
        "elapsed_seconds": elapsed_seconds,
    }, path)
    return elapsed_seconds

def load_full_checkpoint(self, path: str) -> float:
    """Restore complete training state from a full checkpoint.

    Returns:
        Accumulated elapsed seconds stored in the checkpoint.
    """
    state = torch.load(path, map_location=self.device, weights_only=False)
    self.online_net.load_state_dict(state["online_net"])
    self.target_net.load_state_dict(state["target_net"])
    self.optimizer.load_state_dict(state["optimizer"])
    self.replay_buffer.load_state_dict(state["replay_buffer"])
    self._episode_count = state["episode_count"]
    self._step_count = state["step_count"]
    self._best_val_profit = state["best_val_profit"]
    self._val_profits_history = state.get("val_profits_history", [])
    self._best_state_dict = state.get("best_state_dict")
    return float(state.get("elapsed_seconds", 0.0))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_trainer.py -v
```
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trainer.py tests/test_trainer.py
git commit -m "feat: add save_full_checkpoint/load_full_checkpoint to Trainer"
```

---

## Task 4: Create `TrainLogger`

**Files:**
- Create: `src/train_logger.py`
- Create: `tests/test_train_logger.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_train_logger.py`:

```python
"""Tests for TrainLogger JSONL writer."""
import json
import pytest
from src.train_logger import TrainLogger


class TestTrainLogger:
    def test_append_creates_file(self, tmp_path):
        logger = TrainLogger(str(tmp_path / "log.jsonl"))
        logger.append(
            checkpoint=1, episode=50, elapsed_seconds=30.0,
            val_profit_cents=12.5, best_profit_cents=12.5,
            median_profit_cents=12.5, epsilon=0.9,
            action_distribution={"do_nothing": 1.0},
        )
        assert (tmp_path / "log.jsonl").exists()

    def test_append_writes_valid_json(self, tmp_path):
        logger = TrainLogger(str(tmp_path / "log.jsonl"))
        logger.append(
            checkpoint=1, episode=50, elapsed_seconds=30.0,
            val_profit_cents=12.5, best_profit_cents=12.5,
            median_profit_cents=12.5, epsilon=0.9,
            action_distribution={"do_nothing": 1.0},
        )
        line = (tmp_path / "log.jsonl").read_text().strip()
        entry = json.loads(line)
        assert entry["checkpoint"] == 1
        assert entry["episode"] == 50
        assert entry["val_profit_cents"] == pytest.approx(12.5)

    def test_append_multiple_entries_each_valid_json(self, tmp_path):
        logger = TrainLogger(str(tmp_path / "log.jsonl"))
        for i in range(3):
            logger.append(
                checkpoint=i+1, episode=(i+1)*50, elapsed_seconds=float(i*30),
                val_profit_cents=float(i), best_profit_cents=float(i),
                median_profit_cents=float(i), epsilon=0.9 - i*0.1,
                action_distribution={"do_nothing": 1.0},
            )
        lines = (tmp_path / "log.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            json.loads(line)  # must not raise

    def test_append_does_not_overwrite_on_resume(self, tmp_path):
        path = str(tmp_path / "log.jsonl")
        logger1 = TrainLogger(path)
        logger1.append(
            checkpoint=1, episode=50, elapsed_seconds=30.0,
            val_profit_cents=1.0, best_profit_cents=1.0,
            median_profit_cents=1.0, epsilon=0.9,
            action_distribution={},
        )
        logger2 = TrainLogger(path)
        logger2.append(
            checkpoint=2, episode=100, elapsed_seconds=60.0,
            val_profit_cents=2.0, best_profit_cents=2.0,
            median_profit_cents=1.5, epsilon=0.8,
            action_distribution={},
        )
        lines = (tmp_path / "log.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_train_logger.py -v
```
Expected: FAIL — `src.train_logger` does not exist.

- [ ] **Step 3: Create `src/train_logger.py`**

```python
"""JSON Lines log writer for single-run training sessions."""
from __future__ import annotations

import json
import os
from datetime import datetime


class TrainLogger:
    """Appends one JSON entry per validation checkpoint to a .jsonl file.

    Safe to construct multiple times on the same path (entries accumulate).
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def append(
        self,
        checkpoint: int,
        episode: int,
        elapsed_seconds: float,
        val_profit_cents: float,
        best_profit_cents: float,
        median_profit_cents: float,
        epsilon: float,
        action_distribution: dict[str, float],
    ) -> None:
        """Append one checkpoint entry to the log file."""
        entry = {
            "checkpoint": checkpoint,
            "episode": episode,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed_seconds),
            "val_profit_cents": round(val_profit_cents, 2),
            "best_profit_cents": round(best_profit_cents, 2),
            "median_profit_cents": round(median_profit_cents, 2),
            "epsilon": round(epsilon, 4),
            "action_distribution": {
                k: round(v, 4) for k, v in action_distribution.items()
            },
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_train_logger.py -v
```
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/train_logger.py tests/test_train_logger.py
git commit -m "feat: add TrainLogger for JSONL checkpoint logging"
```

---

## Task 5: Create `TrainDisplay`

**Files:**
- Create: `src/train_display.py`
- Create: `tests/test_train_display.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_train_display.py`:

```python
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
            median_profit=10.0, epsilon=0.9,
            action_distribution=self._DIST, checkpoint_num=1,
            is_new_best=True,
        )
        assert len(display._history) == 1

    def test_history_capped_at_ten(self):
        display = TrainDisplay(config=self._CONFIG, max_hours=12.0)
        for i in range(15):
            display.update(
                episode=(i+1)*50, val_profit=float(i), best_profit=float(i),
                median_profit=float(i), epsilon=0.9,
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_train_display.py -v
```
Expected: FAIL — `src.train_display` does not exist.

- [ ] **Step 3: Create `src/train_display.py`**

```python
"""Rich Live terminal display for single-run training progress."""
from __future__ import annotations

from datetime import datetime, timedelta
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
    return f"{sign}${abs(dollars):.2f}"


def _action_bar(fraction: float, width: int = _BAR_WIDTH) -> str:
    """Render a text progress bar for an action fraction."""
    filled = int(round(fraction * width))
    filled = max(0, min(filled, width))
    return "█" * filled + "░" * (width - filled)


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
        # Compute speed (eps/sec) since last update
        now = datetime.now()
        dt = (now - self._last_update_time).total_seconds()
        if dt > 0 and episode > self._last_ep_count:
            self._speed_eps_per_sec = (episode - self._last_ep_count) / dt
        self._last_ep_count = episode
        self._last_update_time = now

        self._episode_count = episode
        self._epsilon = epsilon
        self._latest_dist = action_distribution
        self._history.append({
            "checkpoint": checkpoint_num,
            "episode": episode,
            "val_profit": val_profit,
            "best_profit": best_profit,
            "median_profit": median_profit,
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
            f"seq={cfg.get('seq_len')}  ε-decay={cfg.get('epsilon_decay')}  "
            f"workers={cfg.get('num_gpus', 1)}\n"
            f"Started: {self._start_wall.strftime('%Y-%m-%d %H:%M')}  │  "
            f"Elapsed: {elapsed_td}  │  Remaining: {remaining_str}\n"
            f"Episodes: {self._episode_count:,}  │  Speed: {self._speed_eps_per_sec:.1f} eps/sec  │  ε: {self._epsilon:.3f}"
        )
        return Panel(content, title="Single Run Training")

    def _history_panel(self) -> Panel:
        table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        table.add_column("#", style="dim", width=5)
        table.add_column("Episode", width=9)
        table.add_column("Val Profit", width=12)
        table.add_column("Best Profit", width=12)
        table.add_column("Median", width=10)
        table.add_column("ε", width=7)

        rows = self._history[-_MAX_HISTORY_ROWS:]
        for r in rows:
            star = " ★" if r["is_best"] else ""
            style = "bold green" if r["is_best"] else ""
            table.add_row(
                str(r["checkpoint"]),
                str(r["episode"]),
                f"{_format_profit(r['val_profit'])}{star}",
                _format_profit(r["best_profit"]),
                _format_profit(r["median_profit"]),
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_train_display.py -v
```
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/train_display.py tests/test_train_display.py
git commit -m "feat: add TrainDisplay Rich terminal display for single-run training"
```

---

## Task 6: Create `rollout_worker`

**Files:**
- Create: `src/rollout_worker.py`
- Create: `tests/test_rollout_worker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_rollout_worker.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_rollout_worker.py -v
```
Expected: FAIL — `src.rollout_worker` does not exist.

- [ ] **Step 3: Create `src/rollout_worker.py`**

```python
"""Rollout worker for parallel single-run training.

Top-level function required for multiprocessing pickle compatibility.
"""
from __future__ import annotations

import numpy as np
import torch

from src.models.lstm_dqn import LSTMDQN
from src.normalizer import Normalizer
from src.trainer import Trainer


def run_rollout_worker(
    state_dict: dict,
    episodes: list[dict],
    normalizer: Normalizer,
    config: dict,
    episode_count: int,
    device_str: str,
) -> list[tuple[float, np.ndarray, list[dict]]]:
    """Run a list of episodes and return collected transitions.

    Does NOT add transitions to a replay buffer. The coordinator merges
    returned transitions into its own buffer.

    Args:
        state_dict: Online network weights (CPU tensors).
        episodes: Episodes to run (subset of train_eps for this round).
        normalizer: Fitted feature normalizer.
        config: Training config dict (same as Trainer config).
        episode_count: Current total episode count (used for epsilon calc).
        device_str: Device string, e.g. "cuda:0" or "cpu".

    Returns:
        List of (reward, action_counts, transitions) tuples, one per episode.
    """
    device = torch.device(device_str)
    model = LSTMDQN(lstm_hidden_size=config.get("lstm_hidden", 32))
    trainer = Trainer(model=model, normalizer=normalizer, config=config, device=device)
    trainer.online_net.load_state_dict(
        {k: v.to(device) for k, v in state_dict.items()}
    )
    trainer._episode_count = episode_count  # sets correct epsilon

    results = []
    for ep in episodes:
        reward, action_counts, transitions = trainer.collect_episode(ep)
        results.append((reward, action_counts, transitions))

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_rollout_worker.py -v
```
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rollout_worker.py tests/test_rollout_worker.py
git commit -m "feat: add rollout_worker for parallel episode collection"
```

---

## Task 7: Wire up `run_training_session()` in `train.py`

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add new CLI args to `parse_args()`**

In `parse_args()`, add after the existing `--num-workers` argument:

```python
parser.add_argument(
    "--max-hours", type=float, default=12.0,
    help="Maximum training duration in hours (default: 12)",
)
parser.add_argument(
    "--num-gpus", type=int, default=None,
    help="Number of GPUs (rollout workers) for single-run training. "
         "Defaults to all available GPUs, or 1 CPU worker if none.",
)
parser.add_argument(
    "--checkpoint-dir", type=str, default="checkpoints/single_run",
    help="Directory for checkpoints and train_log.jsonl",
)
parser.add_argument(
    "--resume", action="store_true",
    help="Resume training from checkpoint in --checkpoint-dir",
)
```

Remove `--log-dir` from the parser — TensorBoard is gone. Keep `--save-path` because `main()` passes it to `grid_search()` and removing it would break that path. The single-run mode uses `--checkpoint-dir` instead.

- [ ] **Step 2: Add `_handle_checkpoint_startup()` helper**

Add before `main()`:

```python
def _handle_checkpoint_startup(checkpoint_dir: str, resume: bool) -> bool:
    """Handle checkpoint startup logic.

    Returns:
        True if training should proceed, False if user chose to exit.
    """
    import glob as _glob

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
    log_path = os.path.join(checkpoint_dir, "train_log.jsonl")

    if resume:
        return True  # load happens in run_training_session

    existing = [p for p in [checkpoint_path, best_path] if os.path.exists(p)]
    if not existing:
        return True

    answer = input(
        f"Checkpoints found in {checkpoint_dir!r}. "
        "Clear them and start fresh? [y/N] "
    ).strip().lower()
    if answer == "y":
        for p in existing:
            os.remove(p)
        if os.path.exists(log_path):
            os.remove(log_path)
        print("Checkpoints cleared. Starting fresh.")
        return True

    print("Exiting. Use --resume to continue from an existing checkpoint.")
    return False
```

- [ ] **Step 3: Add `run_training_session()` function**

Add before `main()`:

```python
def run_training_session(
    train_eps: list,
    val_eps: list,
    config: dict,
    checkpoint_dir: str,
    max_hours: float,
    num_gpus: int | None,
    resume: bool,
) -> None:
    """Open-ended single-run training with parallel rollout workers.

    Uses synchronous parallel rollouts: all workers collect episodes, then
    coordinator trains, then validates — repeat until time limit or Ctrl+C.
    """
    import copy
    import multiprocessing
    import statistics
    import time
    from concurrent.futures import ProcessPoolExecutor

    from src.train_display import TrainDisplay
    from src.train_logger import TrainLogger
    from src.rollout_worker import run_rollout_worker

    if not _handle_checkpoint_startup(checkpoint_dir, resume):
        return

    os.makedirs(checkpoint_dir, exist_ok=True)

    normalizer = Normalizer()
    normalizer.fit(train_eps)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LSTMDQN(lstm_hidden_size=config.get("lstm_hidden", 32))
    # Note: Trainer uses "epsilon_decay_episodes" but the CLI arg is "epsilon_decay".
    # Map explicitly here so the CLI arg takes effect.
    trainer = Trainer(
        model=model,
        normalizer=normalizer,
        config={
            "lr": config.get("lr", 1e-4),
            "seq_len": config.get("seq_len", 20),
            "epsilon_decay_episodes": config.get("epsilon_decay", 300),
        },
        device=device,
    )

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
    log_path = os.path.join(checkpoint_dir, "train_log.jsonl")

    # Count existing log entries to continue checkpoint numbering on resume
    elapsed_seconds = 0.0
    checkpoint_num = 0
    if resume and os.path.exists(checkpoint_path):
        elapsed_seconds = trainer.load_full_checkpoint(checkpoint_path)
        if os.path.exists(log_path):
            with open(log_path) as f:
                checkpoint_num = sum(1 for _ in f)
        print(
            f"Resumed from checkpoint #{checkpoint_num}. "
            f"Episode {trainer._episode_count}. "
            f"Elapsed: {elapsed_seconds:.0f}s"
        )

    # Determine worker devices
    n_cuda = torch.cuda.device_count()
    if n_cuda == 0:
        worker_devices = ["cpu"]
    else:
        n = min(num_gpus, n_cuda) if num_gpus else n_cuda
        worker_devices = [f"cuda:{i}" for i in range(n)]

    n_workers = len(worker_devices)
    val_every = trainer.config["val_every_episodes"]
    episodes_per_worker = max(1, val_every // n_workers)
    max_elapsed = max_hours * 3600

    # Required for CUDA in subprocesses
    if n_cuda > 0:
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    rng = np.random.default_rng(seed=0)
    logger = TrainLogger(log_path)
    start_wall = time.time()
    display_config = {**config, "num_gpus": n_workers}

    try:
        with TrainDisplay(
            config=display_config,
            max_hours=max_hours,
            elapsed_offset=elapsed_seconds,
        ) as display:
            while True:
                current_elapsed = elapsed_seconds + (time.time() - start_wall)
                if current_elapsed >= max_elapsed:
                    print(f"\nTime limit reached ({max_hours}h). Stopping.")
                    break

                # Broadcast current weights (CPU tensors for pickle)
                state_dict = {
                    k: v.cpu() for k, v in trainer.online_net.state_dict().items()
                }
                episode_count_snap = trainer._episode_count

                # Sample episodes for each worker
                indices = rng.choice(len(train_eps), size=n_workers * episodes_per_worker, replace=True)
                worker_batches = [
                    [train_eps[i] for i in indices[w * episodes_per_worker:(w + 1) * episodes_per_worker]]
                    for w in range(n_workers)
                ]

                # Collect rollouts (parallel or single-process)
                all_results: list[tuple] = []
                if n_workers == 1 and worker_devices[0] == "cpu":
                    # Single-process inline (no subprocess overhead)
                    all_results = run_rollout_worker(
                        state_dict=state_dict,
                        episodes=worker_batches[0],
                        normalizer=normalizer,
                        config=trainer.config,
                        episode_count=episode_count_snap,
                        device_str="cpu",
                    )
                else:
                    with ProcessPoolExecutor(max_workers=n_workers) as pool:
                        futures = [
                            pool.submit(
                                run_rollout_worker,
                                state_dict=state_dict,
                                episodes=worker_batches[w],
                                normalizer=normalizer,
                                config=trainer.config,
                                episode_count=episode_count_snap,
                                device_str=worker_devices[w],
                            )
                            for w in range(n_workers)
                        ]
                        for fut in futures:
                            all_results.extend(fut.result())

                # Merge experiences into replay buffer
                for reward, action_counts, transitions in all_results:
                    trainer.replay_buffer.add_episode(transitions)
                    trainer._episode_count += 1

                # Run one training step per episode collected (mirrors original loop)
                for _ in range(len(all_results)):
                    if len(trainer.replay_buffer) >= trainer.config["min_buffer_size"]:
                        trainer._train_step()
                        trainer._step_count += 1

                # Validate
                val_profit, action_dist = trainer.evaluate_with_actions(val_eps)
                trainer._val_profits_history.append(val_profit)
                is_new_best = val_profit > trainer._best_val_profit
                if is_new_best:
                    trainer._best_val_profit = val_profit
                    trainer._best_state_dict = copy.deepcopy(
                        trainer.online_net.state_dict()
                    )

                best_profit = trainer._best_val_profit
                median_profit = statistics.median(trainer._val_profits_history)
                checkpoint_num += 1
                current_elapsed = elapsed_seconds + (time.time() - start_wall)

                # Update display
                display.update(
                    episode=trainer._episode_count,
                    val_profit=val_profit,
                    best_profit=best_profit,
                    median_profit=median_profit,
                    epsilon=trainer.epsilon,
                    action_distribution=action_dist,
                    checkpoint_num=checkpoint_num,
                    is_new_best=is_new_best,
                )

                # Append log entry
                logger.append(
                    checkpoint=checkpoint_num,
                    episode=trainer._episode_count,
                    elapsed_seconds=current_elapsed,
                    val_profit_cents=val_profit,
                    best_profit_cents=best_profit,
                    median_profit_cents=median_profit,
                    epsilon=trainer.epsilon,
                    action_distribution=action_dist,
                )

                # Save checkpoints
                trainer.save_full_checkpoint(checkpoint_path, elapsed_seconds=current_elapsed)
                if is_new_best:
                    trainer.save_full_checkpoint(best_path, elapsed_seconds=current_elapsed)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Last checkpoint saved.")
```

- [ ] **Step 4: Update `main()` to route single-run to `run_training_session()`**

Replace the `else:` branch in `main()` (currently calls `train_single`):

```python
    if args.grid_search:
        grid_search(train_eps, val_eps, test_eps, "checkpoints/model.pt",
                    num_workers=args.num_workers)
    else:
        config = {
            "lr": args.lr,
            "lstm_hidden": args.lstm_hidden,
            "seq_len": args.seq_len,
            "epsilon_decay": args.epsilon_decay,
        }
        run_training_session(
            train_eps=train_eps,
            val_eps=val_eps,
            config=config,
            checkpoint_dir=args.checkpoint_dir,
            max_hours=args.max_hours,
            num_gpus=args.num_gpus,
            resume=args.resume,
        )
```

- [ ] **Step 5: Note on `train_single`**

`train_single` is **not** removed — it is still called by `run_config_worker` in the grid search path. The spec's intent is that it is no longer the user-facing single-run entry point (replaced by `run_training_session`). Leave the function in place.

- [ ] **Step 6: Run the full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: ALL PASS. Pay special attention to `test_trainer.py` and `test_grid_parallel.py` — grid search must be unaffected.

- [ ] **Step 7: Smoke test single-GPU single run (requires data file)**

```bash
python train.py --max-hours 0.01 --num-gpus 1 --checkpoint-dir checkpoints/smoke_test
```
Expected: runs for ~36 seconds, Rich display shows three panels, exits cleanly with "Time limit reached".

- [ ] **Step 8: Commit**

```bash
git add train.py
git commit -m "feat: add run_training_session with parallel rollouts, Rich display, and resumable checkpoints"
```
