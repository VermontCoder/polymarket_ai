# Single-Run Training with Parallel Rollouts and Rich Display

## Goal

Replace the multi-GPU grid-search-only training path with a single-run training mode that uses synchronous parallel rollout workers across available GPUs, displays live progress in the terminal via Rich, and persists a JSON training log — while keeping the existing grid search untouched.

---

## Architecture

Training is structured as a **coordinator + worker** pattern:

- **Coordinator** (main process, GPU 0) — owns the replay buffer, model weights, and optimizer. Runs training steps and validation.
- **Rollout workers** (one subprocess per GPU, GPU 0..N-1) — each receives current model weights, runs episodes, and returns collected experiences.

### Round Loop

Each round:
1. Broadcast current model weights to all workers
2. Workers each run `val_every_episodes / num_gpus` episodes, return experiences
3. Coordinator merges all experiences into the replay buffer
4. Coordinator runs training steps
5. Coordinator runs validation
6. Update Rich terminal display
7. Append entry to JSON log
8. Save checkpoint
9. Check time limit — if exceeded, exit cleanly

One round = `val_every_episodes` total episodes (default 50), regardless of GPU count.

### GPU Scaling

- **N GPUs:** N rollout workers, each on their own GPU. Coordinator also on GPU 0.
- **1 GPU:** Single worker and coordinator both on GPU 0. No parallelism, but identical code path — suitable for testing.
- **CPU only:** Single-process mode — no worker subprocess. The coordinator runs the same round loop directly on CPU, calling `collect_episodes` and `train_steps` inline without spawning workers. The old `train_single` function is removed.

---

## Stopping Criteria

- **Time limit:** Stop after `--max-hours` (default 12) of elapsed training time. Checked after each round.
- **Manual:** Ctrl+C triggers a clean exit, saving the current checkpoint first.
- No fixed epoch count — training is open-ended within the time cap.

---

## Checkpoint & Resume

### Saving
Checkpoint written after every round (post-validation, post-log-append) to `<checkpoint-dir>/checkpoint.pt` (single file, overwritten each round). Contains:
- Model weights
- Optimizer state
- Replay buffer state
- Episode count
- Accumulated elapsed seconds

### Startup Logic
1. `--resume` flag provided → load latest checkpoint from `--checkpoint-dir`, continue training
2. No `--resume`, checkpoints exist → prompt: *"Checkpoints found in `<path>`. Clear them and start fresh? [y/N]"*
   - `y`: delete all checkpoints, start fresh
   - `n`: exit
3. No `--resume`, no checkpoints → start fresh immediately

### Resume Behaviour
Elapsed time accumulates across sessions. A run stopped at hour 4 and resumed will stop again at hour 8 (if `--max-hours 12` and 4 hours already spent).

---

## Terminal Display (Rich)

Updated at the end of every round (every 50 episodes by default). Three panels rendered via `rich.live.Live`:

### Panel 1 — Training Status
```
┌─ Single Run Training ──────────────────────────────────────────────────────┐
│ lr=2e-4  hidden=64  seq=20  ε-decay=150  workers=4                         │
│ Started: 2026-03-22 14:30  │  Elapsed: 2:14:35  │  Remaining: 9:45:25     │
│ Episodes: 12,450  │  Speed: 45 eps/sec  │  ε: 0.127                        │
└────────────────────────────────────────────────────────────────────────────┘
```

### Panel 2 — Validation History (newest at bottom, scrolling)
```
┌─ Validation History ───────────────────────────────────────────────────────┐
│  #    Episode   Val Profit   Best Profit   Median     ε                    │
│  1    50        +$1.23       +$1.23        +$1.23     0.842                │
│  2    100       -$0.45       +$1.23        +$0.39     0.756                │
│  3    150       +$2.10 ★     +$2.10        +$0.83     0.671                │
└────────────────────────────────────────────────────────────────────────────┘
```
Rows where a new best profit is achieved are highlighted.

### Panel 3 — Latest Action Distribution
```
┌─ Action Distribution (checkpoint #24) ─────────────────────────────────────┐
│  Hold            ████████████████░░░░  64.2%                               │
│  Buy  (taker)    ████░░░░░░░░░░░░░░░░  12.1%                               │
│  Buy  (maker)    ███░░░░░░░░░░░░░░░░░   8.7%                               │
│  Sell (taker)    ██░░░░░░░░░░░░░░░░░░   7.3%                               │
│  Sell (maker)    ██░░░░░░░░░░░░░░░░░░   5.4%                               │
│  ... (all 9 actions shown)                                                  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## JSON Training Log

File: `<checkpoint-dir>/train_log.jsonl` (newline-delimited JSON)

One entry appended per validation checkpoint. Each line is a self-contained JSON object. New entries are appended on resume — existing lines are never modified or deleted. This format handles appends trivially without read-modify-write.

```json
{
  "checkpoint": 24,
  "episode": 1200,
  "timestamp": "2026-03-22T16:44:35",
  "elapsed_seconds": 8075,
  "val_profit_cents": 345,
  "best_profit_cents": 345,
  "median_profit_cents": 167,
  "epsilon": 0.127,
  "action_distribution": {
    "hold": 0.642,
    "buy_taker": 0.121,
    "buy_maker": 0.087,
    "sell_taker": 0.073,
    "sell_maker": 0.054,
    "buy_cancel": 0.012,
    "sell_cancel": 0.008,
    "buy_amend": 0.002,
    "sell_amend": 0.001
  }
}
```

---

## CLI

Single-run mode (default when `--grid-search` is not passed):

```bash
python train.py \
  --lr 2e-4 \
  --lstm-hidden 64 \
  --seq-len 20 \
  --epsilon-decay 150 \
  --max-hours 12 \
  --num-gpus 4 \
  --checkpoint-dir checkpoints/ \
  --resume
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | `2e-4` | Learning rate |
| `--lstm-hidden` | `64` | LSTM hidden size |
| `--seq-len` | `20` | Sequence length |
| `--epsilon-decay` | `150` | Epsilon decay steps |
| `--max-hours` | `12` | Time cap in hours |
| `--num-gpus` | all available | Number of rollout workers |
| `--checkpoint-dir` | `checkpoints/` | Directory for checkpoints and log |
| `--resume` | off | Resume from latest checkpoint in dir |
| `--grid-search` | off | Run grid search mode (unchanged) |

---

## File Structure

| File | Change | Responsibility |
|------|--------|---------------|
| `train.py` | Modified | CLI entry point; parallel rollout loop; checkpoint startup logic; time cap |
| `src/trainer.py` | Modified | Remove TensorBoard; expose `collect_episodes(device)` and `train_steps()` separately; add `save_checkpoint()` / `load_checkpoint()` |
| `src/train_display.py` | New | Rich Live display — three-panel layout, updated each validation round |
| `src/train_logger.py` | New | Appends JSON entries to `train_log.json` |
| `src/rollout_worker.py` | New | Subprocess worker: receives model weights, runs episodes, returns experiences |
| `src/grid_display.py` | Unchanged | Grid search display (untouched) |

---

## Out of Scope

- Grid search changes of any kind
- TensorBoard (removed entirely from single-run path)
- Early stopping / patience-based stopping (open-ended with time cap only)
- Model architecture changes
