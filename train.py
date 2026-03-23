"""Training entry point for Polymarket BTC 5-minute RL trading agent.

Usage:
    python train.py                        # uses first JSON in data/
    python train.py --data data/foo.json   # explicit file
    python train.py --epochs 3 --lr 1e-4
    python train.py --grid-search
"""

import argparse
import concurrent.futures
import glob
import itertools
import json
import os

import numpy as np
import torch

from src.data_loader import load_episodes, split_episodes
from src.grid_utils import config_key as _config_key
from src.models.lstm_dqn import LSTMDQN
from src.normalizer import Normalizer
from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RL agent for Polymarket BTC 5-minute market"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to JSON episodes file (default: first JSON found in data/)",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-path", type=str, default="checkpoints/model.pt",
        help="Path to save best model checkpoint",
    )
    parser.add_argument(
        "--grid-search", action="store_true",
        help="Run hyperparameter grid search",
    )
    parser.add_argument(
        "--lstm-hidden", type=int, default=32,
        help="LSTM hidden size",
    )
    parser.add_argument(
        "--seq-len", type=int, default=20,
        help="Sub-sequence length for DRQN sampling",
    )
    parser.add_argument(
        "--epsilon-decay", type=int, default=300,
        help="Episodes over which to decay epsilon",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of parallel worker processes for grid search. "
             "Defaults to os.cpu_count().",
    )
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
    return parser.parse_args()


def train_single(
    train_eps, val_eps, test_eps, config, seed, save_path, log_dir=None,
    on_validation=None, device=None,
):
    """Train a single configuration and return validation profit."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    normalizer = Normalizer()
    normalizer.fit(train_eps)

    model = LSTMDQN(
        lstm_hidden_size=config.get("lstm_hidden", 32),
    )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | Config: {config} | Seed: {seed}")

    trainer = Trainer(
        model=model,
        normalizer=normalizer,
        config={
            "lr": config.get("lr", 1e-4),
            "seq_len": config.get("seq_len", 20),
            "epsilon_decay_episodes": config.get("epsilon_decay", 300),
        },
        device=device,
        on_validation=on_validation,
    )

    stats = trainer.train(
        train_episodes=train_eps,
        val_episodes=val_eps,
        num_epochs=config.get("epochs", 1),
    )

    # Final test evaluation
    test_profit = trainer.evaluate(test_eps)

    print(f"\nTraining complete. Stats: {stats}")
    print(f"Test profit: {test_profit:.2f}c")

    # Save checkpoint
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    trainer.save_checkpoint(save_path)
    print(f"Model saved to {save_path}")

    return stats["best_val_profit"]


def _load_grid_results(path):
    """Load existing grid search results from JSON file."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_grid_results(path, results):
    """Save grid search results to JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


GRID_RESULTS_PATH = "checkpoints/grid_results_p2.json"

# Phase 2 grid: focused on productive region found in Phase 1.
# Dropped: lr=5e-5 (consistently weak), seq_len=40 (no signal), h=16/32 (collapse/underfit).
# Expanded: lstm_hidden up to 128 (h=48 was the floor, not the ceiling).
PARAM_GRID = {
    "lr": [1e-4, 2e-4, 3e-4],
    "epsilon_decay": [150, 300],
    "seq_len": [10, 20],
    "lstm_hidden": [48, 64, 96, 128],
}


def run_config_worker(
    config: dict,
    seeds: list,
    train_eps: list,
    val_eps: list,
    test_eps: list,
    status_queue=None,
    worker_id: int = 0,
) -> tuple:
    """Worker function for parallel grid search.

    Runs all seeds for one config sequentially and returns results.
    Must be a top-level function for multiprocessing pickling on Windows.

    When status_queue is provided:
    - Redirects stdout to devnull (prevents raw prints corrupting Rich display)
    - Pushes seed_start, val, seed_done events to the queue

    Args:
        config: Hyperparameter dict (including 'epochs').
        seeds: List of random seeds to run.
        train_eps: Training episodes.
        val_eps: Validation episodes.
        test_eps: Test episodes.
        status_queue: Optional queue for pushing status events to the parent.
        worker_id: Worker index for GPU assignment (round-robin across available GPUs).

    Returns:
        Tuple of (config_key, seed_profits, median_val_profit).
    """
    import sys
    import os as _os

    key = _config_key(config)
    seed_profits = []
    total_seeds = len(seeds)

    # Assign GPU round-robin across available GPUs
    num_gpus = torch.cuda.device_count()
    gpu_id = worker_id % num_gpus if num_gpus > 0 else None
    device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cpu")

    # Suppress stdout in workers so the Rich display in the parent isn't corrupted.
    # stderr is left alone so tracebacks remain visible.
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
                device=device,
            )
            seed_profits.append(val_profit)

            try:
                os.remove(temp_path)
            except OSError:
                pass

            if status_queue is not None:
                status_queue.put({
                    "key": key,
                    "event": "seed_done",
                    "seed": seed,
                    "seeds_done": seed_idx + 1,
                })

    finally:
        if _devnull is not None:
            sys.stdout = _orig_stdout
            _devnull.close()

    median_profit = float(np.median(seed_profits))
    return key, seed_profits, median_profit


def grid_search(train_eps, val_eps, test_eps, save_path, seeds=None, num_workers=None):
    """Run hyperparameter grid search with parallel workers and Rich live display.

    Args:
        train_eps: Training episodes.
        val_eps: Validation episodes.
        test_eps: Test episodes.
        save_path: Where to save the final best model.
        seeds: List of random seeds (default: [42, 123, 456]).
        num_workers: Number of parallel worker processes. Defaults to os.cpu_count().
    """
    import multiprocessing
    import threading
    from src.grid_display import GridDisplay

    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This must be done before creating any ProcessPoolExecutor
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set, ignore
        pass

    if seeds is None:
        seeds = [42, 123, 456, 789, 999]

    results = _load_grid_results(GRID_RESULTS_PATH)

    keys_list = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    all_combos = list(itertools.product(*values))
    total = len(all_combos)

    # Build list of pending configs (skip already completed)
    pending = []
    for combo in all_combos:
        config = dict(zip(keys_list, combo))
        config["epochs"] = 3
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
                        config, seeds, train_eps, val_eps, test_eps, status_queue, worker_id,
                    ): config
                    for worker_id, config in enumerate(pending)
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
        manager.shutdown()

    print(f"\n{'='*60}")
    print(f"Best config: {best_config}")
    print(f"Best median val profit: {best_median:.2f}c")

    if best_config is None:
        print("No configs completed; skipping final retrain.")
        return

    print(f"\nRetraining best config with seed 42...")

    train_single(train_eps, val_eps, test_eps, best_config, seed=42, save_path=save_path)


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
            "lstm_hidden": config.get("lstm_hidden", 32),
            "episodes_per_epoch": len(train_eps),
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

    episodes_per_epoch = len(train_eps)
    current_epoch_num = trainer._episode_count // episodes_per_epoch
    epoch_val_profits: list[float] = []

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

                # Epoch median: reset when episode count crosses an epoch boundary
                new_epoch_num = trainer._episode_count // episodes_per_epoch
                if new_epoch_num != current_epoch_num:
                    epoch_val_profits = []
                    current_epoch_num = new_epoch_num
                epoch_val_profits.append(val_profit)
                epoch_median = statistics.median(epoch_val_profits)

                checkpoint_num += 1
                current_elapsed = elapsed_seconds + (time.time() - start_wall)

                # Update display
                display.update(
                    episode=trainer._episode_count,
                    val_profit=val_profit,
                    best_profit=best_profit,
                    median_profit=median_profit,
                    epoch_median=epoch_median,
                    epoch=current_epoch_num,
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
                    epoch_median_cents=epoch_median,
                    epsilon=trainer.epsilon,
                    action_distribution=action_dist,
                )

                # Save checkpoints
                trainer.save_full_checkpoint(checkpoint_path, elapsed_seconds=current_elapsed)
                if is_new_best:
                    trainer.save_full_checkpoint(best_path, elapsed_seconds=current_elapsed)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Last checkpoint saved.")


def main():
    args = parse_args()

    data_path = args.data
    if data_path is None:
        files = glob.glob("data/*.json")
        if not files:
            raise FileNotFoundError("No JSON files found in data/ and --data not specified")
        data_path = files[0]

    print(f"Loading episodes from {data_path}...")
    episodes = load_episodes(data_path)
    print(f"Loaded {len(episodes)} episodes")

    train_eps, val_eps, test_eps = split_episodes(episodes, seed=args.seed)
    print(
        f"Split: {len(train_eps)} train / {len(val_eps)} val / {len(test_eps)} test"
    )

    if args.grid_search:
        grid_search(train_eps, val_eps, test_eps, args.save_path,
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


if __name__ == "__main__":
    main()
