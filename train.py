"""Training entry point for Polymarket BTC 5-minute RL trading agent.

Usage:
    python train.py --data data/episodes.json
    python train.py --data data/episodes.json --epochs 3 --lr 1e-4
    python train.py --data data/episodes.json --grid-search
"""

import argparse
import itertools
import json
import os

import numpy as np
import torch

from src.data_loader import load_episodes, split_episodes
from src.models.lstm_dqn import LSTMDQN
from src.normalizer import Normalizer
from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RL agent for Polymarket BTC 5-minute market"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to JSON episodes file",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-path", type=str, default="checkpoints/model.pt",
        help="Path to save best model checkpoint",
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="TensorBoard log directory",
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
    return parser.parse_args()


def train_single(
    train_eps, val_eps, test_eps, config, seed, save_path, log_dir=None,
):
    """Train a single configuration and return validation profit."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    normalizer = Normalizer()
    normalizer.fit(train_eps)

    model = LSTMDQN(
        lstm_hidden_size=config.get("lstm_hidden", 32),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    )

    stats = trainer.train(
        train_episodes=train_eps,
        val_episodes=val_eps,
        num_epochs=config.get("epochs", 1),
        log_dir=log_dir,
    )

    # Final test evaluation
    test_profit = trainer.evaluate(test_eps)

    print(f"\nTraining complete. Stats: {stats}")
    print(f"Test profit: {test_profit:.2f}c")

    # Save checkpoint
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    trainer.save_checkpoint(save_path)
    print(f"Model saved to {save_path}")

    trainer.close()
    return stats["best_val_profit"]


def _config_key(config):
    """Create a stable string key for a config dict (excludes 'epochs')."""
    return (
        f"lr={config['lr']}_ed={config['epsilon_decay']}"
        f"_sl={config['seq_len']}_h={config['lstm_hidden']}"
    )


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


def grid_search(train_eps, val_eps, test_eps, save_path):
    """Run hyperparameter grid search per spec. Supports resume via JSON checkpoint."""
    param_grid = {
        "lr": [5e-5, 1e-4, 3e-4],
        "epsilon_decay": [150, 300],
        "seq_len": [10, 20, 40],
        "lstm_hidden": [16, 32, 48],
    }
    seeds = [42, 123, 456]

    results_path = "checkpoints/grid_results.json"
    results = _load_grid_results(results_path)

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    all_combos = list(itertools.product(*values))
    completed = len(results)
    total = len(all_combos)
    if completed > 0:
        print(f"Resuming grid search: {completed}/{total} configs already done")

    best_median = -float("inf")
    best_config = None

    # Restore best from previous results
    for key, entry in results.items():
        if entry["median_val_profit"] > best_median:
            best_median = entry["median_val_profit"]
            best_config = entry["config"]

    for i, combo in enumerate(all_combos):
        config = dict(zip(keys, combo))
        config["epochs"] = 1
        key = _config_key(config)

        if key in results:
            continue

        print(f"\n[Config {i+1}/{total}] {config}")

        seed_profits = []
        for seed in seeds:
            log_dir = (
                f"runs/grid_lr{config['lr']}_ed{config['epsilon_decay']}"
                f"_sl{config['seq_len']}_h{config['lstm_hidden']}_s{seed}"
            )
            val_profit = train_single(
                train_eps, val_eps, test_eps, config, seed,
                save_path="checkpoints/grid_temp.pt",
                log_dir=log_dir,
            )
            seed_profits.append(val_profit)

        median_profit = float(np.median(seed_profits))
        print(f"\nConfig: {config} | Median val profit: {median_profit:.2f}c")
        print(f"  Per-seed profits: {seed_profits}")

        # Save after each config completes
        results[key] = {
            "config": {k: v for k, v in config.items() if k != "epochs"},
            "seed_profits": seed_profits,
            "median_val_profit": median_profit,
        }
        _save_grid_results(results_path, results)

        if median_profit > best_median:
            best_median = median_profit
            best_config = config

    print(f"\n{'='*60}")
    print(f"Best config: {best_config}")
    print(f"Best median val profit: {best_median:.2f}c")
    print(f"\nRetraining best config with seed 42...")

    # Retrain best config for final model
    train_single(
        train_eps, val_eps, test_eps, best_config, seed=42,
        save_path=save_path,
    )


def main():
    args = parse_args()

    print(f"Loading episodes from {args.data}...")
    episodes = load_episodes(args.data)
    print(f"Loaded {len(episodes)} episodes")

    train_eps, val_eps, test_eps = split_episodes(episodes, seed=args.seed)
    print(
        f"Split: {len(train_eps)} train / {len(val_eps)} val / {len(test_eps)} test"
    )

    if args.grid_search:
        grid_search(train_eps, val_eps, test_eps, args.save_path)
    else:
        config = {
            "lr": args.lr,
            "lstm_hidden": args.lstm_hidden,
            "seq_len": args.seq_len,
            "epsilon_decay": args.epsilon_decay,
            "epochs": args.epochs,
        }
        train_single(
            train_eps, val_eps, test_eps, config, args.seed,
            args.save_path, args.log_dir,
        )


if __name__ == "__main__":
    main()
