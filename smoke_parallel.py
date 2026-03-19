"""Smoke test for parallel grid search with Rich display. Run from project root.

Usage:
    python smoke_parallel.py
"""
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
    # Use a separate results file so we don't corrupt the real one
    train.GRID_RESULTS_PATH = "checkpoints/smoke_grid_results.json"

    eps = load_episodes("data/episodes.json")
    train_eps, val_eps, test_eps = split_episodes(eps)
    train.grid_search(
        train_eps, val_eps, test_eps,
        save_path="checkpoints/smoke_test.pt",
        seeds=[42, 123],   # 2 seeds so seed progress column updates
        num_workers=2,     # 2 workers running simultaneously
    )
    print("Smoke test passed.")
