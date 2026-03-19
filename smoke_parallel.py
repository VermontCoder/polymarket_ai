"""Smoke test for parallel grid search with Rich display. Run from project root.

Usage:
    python smoke_parallel.py

Always runs fresh — deletes the smoke results file at startup so repeated runs
actually exercise the code rather than skipping completed configs.
"""
import glob
import os

import train
from src.data_loader import load_episodes, split_episodes

SMOKE_RESULTS_PATH = "checkpoints/smoke_grid_results.json"

if __name__ == "__main__":
    # Always start fresh so repeated runs don't skip everything
    if os.path.exists(SMOKE_RESULTS_PATH):
        os.remove(SMOKE_RESULTS_PATH)

    # Two configs so both rows of the display get exercised
    train.PARAM_GRID = {
        "lr": [1e-4, 3e-4],
        "epsilon_decay": [150],
        "seq_len": [10],
        "lstm_hidden": [16],
    }
    # Use a separate results file so we don't corrupt the real grid results
    train.GRID_RESULTS_PATH = SMOKE_RESULTS_PATH

    data_files = glob.glob("data/*.json")
    if not data_files:
        raise FileNotFoundError("No JSON files found in data/")
    eps = load_episodes(data_files[0])
    train_eps, val_eps, test_eps = split_episodes(eps)
    train.grid_search(
        train_eps, val_eps, test_eps,
        save_path="checkpoints/smoke_test.pt",
        seeds=[42, 123],   # 2 seeds so seed progress column updates
        num_workers=2,     # 2 workers running simultaneously
    )
    print("Smoke test passed.")
