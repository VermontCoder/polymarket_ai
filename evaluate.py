"""Evaluation / visibility mode entry point for Polymarket BTC RL trading agent.

Usage:
    # Random agent on all episodes (auto-detects JSON in data/)
    python evaluate.py --player random

    # Random agent on specific episode count
    python evaluate.py --player random --num-episodes 10

    # Trained agent with checkpoint
    python evaluate.py --player dqn --checkpoint checkpoints/model.pt

    # Explicit data file
    python evaluate.py --data data/episodes.json --player random

    # Specific episode indices
    python evaluate.py --player random --episode-ids 0 5 10
"""

import argparse
import glob
import random

import torch

from src.data_loader import load_episodes, split_episodes
from src.models.lstm_dqn import LSTMDQN
from src.normalizer import Normalizer
from src.agents.dqn_agent import DQNAgent
from src.visibility import run_visibility


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RL agent with console visibility"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to JSON episodes file (default: first JSON found in data/)",
    )
    parser.add_argument(
        "--player", type=str, default="random", choices=["random", "dqn"],
        help="Player type: random or dqn",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (required for dqn player)",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=None,
        help="Number of episodes to run (default: all)",
    )
    parser.add_argument(
        "--episode-ids", type=int, nargs="+", default=None,
        help="Specific episode indices to run",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test", "all"],
        help="Which data split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for data splitting",
    )
    parser.add_argument(
        "--lstm-hidden", type=int, default=32,
        help="LSTM hidden size (must match checkpoint)",
    )
    return parser.parse_args()


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

    # Select split
    if args.split == "all":
        selected = episodes
    else:
        train_eps, val_eps, test_eps = split_episodes(episodes, seed=args.seed)
        split_map = {"train": train_eps, "val": val_eps, "test": test_eps}
        selected = split_map[args.split]
        print(f"Using {args.split} split: {len(selected)} episodes")

    # Filter by episode IDs or count
    if args.episode_ids is not None:
        selected = [selected[i] for i in args.episode_ids if i < len(selected)]
    elif args.num_episodes is not None:
        rng = random.Random(args.seed)
        selected = rng.sample(selected, min(args.num_episodes, len(selected)))

    print(f"Running {len(selected)} episodes with {args.player} agent")
    print(f"Seed: {args.seed}\n")

    # Set up normalizer (needed for both agents for consistency)
    normalizer = Normalizer()
    if args.split == "all":
        # Fit on first 80% as proxy for training set
        n_train = int(len(episodes) * 0.8)
        normalizer.fit(episodes[:n_train])
    else:
        train_eps, _, _ = split_episodes(episodes, seed=args.seed)
        normalizer.fit(train_eps)

    # Set up DQN agent if needed
    dqn_agent = None
    if args.player == "dqn":
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for dqn player")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMDQN(lstm_hidden_size=args.lstm_hidden)
        dqn_agent = DQNAgent.from_checkpoint(model, args.checkpoint, device)

    total_profit = run_visibility(
        episodes=selected,
        player=args.player,
        normalizer=normalizer,
        dqn_agent=dqn_agent,
    )

    print(f"\n{'='*69}")
    print(f"Final Total Profit: {total_profit:+.2f}c across {len(selected)} episodes")
    print(f"Average Profit: {total_profit / len(selected):+.2f}c per episode")


if __name__ == "__main__":
    main()
