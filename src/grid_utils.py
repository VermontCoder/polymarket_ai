"""Shared utilities for grid search — importable by both train.py and src modules."""


def config_key(config: dict) -> str:
    """Create a stable string key for a config dict (excludes 'epochs')."""
    return (
        f"lr={config['lr']}_ed={config['epsilon_decay']}"
        f"_sl={config['seq_len']}_h={config['lstm_hidden']}"
    )
