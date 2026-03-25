"""Double DQN + DRQN training loop for Polymarket BTC RL trading agent.

Implements:
  - Double DQN: online network selects best action, target network evaluates Q-value
  - DRQN-style sequence training via PER sub-sequence sampling
  - Soft target updates (Polyak averaging, tau=0.005)
  - Epsilon-greedy with linear decay
  - Gradient clipping (max norm 1.0)
  - Validation evaluation and early stopping
"""

import copy
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.environment import Environment, NUM_ACTIONS
from src.models.base import BaseModel
from src.normalizer import Normalizer
from src.replay_buffer import PrioritizedReplayBuffer


_ACTION_NAMES = [
    "do_nothing", "buy_up_taker", "sell_up_taker",
    "buy_down_taker", "sell_down_taker",
    "limit_buy_up", "limit_sell_up",
    "limit_buy_down", "limit_sell_down",
]
assert len(_ACTION_NAMES) == NUM_ACTIONS, (
    f"_ACTION_NAMES has {len(_ACTION_NAMES)} entries but NUM_ACTIONS={NUM_ACTIONS}"
)


class Trainer:
    """Double DQN trainer with PER and DRQN sequence sampling.

    Args:
        model: Online Q-network.
        normalizer: Fitted feature normalizer.
        config: Training hyperparameters (overrides defaults).
    """

    DEFAULT_CONFIG = {
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "gamma": 0.99,
        "tau": 0.005,
        "epsilon_start": 1.0,
        "epsilon_end": 0.15,
        "epsilon_decay_episodes": 300,
        "buffer_capacity": 50_000,
        "seq_len": 20,
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "per_beta_end": 1.0,
        "grad_clip_norm": 1.0,
        "val_every_episodes": 50,
        "early_stop_patience": 50,  # in validation checkpoints
        "min_buffer_size": 100,
    }

    def __init__(
        self,
        model: BaseModel,
        normalizer: Normalizer,
        config: Optional[dict] = None,
        device: Optional[torch.device] = None,
        on_validation: Optional[callable] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._on_validation = on_validation

        # Online and target networks
        self.online_net = model.to(self.device)
        self.target_net = copy.deepcopy(model).to(self.device)
        self.target_net.eval()

        self.normalizer = normalizer
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config["buffer_capacity"],
            alpha=self.config["per_alpha"],
            seq_len=self.config["seq_len"],
        )

        self.env = Environment()

        # Training state
        self._episode_count = 0
        self._step_count = 0
        self._best_val_profit = -float("inf")
        self._val_no_improve = 0
        self._best_state_dict: Optional[dict] = None
        self._val_profits_history: list[float] = []

    @property
    def epsilon(self) -> float:
        """Current epsilon for epsilon-greedy exploration.

        If episodes_per_epoch is set in config, epsilon resets at the start
        of each epoch so every pass through the data gets a fresh exploration
        phase. Otherwise decays monotonically from epsilon_start to epsilon_end.
        """
        cfg = self.config
        episodes_per_epoch = cfg.get("episodes_per_epoch")
        if episodes_per_epoch:
            count = self._episode_count % max(episodes_per_epoch, 1)
        else:
            count = self._episode_count
        frac = min(count / max(cfg["epsilon_decay_episodes"], 1), 1.0)
        return cfg["epsilon_start"] + frac * (
            cfg["epsilon_end"] - cfg["epsilon_start"]
        )

    @property
    def per_beta(self) -> float:
        """Current PER beta (annealed linearly)."""
        cfg = self.config
        total_episodes = cfg.get("total_episodes", cfg["epsilon_decay_episodes"] * 3)
        frac = min(self._episode_count / max(total_episodes, 1), 1.0)
        return cfg["per_beta_start"] + frac * (
            cfg["per_beta_end"] - cfg["per_beta_start"]
        )

    # ------------------------------------------------------------------
    # Core training
    # ------------------------------------------------------------------

    def train(
        self,
        train_episodes: list[dict[str, Any]],
        val_episodes: list[dict[str, Any]],
        num_epochs: int = 1,
    ) -> dict:
        """Run training loop over episodes.

        Args:
            train_episodes: Training episode dicts.
            val_episodes: Validation episode dicts.
            num_epochs: Number of passes over training episodes.

        Returns:
            Dict with training stats.
        """
        total_episodes_needed = num_epochs * len(train_episodes)
        self.config["total_episodes"] = total_episodes_needed

        for epoch in range(num_epochs):
            # Shuffle training episodes each epoch
            indices = np.random.permutation(len(train_episodes))

            for idx in indices:
                ep = train_episodes[idx]
                reward, action_counts = self._run_episode(ep)
                self._episode_count += 1

                # Log episode metrics
                self._log_episode(reward, action_counts)

                # Train on replay buffer
                if len(self.replay_buffer) >= self.config["min_buffer_size"]:
                    loss = self._train_step()
                    self._step_count += 1

                # Validation check
                if (
                    self._episode_count % self.config["val_every_episodes"] == 0
                    and val_episodes
                ):
                    val_profit = self.evaluate(val_episodes)
                    self._log_validation(val_profit)

                    if val_profit > self._best_val_profit:
                        self._best_val_profit = val_profit
                        self._val_no_improve = 0
                        self._best_state_dict = copy.deepcopy(
                            self.online_net.state_dict()
                        )
                    else:
                        self._val_no_improve += 1

                    if self._val_no_improve >= self.config["early_stop_patience"]:
                        print(
                            f"Early stopping at episode {self._episode_count}. "
                            f"Best val profit: {self._best_val_profit:.4f}"
                        )
                        self._restore_best()
                        return self._training_stats()

        self._restore_best()
        return self._training_stats()

    def _run_episode(
        self, episode: dict[str, Any]
    ) -> tuple[float, np.ndarray]:
        """Run one episode, collecting all transitions for the replay buffer.

        The environment returns 0 reward for non-terminal steps and the full
        episode P&L at the terminal step. After the episode we redistribute
        that terminal reward uniformly across all N transitions so every row
        carries an equal share of the final outcome signal.

        Returns:
            Tuple of (episode_reward, action_counts array of shape (9,)).
        """
        self.env.reset(episode)
        static_features = self.normalizer.encode_static(episode)
        action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)

        transitions: list[dict] = []

        for step_idx in range(self.env.num_rows):
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

            transitions.append({
                "static_features": static_features,
                "dynamic_features": dynamic_features,
                "action": action,
                "reward": reward,  # will be overwritten below
                "next_dynamic_features": next_dynamic,
                "done": done,
                "action_mask": action_mask,
                "next_action_mask": next_mask,
            })

            if done:
                break

        # Distribute terminal P&L evenly across all steps so every row
        # receives a reward with the same sign as the final outcome.
        n = len(transitions)
        episode_reward = transitions[-1]["reward"] if n > 0 else 0.0
        per_step = episode_reward / n if n > 0 else 0.0
        for t in transitions:
            t["reward"] = per_step

        self.replay_buffer.add_episode(transitions)
        return episode_reward, action_counts

    def collect_episode(
        self, episode: dict
    ) -> tuple[float, np.ndarray, list[dict]]:
        """Run one episode and return transitions WITHOUT adding to replay buffer.

        Used by rollout workers in parallel single-run training.
        Terminal P&L is redistributed uniformly across all steps (same as
        _run_episode) before returning.

        Returns:
            Tuple of (episode_reward, action_counts, transitions).
        """
        self.env.reset(episode)
        static_features = self.normalizer.encode_static(episode)
        action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
        transitions: list[dict] = []

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

            transitions.append({
                "static_features": static_features,
                "dynamic_features": dynamic_features,
                "action": action,
                "reward": reward,  # will be overwritten below
                "next_dynamic_features": next_dynamic,
                "done": done,
                "action_mask": action_mask,
                "next_action_mask": next_mask,
            })

            if done:
                break

        n = len(transitions)
        episode_reward = transitions[-1]["reward"] if n > 0 else 0.0
        per_step = episode_reward / n if n > 0 else 0.0
        for t in transitions:
            t["reward"] = per_step

        return episode_reward, action_counts, transitions

    def _select_action_train(
        self,
        static_features: np.ndarray,
        dynamic_features: np.ndarray,
        action_mask: np.ndarray,
    ) -> int:
        """Epsilon-greedy action selection during training."""
        if np.random.random() < self.epsilon:
            # Random action from valid set
            valid = np.where(action_mask)[0]
            return int(np.random.choice(valid))

        # Greedy from online network
        self.online_net.eval()
        with torch.no_grad():
            static_t = torch.tensor(
                static_features, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            dynamic_t = torch.tensor(
                dynamic_features, dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)

            q_values, _ = self.online_net(static_t, dynamic_t)
            q_values = q_values.squeeze(0).cpu().numpy()

        q_values[~action_mask] = -np.inf
        return int(np.argmax(q_values))

    def _train_step(self) -> float:
        """Perform one training step on a batch from the replay buffer.

        Uses Double DQN: online net selects best action, target net evaluates.

        Returns:
            Loss value.
        """
        self.online_net.train()
        cfg = self.config

        batch = self.replay_buffer.sample(
            batch_size=cfg["batch_size"],
            beta=self.per_beta,
        )

        # Convert to tensors
        static = torch.tensor(
            batch["static_features"], dtype=torch.float32, device=self.device
        )  # (B, L, 37)
        dynamic = torch.tensor(
            batch["dynamic_features"], dtype=torch.float32, device=self.device
        )  # (B, L, 12)
        actions = torch.tensor(
            batch["actions"], dtype=torch.int64, device=self.device
        )  # (B, L)
        rewards = torch.tensor(
            batch["rewards"], dtype=torch.float32, device=self.device
        )  # (B, L)
        next_dynamic = torch.tensor(
            batch["next_dynamic_features"], dtype=torch.float32, device=self.device
        )  # (B, L, 12)
        dones = torch.tensor(
            batch["dones"], dtype=torch.float32, device=self.device
        )  # (B, L)
        action_masks = torch.tensor(
            batch["action_masks"], dtype=torch.bool, device=self.device
        )  # (B, L, 9)
        next_action_masks = torch.tensor(
            batch["next_action_masks"], dtype=torch.bool, device=self.device
        )  # (B, L, 9)
        weights = torch.tensor(
            batch["weights"], dtype=torch.float32, device=self.device
        )  # (B,)

        B, L = actions.shape

        # Use static features from the first timestep (they're constant per episode)
        static_first = static[:, 0, :]  # (B, 37)

        # Forward through online network with full sequence
        q_all, _ = self.online_net(static_first, dynamic)  # (B, 9) for last step

        # For DRQN, we process the whole sequence but only compute loss
        # on the last timestep of each sub-sequence.
        # Get Q-values for chosen actions at the last timestep
        last_actions = actions[:, -1]  # (B,)
        q_selected = q_all.gather(1, last_actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Target: Double DQN
        # Step 1: Online net picks the best next action
        # Build next-step input: use the last next_dynamic as a single-step input
        next_dyn_last = next_dynamic[:, -1:, :]  # (B, 1, 11)
        with torch.no_grad():
            q_online_next, _ = self.online_net(static_first, next_dyn_last)  # (B, 9)
            # Mask invalid next actions
            next_mask_last = next_action_masks[:, -1, :]  # (B, 9)
            q_online_next[~next_mask_last] = -float("inf")
            best_next_actions = q_online_next.argmax(dim=1)  # (B,)

            # Step 2: Target net evaluates
            q_target_next, _ = self.target_net(static_first, next_dyn_last)  # (B, 9)
            q_target_selected = q_target_next.gather(
                1, best_next_actions.unsqueeze(1)
            ).squeeze(1)  # (B,)

        # Compute TD target
        last_rewards = rewards[:, -1]  # (B,)
        last_dones = dones[:, -1]  # (B,)
        td_target = last_rewards + cfg["gamma"] * q_target_selected * (1 - last_dones)

        # TD error for PER updates
        td_errors = (q_selected - td_target).detach().cpu().numpy()

        # Weighted Huber loss
        loss = nn.functional.smooth_l1_loss(
            q_selected, td_target.detach(), reduction="none"
        )
        loss = (loss * weights).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(
            self.online_net.parameters(), cfg["grad_clip_norm"]
        )

        self.optimizer.step()

        # Soft target update
        self._soft_update()

        # Update PER priorities
        self.replay_buffer.update_priorities(batch["indices"], td_errors)

        # Log training metrics
        self._log_train_step(loss.item(), q_all.detach(), float(grad_norm))

        return loss.item()

    def _soft_update(self) -> None:
        """Polyak averaging: target = tau * online + (1-tau) * target."""
        tau = self.config["tau"]
        for tp, op in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            tp.data.copy_(tau * op.data + (1 - tau) * tp.data)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, episodes: list[dict[str, Any]]) -> float:
        """Evaluate the online network greedily on a set of episodes.

        Args:
            episodes: List of episode dicts.

        Returns:
            Total profit (sum of rewards * 100) across all episodes, in cents.
        """
        self.online_net.eval()
        total_profit = 0.0

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

                    q_values, hidden = self.online_net(
                        static_t, dynamic_t, hidden
                    )
                    q_values = q_values.squeeze(0).cpu().numpy()

                q_values[~action_mask] = -np.inf
                action = int(np.argmax(q_values))

                done, reward = self.env.step(action)

                if done:
                    total_profit += reward * 100.0  # Convert to cents
                    break

        return total_profit

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

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_episode(self, reward: float, action_counts: np.ndarray) -> None:
        """Log per-episode metrics."""
        pass

    def _log_train_step(
        self, loss: float, q_values: torch.Tensor, grad_norm: float
    ) -> None:
        """Log per-step training metrics."""
        pass

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

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _restore_best(self) -> None:
        """Restore the best model weights from validation."""
        if self._best_state_dict is not None:
            self.online_net.load_state_dict(self._best_state_dict)

    def _training_stats(self) -> dict:
        """Return summary of training."""
        return {
            "episodes_trained": self._episode_count,
            "train_steps": self._step_count,
            "best_val_profit": self._best_val_profit,
            "early_stopped": self._val_no_improve >= self.config["early_stop_patience"],
        }

    def save_checkpoint(self, path: str) -> None:
        """Save online network state dict."""
        torch.save(self.online_net.state_dict(), path)

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
