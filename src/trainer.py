"""Double DQN + DRQN training loop for Polymarket BTC RL trading agent.

Implements:
  - Double DQN: online network selects best action, target network evaluates Q-value
  - DRQN-style sequence training via PER sub-sequence sampling
  - Soft target updates (Polyak averaging, tau=0.005)
  - Epsilon-greedy with linear decay
  - Gradient clipping (max norm 1.0)
  - TensorBoard logging
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
        "epsilon_end": 0.05,
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
    ) -> None:
        self.device = device or torch.device("cpu")
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

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

        # TensorBoard writer (lazy init)
        self._writer = None

    def _get_writer(self):
        """Lazy-initialize TensorBoard writer."""
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter()
        return self._writer

    @property
    def epsilon(self) -> float:
        """Current epsilon for epsilon-greedy exploration."""
        cfg = self.config
        frac = min(
            self._episode_count / max(cfg["epsilon_decay_episodes"], 1), 1.0
        )
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
        log_dir: Optional[str] = None,
    ) -> dict:
        """Run training loop over episodes.

        Args:
            train_episodes: Training episode dicts.
            val_episodes: Validation episode dicts.
            num_epochs: Number of passes over training episodes.
            log_dir: TensorBoard log directory override.

        Returns:
            Dict with training stats.
        """
        if log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir)

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
        """Run one episode, collecting transitions for the replay buffer.

        Returns:
            Tuple of (episode_reward, action_counts array of shape (9,)).
        """
        self.env.reset(episode)
        static_features = self.normalizer.encode_static(episode)
        action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)

        transitions: list[dict] = []
        prev_dynamic = None

        for step_idx in range(self.env.num_rows):
            obs = self.env.get_observation()
            dynamic_features = self.normalizer.encode_dynamic(obs)
            action_mask = self.env.get_action_mask()

            # Epsilon-greedy action selection
            action = self._select_action_train(
                static_features, dynamic_features, action_mask
            )
            action_counts[action] += 1

            done, reward = self.env.step(action)

            # Store transition (only pre-action timesteps)
            # After the agent has acted, we still store "do nothing" transitions
            # but only until the agent acts. The spec says "only pre-action timesteps".
            # We collect ALL timesteps but mark appropriately.
            next_obs = None
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
                "reward": reward if done else 0.0,
                "next_dynamic_features": next_dynamic,
                "done": done,
                "action_mask": action_mask,
                "next_action_mask": next_mask,
            })

            if done:
                # Assign the final reward to the transition where the
                # agent acted (if any), not to the terminal transition.
                if reward != 0.0 and len(transitions) > 1:
                    self._assign_reward_to_action_step(transitions, reward)
                break

        # Only store pre-action timesteps (up to and including the action)
        pre_action_transitions = self._filter_pre_action(transitions)
        self.replay_buffer.add_episode(pre_action_transitions)

        final_reward = reward if done else 0.0
        return final_reward, action_counts

    def _assign_reward_to_action_step(
        self, transitions: list[dict], final_reward: float
    ) -> None:
        """Move the terminal reward to the step where the agent acted.

        In this environment, reward is only known at episode end, but
        it logically belongs to the action step. We set intermediate
        rewards to 0 and the action step's reward to the final reward.
        For the last transition (terminal), set reward=0 since the
        action step gets it.
        """
        # Find the step where a non-zero action was taken
        action_idx = None
        for i, t in enumerate(transitions):
            if t["action"] != 0:
                action_idx = i
                break

        if action_idx is not None:
            # Clear terminal reward, assign to action step
            transitions[-1]["reward"] = 0.0
            transitions[action_idx]["reward"] = final_reward

    def _filter_pre_action(self, transitions: list[dict]) -> list[dict]:
        """Return only pre-action transitions (before and including the action).

        Post-action forced "do nothing" steps are excluded from the buffer.
        """
        result = []
        acted = False
        for t in transitions:
            if not acted:
                result.append(t)
                if t["action"] != 0:
                    acted = True
            # Skip post-action steps (except if no action was ever taken,
            # include everything — the agent chose to do nothing throughout)

        # If agent never acted, include all transitions
        if not acted:
            return transitions

        return result

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
        )  # (B, L, 11)
        actions = torch.tensor(
            batch["actions"], dtype=torch.int64, device=self.device
        )  # (B, L)
        rewards = torch.tensor(
            batch["rewards"], dtype=torch.float32, device=self.device
        )  # (B, L)
        next_dynamic = torch.tensor(
            batch["next_dynamic_features"], dtype=torch.float32, device=self.device
        )  # (B, L, 11)
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

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_episode(self, reward: float, action_counts: np.ndarray) -> None:
        """Log per-episode metrics to TensorBoard."""
        writer = self._get_writer()
        ep = self._episode_count
        writer.add_scalar("episode/reward", reward, ep)
        writer.add_scalar("episode/epsilon", self.epsilon, ep)

        # Action distribution
        total_actions = action_counts.sum()
        if total_actions > 0:
            for i in range(NUM_ACTIONS):
                writer.add_scalar(
                    f"actions/action_{i}_frac",
                    action_counts[i] / total_actions,
                    ep,
                )

    def _log_train_step(
        self, loss: float, q_values: torch.Tensor, grad_norm: float
    ) -> None:
        """Log per-step training metrics."""
        writer = self._get_writer()
        step = self._step_count
        writer.add_scalar("train/loss", loss, step)
        writer.add_scalar("train/grad_norm", grad_norm, step)
        writer.add_scalar("train/q_mean", q_values.mean().item(), step)
        writer.add_scalar("train/q_std", q_values.std().item(), step)
        writer.add_scalar("train/q_max", q_values.max().item(), step)
        writer.add_scalar("train/q_min", q_values.min().item(), step)

    def _log_validation(self, val_profit: float) -> None:
        """Log validation profit."""
        writer = self._get_writer()
        writer.add_scalar(
            "val/total_profit_cents", val_profit, self._episode_count
        )
        print(
            f"[Episode {self._episode_count}] "
            f"Val profit: {val_profit:.2f}c | "
            f"Best: {self._best_val_profit:.2f}c | "
            f"Epsilon: {self.epsilon:.3f}"
        )

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

    def close(self) -> None:
        """Close TensorBoard writer."""
        if self._writer is not None:
            self._writer.close()
