"""Prioritized Experience Replay buffer with DRQN-style sub-sequence sampling.

Uses a sum-tree data structure for O(log n) proportional-priority sampling.
Episodes are stored as contiguous blocks; sub-sequences of length L are
sampled without crossing episode boundaries.

Key design:
  - PER alpha = 0.6 (prioritization exponent)
  - PER beta = 0.4 -> 1.0 (importance-sampling correction, annealed externally)
  - Max capacity: 50,000 transitions (configurable)
  - Sub-sequence length L = 20 (configurable)
  - Priority = (|TD error| + epsilon)^alpha
  - New transitions get max priority so they are sampled at least once
  - Only PRE-ACTION timesteps are stored (post-action "do nothing" steps excluded)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Sum Tree
# ---------------------------------------------------------------------------

class SumTree:
    """Binary sum-tree for O(log n) proportional-priority sampling.

    Leaf nodes store priorities. Internal nodes store the sum of their
    children's priorities, enabling efficient sampling proportional to
    priority and O(log n) updates.
    """

    def __init__(self, capacity: int) -> None:
        # Round up to next power of 2 for correct binary tree layout
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2
        # Tree has 2 * capacity nodes (1-indexed style stored in 0-indexed
        # array of size 2 * capacity).
        self._tree = np.zeros(2 * self.capacity, dtype=np.float64)
        self._write_idx = 0
        self._size = 0

    @property
    def total(self) -> float:
        """Return the total priority (root of the tree)."""
        return self._tree[1]

    @property
    def size(self) -> int:
        return self._size

    def min_priority(self) -> float:
        """Return the minimum nonzero priority among filled leaves."""
        if self._size == 0:
            return 0.0
        leaves = self._tree[self.capacity: self.capacity + self._size]
        nonzero = leaves[leaves > 0]
        if len(nonzero) == 0:
            return 0.0
        return float(nonzero.min())

    def add(self, priority: float) -> int:
        """Add a new leaf with the given priority, returning its leaf index."""
        idx = self._write_idx
        self._set(idx, priority)
        self._write_idx = (self._write_idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return idx

    def update(self, leaf_idx: int, priority: float) -> None:
        """Update the priority of an existing leaf."""
        self._set(leaf_idx, priority)

    def get(self, cumsum: float) -> tuple[int, float]:
        """Sample a leaf by cumulative sum.

        Walk down the tree: go left if cumsum <= left child, else go right
        (subtracting left child's value).

        Returns:
            (leaf_index, priority)
        """
        node = 1  # root
        while node < self.capacity:
            left = 2 * node
            right = left + 1
            if cumsum <= self._tree[left]:
                node = left
            else:
                cumsum -= self._tree[left]
                node = right
        leaf_idx = node - self.capacity
        return leaf_idx, self._tree[node]

    def _set(self, leaf_idx: int, priority: float) -> None:
        """Set the priority of a leaf and propagate changes up."""
        tree_idx = leaf_idx + self.capacity
        delta = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        # Propagate up
        tree_idx //= 2
        while tree_idx >= 1:
            self._tree[tree_idx] += delta
            tree_idx //= 2


# ---------------------------------------------------------------------------
# Prioritized Replay Buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with DRQN sub-sequence sampling.

    Stores transitions organized by episode. Sampling returns contiguous
    sub-sequences of length ``seq_len`` that never cross episode boundaries.

    Each transition dict must contain:
        static_features     np.array (35,)
        dynamic_features    np.array (12,)
        action              int
        reward              float
        next_dynamic_features  np.array (12,) or None if terminal
        done                bool
        action_mask         np.array (9,) bool
        next_action_mask    np.array (9,) bool or None if terminal
    """

    STATIC_DIM = 37
    DYNAMIC_DIM = 12
    NUM_ACTIONS = 9

    def __init__(
        self,
        capacity: int = 50_000,
        alpha: float = 0.6,
        seq_len: int = 20,
        epsilon: float = 1e-6,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.seq_len = seq_len
        self.epsilon = epsilon

        # Sum tree for priority-based sampling
        self._tree = SumTree(capacity)

        # Flat storage arrays (pre-allocated for capacity)
        self._static_features = np.zeros(
            (capacity, self.STATIC_DIM), dtype=np.float32
        )
        self._dynamic_features = np.zeros(
            (capacity, self.DYNAMIC_DIM), dtype=np.float32
        )
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_dynamic_features = np.zeros(
            (capacity, self.DYNAMIC_DIM), dtype=np.float32
        )
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._action_masks = np.zeros(
            (capacity, self.NUM_ACTIONS), dtype=np.float32
        )
        self._next_action_masks = np.zeros(
            (capacity, self.NUM_ACTIONS), dtype=np.float32
        )

        # Episode boundary tracking: each transition knows which episode it
        # belongs to and its position within that episode.
        self._episode_ids = np.full(capacity, -1, dtype=np.int64)
        self._positions = np.zeros(capacity, dtype=np.int64)
        self._episode_lengths = np.zeros(capacity, dtype=np.int64)

        # Write cursor and bookkeeping
        self._write_idx = 0
        self._size = 0
        self._max_priority: float = 1.0
        self._episode_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_episode(self, transitions: list[dict]) -> None:
        """Add an episode's pre-action transitions to the buffer.

        Each transition is stored individually but tagged with episode id
        and intra-episode position so that sub-sequences can respect
        episode boundaries.
        """
        if len(transitions) == 0:
            return

        ep_id = self._episode_counter
        self._episode_counter += 1
        ep_len = len(transitions)

        for pos, t in enumerate(transitions):
            idx = self._write_idx

            # Store transition data
            self._static_features[idx] = t["static_features"]
            self._dynamic_features[idx] = t["dynamic_features"]
            self._actions[idx] = t["action"]
            self._rewards[idx] = t["reward"]

            if t["next_dynamic_features"] is not None:
                self._next_dynamic_features[idx] = t["next_dynamic_features"]
            else:
                self._next_dynamic_features[idx] = 0.0

            self._dones[idx] = float(t["done"])
            self._action_masks[idx] = t["action_mask"].astype(np.float32)

            if t["next_action_mask"] is not None:
                self._next_action_masks[idx] = t["next_action_mask"].astype(
                    np.float32
                )
            else:
                self._next_action_masks[idx] = 0.0

            self._episode_ids[idx] = ep_id
            self._positions[idx] = pos
            self._episode_lengths[idx] = ep_len

            # Add to sum tree with max priority (ensures it gets sampled)
            priority = self._max_priority ** self.alpha
            self._tree.add(priority)

            # Advance write cursor
            self._write_idx = (self._write_idx + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int = 32, beta: float = 0.4) -> dict:
        """Sample a batch of sub-sequences using proportional prioritization.

        For each sampled index, we extract a sub-sequence of length
        ``seq_len`` starting from some valid position within the same
        episode, ensuring the sampled transition is included in the
        sub-sequence. If the episode is shorter than ``seq_len``, the
        sub-sequence is zero-padded and marked as done.

        Returns:
            dict with keys:
                static_features:      (batch, seq_len, 35)
                dynamic_features:     (batch, seq_len, 12)
                actions:              (batch, seq_len)
                rewards:              (batch, seq_len)
                next_dynamic_features:(batch, seq_len, 12)
                dones:                (batch, seq_len)
                action_masks:         (batch, seq_len, 9)
                next_action_masks:    (batch, seq_len, 9)
                weights:              (batch,) importance-sampling weights
                indices:              (batch,) leaf indices for priority update
        """
        assert self._size > 0, "Cannot sample from empty buffer"

        L = self.seq_len
        total = self._tree.total

        # Pre-allocate output arrays
        b_static = np.zeros(
            (batch_size, L, self.STATIC_DIM), dtype=np.float32
        )
        b_dynamic = np.zeros(
            (batch_size, L, self.DYNAMIC_DIM), dtype=np.float32
        )
        b_actions = np.zeros((batch_size, L), dtype=np.int64)
        b_rewards = np.zeros((batch_size, L), dtype=np.float32)
        b_next_dynamic = np.zeros(
            (batch_size, L, self.DYNAMIC_DIM), dtype=np.float32
        )
        b_dones = np.zeros((batch_size, L), dtype=np.float32)
        b_action_masks = np.zeros(
            (batch_size, L, self.NUM_ACTIONS), dtype=np.float32
        )
        b_next_action_masks = np.zeros(
            (batch_size, L, self.NUM_ACTIONS), dtype=np.float32
        )
        b_weights = np.zeros(batch_size, dtype=np.float32)
        b_indices = np.zeros(batch_size, dtype=np.int64)

        # Stratified sampling from the sum tree
        segment = total / batch_size
        min_prob = self._tree.min_priority() / total if total > 0 else 1.0
        # max_weight for normalization
        max_weight = (self._size * min_prob) ** (-beta) if min_prob > 0 else 1.0

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            cumsum = np.random.uniform(low, high)
            leaf_idx, priority = self._tree.get(cumsum)

            # Ensure leaf_idx is within valid range
            leaf_idx = min(leaf_idx, self._size - 1)

            b_indices[i] = leaf_idx

            # Importance sampling weight
            prob = priority / total if total > 0 else 1.0
            prob = max(prob, 1e-10)  # avoid division by zero
            weight = (self._size * prob) ** (-beta)
            b_weights[i] = weight / max_weight

            # Extract sub-sequence around this transition
            self._extract_subsequence(
                leaf_idx, i,
                b_static, b_dynamic, b_actions, b_rewards,
                b_next_dynamic, b_dones, b_action_masks, b_next_action_masks,
            )

        return {
            "static_features": b_static,
            "dynamic_features": b_dynamic,
            "actions": b_actions,
            "rewards": b_rewards,
            "next_dynamic_features": b_next_dynamic,
            "dones": b_dones,
            "action_masks": b_action_masks,
            "next_action_masks": b_next_action_masks,
            "weights": b_weights,
            "indices": b_indices,
        }

    def update_priorities(
        self,
        indices: np.ndarray | list,
        td_errors: np.ndarray | list,
    ) -> None:
        """Update priorities based on new TD errors.

        Priority = (|TD error| + epsilon) ^ alpha
        """
        indices = np.asarray(indices)
        td_errors = np.asarray(td_errors, dtype=np.float64)

        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + self.epsilon) ** self.alpha
            self._tree.update(int(idx), priority)
            self._max_priority = max(
                self._max_priority, abs(td) + self.epsilon
            )

    def __len__(self) -> int:
        return self._size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_subsequence(
        self,
        leaf_idx: int,
        batch_idx: int,
        b_static: np.ndarray,
        b_dynamic: np.ndarray,
        b_actions: np.ndarray,
        b_rewards: np.ndarray,
        b_next_dynamic: np.ndarray,
        b_dones: np.ndarray,
        b_action_masks: np.ndarray,
        b_next_action_masks: np.ndarray,
    ) -> None:
        """Extract a sub-sequence of length seq_len containing leaf_idx.

        The sub-sequence is anchored so that the sampled transition is
        included. We find the start of the episode in the buffer, then
        pick a window of seq_len that contains the sampled position.

        If the episode is shorter than seq_len, we left-align the episode
        data and zero-pad the rest (with done=True for padded steps).
        """
        L = self.seq_len
        ep_id = self._episode_ids[leaf_idx]
        pos_in_ep = self._positions[leaf_idx]
        ep_len = self._episode_lengths[leaf_idx]

        # Find the buffer index of the episode's first transition.
        # The episode starts at leaf_idx - pos_in_ep, but we need to
        # handle circular buffer wrap-around.
        ep_start_buf = (leaf_idx - pos_in_ep) % self.capacity

        if ep_len <= L:
            # Episode fits entirely; copy all and pad
            actual_len = ep_len
            subseq_start_in_ep = 0
        else:
            # Pick a window of length L that includes pos_in_ep.
            # The window start must be in [max(0, pos_in_ep - L + 1), pos_in_ep]
            # and the window end must be <= ep_len
            earliest_start = max(0, pos_in_ep - L + 1)
            latest_start = min(pos_in_ep, ep_len - L)
            subseq_start_in_ep = np.random.randint(earliest_start, latest_start + 1)
            actual_len = L

        for j in range(actual_len):
            buf_idx = (ep_start_buf + subseq_start_in_ep + j) % self.capacity

            # Verify this transition belongs to the same episode
            if self._episode_ids[buf_idx] != ep_id:
                # Episode has been partially overwritten; treat rest as padding
                b_dones[batch_idx, j:] = 1.0
                break

            b_static[batch_idx, j] = self._static_features[buf_idx]
            b_dynamic[batch_idx, j] = self._dynamic_features[buf_idx]
            b_actions[batch_idx, j] = self._actions[buf_idx]
            b_rewards[batch_idx, j] = self._rewards[buf_idx]
            b_next_dynamic[batch_idx, j] = self._next_dynamic_features[buf_idx]
            b_dones[batch_idx, j] = self._dones[buf_idx]
            b_action_masks[batch_idx, j] = self._action_masks[buf_idx]
            b_next_action_masks[batch_idx, j] = self._next_action_masks[buf_idx]

        # Pad remaining positions if episode < seq_len
        if ep_len < L:
            b_dones[batch_idx, ep_len:] = 1.0
