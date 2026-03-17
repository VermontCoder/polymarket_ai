"""Tests for Prioritized Experience Replay buffer with DRQN sub-sequence sampling.

Test coverage:
  - SumTree: add, update, sampling proportional to priorities, total tracking
  - PER correctly prioritizes high-TD-error transitions
  - Sub-sequence sampling respects episode boundaries
  - Post-action timesteps are excluded (only pre-action stored)
  - Zero-padding for short episodes
  - Priority updates and importance-sampling weights
  - Circular buffer overflow / capacity limits
  - Edge cases: single transition episodes, exact seq_len episodes
"""

import numpy as np
import pytest

from src.replay_buffer import PrioritizedReplayBuffer, SumTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transition(
    action: int = 0,
    reward: float = 0.0,
    done: bool = False,
    static_val: float = 0.0,
    dynamic_val: float = 0.0,
) -> dict:
    """Create a minimal valid transition dict for testing."""
    static = np.full(35, static_val, dtype=np.float32)
    dynamic = np.full(11, dynamic_val, dtype=np.float32)
    next_dynamic = None if done else np.full(11, dynamic_val + 0.1, dtype=np.float32)
    action_mask = np.ones(9, dtype=bool)
    next_action_mask = None if done else np.ones(9, dtype=bool)

    return {
        "static_features": static,
        "dynamic_features": dynamic,
        "action": action,
        "reward": reward,
        "next_dynamic_features": next_dynamic,
        "done": done,
        "action_mask": action_mask,
        "next_action_mask": next_action_mask,
    }


def _make_episode(
    length: int,
    static_val: float = 1.0,
    reward_at_end: float = 0.5,
) -> list[dict]:
    """Create a list of transitions forming one episode.

    Only the last transition has done=True and carries the reward.
    This mimics "pre-action only" storage.
    """
    transitions = []
    for i in range(length):
        is_last = i == length - 1
        t = _make_transition(
            action=0 if i < length - 1 else 1,
            reward=reward_at_end if is_last else 0.0,
            done=is_last,
            static_val=static_val,
            dynamic_val=float(i),
        )
        transitions.append(t)
    return transitions


# ===========================================================================
# SumTree Tests
# ===========================================================================

class TestSumTree:
    """Tests for the SumTree data structure."""

    def test_empty_tree(self):
        tree = SumTree(capacity=8)
        assert tree.total == 0.0
        assert tree.size == 0

    def test_add_single(self):
        tree = SumTree(capacity=8)
        idx = tree.add(5.0)
        assert idx == 0
        assert tree.total == pytest.approx(5.0)
        assert tree.size == 1

    def test_add_multiple(self):
        tree = SumTree(capacity=8)
        tree.add(1.0)
        tree.add(2.0)
        tree.add(3.0)
        assert tree.total == pytest.approx(6.0)
        assert tree.size == 3

    def test_update(self):
        tree = SumTree(capacity=8)
        tree.add(1.0)
        tree.add(2.0)
        tree.update(0, 10.0)
        assert tree.total == pytest.approx(12.0)

    def test_get_sampling(self):
        """Sampling should return indices proportional to their priorities."""
        tree = SumTree(capacity=4)
        tree.add(1.0)  # idx 0
        tree.add(3.0)  # idx 1

        # cumsum in [0, 1] should return idx 0
        idx, pri = tree.get(0.5)
        assert idx == 0
        assert pri == pytest.approx(1.0)

        # cumsum in (1, 4] should return idx 1
        idx, pri = tree.get(2.0)
        assert idx == 1
        assert pri == pytest.approx(3.0)

    def test_wrap_around(self):
        """When capacity is exceeded, old entries are overwritten."""
        tree = SumTree(capacity=2)
        tree.add(1.0)
        tree.add(2.0)
        tree.add(10.0)  # overwrites idx 0
        assert tree.size == 2
        assert tree.total == pytest.approx(12.0)

    def test_min_priority(self):
        tree = SumTree(capacity=4)
        tree.add(5.0)
        tree.add(2.0)
        tree.add(8.0)
        assert tree.min_priority() == pytest.approx(2.0)


# ===========================================================================
# PrioritizedReplayBuffer Tests
# ===========================================================================

class TestPrioritizedReplayBuffer:
    """Tests for the full replay buffer."""

    def test_empty_buffer(self):
        buf = PrioritizedReplayBuffer(capacity=100, seq_len=4)
        assert len(buf) == 0

    def test_add_episode_basic(self):
        buf = PrioritizedReplayBuffer(capacity=100, seq_len=4)
        episode = _make_episode(length=5)
        buf.add_episode(episode)
        assert len(buf) == 5

    def test_add_multiple_episodes(self):
        buf = PrioritizedReplayBuffer(capacity=100, seq_len=4)
        buf.add_episode(_make_episode(10, static_val=1.0))
        buf.add_episode(_make_episode(8, static_val=2.0))
        assert len(buf) == 18

    def test_add_empty_episode(self):
        buf = PrioritizedReplayBuffer(capacity=100, seq_len=4)
        buf.add_episode([])
        assert len(buf) == 0

    def test_capacity_limit(self):
        """Buffer should not exceed capacity."""
        buf = PrioritizedReplayBuffer(capacity=10, seq_len=4)
        buf.add_episode(_make_episode(7))
        buf.add_episode(_make_episode(7))
        assert len(buf) == 10  # capped at capacity

    def test_sample_returns_correct_shapes(self):
        buf = PrioritizedReplayBuffer(capacity=200, seq_len=5)
        buf.add_episode(_make_episode(20))
        buf.add_episode(_make_episode(15))

        batch = buf.sample(batch_size=8, beta=0.4)

        assert batch["static_features"].shape == (8, 5, 35)
        assert batch["dynamic_features"].shape == (8, 5, 11)
        assert batch["actions"].shape == (8, 5)
        assert batch["rewards"].shape == (8, 5)
        assert batch["next_dynamic_features"].shape == (8, 5, 11)
        assert batch["dones"].shape == (8, 5)
        assert batch["action_masks"].shape == (8, 5, 9)
        assert batch["next_action_masks"].shape == (8, 5, 9)
        assert batch["weights"].shape == (8,)
        assert batch["indices"].shape == (8,)

    def test_sample_dtypes(self):
        buf = PrioritizedReplayBuffer(capacity=200, seq_len=5)
        buf.add_episode(_make_episode(20))

        batch = buf.sample(batch_size=4, beta=0.4)

        assert batch["static_features"].dtype == np.float32
        assert batch["dynamic_features"].dtype == np.float32
        assert batch["actions"].dtype == np.int64
        assert batch["rewards"].dtype == np.float32
        assert batch["dones"].dtype == np.float32
        assert batch["weights"].dtype == np.float32
        assert batch["indices"].dtype == np.int64

    def test_sample_weights_positive(self):
        """All importance-sampling weights should be positive."""
        buf = PrioritizedReplayBuffer(capacity=200, seq_len=5)
        buf.add_episode(_make_episode(30))

        batch = buf.sample(batch_size=8, beta=0.4)
        assert np.all(batch["weights"] > 0)

    def test_sample_weights_max_one_with_beta_one(self):
        """With beta=1.0, max weight should be exactly 1.0 (fully corrected)."""
        buf = PrioritizedReplayBuffer(capacity=200, seq_len=5)
        buf.add_episode(_make_episode(30))

        batch = buf.sample(batch_size=8, beta=1.0)
        assert np.all(batch["weights"] <= 1.0 + 1e-6)
        assert np.max(batch["weights"]) >= 1.0 - 1e-6

    # ------------------------------------------------------------------
    # Episode boundary tests
    # ------------------------------------------------------------------

    def test_subsequence_respects_episode_boundaries(self):
        """Sub-sequences must not mix data from different episodes."""
        buf = PrioritizedReplayBuffer(capacity=200, seq_len=5)

        # Two episodes with very different static features
        ep1 = _make_episode(10, static_val=100.0)
        ep2 = _make_episode(10, static_val=200.0)
        buf.add_episode(ep1)
        buf.add_episode(ep2)

        np.random.seed(42)
        for _ in range(50):
            batch = buf.sample(batch_size=16, beta=0.4)
            # Within each sub-sequence, all static features should be identical
            # (same episode)
            for b in range(16):
                # Find the first non-zero static to identify episode
                # (padded positions will be 0; find first real one)
                for t in range(5):
                    if not np.allclose(batch["static_features"][b, t], 0.0):
                        ref_static = batch["static_features"][b, t, 0]
                        break
                else:
                    continue  # all zeros (fully padded), skip

                for t in range(5):
                    sf = batch["static_features"][b, t]
                    if np.allclose(sf, 0.0):
                        # This is a padded position (episode shorter than seq_len)
                        # or a position after done; that's OK
                        continue
                    assert sf[0] == pytest.approx(
                        ref_static, abs=1e-5
                    ), (
                        f"Sub-sequence at batch {b} mixes episodes: "
                        f"expected static_val={ref_static}, got {sf[0]}"
                    )

    def test_short_episode_zero_padded(self):
        """Episodes shorter than seq_len should be zero-padded with done=True."""
        buf = PrioritizedReplayBuffer(capacity=100, seq_len=10)

        # Episode of length 3
        ep = _make_episode(3, static_val=5.0)
        buf.add_episode(ep)

        batch = buf.sample(batch_size=1, beta=0.4)

        # Positions 3..9 should be padded (done=1)
        for t in range(3, 10):
            assert batch["dones"][0, t] == 1.0, (
                f"Padded position {t} should have done=1.0"
            )

    def test_exact_seq_len_episode(self):
        """Episode with exactly seq_len transitions should work without padding."""
        buf = PrioritizedReplayBuffer(capacity=100, seq_len=5)

        ep = _make_episode(5, static_val=7.0)
        buf.add_episode(ep)

        batch = buf.sample(batch_size=1, beta=0.4)

        # All positions should have valid static features
        for t in range(5):
            assert batch["static_features"][0, t, 0] == pytest.approx(7.0)

    def test_single_transition_episode(self):
        """An episode with only 1 transition should be padded to seq_len."""
        buf = PrioritizedReplayBuffer(capacity=100, seq_len=4)

        ep = [_make_transition(action=1, reward=0.3, done=True, static_val=3.0)]
        buf.add_episode(ep)

        batch = buf.sample(batch_size=1, beta=0.4)

        # First position has data, rest padded
        assert batch["static_features"][0, 0, 0] == pytest.approx(3.0)
        assert batch["dones"][0, 0] == 1.0  # the only transition is terminal
        for t in range(1, 4):
            assert batch["dones"][0, t] == 1.0

    # ------------------------------------------------------------------
    # PER priority tests
    # ------------------------------------------------------------------

    def test_high_td_error_sampled_more(self):
        """Transitions with high TD error should be sampled more frequently."""
        buf = PrioritizedReplayBuffer(capacity=200, seq_len=1, alpha=0.6)

        # Add two episodes with distinct static features
        ep_low = _make_episode(50, static_val=1.0)
        ep_high = _make_episode(50, static_val=2.0)
        buf.add_episode(ep_low)
        buf.add_episode(ep_high)

        # Give ep_high transitions very high priority
        indices_high = np.arange(50, 100)
        td_errors_high = np.full(50, 100.0)
        buf.update_priorities(indices_high, td_errors_high)

        # Give ep_low transitions very low priority
        indices_low = np.arange(0, 50)
        td_errors_low = np.full(50, 0.001)
        buf.update_priorities(indices_low, td_errors_low)

        # Sample many times and count
        np.random.seed(123)
        count_high = 0
        count_low = 0
        n_samples = 500
        for _ in range(n_samples):
            batch = buf.sample(batch_size=1, beta=0.4)
            sv = batch["static_features"][0, 0, 0]
            if sv == pytest.approx(2.0, abs=0.1):
                count_high += 1
            elif sv == pytest.approx(1.0, abs=0.1):
                count_low += 1

        # High-priority transitions should be sampled much more often
        assert count_high > count_low * 3, (
            f"High-priority count ({count_high}) should be >> "
            f"low-priority count ({count_low})"
        )

    def test_new_transitions_have_max_priority(self):
        """Newly added transitions should have max priority."""
        buf = PrioritizedReplayBuffer(capacity=100, seq_len=1, alpha=0.6)

        ep1 = _make_episode(10, static_val=1.0)
        buf.add_episode(ep1)

        # Update some priorities to be very high
        buf.update_priorities([0, 1, 2], [100.0, 100.0, 100.0])

        # Now add a new episode — its transitions should get max priority
        ep2 = _make_episode(10, static_val=2.0)
        buf.add_episode(ep2)

        # The new transitions should be competitive with updated ones
        # Since max_priority was updated to 100 + epsilon, new transitions
        # get (100 + eps)^alpha which is comparable
        np.random.seed(42)
        count_new = 0
        for _ in range(200):
            batch = buf.sample(batch_size=1, beta=0.4)
            sv = batch["static_features"][0, 0, 0]
            if sv == pytest.approx(2.0, abs=0.1):
                count_new += 1

        # New transitions (10 out of 20) should be sampled roughly as often
        # as the boosted ones. At minimum, they should appear.
        assert count_new > 10, (
            f"New transitions sampled only {count_new}/200 times; "
            "expected more due to max priority initialization"
        )

    def test_update_priorities(self):
        """update_priorities should change sampling distribution."""
        buf = PrioritizedReplayBuffer(capacity=100, seq_len=1, alpha=0.6)
        ep = _make_episode(10, static_val=1.0)
        buf.add_episode(ep)

        # All start with same max priority; update index 0 to be much higher
        buf.update_priorities([0], [1000.0])

        np.random.seed(7)
        count_idx0 = 0
        for _ in range(300):
            batch = buf.sample(batch_size=1, beta=0.4)
            if batch["indices"][0] == 0:
                count_idx0 += 1

        # Index 0 should appear disproportionately often
        # Expected ~1/10 = 30 without boosting; with boosting >> 30
        assert count_idx0 > 60, (
            f"Boosted index 0 sampled only {count_idx0}/300 times"
        )

    def test_importance_sampling_weights_with_uniform_priorities(self):
        """With uniform priorities, all IS weights should be equal (= 1.0)."""
        buf = PrioritizedReplayBuffer(capacity=100, seq_len=1, alpha=0.6)
        ep = _make_episode(20, static_val=1.0)
        buf.add_episode(ep)

        batch = buf.sample(batch_size=8, beta=0.4)
        # All priorities are the same (max_priority), so all weights should be 1.0
        assert np.allclose(batch["weights"], 1.0, atol=1e-5), (
            f"Expected all weights = 1.0, got {batch['weights']}"
        )

    # ------------------------------------------------------------------
    # Pre-action only storage test
    # ------------------------------------------------------------------

    def test_only_pre_action_stored(self):
        """Verify that only pre-action transitions are stored.

        The buffer stores whatever transitions are passed in. The caller
        (environment/agent) is responsible for only passing pre-action
        timesteps. We verify the buffer faithfully stores and returns
        exactly what was added, with no extra transitions.
        """
        buf = PrioritizedReplayBuffer(capacity=100, seq_len=4)

        # Simulate an episode where agent acts at step 2 (action=1).
        # Pre-action timesteps: steps 0, 1, 2 (action at step 2).
        # Post-action steps (forced "do nothing") are NOT included.
        pre_action_transitions = [
            _make_transition(action=0, reward=0.0, done=False, dynamic_val=0.0),
            _make_transition(action=0, reward=0.0, done=False, dynamic_val=1.0),
            _make_transition(action=1, reward=0.5, done=True, dynamic_val=2.0),
        ]
        buf.add_episode(pre_action_transitions)

        assert len(buf) == 3  # only 3 pre-action steps stored

    # ------------------------------------------------------------------
    # Transition data integrity
    # ------------------------------------------------------------------

    def test_stored_data_matches_input(self):
        """Verify that sampled data matches what was added."""
        buf = PrioritizedReplayBuffer(capacity=100, seq_len=3)

        # Create an episode with known values
        transitions = []
        for i in range(3):
            t = _make_transition(
                action=i,
                reward=float(i) * 0.1,
                done=(i == 2),
                static_val=42.0,
                dynamic_val=float(i) * 10.0,
            )
            transitions.append(t)

        buf.add_episode(transitions)

        # Sample until we get the full episode
        np.random.seed(0)
        batch = buf.sample(batch_size=1, beta=0.4)

        # Static features should all be 42.0
        for t in range(3):
            assert batch["static_features"][0, t, 0] == pytest.approx(42.0)

        # Actions should be 0, 1, 2
        assert list(batch["actions"][0]) == [0, 1, 2]

        # Rewards should be 0.0, 0.1, 0.2
        assert batch["rewards"][0, 0] == pytest.approx(0.0)
        assert batch["rewards"][0, 1] == pytest.approx(0.1)
        assert batch["rewards"][0, 2] == pytest.approx(0.2)

        # Done flags
        assert batch["dones"][0, 0] == 0.0
        assert batch["dones"][0, 1] == 0.0
        assert batch["dones"][0, 2] == 1.0

    def test_terminal_transition_none_fields(self):
        """Terminal transitions with None next fields should be handled."""
        buf = PrioritizedReplayBuffer(capacity=100, seq_len=1)

        t = _make_transition(action=1, reward=0.5, done=True)
        assert t["next_dynamic_features"] is None
        assert t["next_action_mask"] is None

        buf.add_episode([t])
        batch = buf.sample(batch_size=1, beta=0.4)

        # next_dynamic_features should be zeros for terminal
        assert np.allclose(batch["next_dynamic_features"][0, 0], 0.0)
        # next_action_masks should be zeros for terminal
        assert np.allclose(batch["next_action_masks"][0, 0], 0.0)

    # ------------------------------------------------------------------
    # Circular buffer wrap-around
    # ------------------------------------------------------------------

    def test_circular_overwrite(self):
        """Old data should be overwritten when capacity is exceeded."""
        buf = PrioritizedReplayBuffer(capacity=10, seq_len=2)

        # Fill buffer completely with episode A (static=1.0)
        ep_a = _make_episode(10, static_val=1.0)
        buf.add_episode(ep_a)
        assert len(buf) == 10

        # Add episode B (static=2.0) which overwrites first 5 positions
        ep_b = _make_episode(5, static_val=2.0)
        buf.add_episode(ep_b)
        assert len(buf) == 10  # still capped

        # Sampling should be able to find episode B data
        np.random.seed(42)
        found_b = False
        for _ in range(100):
            batch = buf.sample(batch_size=1, beta=0.4)
            if batch["static_features"][0, 0, 0] == pytest.approx(2.0, abs=0.1):
                found_b = True
                break
        assert found_b, "Episode B data should be present after overwrite"

    # ------------------------------------------------------------------
    # Longer episode with sub-sequence windowing
    # ------------------------------------------------------------------

    def test_long_episode_subsequence_window(self):
        """For episodes longer than seq_len, sub-sequences should be valid windows."""
        buf = PrioritizedReplayBuffer(capacity=200, seq_len=5)

        # Episode of length 50 with distinct dynamic values
        transitions = []
        for i in range(50):
            t = _make_transition(
                action=0 if i < 49 else 1,
                reward=0.0 if i < 49 else 1.0,
                done=(i == 49),
                static_val=10.0,
                dynamic_val=float(i),
            )
            transitions.append(t)
        buf.add_episode(transitions)

        np.random.seed(7)
        batch = buf.sample(batch_size=1, beta=0.4)

        # The dynamic values should be 5 consecutive integers
        dyn_vals = batch["dynamic_features"][0, :, 0]
        diffs = np.diff(dyn_vals)
        # All consecutive differences should be 1.0 (contiguous)
        assert np.allclose(diffs, 1.0), (
            f"Dynamic values should be consecutive: {dyn_vals}"
        )

    # ------------------------------------------------------------------
    # Multiple samples consistency
    # ------------------------------------------------------------------

    def test_many_samples_no_crash(self):
        """Buffer should handle many samples without errors."""
        buf = PrioritizedReplayBuffer(capacity=500, seq_len=8)

        for i in range(20):
            ep = _make_episode(
                length=np.random.randint(3, 30),
                static_val=float(i),
            )
            buf.add_episode(ep)

        # Sample many batches
        for _ in range(100):
            batch = buf.sample(batch_size=16, beta=np.random.uniform(0.4, 1.0))
            assert batch["static_features"].shape == (16, 8, 35)

    # ------------------------------------------------------------------
    # Edge case: sample from buffer with fewer transitions than batch_size
    # ------------------------------------------------------------------

    def test_sample_when_few_transitions(self):
        """Sampling should work even when buffer has fewer items than batch_size."""
        buf = PrioritizedReplayBuffer(capacity=100, seq_len=3)
        ep = _make_episode(2, static_val=5.0)
        buf.add_episode(ep)

        # Only 2 transitions but requesting batch of 8
        batch = buf.sample(batch_size=8, beta=0.4)
        assert batch["static_features"].shape == (8, 3, 35)


# ===========================================================================
# Integration-style tests
# ===========================================================================

class TestReplayBufferIntegration:
    """Higher-level integration tests combining add, sample, and update."""

    def test_full_workflow(self):
        """Test the complete add -> sample -> update_priorities workflow."""
        buf = PrioritizedReplayBuffer(capacity=1000, seq_len=10, alpha=0.6)

        # Add several episodes
        for i in range(10):
            ep = _make_episode(
                length=np.random.randint(5, 25),
                static_val=float(i),
                reward_at_end=float(i) * 0.1,
            )
            buf.add_episode(ep)

        # Sample a batch
        batch = buf.sample(batch_size=16, beta=0.5)

        # Simulate TD errors and update priorities
        td_errors = np.random.uniform(0.01, 5.0, size=16)
        buf.update_priorities(batch["indices"], td_errors)

        # Sample again (should not crash, priorities updated)
        batch2 = buf.sample(batch_size=16, beta=0.7)
        assert batch2["weights"].shape == (16,)

    def test_beta_annealing_effect(self):
        """Higher beta produces larger IS correction, spreading weights more.

        With beta=0, all weights are 1.0 (no correction).
        With beta=1, weights fully correct for non-uniform sampling, so
        low-priority items get boosted and high-priority items get dampened,
        resulting in higher weight variance when priorities are non-uniform.

        The key property: at beta=1.0, max(weight) should be 1.0 (normalized).
        At beta close to 0, weights collapse toward 1.0 (less correction).
        """
        buf = PrioritizedReplayBuffer(capacity=200, seq_len=1, alpha=0.6)

        ep = _make_episode(50, static_val=1.0)
        buf.add_episode(ep)

        # Create non-uniform priorities
        for i in range(50):
            buf.update_priorities([i], [float(i + 1)])

        np.random.seed(42)
        batch_low_beta = buf.sample(batch_size=32, beta=0.1)
        batch_high_beta = buf.sample(batch_size=32, beta=1.0)

        # With low beta, weights should be closer to 1.0 (less correction)
        # With high beta, weights spread more (stronger IS correction)
        var_low = np.var(batch_low_beta["weights"])
        var_high = np.var(batch_high_beta["weights"])

        assert var_high >= var_low - 1e-6, (
            f"Higher beta should produce more spread weights (stronger correction). "
            f"var(beta=0.1)={var_low:.4f}, var(beta=1.0)={var_high:.4f}"
        )

    def test_priority_epsilon_prevents_zero(self):
        """Even with TD error = 0, priority should be > 0 due to epsilon."""
        buf = PrioritizedReplayBuffer(
            capacity=100, seq_len=1, alpha=0.6, epsilon=1e-6
        )
        ep = _make_episode(5, static_val=1.0)
        buf.add_episode(ep)

        # Set TD error to exactly 0
        buf.update_priorities([0], [0.0])

        # Priority should be epsilon^alpha > 0
        # Sampling should still work (no zero-division)
        batch = buf.sample(batch_size=1, beta=0.4)
        assert batch["weights"][0] > 0
