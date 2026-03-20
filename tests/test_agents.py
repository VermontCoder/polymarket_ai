"""Tests for Random Agent and DQN Agent."""

import numpy as np
import pytest
import torch

from src.agents.random_agent import RandomAgent
from src.agents.dqn_agent import DQNAgent
from src.environment import Environment, compute_action_mask
from src.models.lstm_dqn import LSTMDQN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(up_bid=55.0, up_ask=56.0, down_bid=44.0, down_ask=45.0,
              diff_pct=0.01, time_to_close=150000):
    return {
        "timestamp": "2026-03-14T17:23:00Z",
        "up_bid": up_bid, "up_ask": up_ask,
        "down_bid": down_bid, "down_ask": down_ask,
        "current_price": 70000.0, "diff_pct": diff_pct,
        "diff_usd": 5.0, "time_to_close": time_to_close,
    }


def _make_episode(outcome="UP", num_rows=5, rows=None, **kwargs):
    if rows is None:
        rows = [_make_row(**kwargs) for _ in range(num_rows)]
    return {
        "session_id": "test-session", "outcome": outcome,
        "hour": 12, "day": 2,
        "start_price": 70000.0, "end_price": 70100.0,
        "diff_pct_prev_session": 0.05, "diff_pct_hour": 0.02,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Tests: Random Agent
# ---------------------------------------------------------------------------

class TestRandomAgent:
    """Random agent selects from unmasked actions only."""

    def test_selects_from_valid_actions_only(self):
        """Random agent only picks actions where mask is True."""
        agent = RandomAgent()
        # Mask: only actions 0 and 3 allowed
        mask = np.array([True, False, False, True, False, False, False, False, False])
        actions_seen = set()
        for _ in range(200):
            action = agent.select_action(mask)
            actions_seen.add(action)
            assert mask[action], f"Selected masked action {action}"
        # Should have selected both valid actions at some point
        assert actions_seen == {0, 3}

    def test_selects_only_action0_when_only_valid(self):
        """When only action 0 is valid, always selects 0."""
        agent = RandomAgent()
        mask = np.zeros(9, dtype=bool)
        mask[0] = True
        for _ in range(50):
            assert agent.select_action(mask) == 0

    def test_at_most_one_action_per_episode(self):
        """Random agent takes at most 1 non-zero action per episode."""
        agent = RandomAgent()
        env = Environment()
        ep = _make_episode(num_rows=10, outcome="UP")
        env.reset(ep)

        non_zero_actions = 0
        for _ in range(env.num_rows):
            mask = env.get_action_mask()
            action = agent.select_action(mask)
            if action != 0:
                non_zero_actions += 1
            done, _ = env.step(action)
            if done:
                break

        assert non_zero_actions <= 1

    def test_handles_all_actions_valid(self):
        """Random agent works when all actions are valid."""
        agent = RandomAgent()
        mask = np.ones(9, dtype=bool)
        action = agent.select_action(mask)
        assert 0 <= action <= 8

    def test_action_distribution(self):
        """Action 0 is selected ~95% of the time, others share ~5%."""
        agent = RandomAgent()
        mask = np.ones(9, dtype=bool)
        counts = np.zeros(9)
        n = 10000
        for _ in range(n):
            counts[agent.select_action(mask)] += 1
        # Action 0 should be ~95%
        assert counts[0] / n > 0.90
        assert counts[0] / n < 0.99
        # Non-zero actions should each get roughly 5%/8 ≈ 0.625%
        for c in counts[1:]:
            assert c > 0  # each action selected at least once in 10k trials


# ---------------------------------------------------------------------------
# Tests: DQN Agent
# ---------------------------------------------------------------------------

class TestDQNAgent:
    """DQN agent produces valid action selections."""

    @pytest.fixture
    def model_and_agent(self):
        model = LSTMDQN()
        device = torch.device("cpu")
        agent = DQNAgent(model=model, device=device)
        return model, agent

    def test_selects_valid_action(self, model_and_agent):
        """DQN agent selects an action within valid range."""
        _, agent = model_and_agent
        agent.reset()

        static = np.random.randn(37).astype(np.float32)
        dynamic = np.random.randn(12).astype(np.float32)
        mask = np.ones(9, dtype=bool)

        action = agent.select_action(static, dynamic, mask)
        assert 0 <= action <= 8

    def test_respects_action_mask(self, model_and_agent):
        """DQN agent never selects a masked action."""
        _, agent = model_and_agent
        agent.reset()

        static = np.random.randn(37).astype(np.float32)
        dynamic = np.random.randn(12).astype(np.float32)
        # Only allow actions 0 and 1
        mask = np.array([True, True, False, False, False, False, False, False, False])

        for _ in range(50):
            action = agent.select_action(static, dynamic, mask)
            assert mask[action], f"DQN selected masked action {action}"

    def test_deterministic_greedy(self, model_and_agent):
        """DQN agent is deterministic (greedy, no epsilon)."""
        _, agent = model_and_agent
        agent.reset()

        static = np.random.randn(37).astype(np.float32)
        dynamic = np.random.randn(12).astype(np.float32)
        mask = np.ones(9, dtype=bool)

        # Same input should give same output
        actions = set()
        for _ in range(10):
            agent.reset()
            action = agent.select_action(static, dynamic, mask)
            actions.add(action)
        assert len(actions) == 1

    def test_hidden_state_maintained_across_steps(self, model_and_agent):
        """Hidden state changes across timesteps within an episode."""
        _, agent = model_and_agent
        agent.reset()

        static = np.random.randn(37).astype(np.float32)
        mask = np.ones(9, dtype=bool)

        # Step through multiple timesteps with different dynamic features
        for _ in range(5):
            dynamic = np.random.randn(12).astype(np.float32)
            agent.select_action(static, dynamic, mask)

        # Hidden state should have been updated (not all zeros)
        h, c = agent._hidden_state
        assert not torch.allclose(h, torch.zeros_like(h))

    def test_reset_clears_hidden_state(self, model_and_agent):
        """Resetting the agent zeros the hidden state."""
        _, agent = model_and_agent
        agent.reset()

        static = np.random.randn(37).astype(np.float32)
        dynamic = np.random.randn(12).astype(np.float32)
        mask = np.ones(9, dtype=bool)

        agent.select_action(static, dynamic, mask)
        agent.reset()

        h, c = agent._hidden_state
        assert torch.allclose(h, torch.zeros_like(h))
        assert torch.allclose(c, torch.zeros_like(c))

    def test_only_action0_when_mask_forces(self, model_and_agent):
        """When only action 0 is valid, DQN must select 0."""
        _, agent = model_and_agent
        agent.reset()

        static = np.random.randn(37).astype(np.float32)
        dynamic = np.random.randn(12).astype(np.float32)
        mask = np.zeros(9, dtype=bool)
        mask[0] = True

        action = agent.select_action(static, dynamic, mask)
        assert action == 0
