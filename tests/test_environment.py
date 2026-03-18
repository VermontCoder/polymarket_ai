"""Tests for the Environment episode simulation, actions, rewards, and fees."""

import numpy as np
import pytest

from src.environment import (
    Environment,
    compute_action_mask,
    compute_reward,
    maker_rebate,
    taker_fee,
    NUM_ACTIONS,
)


# ---------------------------------------------------------------------------
# Helpers: episode and row factories
# ---------------------------------------------------------------------------

def _make_row(
    up_bid=55.0,
    up_ask=56.0,
    down_bid=44.0,
    down_ask=45.0,
    diff_pct=0.01,
    time_to_close=150000,
):
    """Create a minimal row dict for testing."""
    return {
        "timestamp": "2026-03-14T17:23:00Z",
        "up_bid": up_bid,
        "up_ask": up_ask,
        "down_bid": down_bid,
        "down_ask": down_ask,
        "current_price": 70000.0,
        "diff_pct": diff_pct,
        "diff_usd": 5.0,
        "time_to_close": time_to_close,
    }


def _make_episode(outcome="UP", num_rows=5, rows=None, **kwargs):
    """Create a minimal episode dict for testing.

    Args:
        outcome: Episode outcome, "UP" or "DOWN".
        num_rows: Number of rows if rows is not provided.
        rows: Explicit list of row dicts.
        **kwargs: Override row fields for all auto-generated rows.
    """
    if rows is None:
        rows = [_make_row(**kwargs) for _ in range(num_rows)]
    return {
        "session_id": "test-session",
        "outcome": outcome,
        "hour": 12,
        "day": 2,
        "start_price": 70000.0,
        "end_price": 70100.0,
        "diff_pct_prev_session": 0.05,
        "diff_pct_hour": 0.02,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Tests: Taker Fee
# ---------------------------------------------------------------------------

class TestTakerFee:
    """Taker fee matches formula at various prices."""

    def test_fee_at_1c(self):
        # fee = 0.02 * 1 * (1 - 1/100) = 0.02 * 0.99 = 0.0198
        assert taker_fee(1) == pytest.approx(0.0198)

    def test_fee_at_25c(self):
        # fee = 0.02 * 25 * (1 - 25/100) = 0.5 * 0.75 = 0.375
        assert taker_fee(25) == pytest.approx(0.375)

    def test_fee_at_50c(self):
        # fee = 0.02 * 50 * (1 - 50/100) = 1.0 * 0.5 = 0.5
        assert taker_fee(50) == pytest.approx(0.5)

    def test_fee_at_75c(self):
        # fee = 0.02 * 75 * (1 - 75/100) = 1.5 * 0.25 = 0.375
        assert taker_fee(75) == pytest.approx(0.375)

    def test_fee_at_99c(self):
        # fee = 0.02 * 99 * (1 - 99/100) = 1.98 * 0.01 = 0.0198
        assert taker_fee(99) == pytest.approx(0.0198)

    def test_fee_peak_at_50c(self):
        """Peak fee is at 50c."""
        assert taker_fee(50) >= taker_fee(49)
        assert taker_fee(50) >= taker_fee(51)

    def test_fee_symmetry(self):
        """Fee is symmetric around 50c: fee(x) == fee(100-x)."""
        for p in [1, 10, 25, 40, 49]:
            assert taker_fee(p) == pytest.approx(taker_fee(100 - p))

    def test_fee_rounding(self):
        """Fee is rounded to 4 decimal places."""
        fee = taker_fee(33)
        # 0.02 * 33 * (1 - 33/100) = 0.66 * 0.67 = 0.4422
        assert fee == pytest.approx(0.4422)
        # Ensure exactly 4 decimal places
        assert fee == round(fee, 4)

    def test_fee_minimum_enforced(self):
        """Fee at price 0 should still return minimum 0.0001."""
        # At price 0: fee = 0.25 * 0 * (1 - 0) = 0, clipped to 0.0001
        assert taker_fee(0) == 0.0001

    def test_fee_at_100c(self):
        """At price 100: fee = 0.25 * 100 * (1 - 1) = 0, clipped to 0.0001."""
        assert taker_fee(100) == 0.0001


# ---------------------------------------------------------------------------
# Tests: Maker Rebate
# ---------------------------------------------------------------------------

class TestMakerRebate:
    """Maker rebate = 20% of the taker fee."""

    def test_rebate_at_50c(self):
        expected = round(0.2 * taker_fee(50), 4)
        assert maker_rebate(50) == pytest.approx(expected)

    def test_rebate_at_25c(self):
        expected = round(0.2 * taker_fee(25), 4)
        assert maker_rebate(25) == pytest.approx(expected)

    def test_rebate_at_75c(self):
        expected = round(0.2 * taker_fee(75), 4)
        assert maker_rebate(75) == pytest.approx(expected)

    def test_rebate_is_20pct_of_taker(self):
        """Maker rebate is exactly 20% of the taker fee at any price."""
        for price in [1, 10, 25, 33, 50, 67, 75, 90, 99]:
            tf = taker_fee(price)
            mr = maker_rebate(price)
            assert mr == pytest.approx(round(0.2 * tf, 4))


# ---------------------------------------------------------------------------
# Tests: Action Mask
# ---------------------------------------------------------------------------

class TestActionMask:
    """Action masking correctly blocks invalid actions."""

    def test_all_valid_normal_row(self):
        """All actions allowed when all bid/ask present and not acted."""
        row = _make_row()
        mask = compute_action_mask(row, has_acted=False)
        assert mask.shape == (9,)
        assert mask.all()  # All True

    def test_action0_always_allowed(self):
        """Action 0 (do nothing) is always allowed, even after acting."""
        row = _make_row()
        mask = compute_action_mask(row, has_acted=True)
        assert mask[0] is np.True_

    def test_has_acted_blocks_all_except_0(self):
        """After acting, only action 0 is allowed."""
        row = _make_row()
        mask = compute_action_mask(row, has_acted=True)
        assert mask[0] is np.True_
        for i in range(1, 9):
            assert mask[i] is np.False_

    def test_null_up_ask_masks_1_and_5(self):
        """Null up_ask blocks actions 1 (buy UP taker) and 5 (limit buy UP)."""
        row = _make_row(up_ask=None)
        mask = compute_action_mask(row, has_acted=False)
        assert mask[1] is np.False_
        assert mask[5] is np.False_
        # Others should still be available
        assert mask[0] is np.True_
        assert mask[2] is np.True_
        assert mask[3] is np.True_
        assert mask[4] is np.True_

    def test_null_up_bid_masks_2_and_6(self):
        """Null up_bid blocks actions 2 (sell UP taker) and 6 (limit sell UP)."""
        row = _make_row(up_bid=None)
        mask = compute_action_mask(row, has_acted=False)
        assert mask[2] is np.False_
        assert mask[6] is np.False_
        assert mask[0] is np.True_
        assert mask[1] is np.True_

    def test_null_down_ask_masks_3_and_7(self):
        """Null down_ask blocks actions 3 and 7."""
        row = _make_row(down_ask=None)
        mask = compute_action_mask(row, has_acted=False)
        assert mask[3] is np.False_
        assert mask[7] is np.False_
        assert mask[0] is np.True_

    def test_null_down_bid_masks_4_and_8(self):
        """Null down_bid blocks actions 4 and 8."""
        row = _make_row(down_bid=None)
        mask = compute_action_mask(row, has_acted=False)
        assert mask[4] is np.False_
        assert mask[8] is np.False_

    def test_all_null_only_action0(self):
        """When all bid/ask are null, only action 0 is allowed."""
        row = _make_row(up_bid=None, up_ask=None, down_bid=None, down_ask=None)
        mask = compute_action_mask(row, has_acted=False)
        assert mask[0] is np.True_
        for i in range(1, 9):
            assert mask[i] is np.False_

    def test_limit_buy_up_boundary_low(self):
        """Limit buy UP masked when up_ask - 1 < 1 (i.e., up_ask <= 1)."""
        row = _make_row(up_ask=1.0)
        mask = compute_action_mask(row, has_acted=False)
        assert mask[5] is np.False_  # limit buy UP masked
        assert mask[1] is np.True_   # taker buy UP still OK

    def test_limit_sell_up_boundary_high(self):
        """Limit sell UP masked when up_bid + 1 > 99 (i.e., up_bid >= 99)."""
        row = _make_row(up_bid=99.0)
        mask = compute_action_mask(row, has_acted=False)
        assert mask[6] is np.False_  # limit sell UP masked
        assert mask[2] is np.True_   # taker sell UP still OK

    def test_limit_buy_down_boundary_low(self):
        """Limit buy DOWN masked when down_ask - 1 < 1."""
        row = _make_row(down_ask=1.0)
        mask = compute_action_mask(row, has_acted=False)
        assert mask[7] is np.False_  # limit buy DOWN masked
        assert mask[3] is np.True_   # taker buy DOWN still OK

    def test_limit_sell_down_boundary_high(self):
        """Limit sell DOWN masked when down_bid + 1 > 99."""
        row = _make_row(down_bid=99.0)
        mask = compute_action_mask(row, has_acted=False)
        assert mask[8] is np.False_  # limit sell DOWN masked
        assert mask[4] is np.True_   # taker sell DOWN still OK

    def test_limit_buy_up_boundary_ok(self):
        """Limit buy UP is allowed when up_ask - 1 >= 1 (e.g., up_ask = 2)."""
        row = _make_row(up_ask=2.0)
        mask = compute_action_mask(row, has_acted=False)
        assert mask[5] is np.True_

    def test_limit_sell_up_boundary_ok(self):
        """Limit sell UP is allowed when up_bid + 1 <= 99 (e.g., up_bid = 98)."""
        row = _make_row(up_bid=98.0)
        mask = compute_action_mask(row, has_acted=False)
        assert mask[6] is np.True_


# ---------------------------------------------------------------------------
# Tests: Reward Computation (all 8 trade/outcome combinations)
# ---------------------------------------------------------------------------

class TestRewardComputation:
    """All 8 trade/outcome combinations produce correct profit/loss."""

    def test_buy_up_outcome_up(self):
        """Buy UP at 60c, outcome UP -> payout 100c, profit."""
        price = 60.0
        fee = taker_fee(price)
        cost = price + fee
        payout = 100.0
        expected = (payout - cost) / 100.0
        reward = compute_reward(1, price, "UP", is_maker=False, filled=True)
        assert reward == pytest.approx(expected)

    def test_buy_up_outcome_down(self):
        """Buy UP at 60c, outcome DOWN -> payout 0c, loss."""
        price = 60.0
        fee = taker_fee(price)
        cost = price + fee
        payout = 0.0
        expected = (payout - cost) / 100.0
        reward = compute_reward(1, price, "DOWN", is_maker=False, filled=True)
        assert reward == pytest.approx(expected)

    def test_sell_up_outcome_up(self):
        """Sell UP at 60c, outcome UP -> owe 100c, loss."""
        price = 60.0
        fee = taker_fee(price)
        received = price - fee
        payout_owed = 100.0
        expected = (received - payout_owed) / 100.0
        reward = compute_reward(2, price, "UP", is_maker=False, filled=True)
        assert reward == pytest.approx(expected)

    def test_sell_up_outcome_down(self):
        """Sell UP at 60c, outcome DOWN -> owe 0c, profit."""
        price = 60.0
        fee = taker_fee(price)
        received = price - fee
        payout_owed = 0.0
        expected = (received - payout_owed) / 100.0
        reward = compute_reward(2, price, "DOWN", is_maker=False, filled=True)
        assert reward == pytest.approx(expected)

    def test_buy_down_outcome_down(self):
        """Buy DOWN at 40c, outcome DOWN -> payout 100c, profit."""
        price = 40.0
        fee = taker_fee(price)
        cost = price + fee
        payout = 100.0
        expected = (payout - cost) / 100.0
        reward = compute_reward(3, price, "DOWN", is_maker=False, filled=True)
        assert reward == pytest.approx(expected)

    def test_buy_down_outcome_up(self):
        """Buy DOWN at 40c, outcome UP -> payout 0c, loss."""
        price = 40.0
        fee = taker_fee(price)
        cost = price + fee
        payout = 0.0
        expected = (payout - cost) / 100.0
        reward = compute_reward(3, price, "UP", is_maker=False, filled=True)
        assert reward == pytest.approx(expected)

    def test_sell_down_outcome_down(self):
        """Sell DOWN at 40c, outcome DOWN -> owe 100c, loss."""
        price = 40.0
        fee = taker_fee(price)
        received = price - fee
        payout_owed = 100.0
        expected = (received - payout_owed) / 100.0
        reward = compute_reward(4, price, "DOWN", is_maker=False, filled=True)
        assert reward == pytest.approx(expected)

    def test_sell_down_outcome_up(self):
        """Sell DOWN at 40c, outcome UP -> owe 0c, profit."""
        price = 40.0
        fee = taker_fee(price)
        received = price - fee
        payout_owed = 0.0
        expected = (received - payout_owed) / 100.0
        reward = compute_reward(4, price, "UP", is_maker=False, filled=True)
        assert reward == pytest.approx(expected)

    def test_no_action_reward_zero(self):
        """No action (unfilled or action 0) produces reward = 0."""
        # unfilled limit order
        reward = compute_reward(5, 55.0, "UP", is_maker=True, filled=False)
        assert reward == 0.0

    def test_reward_normalized(self):
        """Reward is divided by 100 to scale to approximately [-1, 1]."""
        # Buy UP at 1c, outcome UP: payout=100, cost=1+fee
        # Reward should be close to +1 (max gain)
        reward = compute_reward(1, 1.0, "UP", is_maker=False, filled=True)
        assert -1.0 <= reward <= 1.0


# ---------------------------------------------------------------------------
# Tests: Maker Reward
# ---------------------------------------------------------------------------

class TestMakerReward:
    """Maker orders get rebate instead of paying fee."""

    def test_maker_buy_up_outcome_up(self):
        """Maker buy UP at 55c, outcome UP -> rebate reduces cost."""
        price = 55.0
        rebate = maker_rebate(price)
        cost = price - rebate
        payout = 100.0
        expected = (payout - cost) / 100.0
        reward = compute_reward(5, price, "UP", is_maker=True, filled=True)
        assert reward == pytest.approx(expected)

    def test_maker_sell_up_outcome_down(self):
        """Maker sell UP at 56c, outcome DOWN -> rebate adds to received."""
        price = 56.0
        rebate = maker_rebate(price)
        received = price + rebate
        payout_owed = 0.0
        expected = (received - payout_owed) / 100.0
        reward = compute_reward(6, price, "DOWN", is_maker=True, filled=True)
        assert reward == pytest.approx(expected)
        # Should be positive (profit from selling UP when outcome is DOWN)
        assert reward > 0

    def test_maker_buy_down_outcome_down(self):
        """Maker buy DOWN at 44c, outcome DOWN."""
        price = 44.0
        rebate = maker_rebate(price)
        cost = price - rebate
        payout = 100.0
        expected = (payout - cost) / 100.0
        reward = compute_reward(7, price, "DOWN", is_maker=True, filled=True)
        assert reward == pytest.approx(expected)

    def test_maker_sell_down_outcome_up(self):
        """Maker sell DOWN at 45c, outcome UP -> owe 0, profit."""
        price = 45.0
        rebate = maker_rebate(price)
        received = price + rebate
        payout_owed = 0.0
        expected = (received - payout_owed) / 100.0
        reward = compute_reward(8, price, "UP", is_maker=True, filled=True)
        assert reward == pytest.approx(expected)

    def test_unfilled_maker_reward_zero(self):
        """Unfilled maker orders yield reward = 0."""
        for action in [5, 6, 7, 8]:
            reward = compute_reward(action, 50.0, "UP", is_maker=True, filled=False)
            assert reward == 0.0


# ---------------------------------------------------------------------------
# Tests: Environment - Basic Episode Flow
# ---------------------------------------------------------------------------

class TestEnvironmentFlow:
    """Test the episode flow and basic environment behavior."""

    def test_reset_initializes_state(self):
        env = Environment()
        ep = _make_episode(num_rows=5)
        env.reset(ep)
        assert env.current_step == 0
        assert not env.has_acted
        assert env.num_rows == 5

    def test_full_episode_do_nothing(self):
        """Doing nothing every step gives reward 0."""
        env = Environment()
        ep = _make_episode(num_rows=5, outcome="UP")
        env.reset(ep)

        for _ in range(4):
            done, reward = env.step(0)
            assert not done

        done, reward = env.step(0)
        assert done
        assert reward == 0.0

    def test_at_most_one_action(self):
        """Agent can make at most one trade per episode."""
        env = Environment()
        ep = _make_episode(num_rows=5, outcome="UP")
        env.reset(ep)

        # Take action on first step
        env.step(1)  # Buy UP
        assert env.has_acted

        # All subsequent steps: only action 0 should be valid
        for _ in range(4):
            mask = env.get_action_mask()
            assert mask[0] is np.True_
            for i in range(1, 9):
                assert mask[i] is np.False_
            done, _ = env.step(0)
            if done:
                break

    def test_never_two_actions(self):
        """Trying to take two non-zero actions raises assertion."""
        env = Environment()
        ep = _make_episode(num_rows=5, outcome="UP")
        env.reset(ep)

        env.step(1)  # First action OK

        # Second non-zero action should be masked -> assertion error
        with pytest.raises(AssertionError):
            env.step(1)

    def test_observation_excludes_forbidden_fields(self):
        """Observation does not contain forbidden fields."""
        env = Environment()
        ep = _make_episode(num_rows=3)
        env.reset(ep)
        obs = env.get_observation()
        for field in ["outcome", "end_price", "current_price", "diff_usd",
                       "start_price", "session_id", "timestamp"]:
            assert field not in obs

    def test_observation_contains_allowed_fields(self):
        """Observation contains allowed dynamic fields."""
        env = Environment()
        ep = _make_episode(num_rows=3)
        env.reset(ep)
        obs = env.get_observation()
        for field in ["up_bid", "up_ask", "down_bid", "down_ask",
                       "diff_pct", "time_to_close"]:
            assert field in obs

    def test_step_advances_current_step(self):
        env = Environment()
        ep = _make_episode(num_rows=3)
        env.reset(ep)
        assert env.current_step == 0
        env.step(0)
        assert env.current_step == 1
        env.step(0)
        assert env.current_step == 2

    def test_done_on_last_step(self):
        env = Environment()
        ep = _make_episode(num_rows=3)
        env.reset(ep)
        done, _ = env.step(0)
        assert not done
        done, _ = env.step(0)
        assert not done
        done, _ = env.step(0)
        assert done

    def test_reset_clears_state(self):
        """Resetting with new episode clears all previous state."""
        env = Environment()
        ep1 = _make_episode(num_rows=3, outcome="UP")
        env.reset(ep1)
        env.step(1)  # Act
        assert env.has_acted

        ep2 = _make_episode(num_rows=5, outcome="DOWN")
        env.reset(ep2)
        assert not env.has_acted
        assert env.current_step == 0
        assert env.num_rows == 5


# ---------------------------------------------------------------------------
# Tests: Environment - Taker Actions with Rewards
# ---------------------------------------------------------------------------

class TestEnvironmentTakerRewards:
    """End-to-end reward computation for taker actions."""

    def test_buy_up_correct_outcome(self):
        """Buy UP at ask, outcome UP -> correct positive reward."""
        env = Environment()
        ep = _make_episode(num_rows=2, outcome="UP", up_ask=60.0)
        env.reset(ep)

        env.step(1)  # Buy UP at 60c
        done, reward = env.step(0)
        assert done

        fee = taker_fee(60.0)
        expected = (100.0 - (60.0 + fee)) / 100.0
        assert reward == pytest.approx(expected)

    def test_buy_up_wrong_outcome(self):
        """Buy UP at ask, outcome DOWN -> negative reward."""
        env = Environment()
        ep = _make_episode(num_rows=2, outcome="DOWN", up_ask=60.0)
        env.reset(ep)

        env.step(1)
        done, reward = env.step(0)
        assert done

        fee = taker_fee(60.0)
        expected = (0.0 - (60.0 + fee)) / 100.0
        assert reward == pytest.approx(expected)
        assert reward < 0

    def test_sell_up_outcome_down(self):
        """Sell UP at bid, outcome DOWN -> profit (owe nothing)."""
        env = Environment()
        ep = _make_episode(num_rows=2, outcome="DOWN", up_bid=55.0)
        env.reset(ep)

        env.step(2)  # Sell UP at 55c
        done, reward = env.step(0)
        assert done

        fee = taker_fee(55.0)
        expected = (55.0 - fee - 0.0) / 100.0
        assert reward == pytest.approx(expected)
        assert reward > 0

    def test_buy_down_correct_outcome(self):
        """Buy DOWN at ask, outcome DOWN -> profit."""
        env = Environment()
        ep = _make_episode(num_rows=2, outcome="DOWN", down_ask=40.0)
        env.reset(ep)

        env.step(3)
        done, reward = env.step(0)
        assert done

        fee = taker_fee(40.0)
        expected = (100.0 - (40.0 + fee)) / 100.0
        assert reward == pytest.approx(expected)
        assert reward > 0

    def test_sell_down_outcome_up(self):
        """Sell DOWN at bid, outcome UP -> profit (owe nothing)."""
        env = Environment()
        ep = _make_episode(num_rows=2, outcome="UP", down_bid=42.0)
        env.reset(ep)

        env.step(4)  # Sell DOWN at 42c
        done, reward = env.step(0)
        assert done

        fee = taker_fee(42.0)
        expected = (42.0 - fee - 0.0) / 100.0
        assert reward == pytest.approx(expected)
        assert reward > 0


# ---------------------------------------------------------------------------
# Tests: Limit Order Fill Logic
# ---------------------------------------------------------------------------

class TestLimitOrderFill:
    """Limit order fills when market price reaches order price."""

    def test_limit_buy_up_fills(self):
        """Limit buy UP fills when future ask <= order price."""
        # Place limit buy UP. Order price = up_ask - 1 = 56 - 1 = 55.
        # Next row has up_ask=55, so fill triggers.
        rows = [
            _make_row(up_ask=56.0),
            _make_row(up_ask=55.0),  # ask <= 55, fills
            _make_row(up_ask=54.0),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env = Environment()
        env.reset(ep)

        env.step(5)  # Limit buy UP at 55c
        env.step(0)
        done, reward = env.step(0)
        assert done

        # Filled at 55c with maker rebate
        rebate = maker_rebate(55.0)
        expected = (100.0 - (55.0 - rebate)) / 100.0
        assert reward == pytest.approx(expected)

    def test_limit_buy_up_does_not_fill(self):
        """Limit buy UP does NOT fill when no future ask <= order price."""
        rows = [
            _make_row(up_ask=56.0),
            _make_row(up_ask=57.0),  # ask > 55, no fill
            _make_row(up_ask=58.0),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env = Environment()
        env.reset(ep)

        env.step(5)  # Limit buy UP at 55c
        env.step(0)
        done, reward = env.step(0)
        assert done
        assert reward == 0.0

    def test_limit_sell_up_fills(self):
        """Limit sell UP fills when future bid >= order price."""
        # Place limit sell UP. Order price = up_bid + 1 = 55 + 1 = 56.
        # Next row has up_bid=56, so fill triggers.
        rows = [
            _make_row(up_bid=55.0, up_ask=56.0),
            _make_row(up_bid=56.0),  # bid >= 56, fills
            _make_row(up_bid=57.0),
        ]
        ep = _make_episode(outcome="DOWN", rows=rows)
        env = Environment()
        env.reset(ep)

        env.step(6)  # Limit sell UP at 56c
        env.step(0)
        done, reward = env.step(0)
        assert done

        # Sold UP at 56c, outcome DOWN -> owe 0, receive 56 + rebate
        rebate = maker_rebate(56.0)
        expected = (56.0 + rebate - 0.0) / 100.0
        assert reward == pytest.approx(expected)
        assert reward > 0

    def test_limit_sell_up_does_not_fill(self):
        """Limit sell UP does NOT fill when bid stays below order price."""
        rows = [
            _make_row(up_bid=55.0, up_ask=56.0),
            _make_row(up_bid=54.0),  # bid < 56, no fill
            _make_row(up_bid=53.0),
        ]
        ep = _make_episode(outcome="DOWN", rows=rows)
        env = Environment()
        env.reset(ep)

        env.step(6)  # Limit sell UP at 56c
        env.step(0)
        done, reward = env.step(0)
        assert done
        assert reward == 0.0

    def test_limit_buy_down_fills(self):
        """Limit buy DOWN fills when future down_ask <= order price."""
        rows = [
            _make_row(down_ask=45.0),
            _make_row(down_ask=44.0),  # ask <= 44, fills
            _make_row(down_ask=43.0),
        ]
        ep = _make_episode(outcome="DOWN", rows=rows)
        env = Environment()
        env.reset(ep)

        env.step(7)  # Limit buy DOWN at 44c
        env.step(0)
        done, reward = env.step(0)
        assert done

        rebate = maker_rebate(44.0)
        expected = (100.0 - (44.0 - rebate)) / 100.0
        assert reward == pytest.approx(expected)

    def test_limit_buy_down_does_not_fill(self):
        """Limit buy DOWN does NOT fill when ask stays above order price."""
        rows = [
            _make_row(down_ask=45.0),
            _make_row(down_ask=46.0),  # ask > 44, no fill
            _make_row(down_ask=47.0),
        ]
        ep = _make_episode(outcome="DOWN", rows=rows)
        env = Environment()
        env.reset(ep)

        env.step(7)
        env.step(0)
        done, reward = env.step(0)
        assert done
        assert reward == 0.0

    def test_limit_sell_down_fills(self):
        """Limit sell DOWN fills when future down_bid >= order price."""
        rows = [
            _make_row(down_bid=44.0, down_ask=45.0),
            _make_row(down_bid=45.0),  # bid >= 45, fills
            _make_row(down_bid=46.0),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env = Environment()
        env.reset(ep)

        env.step(8)  # Limit sell DOWN at 45c
        env.step(0)
        done, reward = env.step(0)
        assert done

        rebate = maker_rebate(45.0)
        expected = (45.0 + rebate - 0.0) / 100.0  # outcome UP, owe 0
        assert reward == pytest.approx(expected)

    def test_limit_sell_down_does_not_fill(self):
        """Limit sell DOWN does NOT fill when bid stays below order price."""
        rows = [
            _make_row(down_bid=44.0, down_ask=45.0),
            _make_row(down_bid=43.0),
            _make_row(down_bid=42.0),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env = Environment()
        env.reset(ep)

        env.step(8)
        env.step(0)
        done, reward = env.step(0)
        assert done
        assert reward == 0.0

    def test_limit_order_fills_at_order_price_not_market(self):
        """Limit order executes at order price, not market price at fill."""
        # Place limit buy UP at 55c. Market ask drops to 50c on fill row.
        # Should still be filled at 55c (the order price).
        rows = [
            _make_row(up_ask=56.0),
            _make_row(up_ask=50.0),  # fills, market at 50 but order is 55
            _make_row(up_ask=48.0),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env = Environment()
        env.reset(ep)

        env.step(5)
        env.step(0)
        done, reward = env.step(0)
        assert done

        # Must fill at 55c, not 50c
        rebate = maker_rebate(55.0)
        expected = (100.0 - (55.0 - rebate)) / 100.0
        assert reward == pytest.approx(expected)

    def test_limit_order_does_not_fill_on_action_row(self):
        """Limit order should not check fill on the same row it was placed."""
        # The ask is already <= order price on the placement row,
        # but fill should only check FUTURE rows.
        rows = [
            _make_row(up_ask=55.0),  # Order at 54c. ask=55 > 54. No fill here.
            _make_row(up_ask=56.0),  # ask=56 > 54. No fill.
            _make_row(up_ask=57.0),  # No fill.
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env = Environment()
        env.reset(ep)

        env.step(5)  # Limit buy UP at 54c
        env.step(0)
        done, reward = env.step(0)
        assert done
        assert reward == 0.0  # Never filled

    def test_limit_fill_with_null_market_price(self):
        """If the relevant bid/ask is null, the limit order cannot fill."""
        rows = [
            _make_row(up_ask=56.0),
            _make_row(up_ask=None),  # null, no fill possible
            _make_row(up_ask=None),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env = Environment()
        env.reset(ep)

        env.step(5)  # Limit buy UP at 55c
        env.step(0)
        done, reward = env.step(0)
        assert done
        assert reward == 0.0


# ---------------------------------------------------------------------------
# Tests: Trade Info
# ---------------------------------------------------------------------------

class TestTradeInfo:
    """Test trade_info property."""

    def test_no_trade_info_before_action(self):
        env = Environment()
        ep = _make_episode(num_rows=3)
        env.reset(ep)
        assert env.trade_info is None

    def test_taker_trade_info(self):
        env = Environment()
        ep = _make_episode(num_rows=3, up_ask=60.0)
        env.reset(ep)
        env.step(1)  # Buy UP at 60c

        info = env.trade_info
        assert info is not None
        assert info["action"] == 1
        assert info["price"] == 60.0
        assert info["is_maker"] is False
        assert info["filled"] is True

    def test_maker_trade_info_unfilled(self):
        """Maker order trade info reflects unfilled status."""
        rows = [
            _make_row(up_ask=56.0),
            _make_row(up_ask=57.0),
        ]
        env = Environment()
        ep = _make_episode(outcome="UP", rows=rows)
        env.reset(ep)
        env.step(5)  # Limit buy UP at 55c

        info = env.trade_info
        assert info["action"] == 5
        assert info["price"] == 55.0
        assert info["is_maker"] is True
        assert info["filled"] is False  # Not filled yet


# ---------------------------------------------------------------------------
# Tests: Episode Info
# ---------------------------------------------------------------------------

class TestEpisodeInfo:
    """Test get_episode_info method."""

    def test_returns_non_forbidden_fields(self):
        env = Environment()
        ep = _make_episode()
        env.reset(ep)
        info = env.get_episode_info()
        assert "hour" in info
        assert "day" in info
        assert "diff_pct_prev_session" in info
        assert "diff_pct_hour" in info
        # Forbidden fields excluded
        assert "outcome" not in info
        assert "end_price" not in info
        assert "start_price" not in info
        assert "session_id" not in info
        assert "rows" not in info


# ---------------------------------------------------------------------------
# Tests: Integration with real data
# ---------------------------------------------------------------------------

class TestIntegrationWithRealData:
    """Smoke tests using the actual data file."""

    DATA_PATH = "/workspace/data/btc_polymarket_combined_20260316_230302.json"

    @pytest.fixture
    def real_episodes(self):
        import json
        try:
            with open(self.DATA_PATH) as f:
                return json.load(f)
        except FileNotFoundError:
            pytest.skip("Data file not available")

    def test_do_nothing_all_episodes(self, real_episodes):
        """Doing nothing for all episodes yields reward 0."""
        env = Environment()
        for ep in real_episodes[:20]:
            env.reset(ep)
            for _ in range(len(ep["rows"])):
                done, reward = env.step(0)
                if done:
                    assert reward == 0.0
                    break

    def test_taker_buy_up_first_row(self, real_episodes):
        """Taker buy UP on first row of each episode produces valid reward."""
        env = Environment()
        for ep in real_episodes[:20]:
            env.reset(ep)
            mask = env.get_action_mask()
            if mask[1]:  # Can buy UP
                env.step(1)
                while True:
                    done, reward = env.step(0)
                    if done:
                        # Reward should be in reasonable range
                        assert -1.5 <= reward <= 1.5
                        break

    def test_action_mask_valid_every_row(self, real_episodes):
        """Action mask is valid at every row of every episode."""
        env = Environment()
        for ep in real_episodes[:20]:
            env.reset(ep)
            for _ in range(len(ep["rows"])):
                mask = env.get_action_mask()
                assert mask.shape == (9,)
                assert mask.dtype == bool
                assert mask[0]  # Action 0 always valid
                done, _ = env.step(0)
                if done:
                    break

    def test_observation_no_forbidden_fields(self, real_episodes):
        """No observation contains forbidden fields."""
        env = Environment()
        from src.environment import FORBIDDEN_FIELDS
        for ep in real_episodes[:10]:
            env.reset(ep)
            for _ in range(len(ep["rows"])):
                obs = env.get_observation()
                for field in FORBIDDEN_FIELDS:
                    assert field not in obs
                done, _ = env.step(0)
                if done:
                    break
