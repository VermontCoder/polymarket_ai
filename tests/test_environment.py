"""Tests for the Environment episode simulation, actions, rewards, and fees."""

import numpy as np
import pytest

from src.environment import (
    Environment,
    compute_action_mask,
    maker_rebate,
    taker_fee,
    NUM_ACTIONS,
    compute_buy_shares,
    compute_sell_proceeds,
    SHARES_PER_BUY,
    REWARD_NORMALIZATION,
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
    """Action masking: buy/sell mode, direction, pending limit, null, boundaries."""

    # --- Buy mode (shares_owned == 0) ---

    def test_buy_mode_allows_buys_blocks_sells(self):
        """In buy mode, sell actions 2/4/6/8 are always masked."""
        row = _make_row()
        mask = compute_action_mask(row, shares_owned=0.0)
        assert mask[0] is np.True_
        assert mask[1] is np.True_   # buy UP taker
        assert mask[2] is np.False_  # sell UP taker
        assert mask[3] is np.True_   # buy DOWN taker
        assert mask[4] is np.False_  # sell DOWN taker
        assert mask[5] is np.True_   # limit buy UP
        assert mask[6] is np.False_  # limit sell UP
        assert mask[7] is np.True_   # limit buy DOWN
        assert mask[8] is np.False_  # limit sell DOWN

    def test_buy_mode_null_up_ask_masks_1_and_5(self):
        """Buy mode: null up_ask blocks actions 1 and 5."""
        row = _make_row(up_ask=None)
        mask = compute_action_mask(row, shares_owned=0.0)
        assert mask[1] is np.False_
        assert mask[5] is np.False_
        assert mask[3] is np.True_   # down buy still OK
        assert mask[7] is np.True_

    def test_buy_mode_null_down_ask_masks_3_and_7(self):
        """Buy mode: null down_ask blocks actions 3 and 7."""
        row = _make_row(down_ask=None)
        mask = compute_action_mask(row, shares_owned=0.0)
        assert mask[3] is np.False_
        assert mask[7] is np.False_
        assert mask[1] is np.True_

    def test_buy_mode_limit_buy_boundary_low(self):
        """Buy mode: limit buy UP masked when up_ask - 1 < 1."""
        row = _make_row(up_ask=1.0)
        mask = compute_action_mask(row, shares_owned=0.0)
        assert mask[5] is np.False_
        assert mask[1] is np.True_   # taker buy still OK

    def test_buy_mode_limit_buy_down_boundary_low(self):
        """Buy mode: limit buy DOWN masked when down_ask - 1 < 1."""
        row = _make_row(down_ask=1.0)
        mask = compute_action_mask(row, shares_owned=0.0)
        assert mask[7] is np.False_
        assert mask[3] is np.True_

    def test_buy_mode_all_null_only_action0(self):
        """Buy mode: all bid/ask null -> only action 0."""
        row = _make_row(up_bid=None, up_ask=None, down_bid=None, down_ask=None)
        mask = compute_action_mask(row, shares_owned=0.0)
        assert mask[0] is np.True_
        for i in range(1, 9):
            assert mask[i] is np.False_

    # --- Sell mode (shares_owned > 0) ---

    def test_sell_mode_up_allows_up_sells_blocks_everything_else(self):
        """Sell mode UP: only actions 0, 2 (taker sell UP), 6 (limit sell UP) can be unmasked."""
        row = _make_row()
        mask = compute_action_mask(row, shares_owned=4.99, share_direction="UP")
        assert mask[0] is np.True_
        assert mask[2] is np.True_   # sell UP taker
        assert mask[6] is np.True_   # limit sell UP
        # All buys masked
        for buy_action in [1, 3, 5, 7]:
            assert mask[buy_action] is np.False_
        # DOWN sells also masked (wrong direction)
        assert mask[4] is np.False_
        assert mask[8] is np.False_

    def test_sell_mode_down_allows_down_sells_blocks_everything_else(self):
        """Sell mode DOWN: only actions 0, 4 (taker sell DOWN), 8 (limit sell DOWN) can be unmasked."""
        row = _make_row()
        mask = compute_action_mask(row, shares_owned=4.99, share_direction="DOWN")
        assert mask[0] is np.True_
        assert mask[4] is np.True_   # sell DOWN taker
        assert mask[8] is np.True_   # limit sell DOWN
        for buy_action in [1, 3, 5, 7]:
            assert mask[buy_action] is np.False_
        assert mask[2] is np.False_
        assert mask[6] is np.False_

    def test_sell_mode_null_up_bid_masks_sell_up(self):
        """Sell mode UP: null up_bid masks sell UP actions 2 and 6."""
        row = _make_row(up_bid=None)
        mask = compute_action_mask(row, shares_owned=4.99, share_direction="UP")
        assert mask[2] is np.False_
        assert mask[6] is np.False_

    def test_sell_mode_null_down_bid_masks_sell_down(self):
        """Sell mode DOWN: null down_bid masks sell DOWN actions 4 and 8."""
        row = _make_row(down_bid=None)
        mask = compute_action_mask(row, shares_owned=4.99, share_direction="DOWN")
        assert mask[4] is np.False_
        assert mask[8] is np.False_

    def test_sell_mode_limit_sell_up_boundary_high(self):
        """Sell mode UP: limit sell UP masked when up_bid + 1 > 99."""
        row = _make_row(up_bid=99.0)
        mask = compute_action_mask(row, shares_owned=4.99, share_direction="UP")
        assert mask[6] is np.False_
        assert mask[2] is np.True_   # taker sell still OK

    def test_sell_mode_limit_sell_down_boundary_high(self):
        """Sell mode DOWN: limit sell DOWN masked when down_bid + 1 > 99."""
        row = _make_row(down_bid=99.0)
        mask = compute_action_mask(row, shares_owned=4.99, share_direction="DOWN")
        assert mask[8] is np.False_
        assert mask[4] is np.True_

    # --- Pending limit ---

    def test_pending_limit_blocks_all_except_0(self):
        """Pending limit order -> only action 0 allowed."""
        row = _make_row()
        pending = {"action": 5, "price": 55.0, "market": "UP", "order_type": "buy", "placed_at_step": 0}
        mask = compute_action_mask(row, shares_owned=0.0, pending_limit=pending)
        assert mask[0] is np.True_
        for i in range(1, 9):
            assert mask[i] is np.False_

    def test_action0_always_allowed(self):
        """Action 0 is always allowed regardless of mode."""
        row = _make_row()
        # buy mode
        assert compute_action_mask(row, shares_owned=0.0)[0] is np.True_
        # sell mode
        assert compute_action_mask(row, shares_owned=4.99, share_direction="UP")[0] is np.True_
        # pending limit
        pending = {"action": 5, "price": 55.0, "market": "UP", "order_type": "buy", "placed_at_step": 0}
        assert compute_action_mask(row, shares_owned=0.0, pending_limit=pending)[0] is np.True_

    def test_sell_mode_missing_direction_raises(self):
        """Sell mode with no direction raises AssertionError."""
        row = _make_row()
        with pytest.raises(AssertionError):
            compute_action_mask(row, shares_owned=4.99)  # share_direction defaults to ""


# ---------------------------------------------------------------------------
# Tests: Environment — Basic Flow (new multi-trade mechanics)
# ---------------------------------------------------------------------------

class TestEnvironmentBasicFlow:
    """Basic episode flow under new multi-trade mechanics."""

    def test_reset_initializes_state(self):
        env = Environment()
        ep = _make_episode(num_rows=5)
        env.reset(ep)
        assert env.current_step == 0
        assert env.shares_owned == 0.0
        assert env.num_rows == 5

    def test_full_episode_do_nothing_reward_zero(self):
        env = Environment()
        ep = _make_episode(num_rows=5, outcome="UP")
        env.reset(ep)
        for _ in range(4):
            done, reward = env.step(0)
            assert not done
        done, reward = env.step(0)
        assert done
        assert reward == pytest.approx(0.0)

    def test_step_advances_current_step(self):
        env = Environment()
        ep = _make_episode(num_rows=3)
        env.reset(ep)
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

    def test_reset_clears_all_state(self):
        env = Environment()
        ep1 = _make_episode(num_rows=3, outcome="UP")
        env.reset(ep1)
        env.step(1)  # buy UP
        assert env.shares_owned > 0

        ep2 = _make_episode(num_rows=5, outcome="DOWN")
        env.reset(ep2)
        assert env.shares_owned == 0.0
        assert env.current_step == 0
        assert env.num_rows == 5

    def test_observation_excludes_forbidden_fields(self):
        env = Environment()
        ep = _make_episode(num_rows=3)
        env.reset(ep)
        obs = env.get_observation()
        for field in ["outcome", "end_price", "current_price", "diff_usd",
                       "start_price", "session_id", "timestamp"]:
            assert field not in obs

    def test_observation_contains_is_sell_mode(self):
        """is_sell_mode is always present in observation."""
        env = Environment()
        ep = _make_episode(num_rows=3)
        env.reset(ep)
        obs = env.get_observation()
        assert "is_sell_mode" in obs
        assert obs["is_sell_mode"] == 0.0  # buy mode initially

    def test_invalid_action_raises(self):
        env = Environment()
        ep = _make_episode(num_rows=3)
        env.reset(ep)
        with pytest.raises(AssertionError):
            env.step(2)  # sell in buy mode — masked


# ---------------------------------------------------------------------------
# Tests: Environment — Buy/Sell Mode
# ---------------------------------------------------------------------------

class TestBuySellMode:
    """Environment enforces buy → sell → buy alternation via shares_owned."""

    def test_buy_sets_shares_owned(self):
        """Taker buy UP sets shares_owned > 0 and is_sell_mode=1."""
        env = Environment()
        ep = _make_episode(num_rows=3, outcome="UP", up_ask=50.0)
        env.reset(ep)
        env.step(1)  # buy UP
        expected = compute_buy_shares(50.0, is_maker=False)
        assert env.shares_owned == pytest.approx(expected)
        obs = env.get_observation()
        assert obs["is_sell_mode"] == 1.0

    def test_sell_clears_shares_owned(self):
        """Taker sell after buying clears shares_owned to 0."""
        env = Environment()
        rows = [
            _make_row(up_ask=50.0, up_bid=49.0),
            _make_row(up_ask=51.0, up_bid=50.0),
            _make_row(up_ask=52.0, up_bid=51.0),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env.reset(ep)
        env.step(1)   # buy UP at row 0
        assert env.shares_owned > 0
        env.step(2)   # sell UP at row 1
        assert env.shares_owned == 0.0

    def test_is_sell_mode_false_after_sell(self):
        """is_sell_mode returns 0.0 after selling all shares."""
        env = Environment()
        rows = [
            _make_row(up_ask=50.0, up_bid=49.0),
            _make_row(up_ask=51.0, up_bid=50.0),
            _make_row(),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env.reset(ep)
        env.step(1)   # buy UP
        env.step(2)   # sell UP
        obs = env.get_observation()
        assert obs["is_sell_mode"] == 0.0

    def test_buy_mode_blocks_sell_action(self):
        """In buy mode, attempting to sell raises AssertionError."""
        env = Environment()
        ep = _make_episode(num_rows=3)
        env.reset(ep)
        with pytest.raises(AssertionError):
            env.step(2)  # sell UP in buy mode

    def test_sell_mode_blocks_buy_action(self):
        """In sell mode, attempting to buy raises AssertionError."""
        env = Environment()
        rows = [_make_row(up_ask=50.0), _make_row(up_ask=50.0)]
        ep = _make_episode(outcome="UP", rows=rows)
        env.reset(ep)
        env.step(1)  # buy UP → now in sell mode
        with pytest.raises(AssertionError):
            env.step(1)  # buy again → blocked

    def test_sell_mode_blocks_wrong_direction(self):
        """Holding UP shares: selling DOWN shares is blocked."""
        env = Environment()
        rows = [_make_row(up_ask=50.0), _make_row()]
        ep = _make_episode(outcome="UP", rows=rows)
        env.reset(ep)
        env.step(1)  # buy UP
        with pytest.raises(AssertionError):
            env.step(4)  # sell DOWN — wrong direction

    def test_buy_sell_buy_cycle(self):
        """Agent can buy→sell→buy within one episode."""
        env = Environment()
        rows = [
            _make_row(up_ask=50.0, up_bid=49.0),
            _make_row(up_ask=51.0, up_bid=52.0),
            _make_row(up_ask=52.0, up_bid=51.0),
            _make_row(up_ask=53.0, up_bid=52.0),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env.reset(ep)
        env.step(1)   # buy UP (row 0) → sell mode
        assert env.shares_owned > 0
        env.step(2)   # sell UP (row 1) → buy mode
        assert env.shares_owned == 0.0
        env.step(1)   # buy UP again (row 2) → sell mode
        assert env.shares_owned > 0

    def test_multiple_actions_per_episode(self):
        """Episode can have more than one trade (buy + sell)."""
        env = Environment()
        rows = [
            _make_row(up_ask=50.0, up_bid=49.0),
            _make_row(up_ask=51.0, up_bid=55.0),
            _make_row(),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env.reset(ep)
        env.step(1)   # buy
        env.step(2)   # sell
        done, reward = env.step(0)
        assert done


# ---------------------------------------------------------------------------
# Tests: Environment — Multi-Trade P&L
# ---------------------------------------------------------------------------

class TestMultiTradePnL:
    """Reward reflects full episode P&L including intra-episode trades."""

    def test_buy_and_hold_win_positive_reward(self):
        """Buy UP, hold to end, outcome UP → positive total reward."""
        env = Environment()
        price = 50.0
        ep = _make_episode(num_rows=2, outcome="UP", up_ask=price)
        env.reset(ep)
        _, r1 = env.step(1)   # buy UP at 50c
        done, r2 = env.step(0)
        assert done

        shares = compute_buy_shares(price, is_maker=False)
        cash_out = SHARES_PER_BUY * price
        payout = shares * 100.0  # UP outcome, UP shares
        expected = (payout - cash_out) / REWARD_NORMALIZATION
        total_reward = r1 + r2
        assert total_reward == pytest.approx(expected)
        assert total_reward > 0

    def test_buy_and_hold_loss_negative_reward(self):
        """Buy UP, hold to end, outcome DOWN → negative total reward."""
        env = Environment()
        price = 50.0
        ep = _make_episode(num_rows=2, outcome="DOWN", up_ask=price)
        env.reset(ep)
        _, r1 = env.step(1)
        done, r2 = env.step(0)
        assert done

        cash_out = SHARES_PER_BUY * price
        payout = 0.0  # wrong outcome
        expected = (payout - cash_out) / REWARD_NORMALIZATION
        total_reward = r1 + r2
        assert total_reward == pytest.approx(expected)
        assert total_reward < 0

    def test_buy_sell_intra_episode_profit(self):
        """Buy UP at 50c, sell UP at 60c → intra-episode profit, no end payout."""
        env = Environment()
        buy_price, sell_price = 50.0, 60.0
        rows = [
            _make_row(up_ask=buy_price, up_bid=49.0),
            _make_row(up_ask=61.0, up_bid=sell_price),
            _make_row(),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env.reset(ep)
        _, r1 = env.step(1)   # buy UP at 50c
        _, r2 = env.step(2)   # sell UP at 60c
        done, r3 = env.step(0)
        assert done

        shares = compute_buy_shares(buy_price, is_maker=False)
        cash_out = SHARES_PER_BUY * buy_price
        cash_in = compute_sell_proceeds(shares, sell_price, is_maker=False)
        expected = (cash_in - cash_out) / REWARD_NORMALIZATION
        total_reward = r1 + r2 + r3
        assert total_reward == pytest.approx(expected, rel=1e-4)
        assert total_reward > 0  # sold higher than bought

    def test_no_trade_reward_zero(self):
        """No trades → all step rewards = 0."""
        env = Environment()
        ep = _make_episode(num_rows=3, outcome="UP")
        env.reset(ep)
        for _ in range(2):
            done, r = env.step(0)
            assert not done
            assert r == pytest.approx(0.0)
        done, reward = env.step(0)
        assert done
        assert reward == pytest.approx(0.0)

    def test_reward_normalized_by_500(self):
        """Total episode reward is divided by REWARD_NORMALIZATION (500), not 100."""
        env = Environment()
        price = 1.0  # cheapest buy, max win
        ep = _make_episode(num_rows=2, outcome="UP", up_ask=price)
        env.reset(ep)
        _, r1 = env.step(1)
        done, r2 = env.step(0)
        assert done
        # Max possible total reward ≈ (5 * 100 - 5 * 1) / 500 ≈ 0.99
        total_reward = r1 + r2
        assert total_reward < 1.0
        assert total_reward > 0.5  # but still substantial

    def test_maker_buy_hold_win(self):
        """Maker limit buy fills, hold to end, outcome matches → positive total reward."""
        price = 50.0  # limit placed at up_ask - 1 = 56 - 1 = 55
        rows = [
            _make_row(up_ask=56.0),
            _make_row(up_ask=55.0),  # fills at limit price 55
            _make_row(),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env = Environment()
        env.reset(ep)
        _, r1 = env.step(5)  # limit buy UP at 55c
        _, r2 = env.step(0)  # wait for fill check
        done, r3 = env.step(0)
        assert done

        limit_price = 55.0
        shares = compute_buy_shares(limit_price, is_maker=True)
        cash_out = SHARES_PER_BUY * limit_price
        payout = shares * 100.0
        expected = (payout - cash_out) / REWARD_NORMALIZATION
        total_reward = r1 + r2 + r3
        assert total_reward == pytest.approx(expected, rel=1e-4)

    def test_unfilled_limit_buy_no_reward(self):
        """Unfilled limit buy order → 0 cash flow → reward 0."""
        rows = [_make_row(up_ask=56.0)] * 3  # ask stays above 55, never fills
        ep = _make_episode(outcome="UP", rows=rows)
        env = Environment()
        env.reset(ep)
        env.step(5)  # limit buy UP at 55c
        env.step(0)
        done, reward = env.step(0)
        assert done
        assert reward == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: Environment — Limit Orders (new mechanics)
# ---------------------------------------------------------------------------

class TestLimitOrdersMechanic:
    """Limit orders block actions while pending; fill switches mode correctly."""

    def test_pending_limit_buy_blocks_other_actions(self):
        """After placing a limit buy, only action 0 is available until fill."""
        rows = [
            _make_row(up_ask=56.0),
            _make_row(up_ask=57.0),  # no fill
            _make_row(up_ask=58.0),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env = Environment()
        env.reset(ep)
        env.step(5)  # limit buy UP at 55c

        # Next row: pending limit, only action 0 valid
        mask = env.get_action_mask()
        assert mask[0] is np.True_
        for i in range(1, 9):
            assert mask[i] is np.False_

    def test_limit_buy_fill_sets_shares(self):
        """When a limit buy fills, shares_owned becomes positive."""
        rows = [
            _make_row(up_ask=56.0),
            _make_row(up_ask=55.0),  # fills at 55c
            _make_row(),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env = Environment()
        env.reset(ep)
        env.step(5)  # limit buy UP at 55c
        env.step(0)  # row 1: fill check runs
        assert env.shares_owned == pytest.approx(compute_buy_shares(55.0, is_maker=True))

    def test_limit_sell_fill_clears_shares(self):
        """When a limit sell fills, shares_owned drops to 0."""
        rows = [
            _make_row(up_ask=50.0, up_bid=49.0),
            _make_row(up_ask=51.0, up_bid=50.0),  # buy row
            _make_row(up_ask=52.0, up_bid=51.0),
            _make_row(up_ask=53.0, up_bid=57.0),  # bid >= 51 → limit sell fills
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env = Environment()
        env.reset(ep)
        env.step(1)   # taker buy UP at 50c (row 0)
        env.step(6)   # limit sell UP at 50c (row 1, bid=50 → order at 51c)
        env.step(0)   # row 2: bid=51 >= 51, should fill
        done, reward = env.step(0)
        assert done or env.shares_owned == 0.0

    def test_limit_fill_only_after_placement_row(self):
        """Fill check does not fire on the same row the limit order was placed."""
        rows = [
            _make_row(up_ask=55.0),
            _make_row(up_ask=56.0),  # ask went up, no fill
            _make_row(up_ask=57.0),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env = Environment()
        env.reset(ep)
        env.step(5)   # limit buy UP at 54c
        env.step(0)
        done, reward = env.step(0)
        assert done
        assert reward == pytest.approx(0.0)  # never filled

    def test_null_market_price_prevents_fill(self):
        """Null bid/ask prevents limit fill on that row."""
        rows = [
            _make_row(up_ask=56.0),
            _make_row(up_ask=None),  # null → no fill
            _make_row(up_ask=None),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env = Environment()
        env.reset(ep)
        env.step(5)  # limit buy UP at 55c
        env.step(0)
        done, reward = env.step(0)
        assert done
        assert reward == pytest.approx(0.0)


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

from tests.conftest import find_data_file as _find_data_file


class TestIntegrationWithRealData:
    """Smoke tests using the actual data file."""

    @pytest.fixture
    def real_episodes(self):
        import json
        try:
            with open(_find_data_file()) as f:
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


# ---------------------------------------------------------------------------
# Tests: Share Calculations
# ---------------------------------------------------------------------------

class TestShareCalculations:
    """compute_buy_shares and compute_sell_proceeds match design spec."""

    def test_taker_buy_shares_reduces_by_fee(self):
        """Taker buy: shares = round(5 * (1 - fee/price), 2)."""
        price = 50.0
        expected = round(SHARES_PER_BUY * (1 - taker_fee(price) / price), 2)
        assert compute_buy_shares(price, is_maker=False) == pytest.approx(expected)

    def test_maker_buy_shares_increases_by_rebate(self):
        """Maker buy: shares = round(5 * (1 + rebate/price), 2)."""
        price = 50.0
        expected = round(SHARES_PER_BUY * (1 + maker_rebate(price) / price), 2)
        assert compute_buy_shares(price, is_maker=True) == pytest.approx(expected)

    def test_taker_buy_shares_less_than_five(self):
        """Taker buy yields < 5.0 shares (fee is positive)."""
        for price in [1.0, 25.0, 50.0]:
            assert compute_buy_shares(price, is_maker=False) < 5.0

    def test_maker_buy_shares_more_than_five(self):
        """Maker buy yields > 5.0 shares (rebate is positive)."""
        for price in [1.0, 25.0, 50.0]:
            assert compute_buy_shares(price, is_maker=True) > 5.0

    def test_shares_precision_two_decimal_places(self):
        """Share counts are rounded to 2 decimal places."""
        shares = compute_buy_shares(33.0, is_maker=False)
        assert shares == round(shares, 2)

    def test_taker_sell_proceeds(self):
        """Taker sell: proceeds = shares * (price - taker_fee(price))."""
        shares, price = 4.99, 55.0
        expected = round(shares * (price - taker_fee(price)), 4)
        assert compute_sell_proceeds(shares, price, is_maker=False) == pytest.approx(expected)

    def test_maker_sell_proceeds(self):
        """Maker sell: proceeds = shares * (price + maker_rebate(price))."""
        shares, price = 5.01, 56.0
        expected = round(shares * (price + maker_rebate(price)), 4)
        assert compute_sell_proceeds(shares, price, is_maker=True) == pytest.approx(expected)

    def test_taker_sell_proceeds_less_than_gross(self):
        """Taker sell proceeds < shares * price (fee reduces payout)."""
        shares, price = 4.99, 55.0
        gross = shares * price
        assert compute_sell_proceeds(shares, price, is_maker=False) < gross

    def test_maker_sell_proceeds_more_than_gross(self):
        """Maker sell proceeds > shares * price (rebate boosts payout)."""
        shares, price = 5.01, 55.0
        gross = shares * price
        assert compute_sell_proceeds(shares, price, is_maker=True) > gross

    def test_constants_exist(self):
        """SHARES_PER_BUY and REWARD_NORMALIZATION are defined."""
        assert SHARES_PER_BUY == 5.0
        assert REWARD_NORMALIZATION == 500.0
