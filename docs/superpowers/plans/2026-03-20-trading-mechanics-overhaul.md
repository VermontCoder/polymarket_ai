# Trading Mechanics Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overhaul the trading environment from single-trade-per-episode to multi-trade buy/sell alternation with share tracking, intra-episode P&L, and a new `is_sell_mode` observation feature.

**Architecture:** `Environment` state replaces `has_acted: bool` with `shares_owned: float` + `share_direction: str` + `net_cash: float` + `pending_limit: dict|None`. Buy mode (shares=0) masks all sells; sell mode (shares>0) masks all buys and only allows selling the held direction. `skip_to_end()` is deleted — the trainer runs all rows. Reward = `(net_cash + end_payout) / 500`. A new `is_sell_mode` binary feature (0.0/1.0) is appended to the dynamic observation, expanding `DYNAMIC_DIM` from 11 to 12.

**Tech Stack:** Python, NumPy, PyTorch, existing DRQN/PER infrastructure.

---

## Design Decisions (locked in)

| Decision | Choice |
|----------|--------|
| Shares per buy | `SHARES_PER_BUY = 5.0` (always) |
| Taker buy shares | `round(5 * (1 - taker_fee(price)/price), 2)` |
| Maker buy shares | `round(5 * (1 + maker_rebate(price)/price), 2)` |
| Buy cash outflow | `SHARES_PER_BUY * price` (fee baked into share count) |
| Taker sell proceeds | `shares * (price - taker_fee(price))` |
| Maker sell proceeds | `shares * (price + maker_rebate(price))` |
| Reward normalization | `total_pnl_cents / 500.0` |
| Pending limit → | Only action 0 allowed |
| Sell direction mask | Can only sell the direction you hold (UP or DOWN) |
| `is_sell_mode` | `1.0` if `shares_owned > 0` else `0.0`; appended to dynamic obs |
| Old `compute_reward` | **Deleted** — replaced by episode-level `_compute_final_reward` |
| `skip_to_end()` | **Deleted** — trainer runs all rows |
| Trainer reward assignment | Terminal reward stays at terminal transition; TD discounting handles credit |

---

## File Map

| File | Change |
|------|--------|
| `src/environment.py` | Add `SHARES_PER_BUY`, `REWARD_NORMALIZATION`; add `compute_buy_shares()`, `compute_sell_proceeds()`; replace `compute_action_mask(row, has_acted)` → `compute_action_mask(row, shares_owned, share_direction, pending_limit)`; delete `compute_reward()`; rewrite `Environment` class; delete `skip_to_end()`; add `is_sell_mode` to `get_observation()` |
| `src/normalizer.py` | `DYNAMIC_DIM = 11 → 12`; append `is_sell_mode` (dim 11) to `encode_dynamic()`; update `encode_episode_dynamic()` to default `is_sell_mode=0.0` |
| `src/trainer.py` | Delete `skip_to_end()` call, `_filter_pre_action()`, `_assign_reward_to_action_step()`; rewrite `_run_episode()` to run all rows |
| `src/visibility.py` | Remove early termination; track transaction list; show per-trade accounting; show episode end settlement |
| `src/agents/random_agent.py` | Update docstring comment only — behavior unchanged (mask handles mode) |
| `tests/test_environment.py` | Delete `TestSkipToEnd`; delete single-action tests; rewrite `TestActionMask`; add `TestShareCalculations`, `TestBuySellMode`, `TestMultiTradePnL` |
| `tests/test_normalizer.py` | Update dimension checks for `DYNAMIC_DIM=12`; add `is_sell_mode` encoding test |
| `tests/test_trainer.py` | Update `_run_episode` tests for multi-trade flow |

---

## Task 1: New constants and share calculation functions

**Files:**
- Modify: `src/environment.py` (add after existing fee functions)

These are pure functions — easy to test in isolation before touching the `Environment` class.

- [ ] **Step 1: Write the failing tests**

Add a new class to `tests/test_environment.py` (after the existing imports and helpers):

```python
# ---------------------------------------------------------------------------
# Tests: Share Calculations
# ---------------------------------------------------------------------------

class TestShareCalculations:
    """compute_buy_shares and compute_sell_proceeds match design spec."""

    def test_taker_buy_shares_reduces_by_fee(self):
        """Taker buy: shares = round(5 * (1 - fee/price), 2)."""
        from src.environment import compute_buy_shares, taker_fee
        price = 50.0
        expected = round(5 * (1 - taker_fee(price) / price), 2)
        assert compute_buy_shares(price, is_maker=False) == pytest.approx(expected)

    def test_maker_buy_shares_increases_by_rebate(self):
        """Maker buy: shares = round(5 * (1 + rebate/price), 2)."""
        from src.environment import compute_buy_shares, maker_rebate
        price = 50.0
        expected = round(5 * (1 + maker_rebate(price) / price), 2)
        assert compute_buy_shares(price, is_maker=True) == pytest.approx(expected)

    def test_taker_buy_shares_less_than_five(self):
        """Taker buy always yields fewer than 5.0 shares (fee is positive)."""
        from src.environment import compute_buy_shares
        for price in [1.0, 25.0, 50.0, 75.0, 99.0]:
            assert compute_buy_shares(price, is_maker=False) < 5.0

    def test_maker_buy_shares_more_than_five(self):
        """Maker buy always yields more than 5.0 shares (rebate is positive)."""
        from src.environment import compute_buy_shares
        for price in [1.0, 25.0, 50.0, 75.0, 99.0]:
            assert compute_buy_shares(price, is_maker=True) > 5.0

    def test_shares_precision_two_decimal_places(self):
        """Share counts are rounded to 2 decimal places."""
        from src.environment import compute_buy_shares
        shares = compute_buy_shares(33.0, is_maker=False)
        assert shares == round(shares, 2)

    def test_taker_sell_proceeds(self):
        """Taker sell: proceeds = shares * (price - taker_fee(price))."""
        from src.environment import compute_sell_proceeds, taker_fee
        shares, price = 4.99, 55.0
        expected = round(shares * (price - taker_fee(price)), 4)
        assert compute_sell_proceeds(shares, price, is_maker=False) == pytest.approx(expected)

    def test_maker_sell_proceeds(self):
        """Maker sell: proceeds = shares * (price + maker_rebate(price))."""
        from src.environment import compute_sell_proceeds, maker_rebate
        shares, price = 5.01, 56.0
        expected = round(shares * (price + maker_rebate(price)), 4)
        assert compute_sell_proceeds(shares, price, is_maker=True) == pytest.approx(expected)

    def test_taker_sell_proceeds_less_than_gross(self):
        """Taker sell proceeds < shares * price (fee reduces payout)."""
        from src.environment import compute_sell_proceeds
        shares, price = 4.99, 55.0
        gross = shares * price
        assert compute_sell_proceeds(shares, price, is_maker=False) < gross

    def test_maker_sell_proceeds_more_than_gross(self):
        """Maker sell proceeds > shares * price (rebate boosts payout)."""
        from src.environment import compute_sell_proceeds
        shares, price = 5.01, 55.0
        gross = shares * price
        assert compute_sell_proceeds(shares, price, is_maker=True) > gross

    def test_constants_exist(self):
        """SHARES_PER_BUY and REWARD_NORMALIZATION are defined."""
        from src.environment import SHARES_PER_BUY, REWARD_NORMALIZATION
        assert SHARES_PER_BUY == 5.0
        assert REWARD_NORMALIZATION == 500.0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_environment.py::TestShareCalculations -v
```
Expected: `ImportError` — `compute_buy_shares`, `compute_sell_proceeds`, `SHARES_PER_BUY`, `REWARD_NORMALIZATION` don't exist yet.

- [ ] **Step 3: Add constants and functions to `src/environment.py`**

Add directly after the existing `maker_rebate()` function:

```python
SHARES_PER_BUY: float = 5.0
REWARD_NORMALIZATION: float = 500.0


def compute_buy_shares(price: float, is_maker: bool) -> float:
    """Compute effective shares received when buying SHARES_PER_BUY nominal shares.

    Fee (taker) or rebate (maker) is absorbed into the share count rather than
    tracked as a separate cash outflow. Cash outflow is always SHARES_PER_BUY * price.

    Args:
        price: Trade price in cents (1-99).
        is_maker: True if this is a maker/limit order.

    Returns:
        Effective shares, rounded to 2 decimal places.
    """
    if is_maker:
        return round(SHARES_PER_BUY * (1.0 + maker_rebate(price) / price), 2)
    return round(SHARES_PER_BUY * (1.0 - taker_fee(price) / price), 2)


def compute_sell_proceeds(shares: float, price: float, is_maker: bool) -> float:
    """Compute cash received from selling `shares` at `price`.

    Args:
        shares: Number of shares to sell.
        price: Trade price in cents.
        is_maker: True if this is a maker/limit order.

    Returns:
        Cash received in cents, rounded to 4 decimal places.
    """
    if is_maker:
        net_price = price + maker_rebate(price)
    else:
        net_price = price - taker_fee(price)
    return round(shares * net_price, 4)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_environment.py::TestShareCalculations -v
```
Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/environment.py tests/test_environment.py
git commit -m "feat: add compute_buy_shares, compute_sell_proceeds, SHARES_PER_BUY, REWARD_NORMALIZATION"
```

---

## Task 2: Replace `compute_action_mask` for buy/sell mode

**Files:**
- Modify: `src/environment.py` — replace `compute_action_mask` signature and body
- Modify: `tests/test_environment.py` — rewrite `TestActionMask`

**New signature:** `compute_action_mask(row, shares_owned, share_direction="", pending_limit=None)`

**New rules:**
1. Action 0 always allowed.
2. If `pending_limit` is not None → only action 0 (limit order is active).
3. If `shares_owned == 0` (buy mode): mask actions 2, 4, 6, 8 (all sells); apply null/boundary rules to buy actions only.
4. If `shares_owned > 0` (sell mode): mask actions 1, 3, 5, 7 (all buys); mask sells for the wrong direction (`share_direction == "UP"` → mask 4, 8; `== "DOWN"` → mask 2, 6); apply null/boundary rules to remaining sell actions.

- [ ] **Step 1: Rewrite `TestActionMask` in `tests/test_environment.py`**

Replace the entire `TestActionMask` class with:

```python
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
        """Buy mode: all bid/ask null → only action 0."""
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
        """Pending limit order → only action 0 allowed."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_environment.py::TestActionMask -v
```
Expected: Most fail — `compute_action_mask` still takes `has_acted` parameter.

- [ ] **Step 3: Replace `compute_action_mask` in `src/environment.py`**

Replace the entire `compute_action_mask` function:

```python
def compute_action_mask(
    row: dict[str, Any],
    shares_owned: float,
    share_direction: str = "",
    pending_limit: dict | None = None,
) -> np.ndarray:
    """Build a boolean action mask (9 elements, True = allowed).

    Rules:
      1. Action 0 (do nothing) is always allowed.
      2. If pending_limit is set, only action 0 is allowed.
      3. If shares_owned == 0 (buy mode):
           - Sell actions (2, 4, 6, 8) are masked.
           - Null up_ask -> mask 1, 5.
           - Null down_ask -> mask 3, 7.
           - Limit buy boundary (ask - 1 < 1): mask 5 or 7.
      4. If shares_owned > 0 (sell mode):
           - Buy actions (1, 3, 5, 7) are masked.
           - If share_direction == "UP": mask 4, 8 (wrong direction sells).
           - If share_direction == "DOWN": mask 2, 6 (wrong direction sells).
           - Null up_bid -> mask 2, 6.
           - Null down_bid -> mask 4, 8.
           - Limit sell boundary (bid + 1 > 99): mask 6 or 8.
    """
    mask = np.ones(NUM_ACTIONS, dtype=bool)

    # Rule 2: pending limit order — wait for fill or episode end
    if pending_limit is not None:
        mask[1:] = False
        return mask

    if shares_owned == 0.0:
        # Buy mode: mask all sells
        mask[2] = mask[4] = mask[6] = mask[8] = False

        up_ask = row.get("up_ask")
        down_ask = row.get("down_ask")

        if up_ask is None:
            mask[1] = mask[5] = False
        elif up_ask - 1 < 1:
            mask[5] = False

        if down_ask is None:
            mask[3] = mask[7] = False
        elif down_ask - 1 < 1:
            mask[7] = False

    else:
        # Sell mode: mask all buys
        mask[1] = mask[3] = mask[5] = mask[7] = False

        # Mask sells for the wrong direction
        if share_direction == "UP":
            mask[4] = mask[8] = False  # can't sell DOWN shares
        elif share_direction == "DOWN":
            mask[2] = mask[6] = False  # can't sell UP shares

        up_bid = row.get("up_bid")
        down_bid = row.get("down_bid")

        if up_bid is None:
            mask[2] = mask[6] = False
        elif up_bid + 1 > 99:
            mask[6] = False

        if down_bid is None:
            mask[4] = mask[8] = False
        elif down_bid + 1 > 99:
            mask[8] = False

    return mask
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_environment.py::TestActionMask -v
```
Expected: All tests PASS.

- [ ] **Step 5: Run full suite — expect some failures (dependent code)**

```
pytest tests/test_environment.py -v
```
Expected: `TestActionMask` passes; other tests that call `compute_action_mask(row, has_acted=...)` or use `Environment` will fail. Note which ones — they'll be fixed in Task 3.

- [ ] **Step 6: Commit**

```bash
git add src/environment.py tests/test_environment.py
git commit -m "feat: replace compute_action_mask with buy/sell mode and direction masking"
```

---

## Task 3: Rewrite the `Environment` class

**Files:**
- Modify: `src/environment.py` — full `Environment` class rewrite; delete `compute_reward()` and `skip_to_end()`
- Modify: `tests/test_environment.py` — delete `TestRewardComputation`, `TestMakerReward`, `TestSkipToEnd`, the old `TestEnvironmentFlow`, `TestEnvironmentTakerRewards`, `TestLimitOrderFill`, `TestTradeInfo`; add new test classes

This is the core change. Read the old class carefully before replacing.

- [ ] **Step 1: Write the new test classes**

Delete the following old test classes entirely from `tests/test_environment.py`:
- `TestRewardComputation` (old `compute_reward` function is gone)
- `TestMakerReward` (same)
- `TestEnvironmentFlow` (replace with new version below)
- `TestEnvironmentTakerRewards` (replace)
- `TestLimitOrderFill` (replace)
- `TestTradeInfo` (replace)
- `TestSkipToEnd` (deleted — method removed)

Add these new test classes:

```python
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
        from src.environment import compute_buy_shares
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
        """Buy UP, hold to end, outcome UP → positive reward."""
        from src.environment import compute_buy_shares, SHARES_PER_BUY, REWARD_NORMALIZATION
        env = Environment()
        price = 50.0
        ep = _make_episode(num_rows=2, outcome="UP", up_ask=price)
        env.reset(ep)
        env.step(1)   # buy UP at 50c
        done, reward = env.step(0)
        assert done

        shares = compute_buy_shares(price, is_maker=False)
        cash_out = SHARES_PER_BUY * price
        payout = shares * 100.0  # UP outcome, UP shares
        expected = (payout - cash_out) / REWARD_NORMALIZATION
        assert reward == pytest.approx(expected)
        assert reward > 0

    def test_buy_and_hold_loss_negative_reward(self):
        """Buy UP, hold to end, outcome DOWN → negative reward."""
        from src.environment import compute_buy_shares, SHARES_PER_BUY, REWARD_NORMALIZATION
        env = Environment()
        price = 50.0
        ep = _make_episode(num_rows=2, outcome="DOWN", up_ask=price)
        env.reset(ep)
        env.step(1)
        done, reward = env.step(0)
        assert done

        cash_out = SHARES_PER_BUY * price
        payout = 0.0  # wrong outcome
        expected = (payout - cash_out) / REWARD_NORMALIZATION
        assert reward == pytest.approx(expected)
        assert reward < 0

    def test_buy_sell_intra_episode_profit(self):
        """Buy UP at 50c, sell UP at 60c → intra-episode profit, no end payout."""
        from src.environment import (
            compute_buy_shares, compute_sell_proceeds,
            SHARES_PER_BUY, REWARD_NORMALIZATION
        )
        env = Environment()
        buy_price, sell_price = 50.0, 60.0
        rows = [
            _make_row(up_ask=buy_price, up_bid=49.0),
            _make_row(up_ask=61.0, up_bid=sell_price),
            _make_row(),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env.reset(ep)
        env.step(1)   # buy UP at 50c
        env.step(2)   # sell UP at 60c

        done, reward = env.step(0)
        assert done

        shares = compute_buy_shares(buy_price, is_maker=False)
        cash_out = SHARES_PER_BUY * buy_price
        cash_in = compute_sell_proceeds(shares, sell_price, is_maker=False)
        expected = (cash_in - cash_out) / REWARD_NORMALIZATION
        assert reward == pytest.approx(expected, rel=1e-4)
        assert reward > 0  # sold higher than bought

    def test_no_trade_reward_zero(self):
        """No trades → reward = 0."""
        env = Environment()
        ep = _make_episode(num_rows=3, outcome="UP")
        env.reset(ep)
        for _ in range(2):
            done, r = env.step(0)
            assert not done
        done, reward = env.step(0)
        assert done
        assert reward == pytest.approx(0.0)

    def test_reward_normalized_by_500(self):
        """Reward is divided by REWARD_NORMALIZATION (500), not 100."""
        from src.environment import compute_buy_shares, SHARES_PER_BUY
        env = Environment()
        price = 1.0  # cheapest buy, max win
        ep = _make_episode(num_rows=2, outcome="UP", up_ask=price)
        env.reset(ep)
        env.step(1)
        done, reward = env.step(0)
        assert done
        # Max possible reward ≈ (5 * 100 - 5 * 1) / 500 ≈ 0.99
        assert reward < 1.0
        assert reward > 0.5  # but still substantial

    def test_maker_buy_hold_win(self):
        """Maker limit buy fills, hold to end, outcome matches → reward > taker equivalent."""
        from src.environment import (
            compute_buy_shares, SHARES_PER_BUY, REWARD_NORMALIZATION
        )
        price = 50.0  # limit placed at up_ask - 1 = 56 - 1 = 55
        rows = [
            _make_row(up_ask=56.0),
            _make_row(up_ask=55.0),  # fills at limit price 55
            _make_row(),
        ]
        ep = _make_episode(outcome="UP", rows=rows)
        env = Environment()
        env.reset(ep)
        env.step(5)  # limit buy UP at 55c
        env.step(0)  # wait for fill check
        done, reward = env.step(0)
        assert done

        limit_price = 55.0
        shares = compute_buy_shares(limit_price, is_maker=True)
        cash_out = SHARES_PER_BUY * limit_price
        payout = shares * 100.0
        expected = (payout - cash_out) / REWARD_NORMALIZATION
        assert reward == pytest.approx(expected, rel=1e-4)

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
        from src.environment import compute_buy_shares
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
            _make_row(up_ask=53.0, up_bid=57.0),  # bid >= 57 → limit sell fills
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
        # Row 0: up_ask=55 → limit placed at 54. ask=55 > 54, no fill on row 0.
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_environment.py::TestEnvironmentBasicFlow tests/test_environment.py::TestBuySellMode tests/test_environment.py::TestMultiTradePnL tests/test_environment.py::TestLimitOrdersMechanic -v
```
Expected: Many failures — old `Environment` doesn't match new mechanics.

- [ ] **Step 3: Rewrite the `Environment` class in `src/environment.py`**

Delete the old `compute_reward()` function entirely. Delete `skip_to_end()`. Replace the full `Environment` class:

```python
class Environment:
    """Multi-trade episode simulation environment.

    The agent alternates between buy mode (shares_owned == 0) and sell
    mode (shares_owned > 0). One trade is allowed per row.

    State:
        shares_owned:    Float shares currently held (0.0 = buy mode).
        share_direction: "UP" or "DOWN" — the direction of held shares.
        net_cash:        Intra-episode cash flow (sells_in - buys_out) in cents.
        pending_limit:   Dict describing a pending maker order, or None.

    Usage:
        env = Environment()
        env.reset(episode_dict)
        while True:
            obs  = env.get_observation()
            mask = env.get_action_mask()
            done, reward = env.step(action)
            if done:
                break
    """

    def __init__(self) -> None:
        self._episode: dict[str, Any] | None = None
        self._rows: list[dict[str, Any]] = []
        self._current_step: int = 0
        self._outcome: str = ""

        # Share / cash state
        self._shares_owned: float = 0.0
        self._share_direction: str = ""   # "UP" or "DOWN"
        self._net_cash: float = 0.0       # cents; positive = net received
        self._pending_limit: dict[str, Any] | None = None
        self._trades: list[dict[str, Any]] = []

    def reset(self, episode: dict[str, Any]) -> None:
        """Initialize the environment with an episode dict."""
        self._episode = episode
        self._rows = episode["rows"]
        self._current_step = 0
        self._outcome = episode["outcome"]
        self._shares_owned = 0.0
        self._share_direction = ""
        self._net_cash = 0.0
        self._pending_limit = None
        self._trades = []

    def get_observation(self) -> dict[str, Any]:
        """Return current row data (forbidden fields stripped) plus is_sell_mode."""
        row = self._rows[self._current_step]
        obs = {k: v for k, v in row.items() if k not in FORBIDDEN_FIELDS}
        obs["is_sell_mode"] = 1.0 if self._shares_owned > 0 else 0.0
        return obs

    def get_action_mask(self) -> np.ndarray:
        """Return boolean mask for valid actions at current step."""
        row = self._rows[self._current_step]
        return compute_action_mask(
            row, self._shares_owned, self._share_direction, self._pending_limit
        )

    def get_episode_info(self) -> dict[str, Any]:
        """Return episode-level info (non-forbidden fields) for the agent."""
        assert self._episode is not None
        return {
            k: v for k, v in self._episode.items()
            if k not in FORBIDDEN_FIELDS and k != "rows"
        }

    @property
    def shares_owned(self) -> float:
        return self._shares_owned

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def num_rows(self) -> int:
        return len(self._rows)

    @property
    def trades(self) -> list[dict[str, Any]]:
        """Completed trades this episode (not including pending limit)."""
        return list(self._trades)

    def step(self, action: int) -> tuple[bool, float]:
        """Process one timestep.

        Args:
            action: Integer 0-8.

        Returns:
            Tuple of (done, reward). reward is only meaningful when done=True.
        """
        assert 0 <= action < NUM_ACTIONS, f"Invalid action: {action}"

        mask = self.get_action_mask()
        assert mask[action], (
            f"Action {action} is masked at step {self._current_step} "
            f"(shares={self._shares_owned:.2f}, dir={self._share_direction!r}, "
            f"pending={self._pending_limit is not None})"
        )

        row = self._rows[self._current_step]

        if action != 0:
            self._execute_action(action, row)

        # Check pending limit fill on rows AFTER the order was placed
        if (
            self._pending_limit is not None
            and self._current_step > self._pending_limit["placed_at_step"]
        ):
            self._check_limit_fill(row)

        self._current_step += 1
        done = self._current_step >= len(self._rows)

        if done:
            return True, self._compute_final_reward()
        return False, 0.0

    def _execute_action(self, action: int, row: dict[str, Any]) -> None:
        """Execute a trade action, updating shares and net_cash."""
        if action in (1, 3):
            # Taker buy
            price = row["up_ask"] if action == 1 else row["down_ask"]
            direction = "UP" if action == 1 else "DOWN"
            shares = compute_buy_shares(price, is_maker=False)
            self._shares_owned = shares
            self._share_direction = direction
            self._net_cash -= SHARES_PER_BUY * price
            self._trades.append({
                "type": "buy", "action": action, "price": price,
                "shares": shares, "is_maker": False, "direction": direction,
            })

        elif action in (2, 4):
            # Taker sell
            price = row["up_bid"] if action == 2 else row["down_bid"]
            proceeds = compute_sell_proceeds(self._shares_owned, price, is_maker=False)
            self._trades.append({
                "type": "sell", "action": action, "price": price,
                "shares": self._shares_owned, "proceeds": proceeds,
                "is_maker": False, "direction": self._share_direction,
            })
            self._net_cash += proceeds
            self._shares_owned = 0.0
            self._share_direction = ""

        elif action in (5, 7):
            # Maker limit buy
            price = (row["up_ask"] - 1) if action == 5 else (row["down_ask"] - 1)
            direction = "UP" if action == 5 else "DOWN"
            self._pending_limit = {
                "action": action, "price": price, "market": direction,
                "order_type": "buy", "placed_at_step": self._current_step,
            }

        elif action in (6, 8):
            # Maker limit sell
            price = (row["up_bid"] + 1) if action == 6 else (row["down_bid"] + 1)
            direction = self._share_direction  # sell in the direction we hold
            self._pending_limit = {
                "action": action, "price": price, "market": direction,
                "order_type": "sell", "placed_at_step": self._current_step,
            }

    def _check_limit_fill(self, row: dict[str, Any]) -> None:
        """Check if the pending limit order fills on this row."""
        pending = self._pending_limit
        action = pending["action"]
        price = pending["price"]

        if action in (5, 7):
            # Limit buy: fills if market ask <= order price
            field = "up_ask" if action == 5 else "down_ask"
            market_price = row.get(field)
            if market_price is not None and market_price <= price:
                direction = pending["market"]
                shares = compute_buy_shares(price, is_maker=True)
                self._shares_owned = shares
                self._share_direction = direction
                self._net_cash -= SHARES_PER_BUY * price
                self._trades.append({
                    "type": "buy", "action": action, "price": price,
                    "shares": shares, "is_maker": True, "direction": direction,
                })
                self._pending_limit = None

        elif action in (6, 8):
            # Limit sell: fills if market bid >= order price
            field = "up_bid" if action == 6 else "down_bid"
            market_price = row.get(field)
            if market_price is not None and market_price >= price:
                proceeds = compute_sell_proceeds(self._shares_owned, price, is_maker=True)
                self._trades.append({
                    "type": "sell", "action": action, "price": price,
                    "shares": self._shares_owned, "proceeds": proceeds,
                    "is_maker": True, "direction": self._share_direction,
                })
                self._net_cash += proceeds
                self._shares_owned = 0.0
                self._share_direction = ""
                self._pending_limit = None

    def _compute_final_reward(self) -> float:
        """Compute episode reward at end: (net_cash + end_payout) / REWARD_NORMALIZATION."""
        # Payout for any shares still held at episode end
        if self._shares_owned > 0 and self._share_direction:
            payout = (
                self._shares_owned * 100.0
                if self._outcome == self._share_direction
                else 0.0
            )
        else:
            payout = 0.0

        # Pending unfilled limit → no cash effect (order simply cancels)
        total_pnl = self._net_cash + payout
        return total_pnl / REWARD_NORMALIZATION
```

- [ ] **Step 4: Run new environment tests**

```
pytest tests/test_environment.py -v -k "not Integration"
```
Expected: All new test classes PASS. `TestShareCalculations`, `TestActionMask` should still pass.

- [ ] **Step 5: Run integration tests**

```
pytest tests/test_environment.py::TestIntegrationWithRealData -v
```
Expected: PASS (integration tests use action 0 or action 1 — both still valid in buy mode).

- [ ] **Step 6: Commit**

```bash
git add src/environment.py tests/test_environment.py
git commit -m "feat: rewrite Environment for multi-trade buy/sell mechanics with share tracking"
```

---

## Task 4: Update normalizer for `is_sell_mode`

**Files:**
- Modify: `src/normalizer.py` — `DYNAMIC_DIM = 12`; add dim 11 (`is_sell_mode`) to `encode_dynamic()`
- Modify: `tests/test_normalizer.py` — update dimension checks, add `is_sell_mode` test

`is_sell_mode` is binary (0.0 or 1.0) — no fitting or normalization needed. When absent from the dict (e.g. in `encode_episode_dynamic` which processes raw rows), default to 0.0.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_normalizer.py`:

```python
def test_dynamic_dim_is_12(normalizer):
    """DYNAMIC_DIM is 12 after adding is_sell_mode."""
    from src.normalizer import Normalizer
    assert Normalizer.DYNAMIC_DIM == 12


def test_encode_dynamic_includes_is_sell_mode(normalizer):
    """encode_dynamic includes is_sell_mode at index 11."""
    row = {
        "up_bid": 55.0, "up_ask": 56.0, "down_bid": 44.0, "down_ask": 45.0,
        "diff_pct": 0.01, "time_to_close": 150000, "is_sell_mode": 1.0,
    }
    vec = normalizer.encode_dynamic(row)
    assert vec.shape == (12,)
    assert vec[11] == pytest.approx(1.0)


def test_encode_dynamic_is_sell_mode_buy_mode(normalizer):
    """is_sell_mode=0.0 produces 0 at index 11."""
    row = {
        "up_bid": 55.0, "up_ask": 56.0, "down_bid": 44.0, "down_ask": 45.0,
        "diff_pct": 0.01, "time_to_close": 150000, "is_sell_mode": 0.0,
    }
    vec = normalizer.encode_dynamic(row)
    assert vec[11] == pytest.approx(0.0)


def test_encode_dynamic_is_sell_mode_absent_defaults_to_zero(normalizer):
    """When is_sell_mode absent from row (e.g. raw episode row), defaults to 0.0."""
    row = {
        "up_bid": 55.0, "up_ask": 56.0, "down_bid": 44.0, "down_ask": 45.0,
        "diff_pct": 0.01, "time_to_close": 150000,
    }
    vec = normalizer.encode_dynamic(row)
    assert vec.shape == (12,)
    assert vec[11] == pytest.approx(0.0)


def test_encode_episode_dynamic_shape_12(normalizer, real_episodes):
    """encode_episode_dynamic produces (T, 12) output."""
    ep = real_episodes[0]
    result = normalizer.encode_episode_dynamic(ep)
    assert result.shape[1] == 12
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_normalizer.py -v -k "dim_is_12 or is_sell"
```
Expected: FAIL — `DYNAMIC_DIM` is still 11.

- [ ] **Step 3: Update `src/normalizer.py`**

Change `DYNAMIC_DIM = 11` → `DYNAMIC_DIM = 12`. Update the docstring. Add dim 11 at the end of `encode_dynamic()`:

```python
# is_sell_mode: index 11, binary (0.0 or 1.0), no normalization needed
vec[11] = float(row.get("is_sell_mode", 0.0))
```

Update `encode_episode_dynamic` to use `DYNAMIC_DIM` (it already does via `np.zeros((len(rows), self.DYNAMIC_DIM), ...)`), so raw rows missing `is_sell_mode` will get 0.0 at dim 11 automatically.

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_normalizer.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/normalizer.py tests/test_normalizer.py
git commit -m "feat: add is_sell_mode as dim 11 of dynamic observation (DYNAMIC_DIM 11→12)"
```

---

## Task 5: Update `trainer.py` `_run_episode`

**Files:**
- Modify: `src/trainer.py` — remove `skip_to_end()`, `_filter_pre_action()`, `_assign_reward_to_action_step()`; rewrite `_run_episode()`

In the new system ALL rows are informative — the TD learning naturally propagates credit backward. The terminal reward (at `done=True`) is the episode P&L; intermediate rewards are 0.

- [ ] **Step 1: Write the failing test**

In `tests/test_trainer.py`, add:

```python
def test_run_episode_processes_all_rows_without_skip():
    """_run_episode runs through all rows; no early termination after action."""
    import numpy as np
    from unittest.mock import MagicMock, patch
    from src.trainer import Trainer
    from src.models.lstm_dqn import LSTMDQN
    from src.normalizer import Normalizer

    step_counts = []

    # Minimal episode with 5 rows
    rows = [
        {"up_bid": 48.0, "up_ask": 52.0, "down_bid": 48.0, "down_ask": 52.0,
         "diff_pct": 0.001, "time_to_close": 270000}
    ] * 5
    ep = {
        "hour": 9, "day": 0,
        "diff_pct_prev_session": 0.01, "diff_pct_hour": 0.02,
        "avg_pct_variance_hour": 0.005,
        "outcome": "UP", "rows": rows,
    }

    normalizer = Normalizer()
    normalizer.fit([ep])
    model = LSTMDQN(lstm_hidden_size=16)
    trainer = Trainer(model=model, normalizer=normalizer)

    original_step = trainer.env.step
    def counting_step(action):
        step_counts.append(action)
        return original_step(action)

    trainer.env.step = counting_step
    trainer._run_episode(ep)
    trainer.close()

    # All 5 rows must be processed
    assert len(step_counts) == 5, f"Expected 5 steps, got {len(step_counts)}"
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_trainer.py::test_run_episode_processes_all_rows_without_skip -v
```
Expected: FAIL — current code exits after the first non-zero action.

- [ ] **Step 3: Rewrite `_run_episode` in `src/trainer.py`**

Replace `_run_episode` with:

```python
def _run_episode(
    self, episode: dict[str, Any]
) -> tuple[float, np.ndarray]:
    """Run one episode, collecting all transitions for the replay buffer.

    With multi-trade mechanics, all rows are informative. Reward is 0 for
    non-terminal steps and the episode P&L at the terminal step.

    Returns:
        Tuple of (episode_reward, action_counts array of shape (9,)).
    """
    self.env.reset(episode)
    static_features = self.normalizer.encode_static(episode)
    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)

    final_reward = 0.0

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
        else:
            final_reward = reward

        self.replay_buffer.add_transition({
            "static_features": static_features,
            "dynamic_features": dynamic_features,
            "action": action,
            "reward": reward,  # 0.0 for non-terminal, P&L at terminal
            "next_dynamic_features": next_dynamic,
            "done": done,
            "action_mask": action_mask,
            "next_action_mask": next_mask,
        })

        if done:
            break

    return final_reward, action_counts
```

Also delete the now-dead methods `_assign_reward_to_action_step()` and `_filter_pre_action()` from `trainer.py`.

> **Note on replay buffer API:** The current `_run_episode` calls `self.replay_buffer.add_episode(transitions)`. If `PrioritizedReplayBuffer` only has `add_episode` (not `add_transition`), add a thin `add_transition` wrapper that calls `add_episode` with a single-element list, **or** collect transitions in a list and call `add_episode` at the end. Check `src/replay_buffer.py` and adapt accordingly.

- [ ] **Step 4: Run trainer tests**

```
pytest tests/test_trainer.py -v
```
Expected: All PASS including new test.

- [ ] **Step 5: Run full test suite**

```
pytest tests/ -v
```
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/trainer.py tests/test_trainer.py
git commit -m "feat: update _run_episode to process all rows for multi-trade episodes"
```

---

## Task 6: Update `visibility.py` for multi-trade display

**Files:**
- Modify: `src/visibility.py` — remove `skip_to_end()` and early termination; track transaction list; show per-trade accounting and episode settlement

No automated tests for this (visual output) — verify manually with `evaluate.py`.

- [ ] **Step 1: Rewrite `run_visibility` in `src/visibility.py`**

Key changes:
- Remove the `if action != 0 and not done: env.skip_to_end()` block entirely
- Remove `episode_action`, `episode_action_price`, `is_maker_trade` single-trade tracking
- After each non-zero action: print the trade immediately (price, shares, direction, fee type)
- At episode end: print the full accounting (all trades + end-of-episode settlement)

Replace `run_visibility` with:

```python
def run_visibility(
    episodes: list[dict[str, Any]],
    player: str = "random",
    normalizer: Optional[Normalizer] = None,
    dqn_agent: Optional[DQNAgent] = None,
) -> float:
    """Run episodes with full console visibility (multi-trade)."""
    env = Environment()
    random_agent = RandomAgent() if player == "random" else None
    cumulative_profit = 0.0
    player_name = "Random Agent" if player == "random" else "Trained AI Agent"

    for ep_idx, episode in enumerate(episodes):
        env.reset(episode)

        session_id = episode.get("session_id", "N/A")
        outcome = episode["outcome"]
        start_price = episode.get("start_price")
        end_price = episode.get("end_price")
        hour = episode.get("hour")
        day = episode.get("day")
        diff_prev = episode.get("diff_pct_prev_session")
        diff_hour = episode.get("diff_pct_hour")

        start_str = f"${start_price:,.2f}" if isinstance(start_price, (int, float)) else "N/A"
        end_str = f"${end_price:,.2f}" if isinstance(end_price, (int, float)) else "N/A"
        diff_prev_str = f"{diff_prev:+.3f}%" if diff_prev is not None else "N/A"
        diff_hour_str = f"{diff_hour:+.3f}%" if diff_hour is not None else "N/A"
        day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        day_str = day_names.get(day, str(day)) if day is not None else "N/A"

        print(f"\nEpisode: {session_id} | Outcome: {outcome}")
        print(f"  Start: {start_str} | End: {end_str} | Hour: {hour} | Day: {day_str}")
        print(f"  Prev session: {diff_prev_str} | Hour trend: {diff_hour_str}")
        print(f"Player: {player_name}")
        print("-" * 72)

        if dqn_agent is not None:
            dqn_agent.reset()

        static_features = None
        if normalizer is not None:
            static_features = normalizer.encode_static(episode)

        for step in range(env.num_rows):
            row = episode["rows"][step]
            obs = env.get_observation()
            mask = env.get_action_mask()

            if player == "random":
                action = random_agent.select_action(mask)
            else:
                dynamic_features = normalizer.encode_dynamic(obs)
                action = dqn_agent.select_action(static_features, dynamic_features, mask)

            up_bid = row.get("up_bid")
            up_ask = row.get("up_ask")
            down_bid = row.get("down_bid")
            down_ask = row.get("down_ask")
            diff_pct = row.get("diff_pct")
            time_to_close = row.get("time_to_close")
            mode = "SELL" if obs["is_sell_mode"] else "BUY"

            print(
                f"Row {step:3d} [{mode}] | Time: {_format_time_left(time_to_close)} | "
                f"BTC diff: {_format_diff_pct(diff_pct)}"
            )
            print(
                f"  UP:  bid={_format_price(up_bid)} ask={_format_price(up_ask)} | "
                f"DOWN: bid={_format_price(down_bid)} ask={_format_price(down_ask)}"
            )

            if action == 0:
                print("  Action: DO NOTHING")
            else:
                price = _get_action_price(action, row)
                action_str = ACTION_NAMES[action].format(
                    price=f"{price:.0f}" if price is not None else "?"
                )
                print(f"  Action: {action_str}")

            done, reward = env.step(action)

            # Show any trade that just completed (compare trade count before/after)
            trades = env.trades
            if trades:
                last_trade = trades[-1]
                if last_trade["type"] == "buy":
                    print(
                        f"  >> BUY filled: {last_trade['shares']:.2f} {last_trade['direction']} "
                        f"shares @ {last_trade['price']:.0f}c "
                        f"({'maker' if last_trade['is_maker'] else 'taker'})"
                    )
                elif last_trade["type"] == "sell":
                    print(
                        f"  >> SELL filled: {last_trade['shares']:.2f} {last_trade['direction']} "
                        f"shares @ {last_trade['price']:.0f}c → "
                        f"{last_trade['proceeds']:.2f}c "
                        f"({'maker' if last_trade['is_maker'] else 'taker'})"
                    )

            print("-" * 72)

            if done:
                _print_episode_result(episode, reward, env.trades)
                profit_cents = reward * 500.0
                cumulative_profit += profit_cents
                print("=" * 72)
                print(
                    f"Cumulative: Episodes={ep_idx + 1} | "
                    f"Total Profit: {cumulative_profit:+.2f}c"
                )
                break

    return cumulative_profit
```

Replace `_print_episode_result` with:

```python
def _print_episode_result(
    episode: dict[str, Any],
    reward: float,
    trades: list[dict[str, Any]],
) -> None:
    """Print episode result: trade log, end settlement, total P&L."""
    from src.environment import SHARES_PER_BUY, REWARD_NORMALIZATION
    outcome = episode["outcome"]
    profit_cents = reward * REWARD_NORMALIZATION

    if not trades:
        print(f"Episode Result: {outcome} | No trades made")
        print(f"  Profit: 0.00c")
        return

    print(f"\nEpisode Result: {outcome}")
    print(f"  Trade Log:")
    net_cash = 0.0
    for t in trades:
        if t["type"] == "buy":
            cost = SHARES_PER_BUY * t["price"]
            net_cash -= cost
            tag = "maker" if t["is_maker"] else "taker"
            print(
                f"    BUY  {t['shares']:.2f} {t['direction']} @ {t['price']:.0f}c "
                f"[{tag}] cost={cost:.2f}c | running cash: {net_cash:+.2f}c"
            )
        else:
            net_cash += t["proceeds"]
            tag = "maker" if t["is_maker"] else "taker"
            print(
                f"    SELL {t['shares']:.2f} {t['direction']} @ {t['price']:.0f}c "
                f"[{tag}] proceeds={t['proceeds']:.2f}c | running cash: {net_cash:+.2f}c"
            )

    # Find shares still held (last buy without matching sell)
    # Easiest: look at the last trade — if it was a buy, shares are held
    last = trades[-1]
    if last["type"] == "buy":
        held_shares = last["shares"]
        held_dir = last["direction"]
        payout = held_shares * 100.0 if outcome == held_dir else 0.0
        print(
            f"  End settlement: {held_shares:.2f} {held_dir} shares "
            f"→ payout={payout:.2f}c (outcome={outcome})"
        )
    else:
        print(f"  End settlement: no shares held (all sold within episode)")

    print(f"  Total Profit: {profit_cents:+.2f}c")
```

- [ ] **Step 2: Manually verify with evaluate.py**

```bash
python evaluate.py --episodes 3 --player random
```

Expected: Console output shows rows with `[BUY]`/`[SELL]` mode labels, trade lines appear when executed, episode summary shows trade log + settlement.

- [ ] **Step 3: Commit**

```bash
git add src/visibility.py
git commit -m "feat: update visibility for multi-trade display with per-trade accounting"
```

---

## Task 7: Update `random_agent.py` comment

**Files:**
- Modify: `src/agents/random_agent.py` — update docstring only

- [ ] **Step 1: Update docstring in `random_agent.py`**

Replace the class docstring:

```python
class RandomAgent:
    """Uniform random action selection from valid actions.

    Selects actions based purely on the action mask from the environment.
    In buy mode, only buy actions are available. In sell mode, only sell
    actions matching the held direction are available. The mask enforces
    all trading rules — this agent needs no mode-specific logic.

    95% of the time does nothing (action 0). The remaining 5% selects
    uniformly from the valid non-zero actions.
    """
```

- [ ] **Step 2: Run agent tests to verify no regression**

```
pytest tests/test_agents.py -v
```
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add src/agents/random_agent.py
git commit -m "docs: update RandomAgent docstring for multi-trade mechanics"
```

---

## Task 8: Final verification

- [ ] **Step 1: Run full test suite**

```
pytest tests/ -v
```
Expected: All tests PASS.

- [ ] **Step 2: Run evaluate.py smoke test**

```bash
python evaluate.py --episodes 5 --player random
```
Expected: Runs without error, shows multi-trade display.

- [ ] **Step 3: Update memory file**

Update `project_grid_search_investigation.md` to reflect that Phase 2 is on hold pending this overhaul, and that the model input dimension has changed from `DYNAMIC_DIM=11` to `DYNAMIC_DIM=12`.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: trading mechanics overhaul complete — multi-trade buy/sell with share tracking"
```
