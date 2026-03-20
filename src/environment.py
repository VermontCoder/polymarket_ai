"""Episode simulation environment for Polymarket BTC 5-minute RL trading agent.

Simulates episodes of the BTC 5-minute prediction market. Each episode has
60-150 rows (2-second intervals). The agent can make multiple trades per
episode (one per decision point), with a max of one pending limit order at a time.

9 Actions:
  0: Do nothing
  1: Buy UP at ask (taker)
  2: Sell UP at bid (taker)
  3: Buy DOWN at ask (taker)
  4: Sell DOWN at bid (taker)
  5: Limit buy UP at ask-1 (maker)
  6: Limit sell UP at bid+1 (maker)
  7: Limit buy DOWN at ask-1 (maker)
  8: Limit sell DOWN at bid+1 (maker)
"""

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Fee computation
# ---------------------------------------------------------------------------

NUM_ACTIONS = 9

# Forbidden fields that must never appear in observations
FORBIDDEN_FIELDS = frozenset({
    "outcome", "end_price", "current_price", "diff_usd",
    "start_price", "session_id", "timestamp",
})


def taker_fee(price: float) -> float:
    """Compute taker fee for a given price in cents.

    Formula: fee = 0.25 * price * (1 - price / 100)
    Minimum fee is 0.0001c. Result rounded to 4 decimal places.

    Args:
        price: Trade price in cents (1-99).

    Returns:
        Fee in cents, rounded to 4 decimal places.
    """
    fee = 0.02 * price * (1.0 - price / 100.0)
    fee = round(fee, 4)
    fee = max(fee, 0.0001)
    return fee


def maker_rebate(price: float) -> float:
    """Compute maker rebate: 20% of the taker fee at the trade price.

    Args:
        price: Trade price in cents.

    Returns:
        Rebate in cents, rounded to 4 decimal places.
    """
    return round(0.2 * taker_fee(price), 4)


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


# ---------------------------------------------------------------------------
# Action mask
# ---------------------------------------------------------------------------

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
        # Sell mode: shares_owned > 0 — must have a valid direction
        assert share_direction in ("UP", "DOWN"), (
            f"share_direction must be 'UP' or 'DOWN' when shares_owned > 0, got {share_direction!r}"
        )
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


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_reward(
    action: int,
    trade_price: float,
    outcome: str,
    is_maker: bool,
    filled: bool,
) -> float:
    """Compute the normalized reward for a completed trade.

    Args:
        action: The action taken (1-8). Must not be 0.
        trade_price: The price at which the trade executed (in cents).
        outcome: Episode outcome, "UP" or "DOWN".
        is_maker: True if this was a maker/limit order.
        filled: True if a limit order was filled (always True for taker).

    Returns:
        Reward normalized to roughly [-1, 1] range (divided by 100).
    """
    if action == 0:
        raise ValueError("compute_reward() must not be called with action=0")
    if not filled:
        return 0.0

    # Determine if this is a buy or sell, and the market direction
    is_buy = action in (1, 3, 5, 7)
    is_up_market = action in (1, 2, 5, 6)  # UP market actions
    share_direction = "UP" if is_up_market else "DOWN"

    if is_buy:
        # Cost = price + taker_fee (taker) or price - maker_rebate (maker)
        if is_maker:
            cost = trade_price - maker_rebate(trade_price)
        else:
            cost = trade_price + taker_fee(trade_price)

        # Payout = 100c if outcome matches share direction, else 0
        payout = 100.0 if outcome == share_direction else 0.0
        reward_cents = payout - cost
    else:
        # Sell: received = price - taker_fee (taker) or price + maker_rebate (maker)
        if is_maker:
            received = trade_price + maker_rebate(trade_price)
        else:
            received = trade_price - taker_fee(trade_price)

        # Payout owed = 100c if outcome matches share direction
        payout_owed = 100.0 if outcome == share_direction else 0.0
        reward_cents = received - payout_owed

    return reward_cents / 100.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Environment:
    """Episode simulation environment.

    Simulates one episode at a time. The agent steps through rows and
    can make at most one trade per episode.

    Usage:
        env = Environment()
        env.reset(episode_dict)
        while True:
            obs = env.get_observation()
            mask = env.get_action_mask()
            done, reward = env.step(action)
            if done:
                break
    """

    def __init__(self) -> None:
        self._episode: dict[str, Any] | None = None
        self._rows: list[dict[str, Any]] = []
        self._current_step: int = 0
        self._has_acted: bool = False
        self._outcome: str = ""

        # Trade state
        self._trade_action: int | None = None
        self._trade_price: float | None = None
        self._is_maker: bool = False
        self._limit_filled: bool = False
        self._limit_market: str = ""  # "UP" or "DOWN" for limit orders
        self._action_step: int = 0

    def reset(self, episode: dict[str, Any]) -> None:
        """Initialize the environment with an episode dict.

        Args:
            episode: Episode dict with keys: outcome, rows, hour, day, etc.
        """
        self._episode = episode
        self._rows = episode["rows"]
        self._current_step = 0
        self._has_acted = False
        self._outcome = episode["outcome"]

        # Reset trade state
        self._trade_action = None
        self._trade_price = None
        self._is_maker = False
        self._limit_filled = False
        self._limit_market = ""
        self._action_step = 0

    def get_observation(self) -> dict[str, Any]:
        """Return current row's data with forbidden fields stripped.

        Returns a copy of the current row dict without forbidden fields.
        Does NOT include episode-level forbidden fields either.
        """
        row = self._rows[self._current_step]
        obs = {k: v for k, v in row.items() if k not in FORBIDDEN_FIELDS}
        return obs

    def get_action_mask(self) -> np.ndarray:
        """Return boolean mask for valid actions at current step."""
        row = self._rows[self._current_step]
        return compute_action_mask(row, self._has_acted)

    def get_episode_info(self) -> dict[str, Any]:
        """Return episode-level info (non-forbidden fields) for the agent.

        This provides static features like hour, day, diff_pct_prev_session,
        diff_pct_hour.
        """
        assert self._episode is not None
        return {
            k: v for k, v in self._episode.items()
            if k not in FORBIDDEN_FIELDS and k != "rows"
        }

    @property
    def has_acted(self) -> bool:
        return self._has_acted

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def num_rows(self) -> int:
        return len(self._rows)

    @property
    def trade_info(self) -> dict[str, Any] | None:
        """Return trade details if a trade has been made, else None."""
        if self._trade_action is None:
            return None
        return {
            "action": self._trade_action,
            "price": self._trade_price,
            "is_maker": self._is_maker,
            "filled": self._limit_filled if self._is_maker else True,
        }

    def step(self, action: int) -> tuple[bool, float]:
        """Process one timestep.

        Args:
            action: Integer 0-8.

        Returns:
            Tuple of (done, reward).
            reward is only meaningful when done=True.
        """
        assert 0 <= action < NUM_ACTIONS, f"Invalid action: {action}"

        # Validate action against mask
        mask = self.get_action_mask()
        assert mask[action], (
            f"Action {action} is masked at step {self._current_step}"
        )

        row = self._rows[self._current_step]

        if action != 0 and not self._has_acted:
            self._has_acted = True
            self._trade_action = action
            self._execute_action(action, row)

        # For pending limit orders, check fill on current row
        # (this also checks on the row where the order was placed,
        #  but only AFTER the order is placed, and fill logic checks
        #  future rows. We check on subsequent steps.)
        if self._is_maker and not self._limit_filled and self._has_acted:
            # Only check fill on rows AFTER the action was taken
            if self._current_step > self._action_step:
                self._check_limit_fill(row)

        # Advance step
        self._current_step += 1

        # Check if episode is done
        done = self._current_step >= len(self._rows)

        if done:
            reward = self._compute_final_reward()
            return True, reward

        return False, 0.0

    def _execute_action(self, action: int, row: dict[str, Any]) -> None:
        """Record trade details for the selected action."""
        self._action_step = self._current_step

        if action in (1, 2, 3, 4):
            # Taker actions
            self._is_maker = False
            if action == 1:
                self._trade_price = row["up_ask"]
            elif action == 2:
                self._trade_price = row["up_bid"]
            elif action == 3:
                self._trade_price = row["down_ask"]
            elif action == 4:
                self._trade_price = row["down_bid"]

        elif action in (5, 6, 7, 8):
            # Maker/limit actions
            self._is_maker = True
            self._limit_filled = False
            if action == 5:
                self._trade_price = row["up_ask"] - 1
                self._limit_market = "UP"
            elif action == 6:
                self._trade_price = row["up_bid"] + 1
                self._limit_market = "UP"
            elif action == 7:
                self._trade_price = row["down_ask"] - 1
                self._limit_market = "DOWN"
            elif action == 8:
                self._trade_price = row["down_bid"] + 1
                self._limit_market = "DOWN"

    def _check_limit_fill(self, row: dict[str, Any]) -> None:
        """Check if a pending limit order fills on this row."""
        action = self._trade_action
        price = self._trade_price

        if action in (5, 7):
            # Buy limit: fills if future ask <= order price
            field = "up_ask" if action == 5 else "down_ask"
            market_price = row.get(field)
            if market_price is not None and market_price <= price:
                self._limit_filled = True

        elif action in (6, 8):
            # Sell limit: fills if future bid >= order price
            field = "up_bid" if action == 6 else "down_bid"
            market_price = row.get(field)
            if market_price is not None and market_price >= price:
                self._limit_filled = True

    def skip_to_end(self) -> float:
        """Skip remaining rows and return the final reward.

        Call immediately after a non-zero action has been taken to avoid
        running the model on rows where it can only choose "do nothing".

        For taker actions: O(1) — reward is fully determined by the already-
        recorded trade price and the episode outcome.
        For maker/limit actions: scans remaining rows for a fill check (no
        model inference needed), then computes the final reward.

        Returns:
            Final episode reward (same value as stepping through all rows).
        """
        assert self._has_acted, "skip_to_end() requires has_acted=True"

        if self._is_maker:
            # Scan remaining rows for limit-fill checks only.
            # Once filled, reward is fully determined — no need to continue.
            while self._current_step < len(self._rows):
                self._check_limit_fill(self._rows[self._current_step])
                self._current_step += 1
                if self._limit_filled:
                    self._current_step = len(self._rows)
                    break
        else:
            # Taker: reward is deterministic immediately, no scan needed
            self._current_step = len(self._rows)

        return self._compute_final_reward()

    def _compute_final_reward(self) -> float:
        """Compute reward at episode end."""
        if self._trade_action is None or self._trade_action == 0:
            return 0.0

        filled = not self._is_maker or self._limit_filled

        return compute_reward(
            action=self._trade_action,
            trade_price=self._trade_price,
            outcome=self._outcome,
            is_maker=self._is_maker,
            filled=filled,
        )
