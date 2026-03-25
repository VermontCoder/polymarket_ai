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
# Environment
# ---------------------------------------------------------------------------

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

    def __init__(self, inactivity_penalty: float = 0.0) -> None:
        """
        Args:
            inactivity_penalty: Reward penalty (in cents per share) applied when
                the agent makes no completed trades in an episode. Converted to
                normalized units via SHARES_PER_BUY / REWARD_NORMALIZATION.
                Default 0.0 (no penalty).
        """
        self._inactivity_penalty = inactivity_penalty

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
            Tuple of (done, reward).

        Reward is 0.0 for every non-terminal step. At the terminal step,
        reward equals the normalised episode P&L: (net_cash + end_payout)
        / REWARD_NORMALIZATION. The trainer redistributes this terminal
        reward uniformly across all steps so every row receives equal
        feedback proportional to the final outcome.
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

        reward = self._compute_final_reward() if done else 0.0
        return done, reward

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
        """Compute episode reward at end: (net_cash + end_payout) / REWARD_NORMALIZATION.

        If the agent made no completed trades and inactivity_penalty > 0, a
        penalty of (inactivity_penalty * SHARES_PER_BUY) / REWARD_NORMALIZATION
        is subtracted from the reward to discourage pure do-nothing policies.
        """
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

        # Apply inactivity penalty when no trade was ever completed
        if not self._trades and self._inactivity_penalty > 0.0:
            total_pnl -= self._inactivity_penalty * SHARES_PER_BUY

        return total_pnl / REWARD_NORMALIZATION
