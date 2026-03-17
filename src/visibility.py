"""Console visibility mode for Polymarket BTC 5-minute RL trading agent.

Displays full row-by-row output for every episode, showing prices, actions,
trade details, episode results, and running cumulative profit.
"""

from typing import Any, Optional

import numpy as np

from src.environment import Environment, taker_fee, maker_rebate, NUM_ACTIONS
from src.normalizer import Normalizer
from src.agents.random_agent import RandomAgent
from src.agents.dqn_agent import DQNAgent

# Action names for display
ACTION_NAMES = {
    0: "DO NOTHING",
    1: "BUY UP @ {price}c (taker)",
    2: "SELL UP @ {price}c (taker)",
    3: "BUY DOWN @ {price}c (taker)",
    4: "SELL DOWN @ {price}c (taker)",
    5: "LIMIT BUY UP @ {price}c (maker)",
    6: "LIMIT SELL UP @ {price}c (maker)",
    7: "LIMIT BUY DOWN @ {price}c (maker)",
    8: "LIMIT SELL DOWN @ {price}c (maker)",
}


def _format_price(value: Optional[float]) -> str:
    """Format a bid/ask price for display."""
    if value is None:
        return "N/A"
    return f"{value:.0f}c"


def _format_time_left(ms: Optional[float]) -> str:
    """Format time_to_close in mm:ss."""
    if ms is None:
        return "??:??"
    secs = max(0, ms / 1000.0)
    minutes = int(secs // 60)
    seconds = int(secs % 60)
    return f"{minutes}m {seconds:02d}s"


def _format_diff_pct(value: Optional[float]) -> str:
    """Format diff_pct for display."""
    if value is None:
        return "N/A"
    return f"{value:+.3f}%"


def _get_action_price(action: int, row: dict[str, Any]) -> Optional[float]:
    """Get the trade price for a given action from the row data."""
    prices = {
        1: row.get("up_ask"),
        2: row.get("up_bid"),
        3: row.get("down_ask"),
        4: row.get("down_bid"),
        5: (row.get("up_ask") or 0) - 1 if row.get("up_ask") is not None else None,
        6: (row.get("up_bid") or 0) + 1 if row.get("up_bid") is not None else None,
        7: (row.get("down_ask") or 0) - 1 if row.get("down_ask") is not None else None,
        8: (row.get("down_bid") or 0) + 1 if row.get("down_bid") is not None else None,
    }
    return prices.get(action)


def run_visibility(
    episodes: list[dict[str, Any]],
    player: str = "random",
    normalizer: Optional[Normalizer] = None,
    dqn_agent: Optional[DQNAgent] = None,
) -> float:
    """Run episodes with full console visibility.

    Args:
        episodes: List of episode dicts to run.
        player: "random" or "dqn".
        normalizer: Fitted normalizer (required for DQN agent).
        dqn_agent: Trained DQN agent (required if player="dqn").

    Returns:
        Total cumulative profit in cents.
    """
    env = Environment()
    random_agent = RandomAgent() if player == "random" else None
    cumulative_profit = 0.0

    player_name = "Random Agent" if player == "random" else "Trained AI Agent"

    for ep_idx, episode in enumerate(episodes):
        env.reset(episode)

        # Episode header
        timestamp = episode.get("timestamp", "N/A")
        outcome = episode["outcome"]
        start_price = episode.get("start_price", "N/A")
        if isinstance(start_price, (int, float)):
            start_price = f"${start_price:,.2f}"

        print(f"\nEpisode: {timestamp} | Outcome: {outcome} | Price to beat: {start_price}")
        print(f"Player: {player_name}")
        print("-" * 69)

        # Reset DQN hidden state
        if dqn_agent is not None:
            dqn_agent.reset()

        static_features = None
        if normalizer is not None:
            static_features = normalizer.encode_static(episode)

        episode_action = None
        episode_action_price = None
        is_maker_trade = False
        locked = False

        for step in range(env.num_rows):
            row = episode["rows"][step]
            obs = env.get_observation()
            mask = env.get_action_mask()

            # Select action
            if player == "random":
                action = random_agent.select_action(mask)
            else:
                dynamic_features = normalizer.encode_dynamic(obs)
                action = dqn_agent.select_action(
                    static_features, dynamic_features, mask
                )

            # Display row
            up_bid = row.get("up_bid")
            up_ask = row.get("up_ask")
            down_bid = row.get("down_bid")
            down_ask = row.get("down_ask")
            diff_pct = row.get("diff_pct")
            time_to_close = row.get("time_to_close")

            print(
                f"Row {step:3d} | Time left: {_format_time_left(time_to_close)} | "
                f"BTC diff: {_format_diff_pct(diff_pct)}"
            )
            print(
                f"  UP:  bid={_format_price(up_bid)} ask={_format_price(up_ask)} | "
                f"DOWN: bid={_format_price(down_bid)} ask={_format_price(down_ask)}"
            )

            if locked:
                # Check limit order fill status
                trade_info = env.trade_info
                if trade_info and trade_info["is_maker"] and trade_info["filled"]:
                    print("  [Limit order FILLED]")
                else:
                    print("  [Locked - no action]")
            else:
                # Format action display
                if action == 0:
                    print("  Action: DO NOTHING")
                else:
                    price = _get_action_price(action, row)
                    action_str = ACTION_NAMES[action].format(
                        price=f"{price:.0f}" if price is not None else "?"
                    )
                    print(f"  Action: {action_str}")
                    episode_action = action
                    episode_action_price = price
                    is_maker_trade = action in (5, 6, 7, 8)
                    locked = True
                    print("  >>> Agent locked in. Watching remaining rows...")

            done, reward = env.step(action)

            print("-" * 69)

            if done:
                _print_episode_result(
                    episode, reward, episode_action, episode_action_price,
                    is_maker_trade, env.trade_info
                )
                cumulative_profit += reward * 100.0
                print("=" * 69)
                print(
                    f"Cumulative: Episodes={ep_idx + 1} | "
                    f"Total Profit: {cumulative_profit:+.2f}c"
                )
                break

    return cumulative_profit


def _print_episode_result(
    episode: dict[str, Any],
    reward: float,
    action: Optional[int],
    trade_price: Optional[float],
    is_maker: bool,
    trade_info: Optional[dict],
) -> None:
    """Print the episode result summary."""
    outcome = episode["outcome"]
    profit_cents = reward * 100.0

    if action is None or action == 0:
        print(f"Episode Result: {outcome} | No trade made")
        print(f"  Profit: 0.00c")
        return

    # Determine fill status for maker orders
    filled = True
    if is_maker and trade_info is not None:
        filled = trade_info["filled"]

    if is_maker and not filled:
        print(f"Episode Result: {outcome} | Limit order NOT filled")
        print(f"  Profit: 0.00c")
        return

    # Calculate fee/rebate
    if is_maker:
        fee_val = maker_rebate(trade_price)
        fee_label = "Rebate"
        fee_sign = "+"
    else:
        fee_val = taker_fee(trade_price)
        fee_label = "Fee"
        fee_sign = "-"

    is_buy = action in (1, 3, 5, 7)
    is_up = action in (1, 2, 5, 6)
    share = "UP" if is_up else "DOWN"

    if is_buy:
        payout = 100.0 if outcome == share else 0.0
        print(
            f"Episode Result: {outcome} | Payout: {payout:.0f}c | "
            f"Cost: {trade_price:.0f}c | {fee_label}: {fee_sign}{fee_val:.4f}c"
        )
    else:
        payout_owed = 100.0 if outcome == share else 0.0
        print(
            f"Episode Result: {outcome} | Received: {trade_price:.0f}c | "
            f"Owed: {payout_owed:.0f}c | {fee_label}: {fee_sign}{fee_val:.4f}c"
        )

    print(f"  Profit: {profit_cents:+.2f}c")
