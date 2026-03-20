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
        session_id = episode.get("session_id", "N/A")
        outcome = episode["outcome"]
        start_price = episode.get("start_price")
        end_price = episode.get("end_price")
        hour = episode.get("hour")
        day = episode.get("day")
        diff_prev = episode.get("diff_pct_prev_session")
        diff_hour = episode.get("diff_pct_hour")
        avg_var_hour = episode.get("avg_pct_variance_hour")

        start_str = f"${start_price:,.2f}" if isinstance(start_price, (int, float)) else "N/A"
        end_str = f"${end_price:,.2f}" if isinstance(end_price, (int, float)) else "N/A"
        diff_prev_str = f"{diff_prev:+.3f}%" if diff_prev is not None else "N/A"
        diff_hour_str = f"{diff_hour:+.3f}%" if diff_hour is not None else "N/A"
        avg_var_str = f"{avg_var_hour:.3f}%" if avg_var_hour is not None else "N/A"
        day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        day_str = day_names.get(day, str(day)) if day is not None else "N/A"

        print(f"\nEpisode: {session_id} | Outcome: {outcome}")
        print(f"  Start: {start_str} | End: {end_str} | Hour: {hour} | Day: {day_str}")
        print(f"  Prev session: {diff_prev_str} | Hour trend: {diff_hour_str} | Hour variance: {avg_var_str}")
        print(f"Player: {player_name}")
        print("-" * 69)

        # Reset DQN hidden state
        if dqn_agent is not None:
            dqn_agent.reset()

        static_features = None
        if normalizer is not None:
            static_features = normalizer.encode_static(episode)

        prev_trade_count = 0

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
            mode_str = "SELL" if obs.get("is_sell_mode") else "BUY "

            print(
                f"Row {step:3d} | {_format_time_left(time_to_close)} | "
                f"diff: {_format_diff_pct(diff_pct)} | Mode: {mode_str}"
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
                if action in (5, 6, 7, 8):
                    print("  [Limit order placed - scanning future rows...]")

            done, reward = env.step(action)

            # Show any limit order fills that occurred on this step
            new_trades = env.trades[prev_trade_count:]
            for trade in new_trades:
                if trade["is_maker"]:
                    if trade["type"] == "buy":
                        print(
                            f"  *** LIMIT FILLED: Bought {trade['shares']:.2f} "
                            f"{trade['direction']} shares @ {trade['price']:.0f}c ***"
                        )
                    else:
                        print(
                            f"  *** LIMIT FILLED: Sold {trade['shares']:.2f} "
                            f"{trade['direction']} shares @ {trade['price']:.0f}c "
                            f"-> proceeds={trade['proceeds']:.4f}c ***"
                        )
            prev_trade_count = len(env.trades)

            print("-" * 69)

            if done:
                _print_episode_result(episode, reward, env.trades)
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
    trades: list[dict[str, Any]],
) -> None:
    """Print the episode result summary using the completed trade list."""
    outcome = episode["outcome"]
    profit_cents = reward * 100.0

    if not trades:
        print(f"Episode Result: {outcome} | No trades made")
        print(f"  Profit: 0.00c")
        return

    print(f"Episode Result: {outcome} | {len(trades)} completed trade(s)")

    for i, trade in enumerate(trades):
        t_type = trade["type"].upper()
        direction = trade["direction"]
        price = trade["price"]
        fee_type = "maker" if trade["is_maker"] else "taker"

        if trade["type"] == "buy":
            shares = trade["shares"]
            print(
                f"  Trade {i + 1}: {t_type} {direction} @ {price:.0f}c "
                f"({fee_type}) -> {shares:.2f} shares"
            )
        else:
            shares = trade["shares"]
            proceeds = trade["proceeds"]
            print(
                f"  Trade {i + 1}: {t_type} {direction} @ {price:.0f}c "
                f"({fee_type}), {shares:.2f} shares -> {proceeds:.4f}c"
            )

    print(f"  Profit: {profit_cents:+.2f}c (reward: {reward:.4f})")
