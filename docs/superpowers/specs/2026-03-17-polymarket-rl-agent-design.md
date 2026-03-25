# Polymarket BTC 5-Minute RL Trading Agent — Design Spec

## Overview

Build a reinforcement learning agent that profitably trades on Polymarket's BTC 5-minute prediction market. The agent observes market data row-by-row (2-second intervals) within each 5-minute episode and makes buy/sell decisions to maximize profit. Multiple trades are allowed per episode.

## Problem Definition

- Every 5 minutes, BTC price is noted ("price to beat"). Market resolves UP or DOWN.
- Agent sees ~60-150 rows per episode. It may make multiple trades per episode (one action per row).
- 9 discrete actions: do nothing (0), 4 taker actions (1-4), 4 maker/limit order actions (5-8).
- Reward = episode P&L normalized by 500, redistributed uniformly across all rows.
- Dataset: 1,115+ episodes, expanding to 6,000+.
- 80/10/10 random shuffle split for train/validation/test.

---

## 1. Data Pipeline & Normalization

### Input Features

**Static features (per episode, 37 dims):**

| Field | Raw | Encoding | Dims |
|-------|-----|----------|------|
| `hour` | 0-23 | One-hot | 24 |
| `day` | 0-6 (Mon=0) | One-hot | 7 |
| `diff_pct_prev_session` | float or null | value / training-set std, 0 if null; + `is_null` flag | 2 |
| `diff_pct_hour` | float or null | value / training-set std, 0 if null; + `is_null` flag | 2 |
| `avg_pct_variance_hour` | float or null | value / training-set std, 0 if null; + `is_null` flag | 2 |

**Dynamic features (per row, 12 dims):**

| Field | Raw | Encoding | Dims |
|-------|-----|----------|------|
| `up_bid` | 1-99 cents or null | value / 100, 0 if null; + `is_null` flag | 2 |
| `up_ask` | 1-99 cents or null | value / 100, 0 if null; + `is_null` flag | 2 |
| `down_bid` | 1-99 cents or null | value / 100, 0 if null; + `is_null` flag | 2 |
| `down_ask` | 1-99 cents or null | value / 100, 0 if null; + `is_null` flag | 2 |
| `diff_pct` | float or null | value / training-set std, 0 if null; + `is_null` flag | 2 |
| `time_to_close` | milliseconds | value / 300,000, clamped to [0.0, 1.0] | 1 |
| `is_sell_mode` | synthesized | 1.0 if agent holds shares, 0.0 otherwise | 1 |

### Normalization Rules

- Standard deviation normalization computed on **training set only**, then applied to validation and test sets. Note: the claude.md originally suggested min-max normalization to [-1, 1] for percentage fields, but std-dev normalization was chosen instead as it is more robust to outliers and standard practice in deep learning. User approved this change.
- Null metadata: value = 0, `is_null` = 1.
- Null bid/ask: value = 0, `is_null` = 1. Corresponding action is **masked** (Q-value set to -inf).
- Null `diff_pct`: value = 0, `is_null` = 1.
- `time_to_close`: clamped to [0.0, 1.0] after normalization (some values slightly exceed 300,000ms).

### Forbidden Fields (Excluded from Observations)

The following fields exist in the raw data but must **never** be exposed to the agent, as they would allow cheating:
- `outcome` — the episode result (what we're trying to predict)
- `end_price` — the BTC price at episode end
- `current_price` — absolute BTC price (combined with `start_price`, reveals outcome)
- `diff_usd` — dollar-denominated price change (redundant with diff_pct, reveals absolute price scale)
- `start_price` — the "price to beat" in dollars
- `session_id` — episode identifier (could leak temporal information)
- `timestamp` — absolute time (hour/day and time_to_close suffice)

---

### Step-by-Step Data Cleaning Pipeline

For each episode, before the AI sees any data:

1. **Load episode** from JSON. Extract metadata and rows.
2. **Strip forbidden fields**: Remove `outcome`, `end_price`, `current_price`, `diff_usd`, `start_price`, `session_id`, `timestamp` from the observation.
3. **Encode static features**:
   - `hour` -> 24-dim one-hot vector
   - `day` -> 7-dim one-hot vector
   - `diff_pct_prev_session` -> if null: (0.0, 1.0); else: (value / train_std, 0.0)
   - `diff_pct_hour` -> if null: (0.0, 1.0); else: (value / train_std, 0.0)
4. **For each row, encode dynamic features**:
   - `up_bid` -> if null: (0.0, 1.0); else: (value / 100, 0.0)
   - `up_ask` -> if null: (0.0, 1.0); else: (value / 100, 0.0)
   - `down_bid` -> if null: (0.0, 1.0); else: (value / 100, 0.0)
   - `down_ask` -> if null: (0.0, 1.0); else: (value / 100, 0.0)
   - `diff_pct` -> if null: (0.0, 1.0); else: (value / train_std, 0.0)
   - `time_to_close` -> clamp(value / 300000, 0.0, 1.0)
   - `is_sell_mode` -> 1.0 if agent currently holds shares, 0.0 otherwise
5. **Build action mask** for this row based on null bid/ask values, `shares_owned`, `share_direction`, and `pending_limit` state.
6. **Concatenate**: static features (37 dims) are constant; dynamic features (12 dims) change per row.

### Input Neuron Mapping

**Static encoder input (37 neurons):**

| Neuron Index | Field |
|-------------|-------|
| 0-23 | `hour` one-hot (index = hour value) |
| 24-30 | `day` one-hot (index 24 = Monday, ..., index 30 = Sunday) |
| 31 | `diff_pct_prev_session` normalized value |
| 32 | `diff_pct_prev_session` is_null flag |
| 33 | `diff_pct_hour` normalized value |
| 34 | `diff_pct_hour` is_null flag |
| 35 | `avg_pct_variance_hour` normalized value |
| 36 | `avg_pct_variance_hour` is_null flag |

**LSTM input per timestep (12 neurons):**

| Neuron Index | Field |
|-------------|-------|
| 0 | `up_bid` / 100 |
| 1 | `up_bid` is_null flag |
| 2 | `up_ask` / 100 |
| 3 | `up_ask` is_null flag |
| 4 | `down_bid` / 100 |
| 5 | `down_bid` is_null flag |
| 6 | `down_ask` / 100 |
| 7 | `down_ask` is_null flag |
| 8 | `diff_pct` normalized value |
| 9 | `diff_pct` is_null flag |
| 10 | `time_to_close` normalized and clamped |
| 11 | `is_sell_mode` (1.0 = holding shares, 0.0 = looking to buy) |

---

## 2. Network Architecture: LSTM-DQN

```
Static Encoder:
  Input: 37 static dims
  Linear(37, 16) -> ReLU
  Output: 16-dim static embedding

Dynamic Encoder (LSTM):
  Input per timestep: 12 dynamic dims
  LSTM(input_size=12, hidden_size=32, num_layers=1)
  Output per timestep: 32-dim hidden state

Combiner + Q-Head:
  Concat: [32-dim LSTM output, 16-dim static embedding] = 48 dims
  Linear(48, 32) -> LayerNorm -> ReLU -> Dropout(0.15)
  Linear(32, 9) -> Q-values for 9 actions
```

**Total parameters: ~8,200**

### Key Design Choices

- **1-layer LSTM, hidden=32**: Minimal recurrent capacity to avoid memorizing trajectories while capturing price momentum and timing.
- **LayerNorm** (not BatchNorm): Stable with RL's non-stationary targets and small batch sizes.
- **Dropout 0.15**: Light regularization; heavier dropout destabilizes DQN.
- **Static features encoded separately**: Prevents LSTM from wasting capacity re-encoding constants at every timestep.

### Action Masking

After the Q-head produces 9 values:
1. Set Q = -inf for any action whose required bid/ask is null.
2. If `pending_limit` is set, set Q = -inf for all actions except action 0 (do nothing).
3. If `shares_owned == 0` (buy mode): mask all sell actions (2, 4, 6, 8).
4. If `shares_owned > 0` (sell mode): mask all buy actions (1, 3, 5, 7); also mask sells for the wrong direction.
5. Limit order price constraints: if `bid + 1 > 99` or `ask - 1 < 1`, mask that action.

### Modular Design

An abstract base model interface allows swapping architectures:
- **LSTM-DQN** (primary): For 6K+ episodes with temporal pattern learning.
- **Stacked DQN** (baseline): Feedforward net with last-K observation stacking. Available for comparison or use with smaller datasets.

---

## 3. Environment & Episode Simulation

### Episode Flow

1. Episode starts. `shares_owned = 0.0`, `share_direction = ""`, `net_cash = 0.0`, `pending_limit = None`. LSTM hidden state reset to zeros.
2. Each timestep: environment provides observation (including `is_sell_mode`) -> agent produces Q-values -> action masking -> epsilon-greedy selection.
3. If action is a buy (1, 3, 5, 7): `shares_owned` is set; agent enters sell mode. For limit buys, a pending order is recorded.
4. If action is a sell (2, 4, 6, 8): `shares_owned` is cleared; agent returns to buy mode. For limit sells, a pending order is recorded.
5. While a limit order is pending, only action 0 is available. Each subsequent row checks if the limit fills (buy fills if ask <= order price; sell fills if bid >= order price).
6. This buy→sell cycle may repeat multiple times within the episode.
7. Episode ends. Compute terminal reward.

### The 9 Actions

| # | Action | Type | Price |
|---|--------|------|-------|
| 0 | Do nothing | -- | -- |
| 1 | Buy UP at ask | Taker | Pay `up_ask`c |
| 2 | Sell UP at bid | Taker | Receive `up_bid`c |
| 3 | Buy DOWN at ask | Taker | Pay `down_ask`c |
| 4 | Sell DOWN at bid | Taker | Receive `down_bid`c |
| 5 | Limit buy UP | Maker | Order at `up_ask - 1`c. If filled, pay that price. |
| 6 | Limit sell UP | Maker | Order at `up_bid + 1`c. If filled, receive that price. |
| 7 | Limit buy DOWN | Maker | Order at `down_ask - 1`c. If filled, pay that price. |
| 8 | Limit sell DOWN | Maker | Order at `down_bid + 1`c. If filled, receive that price. |

### Limit Order Fill Logic

- **Sell orders**: Filled if a future row's bid >= order price.
- **Buy orders**: Filled if a future row's ask <= order price.
- Fill at the **order's placed price**, not the market price at fill time.
- Unfilled orders at episode end: cancelled, no cash effect.

### Fee Structure (Crypto Markets)

**Taker fee formula:**
```
fee = 0.02 * price * (1 - price / 100)
```
Where `price` is in cents. Peak fee: ~0.5c at 50c. Rounded to 4 decimal places, minimum 0.0001c.

**Maker rebate:** 20% of the taker fee that would have been charged at the trade price.

**Fee absorption into shares:** Rather than tracking fees as separate cash flows, the fee or rebate is absorbed into the effective share count. Cash outflow on a buy is always `5 * price`. The resulting shares are `5 * (1 - fee/price)` for taker or `5 * (1 + rebate/price)` for maker, rounded to 2 decimal places.

### Reward Calculation

**Terminal reward (end of episode):**
```
total_pnl = net_cash + end_payout
episode_reward = total_pnl / REWARD_NORMALIZATION  (REWARD_NORMALIZATION = 500)
```

Where:
- `net_cash` = sum of all sell proceeds minus sum of all buy costs (in cents)
- `end_payout` = `shares_owned * 100` if the outcome matches `share_direction`, else 0
- `REWARD_NORMALIZATION = 500` (5 shares × $1 maximum payout × 100¢/$)

**Uniform redistribution across all rows:**

The terminal reward is then divided evenly across all N rows in the episode. Every row in the replay buffer receives `episode_reward / N`. This ensures:
- Winning episodes: every step gets a small positive reward
- Losing episodes: every step gets a small negative reward
- Do-nothing episodes: every step gets 0

This prevents a misleading scenario where intermediate per-step rewards have the opposite polarity from the final outcome (e.g., price rises through an episode but crashes at resolution — the agent should not receive positive feedback during those steps).

**No action / unfilled limit order:** Contributes 0 to `net_cash` at episode end.

---

## 4. Training Algorithm: Double DQN + DRQN

### Algorithm

- **Double DQN**: Online network selects best action; target network evaluates its Q-value. Reduces overestimation.
- **DRQN-style sequence training**: Sample sub-sequences of length L from episodes. LSTM hidden state initialized to zeros at sub-sequence start ("burn-in").
- **Prioritized Experience Replay (PER)**: Prioritize transitions with large TD errors.
- All timesteps in an episode are stored in the replay buffer (including post-action rows).

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 (configurable via `--lr`) |
| Optimizer | Adam with weight_decay=1e-4 |
| Batch size | 32 |
| Gamma (discount) | 0.99 |
| Target update tau | 0.005 (soft Polyak, configurable via `--tau`) |
| Epsilon start | 1.0 |
| Epsilon end | 0.15 (configurable via `--epsilon-end`) |
| Epsilon decay | linear over `--epsilon-decay` episodes (default 300), cycling per epoch |
| Replay buffer size | 50,000 transitions (configurable via `--buffer-capacity`) |
| Sub-sequence length L | 20 timesteps (configurable via `--seq-len`) |
| Gradient clip norm | 1.0 |
| PER alpha | 0.6 |
| PER beta | 0.4 -> 1.0 (annealed) |

### Hyperparameter Tuning

Grid search over:
- Learning rate: {5e-5, 1e-4, 3e-4}
- Epsilon decay: {150, 300} episodes
- Sub-sequence length: {10, 20, 40}
- LSTM hidden size: {16, 32, 48}

Protocol:
- 3 seeds per configuration.
- Evaluate on validation set every 50 training episodes.
- Metric: total profit across all validation episodes.
- Select by median validation profit.

### How the Validation Set Tunes the Model

The validation set serves two purposes:

1. **Hyperparameter selection**: Each grid search configuration is evaluated on the validation set. The configuration producing the highest median validation profit (across 3 seeds) is selected. This tunes learning rate, LSTM hidden size, sub-sequence length, and epsilon decay schedule.

2. **Best checkpoint tracking**: During training, the checkpoint with the highest validation profit is saved as `checkpoint_best.pt`. This is the model used for final evaluation.

3. **Structural decisions** (layer count, dropout rate, static encoder size) are fixed by design based on the dataset size and problem constraints. They are not tuned via the validation set because the search space would be too large relative to the data.

The **test set** is held out entirely and used only for final evaluation — the total profit the agent achieves across all test episodes. This is the unbiased estimate of real-world performance.

### TensorBoard Logging

TensorBoard logging was removed in the single-run training path. Progress is tracked via the Rich terminal display and a JSONL training log instead (see single-run training design spec).

---

## 5. Console Visibility Mode

### Display Format

Full row-by-row output showing trades, prices, and fee types as they happen. Limit order fills are announced inline when they occur. Episode summary shows all completed trades with a full accounting.

### Players

- **Random Agent**: Uniform random selection from unmasked actions. For initial environment verification.
- **Trained AI Agent**: Loads model checkpoint, runs greedy inference (epsilon=0).

### Modes

- Can select player type, episode count, and specific episode IDs.
- Always verbose: all rows for all episodes displayed.
- Running cumulative profit total after each episode.

---

## 6. Test Suite

### test_normalizer.py
- Feature encoding produces expected values for known inputs
- Null metadata encodes to (0, is_null=1)
- Null bid/ask encodes to (0, is_null=1)
- Normalization uses training-set statistics only
- One-hot encoding for hour/day is correct dimensions
- `is_sell_mode` encodes correctly at dim 11

### test_environment.py
- Action masking blocks null bid/ask actions
- Action masking blocks sells when shares_owned == 0
- Action masking blocks buys when shares_owned > 0
- Action masking blocks wrong-direction sells
- Only action 0 allowed while a pending limit order is active
- Taker fee matches Polymarket formula at various prices (1c, 25c, 50c, 75c, 99c)
- Maker rebate = 20% of taker fee
- Limit order fills when market price reaches order price
- Limit order does NOT fill when market price doesn't reach order price
- Reward = 0 for no-action and unfilled limit orders
- All 8 trade/outcome combinations produce correct profit/loss
- Limit order price boundary checks (ask-1 >= 1, bid+1 <= 99)
- Multi-trade within an episode accumulates correctly

### test_anti_cheat.py
- Agent cannot access `outcome`, `end_price`, `current_price`, `diff_usd`, `start_price`, `session_id`, `timestamp`, or future rows at decision time
- Allowed observation fields: hour, day, diff_pct_prev_session, diff_pct_hour, up_bid, up_ask, down_bid, down_ask, diff_pct, time_to_close, is_sell_mode

### test_agents.py
- Random agent only selects from unmasked actions
- DQN agent produces valid action selections

### test_replay_buffer.py
- PER correctly prioritizes high-TD-error transitions
- Sub-sequence sampling respects episode boundaries

### test_trainer.py
- `_run_episode()` stores transitions for all rows
- `collect_episode()` stores transitions for all rows
- Uniform reward redistribution: all transitions get `episode_reward / N`
- `evaluate()` returns correct episode P&L (terminal reward × REWARD_NORMALIZATION / 100)

---

## 7. File Structure

```
/workspace/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Load & parse JSON episodes
│   ├── normalizer.py           # Feature normalization pipeline
│   ├── environment.py          # Episode simulation, actions, rewards, fees
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract model interface
│   │   ├── lstm_dqn.py         # LSTM-DQN (primary)
│   │   └── stacked_dqn.py      # Stacked DQN (baseline)
│   ├── replay_buffer.py        # PER with DRQN-style sequence sampling
│   ├── trainer.py              # Training loop, Double DQN
│   ├── train_display.py        # Rich terminal display for single-run training
│   ├── train_logger.py         # JSONL checkpoint logging
│   ├── rollout_worker.py       # Subprocess worker for parallel episode collection
│   ├── grid_display.py         # Rich display for grid search
│   ├── grid_utils.py           # Grid search config utilities
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── random_agent.py     # Random action selection
│   │   └── dqn_agent.py        # Trained model inference
│   └── visibility.py           # Console episode display
├── tests/
│   ├── test_normalizer.py
│   ├── test_environment.py
│   ├── test_replay_buffer.py
│   ├── test_agents.py
│   ├── test_trainer.py
│   └── test_anti_cheat.py
├── train.py                     # Training entry point
├── evaluate.py                  # Visibility mode / evaluation entry point
└── requirements.txt             # torch, rich, pytest, numpy
```
