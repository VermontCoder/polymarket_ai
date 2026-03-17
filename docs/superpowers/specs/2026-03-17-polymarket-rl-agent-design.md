# Polymarket BTC 5-Minute RL Trading Agent — Design Spec

## Overview

Build a reinforcement learning agent that profitably trades on Polymarket's BTC 5-minute prediction market. The agent observes market data row-by-row (2-second intervals) within each 5-minute episode and decides when and how to make a single trade to maximize profit.

## Problem Definition

- Every 5 minutes, BTC price is noted ("price to beat"). Market resolves UP or DOWN.
- Agent sees ~60-150 rows per episode, can make **at most one trade** per episode.
- 9 discrete actions: do nothing (0), 4 taker actions (1-4), 4 maker/limit order actions (5-8).
- Reward = profit/loss from trade, adjusted for Polymarket taker fees and maker rebates.
- Dataset: Currently 620 episodes, user will expand to 6,000+.
- 80/10/10 random shuffle split for train/validation/test. With 620 episodes, val and test are only 62 episodes each — thin but workable. At 6,000+ episodes, this becomes 600+ each, which is statistically robust. The proportions are standard and appropriate.

---

## 1. Data Pipeline & Normalization

### Input Features

**Static features (per episode, 35 dims):**

| Field | Raw | Encoding | Dims |
|-------|-----|----------|------|
| `hour` | 0-23 | One-hot | 24 |
| `day` | 0-6 (Mon=0) | One-hot | 7 |
| `diff_pct_prev_session` | float or null | value / training-set std, 0 if null; + `is_null` flag | 2 |
| `diff_pct_hour` | float or null | value / training-set std, 0 if null; + `is_null` flag | 2 |

**Dynamic features (per row, 11 dims):**

| Field | Raw | Encoding | Dims |
|-------|-----|----------|------|
| `up_bid` | 1-99 cents or null | value / 100, 0 if null; + `is_null` flag | 2 |
| `up_ask` | 1-99 cents or null | value / 100, 0 if null; + `is_null` flag | 2 |
| `down_bid` | 1-99 cents or null | value / 100, 0 if null; + `is_null` flag | 2 |
| `down_ask` | 1-99 cents or null | value / 100, 0 if null; + `is_null` flag | 2 |
| `diff_pct` | float or null | value / training-set std, 0 if null; + `is_null` flag | 2 |
| `time_to_close` | milliseconds | value / 300,000, clamped to [0.0, 1.0] | 1 |

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
5. **Build action mask** for this row based on null bid/ask values and `has_acted` state.
6. **Concatenate**: static features (35 dims) are constant; dynamic features (11 dims) change per row.

### Input Neuron Mapping

**Static encoder input (35 neurons):**

| Neuron Index | Field |
|-------------|-------|
| 0-23 | `hour` one-hot (index = hour value) |
| 24-30 | `day` one-hot (index 24 = Monday, ..., index 30 = Sunday) |
| 31 | `diff_pct_prev_session` normalized value |
| 32 | `diff_pct_prev_session` is_null flag |
| 33 | `diff_pct_hour` normalized value |
| 34 | `diff_pct_hour` is_null flag |

**LSTM input per timestep (11 neurons):**

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

---

## 2. Network Architecture: LSTM-DQN

```
Static Encoder:
  Input: 35 static dims
  Linear(35, 16) -> ReLU
  Output: 16-dim static embedding

Dynamic Encoder (LSTM):
  Input per timestep: 11 dynamic dims
  LSTM(input_size=11, hidden_size=32, num_layers=1)
  Output per timestep: 32-dim hidden state

Combiner + Q-Head:
  Concat: [32-dim LSTM output, 16-dim static embedding] = 48 dims
  Linear(48, 32) -> LayerNorm -> ReLU -> Dropout(0.15)
  Linear(32, 9) -> Q-values for 9 actions
```

**Total parameters: ~8,070** (slightly higher due to 11-dim LSTM input)

### Key Design Choices

- **1-layer LSTM, hidden=32**: Minimal recurrent capacity to avoid memorizing trajectories while capturing price momentum and timing.
- **LayerNorm** (not BatchNorm): Stable with RL's non-stationary targets and small batch sizes.
- **Dropout 0.15**: Light regularization; heavier dropout destabilizes DQN.
- **Static features encoded separately**: Prevents LSTM from wasting capacity re-encoding constants at every timestep.

### Action Masking

After the Q-head produces 9 values:
1. Set Q = -inf for any action whose required bid/ask is null.
2. If `has_acted == True`, set Q = -inf for all actions except action 0 (do nothing).
3. Limit order price constraints: if `bid + 1 > 99` or `ask - 1 < 1`, mask that action.

### Modular Design

An abstract base model interface allows swapping architectures:
- **LSTM-DQN** (primary): For 6K+ episodes with temporal pattern learning.
- **Stacked DQN** (baseline): Feedforward net with last-K observation stacking. Available for comparison or use with smaller datasets.

---

## 3. Environment & Episode Simulation

### Episode Flow

1. Episode starts. `has_acted = False`. LSTM hidden state reset to zeros.
2. Each timestep: environment provides observation -> agent produces Q-values -> action masking -> epsilon-greedy selection.
3. If action != 0: `has_acted = True`. Trade recorded (type, price, direction). For maker orders, record limit price.
4. Remaining timesteps: agent forced to "do nothing". For limit orders, each row checks if order fills.
5. Episode ends. Compute reward.

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
- Unfilled orders at episode end: reward = 0.

### Fee Structure (Crypto Markets)

**Taker fee formula:**
```
fee = 0.25 * price * (1 - price / 100)
```
Where `price` is in cents. Peak fee: 1.5625c at 50c. Rounded to 4 decimal places, minimum 0.0001c.

**Maker rebate:** 20% of the taker fee that would have been charged at the trade price.

### Reward Calculation

All values in cents:

**Buying a share:**
- Cost = share price (+ taker fee, or - maker rebate)
- At episode end: receive 100c if correct side, else 0c
- Reward = payout - cost

**Selling a share:**
- Received = share price (- taker fee, or + maker rebate)
- At episode end: pay 100c if the market resolves in the direction of the share you sold (e.g., sold UP and outcome=UP means you owe 100c). If the market resolves against the share you sold (e.g., sold UP and outcome=DOWN), you owe nothing.
- Reward = received - payout_owed (where payout_owed is 100c or 0c)

**No action / unfilled limit order:** Reward = 0

**Normalization:** Reward divided by 100 (max gain/loss is ~99c) to scale roughly to [-1, 1].

---

## 4. Training Algorithm: Double DQN + DRQN

### Algorithm

- **Double DQN**: Online network selects best action; target network evaluates its Q-value. Reduces overestimation.
- **DRQN-style sequence training**: Sample sub-sequences of length L from episodes. LSTM hidden state initialized to zeros at sub-sequence start ("burn-in").
- **Prioritized Experience Replay (PER)**: Prioritize transitions with large TD errors.
- Only store pre-action timesteps in the replay buffer (not the forced "do nothing" after acting).

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| Optimizer | Adam with weight_decay=1e-4 |
| Batch size | 32 |
| Gamma (discount) | 0.99 |
| Target update tau | 0.005 (soft Polyak) |
| Epsilon | 1.0 -> 0.05, linear decay over 300 episodes |
| Replay buffer size | 50,000 transitions |
| Sub-sequence length L | 20 timesteps |
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
- Early stopping: 50 eval checkpoints without improvement.

### How the Validation Set Tunes the Model

The validation set serves two purposes:

1. **Hyperparameter selection**: Each grid search configuration is evaluated on the validation set. The configuration producing the highest median validation profit (across 3 seeds) is selected. This tunes learning rate, LSTM hidden size, sub-sequence length, and epsilon decay schedule.

2. **Early stopping**: During training, validation profit is checked periodically. If it stops improving, training halts — preventing overfitting to the training data. The checkpoint with the best validation profit is kept as the final model.

3. **Structural decisions** (layer count, dropout rate, static encoder size) are fixed by design based on the dataset size and problem constraints. They are not tuned via the validation set because the search space would be too large relative to the data.

The **test set** is held out entirely and used only for final evaluation — the total profit the agent achieves across all test episodes. This is the unbiased estimate of real-world performance.

### TensorBoard Logging

- Training loss per step
- Q-value distribution
- Epsilon schedule
- Per-episode reward
- Validation profit at each checkpoint
- Gradient norms
- Action distribution (how often each action is chosen)

---

## 5. Console Visibility Mode

### Display Format

Full row-by-row output for **every episode**, even in multi-episode runs:

```
Episode: 2026-03-14T17:20:00Z | Outcome: UP | Price to beat: $70,679.78
Player: Random Agent
---------------------------------------------------------------------
Row 12 | Time left: 2m 01s | BTC diff: +0.018%
  UP:  bid=65c ask=66c | DOWN: bid=34c ask=35c  (N/A shown for null prices)
  Action: DO NOTHING
---------------------------------------------------------------------
Row 13 | Time left: 1m 59s | BTC diff: +0.021%
  UP:  bid=66c ask=67c | DOWN: bid=33c ask=34c
  Action: BUY UP @ 67c (taker)
  >>> Agent locked in. Watching remaining rows...
---------------------------------------------------------------------
Row 14 | Time left: 1m 57s | BTC diff: +0.019%
  UP:  bid=65c ask=66c | DOWN: bid=34c ask=35c
  [Locked - no action]
  ...
---------------------------------------------------------------------
Episode Result: UP | Payout: 100c | Cost: 67c | Fee: -0.55c
  Profit: +32.45c
=====================================================================
Cumulative: Episodes=1 | Total Profit: +32.45c
```

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

### test_environment.py
- At most 0 or 1 action per episode (never 2+)
- Action masking blocks null bid/ask actions
- Action masking blocks all non-zero actions after agent has acted
- Taker fee matches Polymarket formula at various prices (1c, 25c, 50c, 75c, 99c)
- Maker rebate = 20% of taker fee
- Limit order fills when market price reaches order price
- Limit order does NOT fill when market price doesn't reach order price
- Reward = 0 for no-action and unfilled limit orders
- All 8 trade/outcome combinations produce correct profit/loss
- Limit order price boundary checks (ask-1 >= 1, bid+1 <= 99)

### test_anti_cheat.py
- Agent cannot access `outcome`, `end_price`, `current_price`, `diff_usd`, `start_price`, `session_id`, `timestamp`, or future rows at decision time
- Agent only sees the 10 allowed fields: hour, day, diff_pct_prev_session, diff_pct_hour, up_bid, up_ask, down_bid, down_ask, diff_pct, time_to_close

### test_agents.py
- Random agent only selects from unmasked actions
- Random agent takes at most 1 action per episode
- DQN agent produces valid action selections

### test_replay_buffer.py
- PER correctly prioritizes high-TD-error transitions
- Sub-sequence sampling respects episode boundaries
- Post-action timesteps are excluded

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
│   │   └── stacked_dqn.py     # Stacked DQN (baseline)
│   ├── replay_buffer.py        # PER with DRQN-style sequence sampling
│   ├── trainer.py              # Training loop, Double DQN, TensorBoard
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
│   └── test_anti_cheat.py
├── train.py                     # Training entry point
├── evaluate.py                  # Visibility mode / evaluation entry point
└── requirements.txt             # torch, tensorboard, pytest, numpy
```
