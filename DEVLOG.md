# polymarket_ai — Development Container Setup Log
**Date:** 2026-03-16

---

## Project Goal

Build an AI system to make predictions on Polymarket's 5- or 15-minute Bitcoin/Ethereum markets using an Nvidia GPU. Development is done locally first; the container will later be deployed to vast.ai cloud GPU instances.

---

## What Was Built

A Docker development environment extending the `vastai/pytorch` base image with the following additions:

| Addition | Purpose |
|----------|---------|
| `openssh-server` | SSH access for terminal and VS Code Remote SSH |
| `git` | Version control inside container |
| `nodejs` + `npm` | Required for Claude Code CLI |
| `@anthropic-ai/claude-code` v2.1.76 | Claude Code CLI installed globally |
| `setup-ssh-key.sh` | Injects `SSH_PUBLIC_KEY` env var into `authorized_keys` at startup |
| `supervisord-extras.conf` | Runs SSH key injection + `sshd` via Supervisor on boot |
| `NVIDIA_VISIBLE_DEVICES=all` | Forces GPU visibility regardless of launch flags |
| `NVIDIA_DRIVER_CAPABILITIES=all` | Enables full GPU capability passthrough |
| `test_pytorch_gpu.py` | Copied to `/workspace/` for GPU verification |

---

## Files Created

```
polymarket_ai/
├── Dockerfile                  # Main container definition
├── supervisord-extras.conf     # Supervisor configs for sshd + SSH key injection
├── setup-ssh-key.sh            # Reads SSH_PUBLIC_KEY env var → authorized_keys
├── test_pytorch_gpu.py         # GPU verification script (pre-existing)
├── .env.example                # Template for runtime environment variables
├── .gitignore                  # Excludes .env, data files, model weights, etc.
└── DEVLOG.md                   # This file
```

---

## Verification Results

All checks passed on first full run:

```
PyTorch version: 2.5.1+cu121
CUDA available: True
GPU count: 1
  GPU 0: NVIDIA GeForce RTX 4060
    Total memory: 8.0 GB
    Compute capability: 8.9

Tensor device: cuda:0
Matrix multiply result shape: torch.Size([1000, 1000])
Result sample value: 28.4239

PASS: GPU is working correctly.
```

- SSH daemon: running via Supervisor
- Claude Code: v2.1.76 installed
- Ports: 22 (SSH), 6006 (TensorBoard), 18080 (Jupyter)

---

## How to Launch the Container Locally

```bash
# Build
docker build -t polymarket-ai-dev .

# Run (substitute your actual public key)
PUB_KEY=$(cat ~/.ssh/id_rsa.pub)
docker run --gpus all -d --name polymarket-dev \
  -p 2222:22 \
  -p 6006:6006 \
  -e SSH_PUBLIC_KEY="$PUB_KEY" \
  polymarket-ai-dev

# Verify GPU
docker exec polymarket-dev bash -c '. /venv/main/bin/activate && python /workspace/test_pytorch_gpu.py'

# Stop and clean up
docker stop polymarket-dev && docker rm polymarket-dev
```

---

## SSH / VS Code Remote SSH Config

Entry in `~/.ssh/config`:

```
Host polymarket_dev
    HostName 127.0.0.1
    Port 2222
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

`StrictHostKeyChecking no` prevents host key mismatch errors when the container is rebuilt (Docker generates new SSH host keys each time). Safe for localhost dev use.

---

## Issues Encountered & Resolutions

### 1. GPU not enabled on prior vast.ai attempt
**Cause:** Container was launched without `--gpus all` flag, and no GPU ENVs were set.
**Fix:** Added `NVIDIA_VISIBLE_DEVICES=all` and `NVIDIA_DRIVER_CAPABILITIES=all` to the Dockerfile `ENV`. These act as a safety net regardless of how the container is launched.

### 2. SSH host key mismatch error (VS Code)
**Error:** `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED`
**Cause:** Each new container generates fresh SSH host keys. The old key from a prior run was cached in `known_hosts`.
**Fix:** Added `StrictHostKeyChecking no` + `UserKnownHostsFile /dev/null` to `~/.ssh/config` for the `polymarket_dev` host. Cleared the stale entry with `ssh-keygen -R "[127.0.0.1]:2222"`.

### 3. SSH password prompt (VS Code)
**Cause:** `SSH_PUBLIC_KEY` injection is a vast.ai platform feature — it doesn't fire automatically when running Docker locally. The base image's `propagate_ssh_keys.sh` script reads from `/root/.ssh/authorized_keys` which was empty.
**Fix:** Added `setup-ssh-key.sh` to read `SSH_PUBLIC_KEY` env var and write it to `/root/.ssh/authorized_keys` at container startup via Supervisor.

### 4. VS Code server download failing (503)
**Error:** `wget download failed … ERROR 503: Service Unavailable`
**Cause:** VS Code 1.111.0 (commit `ce099c1`) was newly released. Microsoft's CDN had not yet propagated the server binaries for this commit. Both the in-container download and the client-side fallback download returned 503.
**Status:** Not a container issue. Will resolve once Microsoft deploys the artifacts. Test by visiting the server download URL in a browser; if it downloads instead of erroring, VS Code Remote SSH will work immediately.

---

## Insights

**Base image contents:** `vastai/pytorch:latest` ships with PyTorch 2.5.1 + CUDA 12.1 pre-installed — much newer than the Docker Hub docs suggested (which showed CUDA 10.0). No need to install any ML libraries; they're already present in `/venv/main/`.

**GPU passthrough:** The "GPU not enabled" class of problems is almost always a launch-time flag issue (`--gpus all`), not a container config problem. Setting `NVIDIA_VISIBLE_DEVICES=all` in the Dockerfile makes the intent explicit and acts as a safety net.

**SSH key injection:** vast.ai's SSH key injection (`SSH_PUBLIC_KEY`) is platform-level infrastructure — it happens outside the container via the vast.ai agent, not inside it. Running the same image locally requires a startup script to replicate this behavior. The `setup-ssh-key.sh` + Supervisor approach mirrors how vast.ai does it in production.

**VS Code server download:** VS Code Remote SSH installs a server binary on the remote host at connection time, matched exactly to the local VS Code commit ID. If Microsoft's CDN doesn't have that artifact yet (common immediately after a release), connections fail. Pre-installing the server in the Dockerfile is possible but tightly couples the image to a specific VS Code version, so waiting for CDN propagation is preferable.

**Python venv in vast.ai images:** Always use `. /venv/main/bin/activate && python` instead of bare `python` inside the container. The base image's PyTorch lives in the venv, not the system Python. Running bare `python` will resolve to a system Python without CUDA-enabled torch.

**Supervisor priority:** `priority=5` on the SSH key injection script vs `priority=10` on `sshd` ensures the key is written before the SSH daemon starts accepting connections — important because VS Code Remote SSH connects immediately after the container reports ready.

---

## Session: 2026-03-20 — Trading Mechanics Overhaul (Complete)

All 8 tasks from `docs/superpowers/plans/2026-03-20-trading-mechanics-overhaul.md` were executed and committed. 251 tests pass.

### What Changed

| File | Change |
|------|--------|
| `src/environment.py` | Deleted `compute_reward()` and `skip_to_end()`. Replaced single-trade `has_acted` state with multi-trade `shares_owned / share_direction / net_cash / pending_limit`. `get_observation()` now appends `is_sell_mode`. Reward = `(net_cash + end_payout) / 500`. |
| `src/normalizer.py` | `DYNAMIC_DIM` 11→12. Added `is_sell_mode` at dim 11 in `encode_dynamic()`. |
| `src/replay_buffer.py` | `DYNAMIC_DIM` 11→12. |
| `src/trainer.py` | Removed `skip_to_end()` from `evaluate()`. `_run_episode()` now processes all rows (no early termination). Deleted `_assign_reward_to_action_step()` and `_filter_pre_action()`. |
| `src/models/lstm_dqn.py` | Default `dynamic_dim` 11→12. |
| `src/models/stacked_dqn.py` | Default `dynamic_dim` 11→12. |
| `src/visibility.py` | Removed early termination. All rows processed. Per-trade display inline (shares, price, fee type). Limit order fills announced when they happen. Episode summary shows all completed trades. |
| `src/agents/random_agent.py` | Updated docstring to reflect multi-trade mechanics. |
| `tests/test_trainer.py` | Replaced old early-termination tests with `TestRunEpisodeAllRows` + `TestEvaluateAllRows`. |
| `tests/test_*` | All `11` dimension references updated to `12`. `ALLOWED_ROW_FIELDS` in `test_anti_cheat.py` now includes `is_sell_mode`. |

### Commits

| Commit | Description |
|--------|-------------|
| `7cd563c` | Environment class rewrite + test suite |
| `d4dfad9` | Normalizer DYNAMIC_DIM 11→12, is_sell_mode |
| `a4ee212` | trainer/replay_buffer/models all-rows processing + DYNAMIC_DIM propagation |
| `d7670cf` | visibility.py per-trade display + random_agent docstring |

---

---

## Session: 2026-03-25 — Reward Function Redesign

Replaced the mark-to-market (MTM) dense reward shaping with uniform episode reward redistribution. 277 tests pass.

### Problem

Two related bugs were discovered:

1. **Broken `evaluate()`**: When MTM dense rewards were added, `evaluate()` and `evaluate_with_actions()` were never updated. They captured only the terminal step reward, which with MTM was a tiny correction term rather than the full P&L. Validation profit showed near-zero regardless of agent quality — model selection and early stopping were effectively non-functional.

2. **Misleading training signal**: MTM rewards gave positive intermediate rewards whenever a held position's bid price rose, even in episodes that ultimately resolved as losses. For example: agent buys UP at 50¢, price drifts to 95¢ over 80 steps (positive MTM rewards each step), then BTC drops below the price-to-beat at resolution — final payout is 0. The agent received ~80 positive training signals for what was ultimately a losing trade. The correct feedback polarity (negative) appeared only at the last step.

### Fix

**`src/environment.py`**:
- Removed `_prev_portfolio_value` state variable and `_portfolio_value_at()` method entirely.
- `step()` reward block simplified to: `reward = self._compute_final_reward() if done else 0.0`

**`src/trainer.py`** (`_run_episode()` and `collect_episode()`):
- After the episode loop, the terminal reward is redistributed uniformly across all N rows:
  ```python
  per_step = episode_reward / n
  for t in transitions:
      t["reward"] = per_step
  ```
- Every row in a winning episode gets `+pnl/N`; every row in a losing episode gets `-loss/N`; do-nothing episodes get 0 everywhere.

`evaluate()` and `evaluate_with_actions()` required no changes — with sparse terminal reward, the terminal step's reward is the full P&L/500, which these functions already read correctly.

### Documentation Updated

- `claude.md` — multi-trade mechanics, reward redistribution description
- `docs/superpowers/specs/2026-03-17-polymarket-rl-agent-design.md` — reward calculation section, static dims (37), dynamic dims (12 with `is_sell_mode`), episode flow, action masking
- `docs/superpowers/specs/2026-03-22-single-run-training-design.md` — added `--epsilon-end`, `--tau`, `--buffer-capacity` to CLI args table; corrected default values

---

## Major discovery as of 3/20/2026. 

Polymarket does not allow selling before the acquisition of shares. So the code needs to be modified such that you cannot sell shares unless you've bought them previously in the episode. Also, since now you can only sell if you've bought, we are going to remove the restriction that you can only take one action per episode. That would limit us to only buying, which maybe isn't the best restriction.

So now, rather than having one transaction possible per episode, there is now no limit besides a single transaction per row of episode. However, in order to keep things simple, the first transaction must be a buy transaction. Then, the A.I. is permitted in later parts of the episode only to sell. It cannot buy more shares. And it must sell all the shares that it bought previously. If it does sell all the shares, once again in a later row it can once again buy five shares, but seeing as it does not have any shares to sell, it cannot sell. Once it has purchased the five shares, if it decides to do so, it then can only sell them again. And so on. 

This means that the AI will have to be assigned shares when they buy them initially. If they are a taker, they will need to receive shares commensurate with the amount of shares they would purchase with the fee being taken out of their initial purchase. During the training runs, let's adjust all the purchases to be five shares less the taker fee or plus the maker fee. For example, if the AI was a taker and bought five shares at 10 cents a piece and the fee was 5 cents, then that would result in the AI purchasing 4.5 shares. Similarly, if the AI was a maker and bought five shares at 10 cents a piece, and the maker bonus was a penny, then that would result in the AI purchasing 5.01 shares.

Of course, the A.I. can choose to do nothing on any particular row, just as before.

At the end of the episode, as happened in the past, the AI will be paid $1 per winning share or fraction thereof. if the AI is on the losing side, it will receive nothing for the shares. However, profit or loss from the sales within the episode will be counted for or against the the reward amount at the conclusion of the episode.

  ### Effects on the current codebase

  - Share tracking will have to be implemented for the AIs. And the random player.
  - There will need to be a "buy-only" and "sell-only" flag that is fed to the AI to indicate whether it can do either of those Types of transactions, as well as giving it another data point for it to think about. 
  - Profits and losses due to transactions within the episode will need to be tracked so that the proper reward function can be computed at the end of the episode.
  - Because we are now doing a standard 5-share buy, the profit and loss on a given episode is more than -1 or +1, so this will have to be normalized. Moreover, because there are transactions within the episode, the amount lost or gained can be more than $5. I think the normalization that's being done already accounts for that to some degree with standard deviations. So if we normalize -5 to +5 to -1 to +1, I think we'll probably be OK.
  - The display of transactions and the behavior of the random player in evaluate.py will have to be modified to show transactions. Multiple transactions within the episode. These will need to show the fee, the amount of shares bought or sold, and the price, etc. With a total accounting at the end of the episode.
  - The behavior of the random player will have to be modified to the new restrictions. But I think we can keep the probability of doing something as opposed to nothing the same.
  - In the real market, there is a possibility of a partial fill on any limit order where not all the shares that are put up to offer are sold, but only some of them are. We are not going to simulate that here, but the code should be able to respond correctly if the AI still owns shares to remain in sell only mode, and conversely, if it doesn't sell all the shares it owns, it should remain in sell only mode. In other words, rather than using history to tell if it's in buy only or sell only mode, you should use the number of shares it owns. If it owns zero shares, it's in buy only mode. If it owns more than that, then it's in sell only. 
  - To keep things simple, the number of shares should be tracked to a precision of 1/100th of a share and rounded off.       
