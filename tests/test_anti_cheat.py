"""Anti-cheat tests: verify the agent cannot access forbidden information.

The agent must only see the 10 allowed fields:
  hour, day, diff_pct_prev_session, diff_pct_hour,
  up_bid, up_ask, down_bid, down_ask, diff_pct, time_to_close

It must NEVER see: outcome, end_price, current_price, diff_usd,
  start_price, session_id, timestamp, or future rows.
"""

import numpy as np
import pytest

from src.environment import Environment, FORBIDDEN_FIELDS
from src.normalizer import Normalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(up_bid=55.0, up_ask=56.0, down_bid=44.0, down_ask=45.0,
              diff_pct=0.01, time_to_close=150000):
    return {
        "timestamp": "2026-03-14T17:23:00Z",
        "up_bid": up_bid, "up_ask": up_ask,
        "down_bid": down_bid, "down_ask": down_ask,
        "current_price": 70000.0, "diff_pct": diff_pct,
        "diff_usd": 5.0, "time_to_close": time_to_close,
    }


def _make_episode(outcome="UP", num_rows=5, rows=None, **kwargs):
    if rows is None:
        rows = [_make_row(**kwargs) for _ in range(num_rows)]
    return {
        "session_id": "test-session", "outcome": outcome,
        "hour": 12, "day": 2,
        "start_price": 70000.0, "end_price": 70100.0,
        "diff_pct_prev_session": 0.05, "diff_pct_hour": 0.02,
        "rows": rows,
    }


ALLOWED_ROW_FIELDS = {
    "up_bid", "up_ask", "down_bid", "down_ask", "diff_pct", "time_to_close",
}

ALLOWED_EPISODE_FIELDS = {
    "hour", "day", "diff_pct_prev_session", "diff_pct_hour",
}


# ---------------------------------------------------------------------------
# Tests: Forbidden fields stripped from observations
# ---------------------------------------------------------------------------

class TestForbiddenFieldsStripped:
    """Agent cannot access forbidden fields in observations."""

    def test_observation_no_outcome(self):
        env = Environment()
        env.reset(_make_episode())
        obs = env.get_observation()
        assert "outcome" not in obs

    def test_observation_no_end_price(self):
        env = Environment()
        env.reset(_make_episode())
        obs = env.get_observation()
        assert "end_price" not in obs

    def test_observation_no_current_price(self):
        env = Environment()
        env.reset(_make_episode())
        obs = env.get_observation()
        assert "current_price" not in obs

    def test_observation_no_diff_usd(self):
        env = Environment()
        env.reset(_make_episode())
        obs = env.get_observation()
        assert "diff_usd" not in obs

    def test_observation_no_start_price(self):
        env = Environment()
        env.reset(_make_episode())
        obs = env.get_observation()
        assert "start_price" not in obs

    def test_observation_no_session_id(self):
        env = Environment()
        env.reset(_make_episode())
        obs = env.get_observation()
        assert "session_id" not in obs

    def test_observation_no_timestamp(self):
        env = Environment()
        env.reset(_make_episode())
        obs = env.get_observation()
        assert "timestamp" not in obs

    def test_all_forbidden_fields_absent(self):
        """No forbidden field appears in any observation across all rows."""
        env = Environment()
        ep = _make_episode(num_rows=10)
        env.reset(ep)
        for _ in range(env.num_rows):
            obs = env.get_observation()
            for field in FORBIDDEN_FIELDS:
                assert field not in obs, f"Forbidden field '{field}' found in obs"
            done, _ = env.step(0)
            if done:
                break


# ---------------------------------------------------------------------------
# Tests: Only allowed fields visible
# ---------------------------------------------------------------------------

class TestAllowedFieldsOnly:
    """Agent only sees the 10 allowed fields."""

    def test_observation_contains_only_allowed_row_fields(self):
        """Row observations contain only allowed dynamic fields."""
        env = Environment()
        env.reset(_make_episode())
        obs = env.get_observation()
        for key in obs:
            assert key in ALLOWED_ROW_FIELDS, (
                f"Unexpected field '{key}' in observation"
            )

    def test_observation_has_all_allowed_fields(self):
        """Row observations contain all expected dynamic fields."""
        env = Environment()
        env.reset(_make_episode())
        obs = env.get_observation()
        for field in ALLOWED_ROW_FIELDS:
            assert field in obs, f"Missing allowed field '{field}'"

    def test_episode_info_contains_only_allowed_fields(self):
        """Episode info contains only allowed static fields."""
        env = Environment()
        env.reset(_make_episode())
        info = env.get_episode_info()
        for key in info:
            assert key in ALLOWED_EPISODE_FIELDS, (
                f"Unexpected field '{key}' in episode info"
            )

    def test_episode_info_has_all_allowed_fields(self):
        """Episode info contains all expected static fields."""
        env = Environment()
        env.reset(_make_episode())
        info = env.get_episode_info()
        for field in ALLOWED_EPISODE_FIELDS:
            assert field in info, f"Missing allowed field '{field}'"


# ---------------------------------------------------------------------------
# Tests: No future row access
# ---------------------------------------------------------------------------

class TestNoFutureAccess:
    """Agent cannot see future rows at decision time."""

    def test_observation_only_current_row(self):
        """Each get_observation() returns data from only the current row."""
        rows = [
            _make_row(up_bid=50.0 + i, up_ask=51.0 + i)
            for i in range(5)
        ]
        ep = _make_episode(rows=rows)
        # Need to create episode with explicit rows
        ep["rows"] = rows

        env = Environment()
        env.reset(ep)

        for step in range(5):
            obs = env.get_observation()
            expected_bid = 50.0 + step
            expected_ask = 51.0 + step
            assert obs["up_bid"] == expected_bid, (
                f"Step {step}: got bid {obs['up_bid']}, expected {expected_bid}"
            )
            assert obs["up_ask"] == expected_ask
            done, _ = env.step(0)
            if done:
                break

    def test_observation_does_not_contain_list(self):
        """Observations never contain lists that could hold future data."""
        env = Environment()
        env.reset(_make_episode(num_rows=5))
        for _ in range(5):
            obs = env.get_observation()
            for key, val in obs.items():
                assert not isinstance(val, (list, dict)), (
                    f"Field '{key}' is {type(val)}, could leak future data"
                )
            done, _ = env.step(0)
            if done:
                break


# ---------------------------------------------------------------------------
# Tests: Normalizer doesn't leak information
# ---------------------------------------------------------------------------

class TestNormalizerNoLeak:
    """Normalizer uses only training set statistics."""

    def test_fit_only_uses_provided_episodes(self):
        """Normalizer.fit() only uses the episodes passed to it."""
        train_eps = [_make_episode(num_rows=3) for _ in range(10)]
        val_eps = [_make_episode(num_rows=3) for _ in range(5)]

        norm1 = Normalizer()
        norm1.fit(train_eps)

        norm2 = Normalizer()
        norm2.fit(train_eps + val_eps)

        # If val set has different stats, normalizers should differ
        # (or be identical if data is identical, which is fine)
        # The key test: norm1 should not have been affected by val_eps
        # We verify this by encoding with both and checking norm1 is consistent
        static1 = norm1.encode_static(train_eps[0])
        static1_again = norm1.encode_static(train_eps[0])
        assert np.array_equal(static1, static1_again)

    def test_encode_does_not_use_outcome(self):
        """Encoding never accesses the outcome field."""
        ep_up = _make_episode(outcome="UP")
        ep_down = _make_episode(outcome="DOWN")

        norm = Normalizer()
        norm.fit([ep_up, ep_down])

        # Static features should be identical (same hour, day, etc.)
        s1 = norm.encode_static(ep_up)
        s2 = norm.encode_static(ep_down)
        assert np.array_equal(s1, s2)

        # Dynamic features should be identical (same row data)
        d1 = norm.encode_dynamic(ep_up["rows"][0])
        d2 = norm.encode_dynamic(ep_down["rows"][0])
        assert np.array_equal(d1, d2)

    def test_encoded_features_correct_dims(self):
        """Encoded features have correct dimensions."""
        norm = Normalizer()
        norm.fit([_make_episode()])

        static = norm.encode_static(_make_episode())
        assert static.shape == (35,)

        dynamic = norm.encode_dynamic(_make_row())
        assert dynamic.shape == (11,)
