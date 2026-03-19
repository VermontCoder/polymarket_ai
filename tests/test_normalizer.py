"""Tests for the Normalizer feature encoding pipeline."""

import math

import numpy as np
import pytest

from src.normalizer import Normalizer, _std


# ---------------------------------------------------------------------------
# Helpers: tiny episode/row factories
# ---------------------------------------------------------------------------

def _make_episode(
    hour=12,
    day=2,
    diff_pct_prev_session=0.05,
    diff_pct_hour=0.02,
    avg_pct_variance_hour=0.08,
    rows=None,
):
    """Create a minimal episode dict for testing."""
    if rows is None:
        rows = [_make_row()]
    return {
        "session_id": "test",
        "outcome": "UP",
        "hour": hour,
        "day": day,
        "start_price": 70000.0,
        "end_price": 70100.0,
        "diff_pct_prev_session": diff_pct_prev_session,
        "diff_pct_hour": diff_pct_hour,
        "avg_pct_variance_hour": avg_pct_variance_hour,
        "rows": rows,
    }


def _make_row(
    up_bid=55.0,
    up_ask=56.0,
    down_bid=44.0,
    down_ask=45.0,
    diff_pct=0.01,
    time_to_close=150000,
):
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


def _fitted_normalizer(train_episodes=None):
    """Return a Normalizer fitted on the given episodes (or sensible defaults)."""
    if train_episodes is None:
        train_episodes = [
            _make_episode(diff_pct_prev_session=0.1, diff_pct_hour=0.04,
                          rows=[_make_row(diff_pct=0.02)]),
            _make_episode(diff_pct_prev_session=-0.1, diff_pct_hour=-0.04,
                          rows=[_make_row(diff_pct=-0.02)]),
        ]
    norm = Normalizer()
    norm.fit(train_episodes)
    return norm


# ---------------------------------------------------------------------------
# Tests: static feature encoding
# ---------------------------------------------------------------------------

class TestStaticEncoding:
    """Tests for encode_static (37-dim vector)."""

    def test_output_shape(self):
        norm = _fitted_normalizer()
        ep = _make_episode()
        static = norm.encode_static(ep)
        assert static.shape == (37,)
        assert static.dtype == np.float32

    def test_hour_one_hot_correct_index(self):
        norm = _fitted_normalizer()
        for hour in [0, 5, 12, 23]:
            ep = _make_episode(hour=hour)
            static = norm.encode_static(ep)
            # Only the correct hour index should be 1
            for i in range(24):
                expected = 1.0 if i == hour else 0.0
                assert static[i] == expected, (
                    f"hour={hour}, index={i}: expected {expected}, got {static[i]}"
                )

    def test_hour_one_hot_dimensions(self):
        """Hour encoding occupies exactly indices 0-23."""
        norm = _fitted_normalizer()
        ep = _make_episode(hour=0)
        static = norm.encode_static(ep)
        # First 24 elements should sum to 1 (one-hot)
        assert static[:24].sum() == 1.0

    def test_day_one_hot_correct_index(self):
        norm = _fitted_normalizer()
        for day in range(7):  # 0=Mon .. 6=Sun
            ep = _make_episode(day=day)
            static = norm.encode_static(ep)
            for i in range(7):
                expected = 1.0 if i == day else 0.0
                assert static[24 + i] == expected, (
                    f"day={day}, index={24+i}: expected {expected}, got {static[24+i]}"
                )

    def test_day_one_hot_dimensions(self):
        """Day encoding occupies exactly indices 24-30."""
        norm = _fitted_normalizer()
        ep = _make_episode(day=3)
        static = norm.encode_static(ep)
        assert static[24:31].sum() == 1.0

    def test_diff_pct_prev_session_normalized(self):
        """Non-null diff_pct_prev_session is divided by training std."""
        train = [
            _make_episode(diff_pct_prev_session=0.10),
            _make_episode(diff_pct_prev_session=-0.10),
        ]
        norm = _fitted_normalizer(train)
        # std of [0.1, -0.1] = 0.1 (population std)
        ep = _make_episode(diff_pct_prev_session=0.05)
        static = norm.encode_static(ep)
        assert static[31] == pytest.approx(0.05 / 0.1, abs=1e-5)
        assert static[32] == 0.0  # not null

    def test_diff_pct_prev_session_null(self):
        """Null diff_pct_prev_session encodes to (0, is_null=1)."""
        norm = _fitted_normalizer()
        ep = _make_episode(diff_pct_prev_session=None)
        static = norm.encode_static(ep)
        assert static[31] == 0.0
        assert static[32] == 1.0

    def test_diff_pct_hour_normalized(self):
        """Non-null diff_pct_hour is divided by training std."""
        train = [
            _make_episode(diff_pct_hour=0.04),
            _make_episode(diff_pct_hour=-0.04),
        ]
        norm = _fitted_normalizer(train)
        # std of [0.04, -0.04] = 0.04
        ep = _make_episode(diff_pct_hour=0.02)
        static = norm.encode_static(ep)
        assert static[33] == pytest.approx(0.02 / 0.04, abs=1e-5)
        assert static[34] == 0.0

    def test_diff_pct_hour_null(self):
        """Null diff_pct_hour encodes to (0, is_null=1)."""
        norm = _fitted_normalizer()
        ep = _make_episode(diff_pct_hour=None)
        static = norm.encode_static(ep)
        assert static[33] == 0.0
        assert static[34] == 1.0

    def test_avg_pct_variance_hour_normalized(self):
        """Non-null avg_pct_variance_hour is divided by training std."""
        train = [
            _make_episode(avg_pct_variance_hour=0.10),
            _make_episode(avg_pct_variance_hour=0.06),
        ]
        norm = _fitted_normalizer(train)
        ep = _make_episode(avg_pct_variance_hour=0.08)
        static = norm.encode_static(ep)
        assert static[35] == pytest.approx(0.08 / norm.std_avg_pct_variance_hour, abs=1e-5)
        assert static[36] == 0.0

    def test_avg_pct_variance_hour_null(self):
        """Null avg_pct_variance_hour encodes to (0, is_null=1)."""
        norm = _fitted_normalizer()
        ep = _make_episode(avg_pct_variance_hour=None)
        static = norm.encode_static(ep)
        assert static[35] == 0.0
        assert static[36] == 1.0


# ---------------------------------------------------------------------------
# Tests: dynamic feature encoding
# ---------------------------------------------------------------------------

class TestDynamicEncoding:
    """Tests for encode_dynamic (11-dim vector)."""

    def test_output_shape(self):
        norm = _fitted_normalizer()
        row = _make_row()
        dynamic = norm.encode_dynamic(row)
        assert dynamic.shape == (11,)
        assert dynamic.dtype == np.float32

    def test_bid_ask_encoding(self):
        """Non-null bid/ask values are divided by 100."""
        norm = _fitted_normalizer()
        row = _make_row(up_bid=65.0, up_ask=66.0, down_bid=34.0, down_ask=35.0)
        dynamic = norm.encode_dynamic(row)
        assert dynamic[0] == pytest.approx(0.65)   # up_bid / 100
        assert dynamic[1] == 0.0                     # not null
        assert dynamic[2] == pytest.approx(0.66)   # up_ask / 100
        assert dynamic[3] == 0.0
        assert dynamic[4] == pytest.approx(0.34)   # down_bid / 100
        assert dynamic[5] == 0.0
        assert dynamic[6] == pytest.approx(0.35)   # down_ask / 100
        assert dynamic[7] == 0.0

    def test_null_up_bid(self):
        """Null up_bid encodes to (0, is_null=1)."""
        norm = _fitted_normalizer()
        row = _make_row(up_bid=None)
        dynamic = norm.encode_dynamic(row)
        assert dynamic[0] == 0.0
        assert dynamic[1] == 1.0

    def test_null_up_ask(self):
        """Null up_ask encodes to (0, is_null=1)."""
        norm = _fitted_normalizer()
        row = _make_row(up_ask=None)
        dynamic = norm.encode_dynamic(row)
        assert dynamic[2] == 0.0
        assert dynamic[3] == 1.0

    def test_null_down_bid(self):
        """Null down_bid encodes to (0, is_null=1)."""
        norm = _fitted_normalizer()
        row = _make_row(down_bid=None)
        dynamic = norm.encode_dynamic(row)
        assert dynamic[4] == 0.0
        assert dynamic[5] == 1.0

    def test_null_down_ask(self):
        """Null down_ask encodes to (0, is_null=1)."""
        norm = _fitted_normalizer()
        row = _make_row(down_ask=None)
        dynamic = norm.encode_dynamic(row)
        assert dynamic[6] == 0.0
        assert dynamic[7] == 1.0

    def test_all_bid_ask_null(self):
        """All bid/ask null: all values 0, all flags 1."""
        norm = _fitted_normalizer()
        row = _make_row(up_bid=None, up_ask=None, down_bid=None, down_ask=None)
        dynamic = norm.encode_dynamic(row)
        for i in range(0, 8, 2):
            assert dynamic[i] == 0.0, f"index {i} should be 0"
            assert dynamic[i + 1] == 1.0, f"index {i+1} should be 1"

    def test_diff_pct_normalized_with_training_std(self):
        """diff_pct is normalized by training-set std."""
        train = [
            _make_episode(rows=[_make_row(diff_pct=0.03)]),
            _make_episode(rows=[_make_row(diff_pct=-0.03)]),
        ]
        norm = _fitted_normalizer(train)
        # std of [0.03, -0.03] = 0.03
        row = _make_row(diff_pct=0.015)
        dynamic = norm.encode_dynamic(row)
        assert dynamic[8] == pytest.approx(0.015 / 0.03, abs=1e-5)
        assert dynamic[9] == 0.0

    def test_diff_pct_null(self):
        """Null diff_pct encodes to (0, is_null=1)."""
        norm = _fitted_normalizer()
        row = _make_row(diff_pct=None)
        dynamic = norm.encode_dynamic(row)
        assert dynamic[8] == 0.0
        assert dynamic[9] == 1.0

    def test_time_to_close_normalization(self):
        """time_to_close is divided by 300000."""
        norm = _fitted_normalizer()
        row = _make_row(time_to_close=150000)
        dynamic = norm.encode_dynamic(row)
        assert dynamic[10] == pytest.approx(0.5)

    def test_time_to_close_zero(self):
        norm = _fitted_normalizer()
        row = _make_row(time_to_close=0)
        dynamic = norm.encode_dynamic(row)
        assert dynamic[10] == 0.0

    def test_time_to_close_at_300000(self):
        norm = _fitted_normalizer()
        row = _make_row(time_to_close=300000)
        dynamic = norm.encode_dynamic(row)
        assert dynamic[10] == pytest.approx(1.0)

    def test_time_to_close_clamped_high(self):
        """time_to_close > 300000 is clamped to 1.0."""
        norm = _fitted_normalizer()
        row = _make_row(time_to_close=500000)
        dynamic = norm.encode_dynamic(row)
        assert dynamic[10] == 1.0

    def test_time_to_close_clamped_low(self):
        """Negative time_to_close is clamped to 0.0."""
        norm = _fitted_normalizer()
        row = _make_row(time_to_close=-100)
        dynamic = norm.encode_dynamic(row)
        assert dynamic[10] == 0.0


# ---------------------------------------------------------------------------
# Tests: episode-level dynamic encoding
# ---------------------------------------------------------------------------

class TestEpisodeDynamic:
    """Tests for encode_episode_dynamic."""

    def test_shape(self):
        norm = _fitted_normalizer()
        ep = _make_episode(rows=[_make_row(), _make_row(), _make_row()])
        result = norm.encode_episode_dynamic(ep)
        assert result.shape == (3, 11)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests: normalization uses training-set statistics only
# ---------------------------------------------------------------------------

class TestTrainingStatsOnly:
    """Verify that normalization statistics come from training set only."""

    def test_std_computed_from_training_only(self):
        """Fitting on different training sets produces different stds."""
        train_a = [
            _make_episode(rows=[_make_row(diff_pct=0.10)]),
            _make_episode(rows=[_make_row(diff_pct=-0.10)]),
        ]
        train_b = [
            _make_episode(rows=[_make_row(diff_pct=0.50)]),
            _make_episode(rows=[_make_row(diff_pct=-0.50)]),
        ]
        norm_a = Normalizer()
        norm_a.fit(train_a)

        norm_b = Normalizer()
        norm_b.fit(train_b)

        # The stds should differ
        assert norm_a.std_diff_pct != norm_b.std_diff_pct

        # Same input row should produce different normalized values
        row = _make_row(diff_pct=0.05)
        dyn_a = norm_a.encode_dynamic(row)
        dyn_b = norm_b.encode_dynamic(row)
        assert dyn_a[8] != dyn_b[8]

    def test_val_episode_uses_train_stats(self):
        """A validation episode is normalized using training-set std."""
        train = [
            _make_episode(
                diff_pct_prev_session=0.10,
                diff_pct_hour=0.04,
                rows=[_make_row(diff_pct=0.02)],
            ),
            _make_episode(
                diff_pct_prev_session=-0.10,
                diff_pct_hour=-0.04,
                rows=[_make_row(diff_pct=-0.02)],
            ),
        ]
        norm = Normalizer()
        norm.fit(train)

        # "Validation" episode with values outside training range
        val_ep = _make_episode(
            diff_pct_prev_session=0.30,
            diff_pct_hour=0.12,
        )
        static = norm.encode_static(val_ep)
        # std_prev = 0.1, so 0.3 / 0.1 = 3.0
        assert static[31] == pytest.approx(3.0, abs=1e-5)
        # std_hour = 0.04, so 0.12 / 0.04 = 3.0
        assert static[33] == pytest.approx(3.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Tests: fit() must be called before encoding
# ---------------------------------------------------------------------------

class TestFitRequired:

    def test_encode_static_without_fit_raises(self):
        norm = Normalizer()
        with pytest.raises(AssertionError, match="fit"):
            norm.encode_static(_make_episode())

    def test_encode_dynamic_without_fit_raises(self):
        norm = Normalizer()
        with pytest.raises(AssertionError, match="fit"):
            norm.encode_dynamic(_make_row())


# ---------------------------------------------------------------------------
# Tests: internal _std helper
# ---------------------------------------------------------------------------

class TestStdHelper:

    def test_empty_returns_one(self):
        assert _std([]) == 1.0

    def test_single_value_returns_one(self):
        """Single value has std=0, so fallback to 1.0."""
        assert _std([5.0]) == 1.0

    def test_identical_values_returns_one(self):
        """All identical values have std=0, so fallback to 1.0."""
        assert _std([3.0, 3.0, 3.0]) == 1.0

    def test_known_std(self):
        """Population std of [1, -1] = 1.0."""
        assert _std([1.0, -1.0]) == pytest.approx(1.0)

    def test_known_std_2(self):
        """Population std of [0.1, -0.1] = 0.1."""
        assert _std([0.1, -0.1]) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Tests: integration with real data file (if available)
# ---------------------------------------------------------------------------

def _find_data_file():
    """Return the first JSON file found in the data/ directory."""
    import glob
    files = glob.glob("data/*.json")
    if not files:
        raise FileNotFoundError("No JSON files found in data/")
    return files[0]


class TestIntegrationWithData:
    """Test normalizer against the actual data file."""

    @pytest.fixture
    def real_data(self):
        import json
        try:
            with open(_find_data_file()) as f:
                return json.load(f)
        except FileNotFoundError:
            pytest.skip("Data file not available")

    def test_fit_and_encode_all_episodes(self, real_data):
        """Smoke test: fit on training split, encode all episodes."""
        from src.data_loader import split_episodes

        train, val, test = split_episodes(real_data, seed=42)
        norm = Normalizer()
        norm.fit(train)

        # All stds should be positive
        assert norm.std_diff_pct_prev_session > 0
        assert norm.std_diff_pct_hour > 0
        assert norm.std_diff_pct > 0

        # Encode every episode without errors
        for ep in train + val + test:
            static = norm.encode_static(ep)
            assert static.shape == (37,)
            dynamic = norm.encode_episode_dynamic(ep)
            assert dynamic.shape[0] == len(ep["rows"])
            assert dynamic.shape[1] == 11

    def test_static_values_in_expected_ranges(self, real_data):
        """Check that encoded static features are in sensible ranges."""
        from src.data_loader import split_episodes

        train, _, _ = split_episodes(real_data, seed=42)
        norm = Normalizer()
        norm.fit(train)

        for ep in real_data[:10]:
            static = norm.encode_static(ep)
            # One-hot sums
            assert static[:24].sum() == 1.0
            assert static[24:31].sum() == 1.0
            # is_null flags are 0 or 1
            assert static[32] in (0.0, 1.0)
            assert static[34] in (0.0, 1.0)
            assert static[36] in (0.0, 1.0)

    def test_dynamic_values_in_expected_ranges(self, real_data):
        """Check that encoded dynamic features are in sensible ranges."""
        from src.data_loader import split_episodes

        train, _, _ = split_episodes(real_data, seed=42)
        norm = Normalizer()
        norm.fit(train)

        ep = real_data[0]
        for row in ep["rows"][:5]:
            dynamic = norm.encode_dynamic(row)
            # bid/ask values should be in [0, 1] when not null
            for i in [0, 2, 4, 6]:
                if dynamic[i + 1] == 0.0:  # not null
                    assert 0.0 <= dynamic[i] <= 1.0
            # time_to_close clamped [0, 1]
            assert 0.0 <= dynamic[10] <= 1.0
