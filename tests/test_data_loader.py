"""Tests for stale leading row reassignment in data_loader."""

import pytest

from src.data_loader import _reassign_stale_leading_rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(time_to_close=297000):
    return {"time_to_close": time_to_close, "up_bid": 50, "up_ask": 51}


def _make_ep(session_id, rows):
    return {
        "session_id": session_id,
        "outcome": "UP",
        "hour": 12,
        "day": 2,
        "start_price": 70000.0,
        "end_price": 70100.0,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Tests: stale row reassignment
# ---------------------------------------------------------------------------

class TestStaleRowReassignment:
    """Stale leading rows (time_to_close < 5s) are moved to the previous episode."""

    def test_stale_row_moved_to_previous(self):
        """A stale row 0 on episode 1 is appended to episode 0."""
        eps = [
            _make_ep("2026-03-14T17:40:00Z", [_make_row(10000), _make_row(5000)]),
            _make_ep("2026-03-14T17:45:00Z", [_make_row(1400), _make_row(299000), _make_row(297000)]),
        ]
        result = _reassign_stale_leading_rows(eps)
        assert len(result[0]["rows"]) == 3  # gained the stale row
        assert len(result[1]["rows"]) == 2  # lost the stale row
        assert result[0]["rows"][-1]["time_to_close"] == 1400
        assert result[1]["rows"][0]["time_to_close"] == 299000

    def test_multiple_stale_rows_moved(self):
        """Multiple consecutive stale leading rows are all moved."""
        eps = [
            _make_ep("2026-03-14T17:40:00Z", [_make_row(10000)]),
            _make_ep("2026-03-14T17:45:00Z", [_make_row(1000), _make_row(500), _make_row(299000)]),
        ]
        result = _reassign_stale_leading_rows(eps)
        assert len(result[0]["rows"]) == 3  # gained 2 stale rows
        assert len(result[1]["rows"]) == 1
        assert result[1]["rows"][0]["time_to_close"] == 299000

    def test_no_stale_rows_unchanged(self):
        """Episodes without stale leading rows are not modified."""
        eps = [
            _make_ep("2026-03-14T17:40:00Z", [_make_row(299000), _make_row(297000)]),
            _make_ep("2026-03-14T17:45:00Z", [_make_row(299000), _make_row(297000)]),
        ]
        result = _reassign_stale_leading_rows(eps)
        assert len(result[0]["rows"]) == 2
        assert len(result[1]["rows"]) == 2

    def test_does_not_mutate_input(self):
        """The original episode list is not modified."""
        original_rows = [_make_row(1400), _make_row(299000)]
        eps = [
            _make_ep("2026-03-14T17:40:00Z", [_make_row(10000)]),
            _make_ep("2026-03-14T17:45:00Z", list(original_rows)),
        ]
        _reassign_stale_leading_rows(eps)
        assert len(eps[0]["rows"]) == 1
        assert len(eps[1]["rows"]) == 2


class TestStaleRowFirstEpisode:
    """When episode 0 has stale leading rows, they are dropped safely."""

    def test_first_episode_stale_row_dropped(self):
        """Stale row on episode 0 is dropped (no previous episode exists)."""
        eps = [
            _make_ep("2026-03-14T17:20:00Z", [_make_row(1400), _make_row(299000), _make_row(297000)]),
            _make_ep("2026-03-14T17:25:00Z", [_make_row(299000), _make_row(297000)]),
        ]
        result = _reassign_stale_leading_rows(eps)
        assert len(result[0]["rows"]) == 2
        assert result[0]["rows"][0]["time_to_close"] == 299000
        # Second episode untouched
        assert len(result[1]["rows"]) == 2

    def test_first_episode_multiple_stale_rows_dropped(self):
        """Multiple stale rows on episode 0 are all dropped."""
        eps = [
            _make_ep("2026-03-14T17:20:00Z", [_make_row(0), _make_row(500), _make_row(299000)]),
        ]
        result = _reassign_stale_leading_rows(eps)
        assert len(result[0]["rows"]) == 1
        assert result[0]["rows"][0]["time_to_close"] == 299000

    def test_first_episode_no_stale_rows_unchanged(self):
        """Episode 0 without stale rows is not modified."""
        eps = [
            _make_ep("2026-03-14T17:20:00Z", [_make_row(121000), _make_row(119000)]),
        ]
        result = _reassign_stale_leading_rows(eps)
        assert len(result[0]["rows"]) == 2


class TestStaleRowGapHandling:
    """Stale rows are dropped (not reassigned) when there is a session gap."""

    def test_gap_drops_stale_rows(self):
        """Stale row is dropped when previous episode is not consecutive."""
        eps = [
            _make_ep("2026-03-14T17:00:00Z", [_make_row(10000)]),
            # 15-minute gap — not consecutive
            _make_ep("2026-03-14T17:15:00Z", [_make_row(1400), _make_row(299000)]),
        ]
        result = _reassign_stale_leading_rows(eps)
        assert len(result[0]["rows"]) == 1  # did NOT gain the stale row
        assert len(result[1]["rows"]) == 1
        assert result[1]["rows"][0]["time_to_close"] == 299000


from tests.conftest import find_data_file as _find_data_file


class TestStaleRowRegression:
    """Regression test: no episode should start with a row from the previous session."""

    def test_no_episode_starts_with_stale_row(self):
        """After reassignment, no episode has row 0 with time_to_close < 5s."""
        from src.data_loader import load_episodes
        eps = load_episodes(_find_data_file())
        for i, ep in enumerate(eps):
            ttc = ep["rows"][0].get("time_to_close")
            assert ttc is None or ttc >= 5000, (
                f"Episode {i} ({ep['session_id']}) starts with stale row: "
                f"time_to_close={ttc}ms"
            )

    def test_no_data_lost(self):
        """Total row count is preserved — stale rows are moved, not dropped."""
        import json
        path = _find_data_file()
        with open(path) as f:
            raw = json.load(f)
        from src.data_loader import load_episodes
        eps = load_episodes(path)

        raw_total = sum(len(ep["rows"]) for ep in raw)
        fixed_total = sum(len(ep["rows"]) for ep in eps)
        assert fixed_total == raw_total
