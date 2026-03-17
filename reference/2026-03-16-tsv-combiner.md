# TSV Combiner Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `combine_tsv.py`, a one-shot script that merges all Polymarket BTC 5-minute TSV snapshot files into a single JSONL-style episode file for RL training.

**Architecture:** Pure functions handle each processing stage (parse → segment → filter → annotate → write); `main()` is wiring only. Every function is tested in isolation before composition. The test file imports functions directly from `combine_tsv.py`.

**Tech Stack:** Python 3.10+, stdlib only (`glob`, `json`, `datetime`, `time`, `unittest`)

---

## Chunk 1: Core Processing Functions + Unit Tests

### Task 1: Project scaffolding

**Files:**
- Create: `combine_tsv.py`
- Create: `test_combine_tsv.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add `btc_polymarket_combined.json` to `.gitignore`**

Open `.gitignore` (create it if absent) and add:
```
btc_polymarket_combined.json
```

- [ ] **Step 2: Create `combine_tsv.py` with imports and constants**

```python
"""
combine_tsv.py
--------------
Merges all Polymarket BTC 5-minute TSV snapshot files into a single
JSON episode file for reinforcement learning training.

Usage:
    python combine_tsv.py
"""

import glob
import json
import time
import sys
from datetime import datetime, timezone

TSV_DIR     = "tsv"
TSV_PATTERN = "btc_polymarket_*.tsv"
OUTPUT_FILE = "btc_polymarket_combined.json"
CLOSE_THRESHOLD_MS = 15_000  # segment must have a row within 15s of close
```

- [ ] **Step 3: Create `test_combine_tsv.py` with imports**

```python
"""
test_combine_tsv.py
-------------------
Unit and integration tests for combine_tsv.py.
Run with: python -m pytest test_combine_tsv.py -v
  or:     python -m unittest test_combine_tsv -v
"""

import unittest
import os
import json
from datetime import datetime, timezone

import combine_tsv
```

- [ ] **Step 4: Commit scaffolding**

```bash
git add combine_tsv.py test_combine_tsv.py .gitignore
git commit -m "feat: scaffold combine_tsv.py and test file"
```

---

### Task 2: `parse_row()` — parse a TSV line into a dict

**Files:**
- Modify: `combine_tsv.py`
- Modify: `test_combine_tsv.py`

The TSV columns in order are:
`Timestamp, Up Bid, Up Ask, Down Bid, Down Ask, Price to Beat, Current Price, Difference %, Difference $, Time to Close (ms)`

Note: some fields may be empty (e.g., a missing ask when the market is nearly resolved). These should be stored as `None`.

- [ ] **Step 1: Write the failing test**

```python
class TestParseRow(unittest.TestCase):

    def _make_row(self, **overrides):
        """Return a TSV line with sensible defaults."""
        defaults = {
            "timestamp":     "2026-03-14T17:23:01.761Z",
            "up_bid":        "55.00",
            "up_ask":        "56.00",
            "down_bid":      "44.00",
            "down_ask":      "45.00",
            "price_to_beat": "70679.78",
            "current_price": "70685.94",
            "diff_pct":      "0.008715",
            "diff_usd":      "6.16",
            "time_to_close": "119733",
        }
        defaults.update(overrides)
        return "\t".join(defaults.values())

    def test_parse_row_basic(self):
        row = self._make_row()
        result = combine_tsv.parse_row(row)
        self.assertEqual(result["timestamp"], "2026-03-14T17:23:01.761Z")
        self.assertAlmostEqual(result["up_bid"], 55.0)
        self.assertAlmostEqual(result["price_to_beat"], 70679.78)
        self.assertAlmostEqual(result["current_price"], 70685.94)
        self.assertEqual(result["time_to_close"], 119733)  # int

    def test_parse_row_empty_field_is_none(self):
        # Simulate a row where up_ask is blank (market nearly resolved)
        row = "2026-03-14T17:24:59.792Z\t99.00\t\t1.00\t70679.78\t70694.50\t0.020826\t14.72\t1592"
        result = combine_tsv.parse_row(row)
        self.assertIsNone(result["up_ask"])

    def test_parse_row_time_to_close_truncated_to_int(self):
        row = self._make_row(time_to_close="119733.9")
        result = combine_tsv.parse_row(row)
        self.assertIsInstance(result["time_to_close"], int)
        self.assertEqual(result["time_to_close"], 119733)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
python -m pytest test_combine_tsv.py::TestParseRow -v
```
Expected: `AttributeError: module 'combine_tsv' has no attribute 'parse_row'`

- [ ] **Step 3: Implement `parse_row()` in `combine_tsv.py`**

```python
_COLUMNS = [
    "timestamp", "up_bid", "up_ask", "down_bid", "down_ask",
    "price_to_beat", "current_price", "diff_pct", "diff_usd", "time_to_close",
]

def parse_row(line: str) -> dict:
    """Parse a single TSV data line into a dict. Empty fields become None."""
    parts = line.rstrip("\n").split("\t")
    row = {}
    for key, raw in zip(_COLUMNS, parts):
        if raw == "":
            row[key] = None
        elif key == "timestamp":
            row[key] = raw
        elif key == "time_to_close":
            row[key] = int(float(raw))
        else:
            row[key] = float(raw)
    return row
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest test_combine_tsv.py::TestParseRow -v
```
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add combine_tsv.py test_combine_tsv.py
git commit -m "feat: add parse_row() with tests"
```

---

### Task 3: `get_window_key()` — compute the 5-minute window a timestamp belongs to

**Files:**
- Modify: `combine_tsv.py`
- Modify: `test_combine_tsv.py`

The window key is `(year, month, day, hour, (minute // 5) * 5)` derived from the UTC timestamp. When the key changes between adjacent rows, a segment boundary has been crossed.

- [ ] **Step 1: Write the failing tests**

```python
class TestGetWindowKey(unittest.TestCase):

    def test_window_key_rounds_minute_down(self):
        # 17:23 → window starts at 17:20
        key = combine_tsv.get_window_key("2026-03-14T17:23:01.761Z")
        self.assertEqual(key, (2026, 3, 14, 17, 20))

    def test_window_key_at_boundary(self):
        # 17:25:00 → new window starting at 17:25
        key = combine_tsv.get_window_key("2026-03-14T17:25:01.837Z")
        self.assertEqual(key, (2026, 3, 14, 17, 25))

    def test_window_key_hour_rollover(self):
        # 18:00:xx → window starts at 18:00
        key = combine_tsv.get_window_key("2026-03-14T18:00:05.000Z")
        self.assertEqual(key, (2026, 3, 14, 18, 0))

    def test_different_windows_give_different_keys(self):
        key1 = combine_tsv.get_window_key("2026-03-14T17:24:59.792Z")
        key2 = combine_tsv.get_window_key("2026-03-14T17:25:01.837Z")
        self.assertNotEqual(key1, key2)

    def test_same_window_gives_same_key(self):
        key1 = combine_tsv.get_window_key("2026-03-14T17:23:01.761Z")
        key2 = combine_tsv.get_window_key("2026-03-14T17:24:59.792Z")
        self.assertEqual(key1, key2)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest test_combine_tsv.py::TestGetWindowKey -v
```
Expected: `AttributeError: module 'combine_tsv' has no attribute 'get_window_key'`

- [ ] **Step 3: Implement `get_window_key()` in `combine_tsv.py`**

```python
def get_window_key(timestamp: str) -> tuple:
    """Return the 5-minute window key (y, m, d, h, floored_minute) for a UTC timestamp."""
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return (dt.year, dt.month, dt.day, dt.hour, (dt.minute // 5) * 5)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest test_combine_tsv.py::TestGetWindowKey -v
```
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add combine_tsv.py test_combine_tsv.py
git commit -m "feat: add get_window_key() with tests"
```

---

### Task 4: `segment_rows()` — group rows into 5-minute segments

**Files:**
- Modify: `combine_tsv.py`
- Modify: `test_combine_tsv.py`

Takes a flat list of parsed row dicts. Returns a list of lists, where each inner list is one segment. The boundary rule: the first row after a window key change starts a new segment.

- [ ] **Step 1: Write the failing tests**

```python
def _make_parsed_row(timestamp, price_to_beat=70000.0, current_price=70010.0, time_to_close=100000):
    """Helper: return a minimal parsed row dict."""
    return {
        "timestamp": timestamp,
        "up_bid": 55.0, "up_ask": 56.0,
        "down_bid": 44.0, "down_ask": 45.0,
        "price_to_beat": price_to_beat,
        "current_price": current_price,
        "diff_pct": 0.01, "diff_usd": 10.0,
        "time_to_close": time_to_close,
    }

class TestSegmentRows(unittest.TestCase):

    def test_single_segment(self):
        rows = [
            _make_parsed_row("2026-03-14T17:23:01Z"),
            _make_parsed_row("2026-03-14T17:23:03Z"),
            _make_parsed_row("2026-03-14T17:24:59Z"),
        ]
        segments = combine_tsv.segment_rows(rows)
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0]), 3)

    def test_two_segments_at_clock_boundary(self):
        rows = [
            _make_parsed_row("2026-03-14T17:24:59Z"),  # last row of 17:20 window
            _make_parsed_row("2026-03-14T17:25:01Z"),  # first row of 17:25 window
        ]
        segments = combine_tsv.segment_rows(rows)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0][0]["timestamp"], "2026-03-14T17:24:59Z")
        self.assertEqual(segments[1][0]["timestamp"], "2026-03-14T17:25:01Z")

    def test_boundary_row_belongs_to_new_segment(self):
        rows = [
            _make_parsed_row("2026-03-14T17:24:55Z"),
            _make_parsed_row("2026-03-14T17:24:59Z"),  # last of old window
            _make_parsed_row("2026-03-14T17:25:01Z"),  # first of new window
            _make_parsed_row("2026-03-14T17:25:03Z"),
        ]
        segments = combine_tsv.segment_rows(rows)
        self.assertEqual(len(segments), 2)
        self.assertEqual(len(segments[0]), 2)  # rows at :55 and :59
        self.assertEqual(len(segments[1]), 2)  # rows at :01 and :03

    def test_same_price_to_beat_across_boundary_still_splits(self):
        # Even if price_to_beat is identical, the clock boundary splits segments
        rows = [
            _make_parsed_row("2026-03-14T17:24:59Z", price_to_beat=70000.0),
            _make_parsed_row("2026-03-14T17:25:01Z", price_to_beat=70000.0),
        ]
        segments = combine_tsv.segment_rows(rows)
        self.assertEqual(len(segments), 2)

    def test_empty_input(self):
        self.assertEqual(combine_tsv.segment_rows([]), [])
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest test_combine_tsv.py::TestSegmentRows -v
```
Expected: `AttributeError: module 'combine_tsv' has no attribute 'segment_rows'`

- [ ] **Step 3: Implement `segment_rows()` in `combine_tsv.py`**

```python
def segment_rows(rows: list[dict]) -> list[list[dict]]:
    """Group a flat list of parsed rows into 5-minute segments by clock boundary."""
    if not rows:
        return []
    segments = []
    current = [rows[0]]
    for row in rows[1:]:
        if get_window_key(row["timestamp"]) != get_window_key(current[-1]["timestamp"]):
            segments.append(current)
            current = [row]
        else:
            current.append(row)
    segments.append(current)
    return segments
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest test_combine_tsv.py::TestSegmentRows -v
```
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add combine_tsv.py test_combine_tsv.py
git commit -m "feat: add segment_rows() with tests"
```

---

### Task 5: `filter_segments()` — discard incomplete segments

**Files:**
- Modify: `combine_tsv.py`
- Modify: `test_combine_tsv.py`

A segment is complete if at least one row has `time_to_close < 15000`. Returns `(kept, dropped_count)`.

- [ ] **Step 1: Write the failing tests**

```python
class TestFilterSegments(unittest.TestCase):

    def test_drops_segment_with_no_close_row(self):
        segment = [
            _make_parsed_row("2026-03-14T17:21:00Z", time_to_close=200000),
            _make_parsed_row("2026-03-14T17:22:00Z", time_to_close=150000),
        ]
        kept, dropped = combine_tsv.filter_segments([segment])
        self.assertEqual(len(kept), 0)
        self.assertEqual(dropped, 1)

    def test_keeps_segment_with_close_row(self):
        segment = [
            _make_parsed_row("2026-03-14T17:21:00Z", time_to_close=200000),
            _make_parsed_row("2026-03-14T17:24:59Z", time_to_close=8000),
        ]
        kept, dropped = combine_tsv.filter_segments([segment])
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 0)

    def test_keeps_segment_with_close_row_at_boundary(self):
        # Exactly at threshold: 14999 is kept, 15000 is dropped
        segment_kept = [_make_parsed_row("2026-03-14T17:24:59Z", time_to_close=14999)]
        segment_drop = [_make_parsed_row("2026-03-14T17:24:59Z", time_to_close=15000)]
        kept, dropped = combine_tsv.filter_segments([segment_kept, segment_drop])
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 1)

    def test_partial_start_segment_kept_if_close_confirmed(self):
        # Segment starts mid-period (only a few rows) but reaches the close
        segment = [
            _make_parsed_row("2026-03-14T17:23:00Z", time_to_close=120000),
            _make_parsed_row("2026-03-14T17:24:58Z", time_to_close=2000),
        ]
        kept, dropped = combine_tsv.filter_segments([segment])
        self.assertEqual(len(kept), 1)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest test_combine_tsv.py::TestFilterSegments -v
```
Expected: `AttributeError: module 'combine_tsv' has no attribute 'filter_segments'`

- [ ] **Step 3: Implement `filter_segments()` in `combine_tsv.py`**

```python
def filter_segments(segments: list[list[dict]]) -> tuple[list[list[dict]], int]:
    """
    Discard segments with no near-close row.
    Returns (kept_segments, dropped_count).
    """
    kept = []
    dropped = 0
    for seg in segments:
        if any(r["time_to_close"] is not None and r["time_to_close"] < CLOSE_THRESHOLD_MS
               for r in seg):
            kept.append(seg)
        else:
            dropped += 1
    return kept, dropped
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest test_combine_tsv.py::TestFilterSegments -v
```
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add combine_tsv.py test_combine_tsv.py
git commit -m "feat: add filter_segments() with tests"
```

---

### Task 6: `annotate_segment()` — compute outcome, metadata for a segment

**Files:**
- Modify: `combine_tsv.py`
- Modify: `test_combine_tsv.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestAnnotateSegment(unittest.TestCase):

    def test_outcome_up_when_current_greater(self):
        rows = [_make_parsed_row("2026-03-14T17:24:59Z",
                                  price_to_beat=70000.0, current_price=70010.0,
                                  time_to_close=500)]
        episode = combine_tsv.annotate_segment(rows)
        self.assertEqual(episode["outcome"], "UP")

    def test_outcome_up_on_exact_tie(self):
        rows = [_make_parsed_row("2026-03-14T17:24:59Z",
                                  price_to_beat=70000.0, current_price=70000.0,
                                  time_to_close=500)]
        episode = combine_tsv.annotate_segment(rows)
        self.assertEqual(episode["outcome"], "UP")

    def test_outcome_down_when_current_less(self):
        rows = [_make_parsed_row("2026-03-14T17:24:59Z",
                                  price_to_beat=70000.0, current_price=69990.0,
                                  time_to_close=500)]
        episode = combine_tsv.annotate_segment(rows)
        self.assertEqual(episode["outcome"], "DOWN")

    def test_start_price_is_price_to_beat(self):
        rows = [_make_parsed_row("2026-03-14T17:24:59Z", price_to_beat=70679.78)]
        episode = combine_tsv.annotate_segment(rows)
        self.assertAlmostEqual(episode["start_price"], 70679.78)

    def test_end_price_is_current_price_of_last_row(self):
        # Two rows with DIFFERENT current_price — ensures end_price is from last, not first
        rows = [
            _make_parsed_row("2026-03-14T17:23:00Z", current_price=70685.94),
            _make_parsed_row("2026-03-14T17:24:59Z", current_price=70694.50),
        ]
        episode = combine_tsv.annotate_segment(rows)
        self.assertAlmostEqual(episode["end_price"], 70694.50)   # last row, not 70685.94

    def test_hour_annotation(self):
        rows = [_make_parsed_row("2026-03-14T17:23:01Z")]
        episode = combine_tsv.annotate_segment(rows)
        self.assertEqual(episode["hour"], 17)

    def test_day_annotation_saturday(self):
        # 2026-03-14 is a Saturday → weekday() == 5
        rows = [_make_parsed_row("2026-03-14T17:23:01Z")]
        episode = combine_tsv.annotate_segment(rows)
        self.assertEqual(episode["day"], 5)

    def test_hour_and_day_from_first_row_not_last(self):
        # Segment spanning midnight: first row at 23:59 (hour=23, day=5 Saturday),
        # last row at 00:00 next day (would be hour=0, day=6 Sunday).
        # hour and day must come from the FIRST row.
        rows = [
            _make_parsed_row("2026-03-14T23:59:01Z"),   # Saturday 23:59
            _make_parsed_row("2026-03-14T23:59:57Z"),   # still Saturday (same window)
        ]
        episode = combine_tsv.annotate_segment(rows)
        self.assertEqual(episode["hour"], 23)
        self.assertEqual(episode["day"], 5)   # Saturday, not Sunday

    def test_rows_are_preserved_in_order(self):
        rows = [
            _make_parsed_row("2026-03-14T17:23:01Z"),
            _make_parsed_row("2026-03-14T17:23:03Z"),
        ]
        episode = combine_tsv.annotate_segment(rows)
        self.assertEqual(len(episode["rows"]), 2)
        self.assertEqual(episode["rows"][0]["timestamp"], "2026-03-14T17:23:01Z")
        self.assertEqual(episode["rows"][1]["timestamp"], "2026-03-14T17:23:03Z")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest test_combine_tsv.py::TestAnnotateSegment -v
```
Expected: `AttributeError: module 'combine_tsv' has no attribute 'annotate_segment'`

- [ ] **Step 3: Implement `annotate_segment()` in `combine_tsv.py`**

```python
def annotate_segment(rows: list[dict]) -> dict:
    """Compute episode metadata for a validated segment."""
    first, last = rows[0], rows[-1]
    dt = datetime.fromisoformat(first["timestamp"].replace("Z", "+00:00"))
    outcome = "UP" if last["current_price"] >= last["price_to_beat"] else "DOWN"
    return {
        "outcome":     outcome,
        "hour":        dt.hour,
        "day":         dt.weekday(),   # 0=Monday, 6=Sunday
        "start_price": last["price_to_beat"],
        "end_price":   last["current_price"],
        "rows":        rows,
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest test_combine_tsv.py::TestAnnotateSegment -v
```
Expected: 8 tests pass.

- [ ] **Step 5: Run all unit tests so far**

```bash
python -m pytest test_combine_tsv.py -v
```
Expected: all tests pass (no failures).

- [ ] **Step 6: Commit**

```bash
git add combine_tsv.py test_combine_tsv.py
git commit -m "feat: add annotate_segment() with tests"
```

---

## Chunk 2: I/O, Stats Output, and Integration Tests

### Task 7: `format_episode()` + `write_output()` — serialize episodes to file

**Files:**
- Modify: `combine_tsv.py`
- Modify: `test_combine_tsv.py`

The output format: one JSON object per episode, each row of `rows` on its own line, episodes separated by exactly one blank line (`\n\n`). No blank lines within an episode block.

- [ ] **Step 1: Write the failing tests**

```python
class TestWriteOutput(unittest.TestCase):

    def _make_episode(self, outcome="UP", hour=17, day=5):
        return {
            "outcome": outcome,
            "hour": hour,
            "day": day,
            "start_price": 70679.78,
            "end_price": 70694.50,
            "rows": [
                _make_parsed_row("2026-03-14T17:23:01Z"),
                _make_parsed_row("2026-03-14T17:24:59Z", time_to_close=1592),
            ],
        }

    def test_output_is_parseable_json_per_block(self):
        episodes = [self._make_episode("UP"), self._make_episode("DOWN")]
        text = combine_tsv.format_output(episodes)
        blocks = text.strip().split("\n\n")
        self.assertEqual(len(blocks), 2)
        parsed = json.loads(blocks[0])
        self.assertEqual(parsed["outcome"], "UP")

    def test_each_row_on_its_own_line(self):
        episodes = [self._make_episode()]
        text = combine_tsv.format_output(episodes)
        block = text.strip().split("\n\n")[0]
        lines = block.split("\n")
        # First line: opening brace with metadata and rows:[
        # Middle lines: one row dict each
        # Last line: closing ]}
        self.assertIn('"outcome"', lines[0])
        self.assertIn('"timestamp"', lines[1])  # first row
        self.assertIn('"timestamp"', lines[2])  # second row

    def test_no_blank_lines_within_episode(self):
        episodes = [self._make_episode()]
        text = combine_tsv.format_output(episodes)
        block = text.strip().split("\n\n")[0]
        self.assertNotIn("\n\n", block)

    def test_episodes_separated_by_exactly_one_blank_line(self):
        episodes = [self._make_episode(), self._make_episode(), self._make_episode()]
        text = combine_tsv.format_output(episodes)
        # Should be exactly 2 blank-line separators for 3 episodes
        self.assertEqual(text.count("\n\n"), 2)

    def test_outcome_and_metadata_in_output(self):
        episodes = [self._make_episode("DOWN", hour=9, day=0)]
        text = combine_tsv.format_output(episodes)
        parsed = json.loads(text.strip().split("\n\n")[0])
        self.assertEqual(parsed["outcome"], "DOWN")
        self.assertEqual(parsed["hour"], 9)
        self.assertEqual(parsed["day"], 0)
        self.assertAlmostEqual(parsed["start_price"], 70679.78)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest test_combine_tsv.py::TestWriteOutput -v
```
Expected: `AttributeError: module 'combine_tsv' has no attribute 'format_output'`

- [ ] **Step 3: Implement `format_episode()` and `format_output()` in `combine_tsv.py`**

```python
def format_episode(episode: dict) -> str:
    """
    Serialize one episode to a string.
    Opening line: JSON object with metadata and 'rows':[
    Middle lines: one row JSON object per line (comma-separated)
    Closing line: ]}
    No blank lines within the block.
    """
    rows = episode["rows"]
    meta = {k: v for k, v in episode.items() if k != "rows"}
    # Build opening line: {"outcome":"UP","hour":17,...,"rows":[
    opening = json.dumps(meta, separators=(",", ":"))[:-1] + ',"rows":['
    row_lines = []
    for i, row in enumerate(rows):
        suffix = "," if i < len(rows) - 1 else ""
        row_lines.append(json.dumps(row, separators=(",", ":")) + suffix)
    closing = "]}"
    return "\n".join([opening] + row_lines + [closing])


def format_output(episodes: list[dict]) -> str:
    """Serialize all episodes separated by a single blank line."""
    return "\n\n".join(format_episode(ep) for ep in episodes)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest test_combine_tsv.py::TestWriteOutput -v
```
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add combine_tsv.py test_combine_tsv.py
git commit -m "feat: add format_output() with tests"
```

---

### Task 8: `collect_files()` — glob TSV files with error handling

**Files:**
- Modify: `combine_tsv.py`
- Modify: `test_combine_tsv.py`

- [ ] **Step 1: Write the failing tests**

```python
import tempfile, os

class TestCollectFiles(unittest.TestCase):

    def test_returns_sorted_list(self):
        with tempfile.TemporaryDirectory() as d:
            # Create two fake TSV files
            for name in ["btc_polymarket_20260315_120000.tsv",
                         "btc_polymarket_20260314_090000.tsv"]:
                open(os.path.join(d, name), "w").close()
            result = combine_tsv.collect_files(d)
        basenames = [os.path.basename(f) for f in result]
        self.assertEqual(basenames, sorted(basenames))

    def test_raises_on_no_files(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as ctx:
                combine_tsv.collect_files(d)
            self.assertNotEqual(ctx.exception.code, 0)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest test_combine_tsv.py::TestCollectFiles -v
```

- [ ] **Step 3: Implement `collect_files()` in `combine_tsv.py`**

```python
def collect_files(tsv_dir: str = TSV_DIR) -> list[str]:
    """Return sorted list of TSV file paths. Exits with error if none found."""
    pattern = f"{tsv_dir}/{TSV_PATTERN}"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Error: no files matching '{pattern}' found.", file=sys.stderr)
        sys.exit(1)
    return files
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest test_combine_tsv.py::TestCollectFiles -v
```
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add combine_tsv.py test_combine_tsv.py
git commit -m "feat: add collect_files() with error handling and tests"
```

---

### Task 9: `read_file_rows()` + `format_duration()` + `print_stats()` — file reading and stats

**Files:**
- Modify: `combine_tsv.py`
- Modify: `test_combine_tsv.py`

- [ ] **Step 1: Write failing tests for `read_file_rows` and `format_duration`**

```python
import io, tempfile, os

class TestReadFileRows(unittest.TestCase):

    def _write_tsv(self, path, lines):
        header = "Timestamp\tUp Bid\tUp Ask\tDown Bid\tDown Ask\tPrice to Beat\tCurrent Price\tDifference %\tDifference $\tTime to Close (ms)\n"
        with open(path, "w") as f:
            f.write(header)
            for line in lines:
                f.write(line + "\n")

    def test_skips_header_line(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as tmp:
            path = tmp.name
        try:
            self._write_tsv(path, [
                "2026-03-14T17:23:01Z\t55.00\t56.00\t44.00\t45.00\t70679.78\t70685.94\t0.008715\t6.16\t119733",
                "2026-03-14T17:23:03Z\t55.00\t56.00\t44.00\t45.00\t70679.78\t70685.94\t0.008715\t6.16\t117661",
            ])
            rows, first_ts, last_ts = combine_tsv.read_file_rows(path)
            self.assertEqual(len(rows), 2)           # header not included
            self.assertEqual(first_ts, "2026-03-14T17:23:01Z")
            self.assertEqual(last_ts, "2026-03-14T17:23:03Z")
        finally:
            os.unlink(path)

    def test_returns_parsed_dicts(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as tmp:
            path = tmp.name
        try:
            self._write_tsv(path, [
                "2026-03-14T17:23:01Z\t55.00\t56.00\t44.00\t45.00\t70679.78\t70685.94\t0.008715\t6.16\t119733",
            ])
            rows, _, _ = combine_tsv.read_file_rows(path)
            self.assertAlmostEqual(rows[0]["up_bid"], 55.0)
            self.assertEqual(rows[0]["time_to_close"], 119733)
        finally:
            os.unlink(path)


class TestFormatDuration(unittest.TestCase):

    def test_multi_hour_duration(self):
        result = combine_tsv.format_duration(
            "2026-03-14T17:00:00Z", "2026-03-14T23:36:00Z"
        )
        self.assertEqual(result, "6h 36m")

    def test_sub_hour_duration(self):
        result = combine_tsv.format_duration(
            "2026-03-14T17:00:00Z", "2026-03-14T17:42:00Z"
        )
        self.assertEqual(result, "0h 42m")

    def test_exactly_one_hour(self):
        result = combine_tsv.format_duration(
            "2026-03-14T17:00:00Z", "2026-03-14T18:00:00Z"
        )
        self.assertEqual(result, "1h 00m")

    def test_midnight_crossing(self):
        result = combine_tsv.format_duration(
            "2026-03-14T23:00:00Z", "2026-03-15T01:30:00Z"
        )
        self.assertEqual(result, "2h 30m")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest test_combine_tsv.py::TestReadFileRows test_combine_tsv.py::TestFormatDuration -v
```
Expected: `AttributeError: module 'combine_tsv' has no attribute 'read_file_rows'`

- [ ] **Step 3: Implement `read_file_rows()` in `combine_tsv.py`**

Add `import os` at the top if not already present.

```python
def read_file_rows(filepath: str) -> tuple[list[dict], str, str]:
    """
    Parse all data rows from a TSV file.
    Returns (rows, first_timestamp, last_timestamp).
    Skips header lines (lines starting with 'Timestamp').
    """
    rows = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Timestamp"):
                continue
            rows.append(parse_row(line))
    first_ts = rows[0]["timestamp"] if rows else ""
    last_ts  = rows[-1]["timestamp"] if rows else ""
    return rows, first_ts, last_ts
```

- [ ] **Step 4: Implement `format_duration()` and `print_stats()` in `combine_tsv.py`**

```python
def format_duration(first_ts: str, last_ts: str) -> str:
    """Return elapsed time between two ISO 8601 UTC timestamps as 'Xh YYm'."""
    dt1 = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
    dt2 = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
    total_minutes = int((dt2 - dt1).total_seconds() // 60)
    return f"{total_minutes // 60}h {total_minutes % 60:02d}m"


def print_stats(files: list[str], file_meta: list[dict],
                episodes_written: int, dropped: int, elapsed: float) -> None:
    """
    Print processing summary to stdout.

    file_meta is a list of dicts, one per file:
        {"name": str, "first_ts": str, "last_ts": str, "episode_count": int}
    """
    bar = "─" * 65
    print(bar)
    print(f"Combined: {OUTPUT_FILE}")
    print(bar)
    print(f"Episodes written:  {episodes_written:>6}")
    print(f"Dropped (no close):{dropped:>6}")
    print(f"Processing time:   {elapsed:.1f}s")
    print()
    print("Date ranges by source file:")
    for m in file_meta:
        duration = format_duration(m["first_ts"], m["last_ts"])
        name = os.path.basename(m["name"])
        print(f"  {name:<42} {m['first_ts']} → {m['last_ts']}   {duration:>7}   {m['episode_count']:>3} episodes")
    print(bar)
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python -m pytest test_combine_tsv.py::TestReadFileRows test_combine_tsv.py::TestFormatDuration -v
```
Expected: 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add combine_tsv.py test_combine_tsv.py
git commit -m "feat: add read_file_rows(), format_duration(), print_stats() with tests"
```

---

### Task 10: `main()` — wire the pipeline together

**Files:**
- Modify: `combine_tsv.py`

**Episode-to-file attribution:** After all rows from all files are merged and segmented, each episode must be credited to the source file whose timestamp range contains the episode's first row. The approach:
1. While reading files, record each file's `(first_ts, last_ts)` in order
2. After segmentation, iterate episodes; for each episode compare its first row's timestamp against each file's range (`first_ts <= ep_ts <= last_ts`); increment that file's counter and stop
3. This works because files are non-overlapping and sorted chronologically

- [ ] **Step 1: Implement `main()` in `combine_tsv.py`**

```python
def main():
    t0 = time.perf_counter()
    files = collect_files()

    all_rows: list[dict] = []
    file_meta: list[dict] = []

    for filepath in files:
        rows, first_ts, last_ts = read_file_rows(filepath)
        file_meta.append({
            "name": filepath,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "episode_count": 0,
        })
        all_rows.extend(rows)

    segments = segment_rows(all_rows)
    kept, dropped = filter_segments(segments)
    episodes = [annotate_segment(seg) for seg in kept]

    # Credit each episode to the source file whose timestamp range contains
    # the episode's first row. Files are non-overlapping and sorted, so at
    # most one file will match.
    for ep in episodes:
        ep_ts = ep["rows"][0]["timestamp"]
        for meta in file_meta:
            if meta["first_ts"] <= ep_ts <= meta["last_ts"]:
                meta["episode_count"] += 1
                break

    output_text = format_output(episodes)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output_text)

    elapsed = time.perf_counter() - t0
    print_stats(files, file_meta, len(episodes), dropped, elapsed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script against the real TSV files**

```bash
python combine_tsv.py
```

Expected: stats table printed to stdout, `btc_polymarket_combined.json` created. The episode count should be non-zero, per-file date ranges and durations should be plausible, and the sum of per-file episode counts should equal episodes written.

- [ ] **Step 3: Validate the output is parseable JSON**

```bash
python -c "
import json
with open('btc_polymarket_combined.json') as f:
    blocks = f.read().strip().split('\n\n')
episodes = [json.loads(b) for b in blocks]
print(f'Episodes parsed OK: {len(episodes)}')
print(f'First: outcome={episodes[0][\"outcome\"]} hour={episodes[0][\"hour\"]} day={episodes[0][\"day\"]}')
print(f'Last:  outcome={episodes[-1][\"outcome\"]} hour={episodes[-1][\"hour\"]} day={episodes[-1][\"day\"]}')
"
```

Expected: no JSON parse errors, a sensible episode count printed.

- [ ] **Step 4: Commit**

```bash
git add combine_tsv.py
git commit -m "feat: implement main() pipeline wiring"
```

---

### Task 11: Integration (spot-check) tests

**Files:**
- Modify: `test_combine_tsv.py`

These tests load `btc_polymarket_combined.json` AND the source TSV files, derive expected values directly from the TSVs, then assert the combined output matches. This approach requires no hardcoded magic numbers — if the source data changes, expected values update automatically.

Tests are skipped if the combined file does not exist (requires `python combine_tsv.py` to have been run first, which Task 10 Step 2 guarantees).

- [ ] **Step 1: Write the spot-check tests**

```python
COMBINED_FILE = "btc_polymarket_combined.json"
TSV_DIR = "tsv"

@unittest.skipUnless(os.path.exists(COMBINED_FILE),
                     "combined file not present — run combine_tsv.py first")
class TestSpotChecks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(COMBINED_FILE) as f:
            cls.episodes = [json.loads(b) for b in f.read().strip().split("\n\n")]

    def test_spot_episode_1_first_window(self):
        """
        First episode is a partial-start window (17:20-17:25 UTC on 2026-03-14).
        Expected values are derived directly from the source TSV.

        Source TSV: btc_polymarket_20260314_132259.tsv
          First row in window: 2026-03-14T17:22:59.879Z  (price_to_beat=70679.78)
          Last row in window:  2026-03-14T17:24:59.792Z  (current_price=70694.50)
        """
        # Read expected values from the source TSV
        source_path = os.path.join(TSV_DIR, "btc_polymarket_20260314_132259.tsv")
        rows, _, _ = combine_tsv.read_file_rows(source_path)
        # Filter to the first window (17:20-17:25 key)
        first_key = combine_tsv.get_window_key(rows[0]["timestamp"])
        window_rows = [r for r in rows if combine_tsv.get_window_key(r["timestamp"]) == first_key]
        expected_start_price = window_rows[0]["price_to_beat"]
        expected_end_price   = window_rows[-1]["current_price"]
        expected_outcome     = "UP" if expected_end_price >= expected_start_price else "DOWN"
        expected_hour        = 17
        expected_day         = 5   # Saturday: datetime(2026,3,14).weekday() == 5

        ep = self.episodes[0]
        self.assertEqual(ep["outcome"], expected_outcome)
        self.assertEqual(ep["hour"], expected_hour)
        self.assertEqual(ep["day"], expected_day)
        self.assertAlmostEqual(ep["start_price"], expected_start_price, places=2)
        self.assertAlmostEqual(ep["end_price"], expected_end_price, places=2)

    def test_spot_episode_2_first_episode_from_file3(self):
        """
        First complete episode from btc_polymarket_20260315_210627.tsv.
        That file's data starts at 2026-03-16T01:06:xx UTC (Monday = day 0).
        Expected values derived from source TSV.
        """
        source_path = os.path.join(TSV_DIR, "btc_polymarket_20260315_210627.tsv")
        rows, _, _ = combine_tsv.read_file_rows(source_path)
        first_key = combine_tsv.get_window_key(rows[0]["timestamp"])
        window_rows = [r for r in rows if combine_tsv.get_window_key(r["timestamp"]) == first_key]

        # This window must be complete (has a near-close row) to appear in output
        has_close = any(r["time_to_close"] is not None and r["time_to_close"] < 15000
                        for r in window_rows)
        if not has_close:
            self.skipTest("First window of file 3 was incomplete — not in combined output")

        expected_start_price = window_rows[0]["price_to_beat"]
        expected_end_price   = window_rows[-1]["current_price"]
        expected_outcome     = "UP" if expected_end_price >= expected_start_price else "DOWN"

        # Find matching episode: first row must fall within file 3's timestamp range
        _, first_ts, last_ts = combine_tsv.read_file_rows(source_path)
        ep = next((e for e in self.episodes
                   if first_ts <= e["rows"][0]["timestamp"] <= last_ts), None)
        self.assertIsNotNone(ep, "No episode found from file 3")

        self.assertEqual(ep["hour"], 1)    # 01:xx UTC
        self.assertEqual(ep["day"], 0)     # Monday: datetime(2026,3,16).weekday() == 0
        self.assertEqual(ep["outcome"], expected_outcome)
        self.assertAlmostEqual(ep["start_price"], expected_start_price, places=2)
        self.assertAlmostEqual(ep["end_price"], expected_end_price, places=2)

    def test_spot_episode_3_last_episode(self):
        """
        Last episode in combined output comes from btc_polymarket_20260316_093828.tsv.
        Verifies end_price matches the last row's current_price in that episode,
        and that the value agrees with the source TSV.
        """
        source_path = os.path.join(TSV_DIR, "btc_polymarket_20260316_093828.tsv")
        rows, _, _ = combine_tsv.read_file_rows(source_path)
        last_key = combine_tsv.get_window_key(rows[-1]["timestamp"])
        window_rows = [r for r in rows if combine_tsv.get_window_key(r["timestamp"]) == last_key]
        expected_end_price = window_rows[-1]["current_price"]

        ep = self.episodes[-1]
        # end_price must match both the episode's last row and the source TSV
        self.assertAlmostEqual(ep["end_price"], ep["rows"][-1]["current_price"], places=4)
        self.assertAlmostEqual(ep["end_price"], expected_end_price, places=2)
```

- [ ] **Step 2: Run the spot-check tests**

```bash
python -m pytest test_combine_tsv.py::TestSpotChecks -v
```
Expected: 3 tests pass (or 1 skipped if file 3's first window was incomplete).

- [ ] **Step 3: Run the full test suite**

```bash
python -m pytest test_combine_tsv.py -v
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add test_combine_tsv.py
git commit -m "test: add integration spot-check tests against real combined output"
```

---

### Task 12: Final verification and cleanup

**Files:** none

- [ ] **Step 1: Run the full test suite one final time**

```bash
python -m pytest test_combine_tsv.py -v
```
Expected: all tests pass, no warnings.

- [ ] **Step 2: Run the script and verify the stats output**

```bash
python combine_tsv.py
```

Verify the console output shows:
- A non-zero episode count
- Per-file date ranges with durations and episode counts
- Sum of per-file episode counts equals episodes written
- Processing time under 10 seconds

- [ ] **Step 3: Confirm `btc_polymarket_combined.json` is gitignored and working tree is clean**

```bash
git status
```
Expected: `btc_polymarket_combined.json` does NOT appear in untracked files. Working tree is clean (all changes committed in prior tasks).
