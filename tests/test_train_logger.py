"""Tests for TrainLogger JSONL writer."""
import json
import pytest
from src.train_logger import TrainLogger


class TestTrainLogger:
    def test_append_creates_file(self, tmp_path):
        logger = TrainLogger(str(tmp_path / "log.jsonl"))
        logger.append(
            checkpoint=1, episode=50, elapsed_seconds=30.0,
            val_profit_cents=12.5, best_profit_cents=12.5,
            median_profit_cents=12.5, epsilon=0.9,
            action_distribution={"do_nothing": 1.0},
        )
        assert (tmp_path / "log.jsonl").exists()

    def test_append_writes_valid_json(self, tmp_path):
        logger = TrainLogger(str(tmp_path / "log.jsonl"))
        logger.append(
            checkpoint=1, episode=50, elapsed_seconds=30.0,
            val_profit_cents=12.5, best_profit_cents=12.5,
            median_profit_cents=12.5, epsilon=0.9,
            action_distribution={"do_nothing": 1.0},
        )
        line = (tmp_path / "log.jsonl").read_text().strip()
        entry = json.loads(line)
        assert entry["checkpoint"] == 1
        assert entry["episode"] == 50
        assert entry["val_profit_cents"] == pytest.approx(12.5)

    def test_append_multiple_entries_each_valid_json(self, tmp_path):
        logger = TrainLogger(str(tmp_path / "log.jsonl"))
        for i in range(3):
            logger.append(
                checkpoint=i+1, episode=(i+1)*50, elapsed_seconds=float(i*30),
                val_profit_cents=float(i), best_profit_cents=float(i),
                median_profit_cents=float(i), epsilon=0.9 - i*0.1,
                action_distribution={"do_nothing": 1.0},
            )
        lines = (tmp_path / "log.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            json.loads(line)  # must not raise

    def test_append_does_not_overwrite_on_resume(self, tmp_path):
        path = str(tmp_path / "log.jsonl")
        logger1 = TrainLogger(path)
        logger1.append(
            checkpoint=1, episode=50, elapsed_seconds=30.0,
            val_profit_cents=1.0, best_profit_cents=1.0,
            median_profit_cents=1.0, epsilon=0.9,
            action_distribution={},
        )
        logger2 = TrainLogger(path)
        logger2.append(
            checkpoint=2, episode=100, elapsed_seconds=60.0,
            val_profit_cents=2.0, best_profit_cents=2.0,
            median_profit_cents=1.5, epsilon=0.8,
            action_distribution={},
        )
        lines = (tmp_path / "log.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
