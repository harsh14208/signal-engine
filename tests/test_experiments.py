import json
import os
import tempfile

import pytest

from signal_engine.backtest import run_backtest
from signal_engine.config import Config
from signal_engine.data import synthetic_prices
from signal_engine.experiments import count_experiments, log_experiment
from signal_engine.markets import symbols


@pytest.fixture
def tmp_log():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    yield path
    os.unlink(path)


def test_count_empty_log(tmp_log):
    assert count_experiments(tmp_log) == 0


def test_log_counts_unique_configs(tmp_log):
    px = synthetic_prices(symbols()[:4], n_days=400, seed=1)
    cfg1 = Config()
    cfg2 = Config(use_governor=False)
    r1 = run_backtest(px, cfg1)
    r2 = run_backtest(px, cfg2)
    log_experiment(cfg1, r1, tmp_log)
    log_experiment(cfg1, r1, tmp_log)  # duplicate
    log_experiment(cfg2, r2, tmp_log)
    assert count_experiments(tmp_log) == 2


def test_log_writes_valid_json(tmp_log):
    px = synthetic_prices(symbols()[:4], n_days=400, seed=2)
    cfg = Config()
    log_experiment(cfg, run_backtest(px, cfg), tmp_log)
    with open(tmp_log, "r", encoding="utf-8") as f:
        record = json.loads(f.readline())
    assert "config_hash" in record
    assert "sharpe" in record
    assert "timestamp" in record
