import pytest

from signal_engine.backtest import run_backtest
from signal_engine.config import Config
from signal_engine.data import synthetic_prices
from signal_engine.markets import symbols


@pytest.fixture(scope="session")
def small_prices():
    """Six instruments, ~3 years — enough to warm up the 64/256 EWMAC."""
    return synthetic_prices(symbols()[:6], n_days=900, seed=3)


@pytest.fixture(scope="session")
def full_prices():
    return synthetic_prices(symbols(), n_days=1500, seed=5)


@pytest.fixture(scope="session")
def config():
    return Config()


@pytest.fixture(scope="session")
def result(full_prices, config):
    return run_backtest(full_prices, config)
