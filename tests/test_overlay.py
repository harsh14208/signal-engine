import numpy as np
import pandas as pd

from signal_engine.portfolio import corr_spike_overlay


def test_overlay_one_instrument_is_neutral():
    rets = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, size=(100, 1)))
    mult = corr_spike_overlay(rets)
    assert (mult == 1.0).all()


def test_overlay_low_correlation_stays_at_one():
    rng = np.random.default_rng(1)
    rets = pd.DataFrame({f"s{i}": rng.normal(0, 0.01, 200) for i in range(5)})
    mult = corr_spike_overlay(rets, span=60, threshold=0.5)
    # After the initial warm-up the multiplier should be very close to 1.0
    # because the independent series are uncorrelated.
    assert mult.iloc[80:].mean() > 0.99


def test_overlay_spike_reduces_multiplier():
    rng = np.random.default_rng(2)
    common = rng.normal(0, 0.01, size=200)
    # Five instruments driven by a single common factor → high correlation.
    rets = pd.DataFrame({f"s{i}": common + rng.normal(0, 0.003, size=200) for i in range(5)})
    mult = corr_spike_overlay(rets, span=60, threshold=0.5, max_degross=0.5)
    # After warm-up the multiplier should drop well below 1.0.
    assert mult.iloc[80:].mean() < 0.90
    assert mult.min() <= 0.75


def test_overlay_no_lookahead():
    rng = np.random.default_rng(3)
    rets = pd.DataFrame({f"s{i}": rng.normal(0, 0.01, 150) for i in range(4)})
    base = corr_spike_overlay(rets)
    # Mutate the final day's returns; only the final multiplier (if any) may change.
    tweaked = rets.copy()
    tweaked.iloc[-1, :] = 0.5
    mod = corr_spike_overlay(tweaked)
    assert np.allclose(base.iloc[:-1].to_numpy(), mod.iloc[:-1].to_numpy())
