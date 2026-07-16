"""Tests for CLI argument wiring and symbol-list building."""

from __future__ import annotations

import pytest

from signal_engine.cli import _build_symbol_list, build_config, main
from signal_engine.markets import symbols


class TestBuildConfig:
    def test_network_momentum_flag(self):
        cfg = build_config(_args(use_network_momentum=True))
        assert cfg.use_network_momentum is True

    def test_max_gross_flag(self):
        cfg = build_config(_args(max_gross=3.0))
        assert cfg.max_gross_notional == pytest.approx(3.0)


class TestSymbolList:
    def test_baseline_core(self):
        syms, tag = _build_symbol_list(_args())
        assert syms == symbols(expanded=False)
        assert tag == "universe"

    def test_semis_pack(self):
        syms, tag = _build_symbol_list(_args(semis=True))
        assert "SMH" in syms
        assert "SOXX" in syms
        assert "XSD" in syms
        assert tag == "options_experiment"
        # Core 19 still present.
        assert "SPY" in syms

    def test_qqq_pack(self):
        syms, tag = _build_symbol_list(_args(qqq=True))
        assert "QQQ" in syms
        assert tag == "options_experiment"

    def test_combined_packs_no_duplicates(self):
        syms, tag = _build_symbol_list(_args(semis=True, qqq=True))
        assert len(syms) == len(set(syms))
        assert all(s in syms for s in ["SMH", "QQQ", "SPY"])

    def test_network_momentum_does_not_change_symbols(self):
        syms, _ = _build_symbol_list(_args(use_network_momentum=True))
        assert syms == symbols(expanded=False)


class TestMainEntryPoint:
    def test_help_returns_zero(self):
        # --help prints and exits 0.
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0


def _args(**kwargs) -> object:
    """Build a minimal argparse Namespace for config/symbol tests."""
    defaults = {
        "capital": 1_000_000.0,
        "vol_target": 0.20,
        "no_breakout": False,
        "carry": False,
        "carry_proxies": False,
        "real_bond_carry": False,
        "curve_steepener": False,
        "curve_steepener_scale": 1.0,
        "curve_steepener_cost_bps": 0.5,
        "equity_momentum_sleeve": False,
        "eq_mom_lookback": 252,
        "eq_mom_rebalance": 21,
        "eq_mom_decile": 0.10,
        "expanded_universe": False,
        "empirical_scalars": False,
        "regime_overlay": False,
        "regime_threshold": 20.0,
        "regime_max_degear": 0.5,
        "regime_smooth": None,
        "vix_term_overlay": False,
        "vix_term_short_thresh": 1.10,
        "vix_term_long_thresh": 0.95,
        "vix_term_max_gear": 1.25,
        "vix_term_max_degear": 0.50,
        "vix_term_smooth": None,
        "credit_overlay": False,
        "credit_upper_thresh": 1.50,
        "credit_lower_thresh": 0.80,
        "credit_lookback": 1260,
        "credit_max_gear": 1.25,
        "credit_max_degear": 0.50,
        "credit_smooth": None,
        "use_hmm_regime_overlay": False,
        "hmm_train_window": 252,
        "hmm_refit_stride": 63,
        "hmm_bull_thresh": 0.75,
        "hmm_bear_thresh": 0.70,
        "hmm_trans_thresh": 0.15,
        "hmm_bull_gear": 1.10,
        "hmm_bear_degear": 0.70,
        "hmm_trans_degear": 0.85,
        "hmm_smooth": None,
        "hmm_random_state": 42,
        "cost_bps": 1.5,
        "cost_scheme": "flat",
        "buffer": 0.30,
        "weight_scheme": "equal",
        "cluster_weights": False,
        "no_governor": False,
        "governor_smooth": None,
        "accel": False,
        "xsmom": False,
        "corr_spike": False,
        "corr_spike_span": 60,
        "corr_spike_threshold": 0.50,
        "corr_spike_max_degross": 0.50,
        "cot": False,
        "cot_momentum": False,
        "core_commodities": False,
        "use_garch_vol": False,
        "garch_weight": 0.0,
        "garch_min_history": 252,
        "garch_refit_step": 63,
        "garch_horizon": 1,
        "use_network_momentum": False,
        "nm_speed": (32, 128),
        "nm_lookback": 256,
        "nm_lag": 1,
        "nm_rebal": 63,
        "nm_top_k": 3,
        "semis": False,
        "qqq": False,
        "diversifier_pack": False,
        "rate_pack": False,
        "max_gross": None,
        "financing_rate": 0.0,
        "financing_threshold": 1.0,
        "max_annual_financing_cost": None,
    }
    defaults.update(kwargs)
    return type("Args", (), defaults)()
