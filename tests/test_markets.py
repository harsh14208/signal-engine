from signal_engine.markets import EXPANDED_UNIVERSE, UNIVERSE, symbols


def test_core_universe_size():
    assert len(UNIVERSE) == 19
    assert len(symbols(expanded=False)) == 19


def test_expanded_universe_is_larger():
    assert len(EXPANDED_UNIVERSE) > len(UNIVERSE)
    assert len(symbols(expanded=True)) > len(symbols(expanded=False))


def test_expanded_contains_core():
    core = {i.symbol for i in UNIVERSE}
    expanded = {i.symbol for i in EXPANDED_UNIVERSE}
    assert core.issubset(expanded)
