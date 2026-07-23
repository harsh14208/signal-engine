#!/bin/bash
set -euo pipefail
# Daily forward loop for signal-engine (Tier A).
# Runs after the US close. Keeps caches warm, generates targets, marks shadow
# returns, and reconciles. Alpaca execution is commented out until shadow-book
# tracking is confirmed.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Activate the project virtual environment.
# shellcheck source=/dev/null
source .venv/bin/activate

# 1. Refresh ETF price cache (yfinance; PIT-stitched). --semis also caches
#    SMH/SOXX/XSD for the challenger_semis shadow book.
python scripts/warm_cache.py --semis

# 2. Generate target positions for the next session. --challenger runs a parallel
#    shadow book with the COT lever flipped, to forward-test it against champion
#    (resumed 2026-07-22 — had been generated once on 2026-07-10 and then dropped
#    from this invocation when --challenger-semis was added, so it accrued no
#    forward evidence for 12 days). --challenger-semis runs a parallel shadow book
#    (champion config + semis pack) to earn FORWARD evidence for the pack;
#    promotion waits on champion_challenger_report (>=60 days). Both are
#    shadow-only (signal_engine.live.mark_all_shadow_returns) — execute_alpaca.py
#    only ever submits real paper orders for the champion book.
python scripts/generate_targets.py --source auto --challenger --challenger-semis

# 3. Mark the shadow return for the day that just closed.
python scripts/shadow_book.py --source auto

# 4. Reconcile shadow returns vs the backtest and update guardrails.
python scripts/reconcile.py --source auto

# 5. Replay stored decisions and detect engine drift (Phase 3). --enforce engages
#    the kill switch on LOGIC drift (code-level divergence). Non-fatal to the loop.
python scripts/detect_drift.py --source auto --enforce || true

# 6. Submit orders to Alpaca PAPER to match the target (reads ALPACA_SE_* creds;
#    respects the kill switch). Enabled 2026-06-29 for forward-test execution.
#    --max-gross-mult raised 1.0 -> 4.0 on 2026-07-22: validated_config() now
#    models the book uncapped (natural gross ~3.9-4.3x median/mean, per the
#    actual backtest distribution) with 1% financing — the configuration that
#    clears the H4 beats_buy_hold gate (CAGR>MaxDD>Sharpe priority) against SPY.
#    A cap of 1.0x had the real paper account running at roughly 1/4 the
#    modeled book's risk, which is why "modeled Sharpe" never matched what was
#    actually held. 4.0x also matches this account's real Reg-T buying power
#    (~4x equity, confirmed via GET /v2/account) — the tail of the gross-
#    exposure distribution (p90 6.7x, p99 9.7x) will still get scaled down on
#    high-conviction days, same mechanism as before, just anchored higher.
#    --use-cash-balance sizes the notional cap against cash instead of equity
#    (long leg fully cash-paid). Short legs, routine for this long/short trend
#    book, still draw Reg-T margin buying power regardless of this flag.
python scripts/execute_alpaca.py --paper --max-gross-mult 4.0 --use-cash-balance
