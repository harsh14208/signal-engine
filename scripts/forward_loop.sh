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

# 1. Refresh ETF price cache (yfinance).
python scripts/warm_cache.py

# 2. Generate target positions for the next session.
python scripts/generate_targets.py --source auto

# 3. Mark the shadow return for the day that just closed.
python scripts/shadow_book.py --source auto

# 4. Reconcile shadow returns vs the backtest and update guardrails.
python scripts/reconcile.py --source auto

# 5. Replay stored decisions and detect engine drift (Phase 3). --enforce engages
#    the kill switch on LOGIC drift (code-level divergence). Non-fatal to the loop.
python scripts/detect_drift.py --source auto --enforce || true

# 6. Submit orders to Alpaca PAPER to match the target (reads ALPACA_SE_* creds;
#    respects the kill switch). Enabled 2026-06-29 for forward-test execution.
python scripts/execute_alpaca.py --paper
