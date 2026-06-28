# Forward-deployment runbook (Tier A)

This is the no-broker shadow paper loop. It answers the question the parent
project never instrumented: **does the live return actually track the backtest?**

Only after the shadow book confirms tracking should you uncomment the Alpaca
paper-execution line in `scripts/forward_loop.sh`.

## What the loop does

1. `scripts/warm_cache.py` — refreshes `data/prices_universe.parquet` from yfinance.
2. `scripts/generate_targets.py` — runs the validated config
   (core 19 + governor + 30% buffer + COT) and appends a target record to
   `data/live_targets.jsonl`.
3. `scripts/shadow_book.py` — marks the next-day shadow return using closing
   prices and appends it to `data/live_returns.csv`.
4. `scripts/reconcile.py` — compares live vs modeled returns, persists a JSON
   report under `data/reconciliation/`, and updates the kill switch if guardrails
   fire.
5. `scripts/execute_alpaca.py` *(commented out)* — submits delta notional orders
   to Alpaca paper once shadow tracking is confirmed.

## Quick start

```bash
# One-time: install the daily LaunchAgent (runs at 17:30 local time).
./scripts/install_launchd.sh

# Or run manually once after the US close:
./scripts/forward_loop.sh
```

## Start / stop / status

```bash
# Start (load agent)
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.signal.engine.forward.plist

# Stop (unload agent)
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.signal.engine.forward.plist

# Check status
launchctl list | grep com.signal.engine.forward

# View recent logs
tail -f data/launchd_out.log data/launchd_err.log
```

Use `./scripts/install_launchd.sh` and `./scripts/uninstall_launchd.sh` for a
safer wrapper that handles old macOS `load/unload` as well as modern
`bootstrap/bootout`.

## Artifacts

| File | Purpose |
|------|---------|
| `data/live_targets.jsonl` | Date-stamped target units, notional, forecast, config snapshot. |
| `data/live_returns.csv` | Realised daily shadow/live returns. |
| `data/reconciliation/YYYY-MM-DD.json` | Daily correlation, tracking error, drift, edge-decay alarm. |
| `data/kill_switch.json` | Set to `{"paused": true, ...}` if a guardrail fires. |
| `data/broker_orders.jsonl` | Submitted Alpaca orders (once enabled). |

## Reading the reconciliation report

A healthy report looks like:

```markdown
- n=252  corr=0.94  tracking_error=2.1%  drift=0.3%  ✅ aligned
- current 0.72  median 0.68  min 0.12  max 1.05  % windows <0: 5%  ✅ healthy
```

Watch for:

- `⚠ NOT aligned` — live returns are diverging from the backtest (corr ≤ 0.80
  or tracking error ≥ 5%). This fires the kill switch.
- `⚠ ALARM` — rolling 1-year live Sharpe has dropped below 0. This also fires
  the kill switch.

## Alarms and the kill switch

`scripts/reconcile.py` writes `data/kill_switch.json` when:

- reconciliation is **not aligned**, or
- the rolling 1-year live Sharpe falls below `--alarm-floor` (default 0.0).

When the kill switch is engaged:

- `execute_alpaca.py` will refuse to submit new orders.
- The shadow book and reconciliation reports keep running so you can diagnose.
- Resolve manually; delete or edit `data/kill_switch.json` to resume trading.

## Disabling broker execution while keeping the shadow book

Leave `execute_alpaca.py` commented out in `scripts/forward_loop.sh`. The shadow
book and reconciliation run without any broker connection, which is the intended
first stage.

## Forward-confirming COT (A7)

Every target record stores `use_cot` and `cot_as_of`. As `data/live_returns.csv`
accrues, you can split live performance by `use_cot=true/false` to verify whether
the +0.02 walk-forward COT edge holds out of sample. The decision to promote COT
to the default config waits on that forward evidence.

## Credentials for Alpaca paper (optional)

Create `.env` in the repo root:

```bash
alpaca_api_key=PK...
alpaca_api_secret=...
```

Then run:

```bash
python scripts/execute_alpaca.py --paper
```

For live trading, use `alpaca_live_api_key` / `alpaca_live_api_secret` and pass
`--live`. Do not enable live until the shadow book has tracked the backtest for
an extended period.

## Holidays / missing data

The scripts are idempotent:

- `generate_targets.py` skips if a target for the latest price date already exists.
- `shadow_book.py` skips if the next-day close is not yet available or already recorded.
- `reconcile.py` reports "insufficient live data" until at least 20 shared dates exist.

If a market holiday causes no new close, the loop records nothing for that day.
