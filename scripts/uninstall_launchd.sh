#!/bin/bash
set -euo pipefail
# Unload and remove the daily forward-reconciliation LaunchAgent.

PLIST="com.signal.engine.forward.plist"
LAUNCHD_DIR="$HOME/Library/LaunchAgents"

if [[ -f "$LAUNCHD_DIR/$PLIST" ]]; then
    if command -v launchctl >/dev/null 2>&1 && launchctl help 2>&1 | grep -q bootstrap; then
        launchctl bootout "gui/$UID/$PLIST" >/dev/null 2>&1 || true
    else
        launchctl unload "$LAUNCHD_DIR/$PLIST" >/dev/null 2>&1 || true
    fi
    rm "$LAUNCHD_DIR/$PLIST"
    echo "Unloaded and removed $PLIST"
else
    echo "$PLIST not installed"
fi
