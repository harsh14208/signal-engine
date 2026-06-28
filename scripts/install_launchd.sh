#!/bin/bash
set -euo pipefail
# Install and load the daily forward-reconciliation LaunchAgent.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST="com.signal.engine.forward.plist"
LAUNCHD_DIR="$HOME/Library/LaunchAgents"

mkdir -p "$LAUNCHD_DIR"
cp "$REPO_ROOT/scripts/launchd/$PLIST" "$LAUNCHD_DIR/"

# macOS 10.10+ uses bootstrap/bootout. Older systems can use load/unload.
if command -v launchctl >/dev/null 2>&1 && launchctl help 2>&1 | grep -q bootstrap; then
    launchctl bootout "gui/$UID/$PLIST" >/dev/null 2>&1 || true
    launchctl bootstrap "gui/$UID" "$LAUNCHD_DIR/$PLIST"
else
    launchctl unload "$LAUNCHD_DIR/$PLIST" >/dev/null 2>&1 || true
    launchctl load "$LAUNCHD_DIR/$PLIST"
fi

echo "Installed and loaded $PLIST (runs daily at 17:30 local time)"
