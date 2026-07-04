#!/bin/zsh
set -eu

launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.tachibanashunta.acos-api.plist" 2>/dev/null || true
launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.tachibanashunta.acos-frontend.plist" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/com.tachibanashunta.acos-api.plist"
rm -f "$HOME/Library/LaunchAgents/com.tachibanashunta.acos-frontend.plist"

echo "ACOS launch agents removed."
