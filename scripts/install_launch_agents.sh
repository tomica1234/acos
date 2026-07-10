#!/bin/zsh
set -eu

mkdir -p "$HOME/Library/LaunchAgents" /Users/tachibanashunta/wip/acos/.acos/logs

cp /Users/tachibanashunta/wip/acos/launchd/com.tachibanashunta.acos-api.plist \
  "$HOME/Library/LaunchAgents/com.tachibanashunta.acos-api.plist"
cp /Users/tachibanashunta/wip/acos/launchd/com.tachibanashunta.acos-frontend.plist \
  "$HOME/Library/LaunchAgents/com.tachibanashunta.acos-frontend.plist"

launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.tachibanashunta.acos-api.plist" 2>/dev/null || true
launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.tachibanashunta.acos-frontend.plist" 2>/dev/null || true

launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.tachibanashunta.acos-api.plist"
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.tachibanashunta.acos-frontend.plist"
launchctl enable "gui/$(id -u)/com.tachibanashunta.acos-api"
launchctl enable "gui/$(id -u)/com.tachibanashunta.acos-frontend"
launchctl kickstart -k "gui/$(id -u)/com.tachibanashunta.acos-api"
launchctl kickstart -k "gui/$(id -u)/com.tachibanashunta.acos-frontend"

echo "ACOS launch agents installed and started."
