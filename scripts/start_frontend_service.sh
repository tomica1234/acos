#!/bin/zsh
set -eu

cd /Users/tachibanashunta/wip/acos/frontend
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

exec /opt/homebrew/bin/npm run dev -- --host 127.0.0.1 --port 5174
