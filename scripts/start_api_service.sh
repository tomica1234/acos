#!/bin/zsh
set -eu

cd /Users/tachibanashunta/wip/acos
mkdir -p .acos/logs .acos/jobs-ui .acos/ui-cycles

exec .venv/bin/uvicorn apps.api.main:create_app --factory --host 127.0.0.1 --port 8080
