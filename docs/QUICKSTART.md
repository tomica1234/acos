# Quickstart

## 1. Install Dependencies

```bash
uv sync --group dev
```

## 2. Configure `.env`

```bash
cp .env.example .env
```

Set the API keys expected by `configs/model_providers.yaml`.

Example:

```bash
export OPENAI_API_KEY=replace-me
export MOCK_API_KEY=dummy
```

## 3. Edit Config

Update these files for your environment:

- `configs/model_providers.yaml`
  - set `base_url`
  - set `api_key_env`
- `configs/agents.yaml`
  - choose primary and fallback models per role
- `configs/model_routing.yaml`
  - tune fallback and escalation rules
- `configs/policies.yaml`
  - confirm workspace sandbox, approval policy, tool policy, and git policy

If you want the workspace sandbox to follow the job file, keep:

```yaml
workspace:
  root_from_job: true
```

Then set `workspace_root` or `repo_path` in your job file.

## 4. Validate Config

```bash
acos validate-config
acos check-provider --provider local_qwen
acos check-model --model qwen_35b
```

## 5. Inspect Available Models

```bash
acos list-models
acos list-agents
acos resolve-model --role implementer
acos resolve-model --role fixer --repeated-failures 2
acos explain-routing --role implementer
```

## 6. Start Durable Worker

Foreground:

```bash
acos daemon start --foreground --workspace .
```

Or direct worker:

```bash
acos worker run --forever --repo .
```

## 7. Submit A Job

```bash
cp job.yaml.example job.yaml
acos jobs submit --file job.yaml --workspace .
acos jobs watch <job_id> --workspace .
```

The example job file supports a friendly shape with fields such as
`requester_input`, `base_branch`, `notification_channel`, `constraints`, and
`workspace_root`.

## 8. Handle Approval Requests

Normal development work inside the configured workspace is auto-allowed. High
risk actions pause the job in `waiting_approval` and emit a notification.

Console approval flow:

```bash
acos approvals list --workspace .
acos approvals show <approval_id> --workspace .
acos approvals approve <approval_id> --workspace .
acos approvals reject <approval_id> --workspace . --reason "not acceptable"
acos jobs resume <job_id> --workspace .
```

## 9. Runtime Wait And Recovery

If the model provider goes down, ACOS pauses the job in `waiting_runtime`
instead of marking it failed.

```bash
acos runtime status --workspace .
acos runtime check --workspace .
acos check-provider --provider local_qwen
acos check-model --model qwen_35b
acos jobs resume <job_id> --workspace .
```

The approve and reject HTTP links are for local/dev convenience. In production,
prefer authenticated POST endpoints over GET links.

## 10. Start API Or Worker

```bash
acos api
acos worker run --repo . --request "READMEにセットアップ手順を追加してください"
```

## 11. Approval Notification Settings

`configs/policies.yaml` controls:

- request TTL
- whether CLI approval is allowed
- whether HTTP approval is allowed
- whether one-time notification links are emitted

The MVP fake notify server prints approval requests to the console-style
notification buffer. Optional Telegram or webhook delivery is future work.

## 12. Run Tests

```bash
python3 -m compileall acos
.venv/bin/pytest
```

## 13. Optional Deterministic Demo

If you want a smoke test without a real model provider:

```bash
python3 -m apps.cli run-demo --workspace /tmp/acos-demo
```
