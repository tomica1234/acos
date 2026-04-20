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
export QWEN_API_KEY=replace-me
export SMALL_MODEL_API_KEY=replace-me
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
```

## 5. Inspect Available Models

```bash
acos list-models
acos list-agents
acos resolve-model --role implementer
acos resolve-model --role fixer --repeated-failures 2
acos explain-routing --role implementer
```

## 6. Run A Job

```bash
cp job.yaml.example job.yaml
acos run-job --file job.yaml
```

The example job file supports a friendly shape with fields such as
`requester_input`, `base_branch`, `notification_channel`, `constraints`, and
`workspace_root`.

## 7. Handle Approval Requests

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

The approve and reject HTTP links are for local/dev convenience. In production,
prefer authenticated POST endpoints over GET links.

## 8. Start API Or Worker

```bash
acos api
acos worker
acos worker --request "READMEにセットアップ手順を追加してください" --repo .
```

## 9. Approval Notification Settings

`configs/policies.yaml` controls:

- request TTL
- whether CLI approval is allowed
- whether HTTP approval is allowed
- whether one-time notification links are emitted

The MVP fake notify server prints approval requests to the console-style
notification buffer. Optional Telegram or webhook delivery is future work.

## 10. Run Tests

```bash
python3 -m compileall acos
.venv/bin/pytest
```

## 11. Optional Deterministic Demo

If you want a smoke test without a real model provider:

```bash
python3 -m apps.cli run-demo --workspace /tmp/acos-demo
```
