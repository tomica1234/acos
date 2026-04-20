# ACOS

ACOS is an Autonomous Coding Operating System. A user provides product
requirements in natural language, and ACOS decomposes the work across explicit
roles such as PM, Architect, Planner, Implementer, Test Writer, Reviewer,
Security Reviewer, Fixer, Summarizer, and Release Manager.

The orchestrator owns state transitions, retries, stuck and blocked handling,
tool permissions, and model selection. LLMs do not own workflow control.

## Architecture Overview

- `packages/orchestrator/`: job runner, policy engine, audit, context builder
- `packages/agents/`: role runner and role-specific prompt/config glue
- `packages/llm/`: provider registry, model adapters, routing, budgets
- `packages/mcp_client/`: local MCP-style router and fake tool environment
- `mcp_servers/`: repo, git, test, memory, and notify server skeletons
- `apps/`: CLI, API, and worker entrypoints

## Model Routing Design

ACOS does not fix itself to one model family. Every role resolves models
through:

- `ModelProviderConfig`: provider endpoint and capability metadata
- `ModelConfig`: concrete model entry with context/output limits
- `AgentModelConfig`: role-to-model mapping
- `ModelRegistry`: YAML-backed provider/model/agent catalog
- `ModelRouter`: role-aware selection, fallback, escalation, capability checks
- `ModelAdapter`: provider-specific execution layer

This allows patterns such as:

- PM and Architect on a large long-context model
- Summarizer on a cheaper model
- Fixer on a smaller model first, then escalation after repeated failures

See [docs/MODEL_ROUTING.md](/Users/tachibanashunta/wip/acos/docs/MODEL_ROUTING.md) for
the detailed routing design.

## Setup

### 1. Install Dependencies

```bash
uv sync --group dev
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Set the API key variables expected by `configs/model_providers.yaml`.

Example:

```bash
export QWEN_API_KEY=replace-me
export SMALL_MODEL_API_KEY=replace-me
export MOCK_API_KEY=dummy
```

### 3. Point ACOS at Your OpenAI-Compatible API

Edit [configs/model_providers.yaml](/Users/tachibanashunta/wip/acos/configs/model_providers.yaml).
For a local Qwen endpoint, update the provider `base_url` fields to match your
OpenAI-compatible server.

Example provider shape:

```yaml
providers:
  local_qwen:
    type: openai_compatible
    base_url: "http://localhost:8000/v1"
    api_key_env: "QWEN_API_KEY"
    supports_tools: true
    supports_json_mode: false
```

## Config Files

### `configs/model_providers.yaml`

Defines:

- provider type
- base URL
- API key env var
- timeout
- tool support
- JSON mode support
- context and output limits

### `configs/agents.yaml`

Defines, per role:

- primary model
- fallback models
- sampling settings
- context budget
- allowed tools
- output schema

Example idea:

- `pm.primary_model = qwen_35b`
- `summarizer.primary_model = qwen_small`
- `fixer.fallback_models = [qwen_35b, mock_structured]`

### `configs/model_routing.yaml`

Controls:

- fallback errors such as `timeout`, `rate_limit`, `invalid_json`
- escalation triggers such as repeated failures
- roles that require tool calling
- roles that require strict JSON

### `configs/policies.yaml`

Defines:

- deny-by-default tool policy
- release branch rules
- test command restrictions
- forbidden patch targets
- protected file and secret-path restrictions

## MCP Server Overview

ACOS routes repo, git, test, memory, and notification actions through MCP-style
tools rather than arbitrary shell execution.

Current MVP state:

- fake in-process MCP environment is implemented and used by tests
- `mcp_servers/*` contains skeleton wrappers for future standalone servers

## Local Execution

### Validate Config

```bash
acos validate-config
```

### Inspect Models

```bash
acos list-models
```

### Inspect Role Assignments

```bash
acos list-agents
```

### Resolve A Model For A Role

```bash
acos resolve-model --role implementer
acos resolve-model --role fixer --repeated-failures 2
```

### Explain Routing In Human Terms

```bash
acos explain-routing --role implementer
```

### Run A Job From YAML

Start from [job.yaml.example](/Users/tachibanashunta/wip/acos/job.yaml.example):

```bash
cp job.yaml.example job.yaml
acos run-job --file job.yaml
```

`run-job` accepts both:

- direct `JobSpec` fields such as `request_text`, `repo_path`, `target_branch`
- a friendlier job file format using `requester_input`, `title`,
  `base_branch`, `notification_channel`, `constraints`, and `workspace_root`

### Approval Commands

When ACOS hits a high-risk action, the job moves to `waiting_approval`.

```bash
acos approvals list --workspace .
acos approvals show <approval_id> --workspace .
acos approvals approve <approval_id> --workspace .
acos approvals reject <approval_id> --workspace . --reason "not acceptable"
acos jobs resume <job_id> --workspace .
```

Inside the configured workspace, normal development work is auto-allowed.
Approval is reserved for high-risk actions such as large patches, mass delete,
release-like operations, or external send operations. Critical actions remain
denied.

### Start The API

```bash
acos api
```

### Start The Worker

```bash
acos worker
acos worker --request "READMEにセットアップ手順を追加してください" --repo .
acos worker --file job.yaml.example
```

In the MVP, the worker is a simple entrypoint for running one job locally.

## CLI Reference

- `acos validate-config`
- `acos list-models`
- `acos list-agents`
- `acos resolve-model --role implementer`
- `acos resolve-model --role fixer --repeated-failures 2`
- `acos explain-routing --role implementer`
- `acos run-job --file job.yaml`
- `acos approvals list`
- `acos approvals show <approval_id>`
- `acos approvals approve <approval_id>`
- `acos approvals reject <approval_id> --reason "..."`
- `acos jobs resume <job_id>`
- `acos api`
- `acos worker`

## Testing

```bash
python3 -m compileall acos
.venv/bin/pytest
```

For a deterministic smoke test without a real provider:

```bash
python3 -m apps.cli run-demo --workspace /tmp/acos-demo
```

## Security Policy

ACOS enforces:

- deny-by-default tool policy
- workspace-only file access
- symlink escape blocking
- workspace-auto approval gateway for high-risk operations
- secret redaction before prompt, memory, audit, and notification use
- allowlisted test commands only
- protected branch restrictions
- release commit restriction to `release_manager`
- patch restrictions on tests and dependency manifests

See [docs/SECURITY.md](/Users/tachibanashunta/wip/acos/docs/SECURITY.md).

## Current Limitations

- `mcp_servers/*` are still skeletons for standalone server mode
- Docker sandbox execution is not implemented yet
- tests do not call real external model APIs
- `run-job` needs a reachable configured provider unless you switch configs to
  mock models or use `run-demo`

## Roadmap

- standalone MCP server transports
- persistent audit and job history
- stronger worker queue and background processing
- real sandbox execution
- richer semantic review and test-quality gates

## More Detail

- [docs/QUICKSTART.md](/Users/tachibanashunta/wip/acos/docs/QUICKSTART.md)
- [docs/ARCHITECTURE.md](/Users/tachibanashunta/wip/acos/docs/ARCHITECTURE.md)
- [docs/MODEL_ROUTING.md](/Users/tachibanashunta/wip/acos/docs/MODEL_ROUTING.md)
- [docs/SECURITY.md](/Users/tachibanashunta/wip/acos/docs/SECURITY.md)
