# ACOS

ACOS is an Autonomous Coding Operating System. A user provides product
requirements in natural language, and ACOS decomposes the work across multiple
specialized roles for requirements analysis, architecture, planning,
implementation, test generation, review, deterministic test execution, fixing,
release finalization, memory summarization, and notification.

The system is explicitly orchestrated. It does not delegate job lifecycle
control to LLMs. Every role is model-routed through a registry and adapter
layer so providers and model sizes remain interchangeable.

## Features

- Orchestrator-owned state machine with retries, blocked and stuck handling
- Role-specific model configuration and dynamic model escalation
- OpenAI-compatible provider support plus a deterministic mock adapter
- Pydantic-validated agent outputs and context packets
- MCP-based tool routing for repo, git, test, memory, and notification actions
- SQLite-backed memory store for MVP persistence
- FastAPI API and CLI entrypoints
- End-to-end vertical slice tests using fake MCP tools

## Setup From Scratch

```bash
uv sync --group dev
cp .env.example .env
```

For local Qwen or any OpenAI-compatible endpoint, set the provider URL and API
key environment variables expected by `configs/model_providers.yaml`.

Example:

```bash
export QWEN_API_KEY=replace-me
export SMALL_MODEL_API_KEY=replace-me
```

## OpenAI-Compatible Qwen Configuration

`configs/model_providers.yaml` defines providers and concrete model records.
Each provider includes:

- provider type
- base URL
- API key environment variable name
- timeout
- tool support
- JSON mode support
- streaming support
- provider-wide token limits

Each model includes:

- provider binding
- display name
- max context and output tokens
- tool calling support
- structured output support
- JSON repair capability
- tags

## Role-Specific Model Settings

`configs/agents.yaml` assigns models by role. This is where you decide that:

- PM uses a long-context model
- Architect uses a design-capable model
- Implementer uses a coding model
- Reviewer uses a strict low-temperature model
- Summarizer uses a smaller cheaper model
- Fixer starts on a smaller model and escalates when failures repeat

`configs/model_routing.yaml` controls:

- fallback errors
- escalation thresholds
- roles that require tools
- roles that require strict JSON

## CLI

Validate all config cross-references:

```bash
acos validate-config
```

List loaded models:

```bash
acos list-models
```

List role-to-model assignments:

```bash
acos list-agents
```

Resolve the currently selected model for a role:

```bash
acos resolve-model --role implementer
```

Resolve the model after repeated failures:

```bash
acos resolve-model --role fixer --repeated-failures 2
```

Explain why routing chose a model and which fallback or escalation rules apply:

```bash
acos explain-routing --role implementer
```

Run a job from YAML:

```bash
acos run-job --file job.yaml
```

Start the API:

```bash
acos api
```

Start the worker:

```bash
acos worker --request "build a small API" --repo .
```

## Repository Layout

- `apps/`: API, worker, and CLI entrypoints
- `packages/`: reusable ACOS core modules
- `mcp_servers/`: MCP server skeletons for repo, git, test, memory, notify
- `configs/`: model, agent, routing, and policy configuration
- `docs/`: implementation and operations documentation
- `tests/`: schema, routing, orchestration, and vertical slice coverage

## Config Files

- `configs/model_providers.yaml`: providers and model catalog
- `configs/agents.yaml`: per-role model defaults and tool permissions
- `configs/model_routing.yaml`: fallback, escalation, and capability rules
- `configs/policies.yaml`: deny-by-default MCP and git policy

## Testing

```bash
make compile
make pytest
python -m apps.cli run-demo --workspace /tmp/acos-demo
```

## Security Policy

- LLMs do not manage orchestration state
- secrets are redacted before prompts, memory, audit, and notifications
- workspace escapes are blocked
- arbitrary shell is blocked
- test execution is allowlisted
- direct `main`/`master` writes are blocked
- force push, production deploy, and destructive migrations are blocked

## Current Limitations

- MCP server implementations are still local skeletons and fake adapters for MVP
- the Docker sandbox runner is structural only
- OpenAI-compatible integration is implemented, but tests do not call real APIs
- branch and patch handling are safe MVP abstractions, not full git-native automation

See [docs/QUICKSTART.md](/Users/tachibanashunta/wip/acos/docs/QUICKSTART.md) for
setup details.
