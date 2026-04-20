# AGENTS.md

## Repo Mission

This repository implements ACOS: Autonomous Coding Operating System.

## Core Architecture Rules

- ACOS must be implemented in this repository, not as Codex-specific settings.
- LLMs must not own state transitions; the Orchestrator owns state, retries,
  stuck detection, blocked detection, tool permissions, and model selection.
- Models must not be fixed in code paths; they are selected through
  `ModelRegistry`, `ModelAdapter`, and `ModelRouter`.
- Different roles must be able to use different models and providers.
- Dynamic model escalation must be supported when task difficulty or repeated
  failures exceed configured thresholds.
- Agent output must always be validated against Pydantic schemas.
- Agents must receive only the necessary information through a Context Packet.
- Real file editing, git operations, test execution, memory storage, and
  notifications must go through MCP servers.
- All tool calls and model calls must be written to the audit trail.

## Security And Safety Rules

- Never put secrets into prompts, memory, logs, or notifications.
- Workspace-external access is forbidden.
- Arbitrary shell execution is forbidden.
- Direct writes to `main` or `master` are forbidden.
- Force push is forbidden.
- Production deploy operations are forbidden.
- Destructive database migrations are forbidden.
- Implementer must not run tests.
- Fixer must not weaken tests.
- Release Manager is the only role allowed to commit.
- Deleting tests to manufacture success is forbidden.

## Implementation Expectations

- Use Python 3.11+.
- Use Pydantic v2 for schemas and validation.
- Use FastAPI for the API surface.
- Use OpenAI-compatible adapters for runtime model calls.
- Use SQLite for MVP memory persistence and in-memory state for MVP job state.
- Keep the Docker sandbox runner skeletal but structurally ready.
- Prefer explicit, inspectable orchestration over opaque agent autonomy.

## Verification

- `uv sync --group dev`
- `make compile`
- `make pytest`
