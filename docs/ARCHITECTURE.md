# Architecture

## Control Plane

`packages.orchestrator.job_runner.JobRunner` is the control plane. It advances a
job through explicit states and owns retries, policy checks, quality gates,
tool execution, and model selection inputs.

## Data Plane

- `packages.schemas.*` define every durable and agent-facing structure.
- `packages.llm.registry.ModelRegistry` loads provider, model, and role config.
- `packages.llm.routing.ModelRouter` selects the model for each role.
- `packages.agents.runner.AgentRunner` reads the role config, asks
  `ModelRouter` for a `ModelSelection`, resolves the concrete adapter through
  `ModelRegistry`, executes policy-approved MCP tools, retries invalid JSON once
  with a repair prompt, then falls back or escalates according to routing
  policy.
- `packages.mcp_client.router.MCPRouter` invokes only registered MCP tools.
- `packages.memory.store.SQLiteMemoryStore` persists redacted memory entries.

## Role Flow

1. PM creates a PRD.
2. Architect produces the architecture plan.
3. Planner produces the task graph.
4. Implementer proposes file patches.
5. Test Writer proposes test patches.
6. Reviewer and Security Reviewer inspect the changes.
7. Test Runner executes deterministic tests.
8. Fixer proposes corrective patches if tests fail.
9. Release Manager proposes the release artifact and commit message.
10. Summarizer stores memory and prepares a user-facing summary.

## Tooling Boundary

Agents do not edit the file system directly. They return structured outputs
that the orchestrator turns into MCP tool calls. This keeps tool execution
auditable and policy-gated.

## Audit Boundary

`packages.orchestrator.audit.AuditRecorder` records:

- model selection events with `model_key`, `provider_key`, and `routing_reason`
- model call events with redacted hashes and token estimates
- tool call events with hashed inputs and outputs

Raw prompts, API keys, and secret-like strings are not stored in audit event
metadata.
