# Architecture

## Control Plane

`packages.orchestrator.job_runner.JobRunner` is the control plane. It advances a
job through explicit states and owns retries, policy checks, quality gates,
tool execution, and model selection inputs.

`packages.orchestrator.autonomy_governor.AutonomyGovernor` decides how ACOS
recovers from non-policy stops. Its default is to keep the job moving by
recording a PM strategy change, adding recovery constraints to the job record,
and passing that plan into the next agent context. Only policy hard stops are
converted into human inspection paths.

## Data Plane

- `packages.schemas.*` define every durable and agent-facing structure.
- `packages.llm.registry.ModelRegistry` loads provider, model, and role config.
- `packages.llm.routing.ModelRouter` selects the model for each role.
- `configs/model_routing.yaml` escalates repeated implementation and fixer
  failures from the default Ornith model to `ncmoe40_q4`; config validation
  rejects no-op escalation rules that point back to the role primary model.
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

When tests, planning quality gates, completion integrity, stage limits, or
supervision stalls fail, ACOS records `failure_analysis`,
`failure_diagnosis` when available, `autonomous_recovery_plan`, and
`pm_interventions`. The next run uses those records as context instead of
asking the user to choose a recovery path.

The runtime separates status classes:

- hard terminal: `DONE`, `CANCELLED`, `POLICY_HARD_STOP`
- waiting: `WAITING_APPROVAL`, `WAITING_RUNTIME`, `PROVIDER_UNAVAILABLE`, `PAUSED`
- recoverable: `BLOCKED`, `STUCK`, `FAILED`
- runnable: submitted, queued, running, planning, implementing, testing,
  recovering, replanning, diagnosing, strategy-change, and provider retry states

`RecoveryGovernor` decides the strategy. `RecoveryExecutor` executes the saved
steps, writes checkpoints, updates constraints, and returns the job to the next
actor. This keeps `max_attempts_exceeded`, repeated failures, bad task graphs,
review rejection, and completion integrity failures inside ACOS instead of
turning them into human-facing terminal states.

`SQLiteJobStore` persists job records, recovery plans, checkpoints, runtime
issues, worker heartbeats, leases, tasks, and notifications. `WorkerDaemon`
polls runnable jobs, acquires a lease, normalizes recoverable statuses to
`RECOVERING`, and resumes after stale heartbeat or provider recovery.

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
