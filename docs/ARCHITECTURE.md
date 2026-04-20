# Architecture

## Control Plane

`packages.orchestrator.job_runner.JobRunner` is the control plane. It advances a
job through explicit states and owns retries, policy checks, quality gates,
tool execution, approval pauses, and model selection inputs.

## Data Plane

- `packages.schemas.*` define every durable and agent-facing structure.
- `packages.llm.registry.ModelRegistry` loads provider, model, and role config.
- `packages.llm.routing.ModelRouter` selects the model for each role.
- `packages.orchestrator.policy.PolicyEngine` classifies every tool request as
  `allow`, `allow_and_audit`, `require_approval`, or `deny`.
- `packages.orchestrator.workspace.WorkspacePolicy` enforces workspace-root
  confinement, symlink escape blocking, and forbidden path checks.
- `packages.orchestrator.approval.ApprovalGateway` persists approval requests in
  SQLite and issues one-time approval links.
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

## Approval Flow

1. A role or orchestrator stage requests an MCP tool.
2. `PolicyEngine.classify_tool_call()` evaluates the request against the
   workspace sandbox and risk rules.
3. `ALLOW` executes immediately.
4. `ALLOW_AND_AUDIT` executes immediately and records an explicit policy event.
5. `REQUIRE_APPROVAL` creates an `ApprovalRequest`, stores it in SQLite, sends a
   notification, and moves the job to `waiting_approval`.
6. CLI or HTTP approval resolves the request.
7. `JobRunner.resume_job()` consumes the approved operation once and resumes from
   the stored phase. Rejected or expired approvals transition the job to
   `blocked`.

Current MVP behavior is job-level pause and resume. Task-level parallel progress
while another task waits for approval is not implemented yet.

## Notification Flow

- `notify_server.send_approval_request` emits:
  - `approval_id`
  - `job_id`
  - `risk_level`
  - `operation`
  - `reason`
  - one-time approve and reject URLs
  - CLI fallback command
- The fake notify server writes these messages to console-style buffers for
  deterministic local testing.

## Audit Boundary

`packages.orchestrator.audit.AuditRecorder` records:

- model selection events with `model_key`, `provider_key`, and `routing_reason`
- model call events with redacted hashes and token estimates
- tool call events with hashed inputs and outputs
- policy decisions with operation, risk level, and approval id when present
- approval lifecycle events such as requested, approved, rejected, and expired

Raw prompts, API keys, and secret-like strings are not stored in audit event
metadata.
