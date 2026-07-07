# Runtime Recovery

ACOS treats most execution failures as strategy-change triggers, not as human
terminal states. After the initial requirements definition, the runtime should
keep responsibility for moving the job toward `DONE` unless a policy hard stop
is reached.

## Status Model

Hard terminal statuses are intentionally narrow:

- `DONE`
- `CANCELLED`
- `POLICY_HARD_STOP`

Recoverable statuses are not settled:

- `BLOCKED`
- `STUCK`
- `FAILED`

Workers normalize recoverable records through `RECOVERING` and then call
`RecoveryGovernor`. Waiting statuses remain paused until their external
condition changes:

- `WAITING_APPROVAL`
- `WAITING_RUNTIME`

## RecoveryGovernor

`packages/orchestrator/recovery_governor.py` converts a failed record into a
recovery plan. The plan is written to:

- `record.runtime_state["recovery_plan"]`
- `record.outputs["recovery_history"]`

The plan records the trigger, strategy, next actor, next status, checkpoint
policy, and constraints that should be passed into the next agent context.
Recoverable failures are also written to
`record.runtime_state["current_recovery_event"]`,
`record.runtime_state["last_recoverable_error"]`, and
`record.outputs["recovery_events"]`. `record.last_error` is reserved for hard
terminal failures such as `POLICY_HARD_STOP`, so the UI can distinguish
"recovering with a strategy" from "stopped and needs a human".

Common mappings:

- `max_attempts_exceeded` -> diagnose failure, then replan the task
- `same_failure_threshold_reached` -> diagnose, expand context, and retry with
  a different strategy
- `design_review_max_attempts_exceeded` -> revise PRD and architecture
- `acceptance_review_max_attempts_exceeded` -> split the task or redefine
  acceptance
- `completion_integrity_failed` -> replan with required artifacts and evidence
- target file missing -> return to implementer
- test patch quality failure -> return to test writer
- provider unavailable -> `WAITING_RUNTIME`, then auto resume when healthy
- critical policy denial -> `POLICY_HARD_STOP`

## Context Retrieval

Recovery depends on better context than a fixed first-N file list. JobRunner now
builds a retrieval trace from:

- `task.target_files`
- `task.required_artifacts`
- git modified files
- failed test output and exception paths
- recent failure diagnoses
- a repo map, when the role may inspect the tree

The selected files and reasons are exposed in
`ContextPacket.metadata["retrieval_trace"]` and in
`__retrieval_trace__.txt` inside the context packet.

When a recovery plan sets `expand_context=True`, retrieval also searches for
symbols and exception fragments from the failure signature, root cause, and test
output. The trace is saved to both `record.runtime_state["retrieval_trace"]`
and `record.outputs["retrieval_trace"]`.

## RecoveryExecutor

`RecoveryGovernor` writes a durable `record.runtime_state["recovery_plan"]`
with `id`, `status`, `trigger`, `strategy`, `current_step_index`, `steps`,
`executed_steps`, `next_actor`, `next_status`, constraints, and timestamps.
`RecoveryExecutor` consumes these steps in order and checkpoints each step.

Supported steps include diagnosis, context expansion/compaction, PRD and
architecture revision, task replanning/splitting, returning to implementer,
test writer, or fixer, retrying with a different strategy or escalated model,
waiting for runtime recovery, avoiding rejected operations, and completion
audit.

`max_attempts_per_task` and `max_same_failure_repeats` are strategy-change
triggers. They are not stop conditions.

## Durable Worker

`SQLiteJobStore` can persist jobs, tasks, recovery plans, checkpoints, worker
heartbeats, leases, runtime issues, and notifications. A worker can run with:

```bash
acos-worker --repo . --sqlite-path .acos/acos.sqlite3 --forever
```

If a process crashes, expired leases and stale heartbeats are moved to
`RECOVERING`. Provider outages move to `WAITING_RUNTIME`; provider recovery
moves jobs to `RESUMING`.

## Safety Boundary

Recovery does not relax policy. Direct `main` or `master` writes, force push,
secret access, workspace escape, sudo, arbitrary shell, production deploy, and
other blocked operations remain hard safety boundaries. Approval rejection is
only a hard stop when the rejected operation is critical; otherwise the planner
should replan around the rejected operation.
