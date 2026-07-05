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

## Safety Boundary

Recovery does not relax policy. Direct `main` or `master` writes, force push,
secret access, workspace escape, sudo, arbitrary shell, production deploy, and
other blocked operations remain hard safety boundaries. Approval rejection is
only a hard stop when the rejected operation is critical; otherwise the planner
should replan around the rejected operation.
