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
- AutonomyGovernor recovery that keeps jobs moving until completion unless a
  policy hard stop is reached
- RecoveryGovernor runtime recovery: `BLOCKED`, `STUCK`, and `FAILED` are
  recoverable strategy-change triggers, while only `DONE`, `CANCELLED`, and
  `POLICY_HARD_STOP` are hard terminal states
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

For local Ornith or any OpenAI-compatible endpoint, set the provider URL and API
key environment variables expected by `configs/model_providers.yaml`. The
default model record targets `ornith-1.0-35b-Q4_K_M.gguf` served from a local
OpenAI-compatible `/v1` endpoint.

Example:

```bash
export ORNITH_API_KEY=replace-me
export LOCAL_ORNITH_BASE_URL=http://localhost:8000/v1
```

## OpenAI-Compatible Ornith Configuration

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
- Summarizer uses the same local Ornith model for memory updates
- Fixer can escalate repeated failures through routing policy while staying on
  Ornith by default, and escalates repeated failures to `ncmoe40_q4`

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

`run-job` applies the same strict quality gates used by the API and worker
entrypoints: PRD quality, task acceptance criteria, required artifacts,
completion integrity, test evidence, test patch evidence, and stage review.

Run a larger autonomous job in guarded stages:

```bash
acos plan-job --request "Build a project tracker with auth and tests" --repo-path . --jobs-dir .acos/jobs --summary-file .acos/plan-summary.json --preflight-provider local_ornith
acos plan-job --request "Build a project tracker with auth and tests" --repo-path . --jobs-dir .acos/jobs --summary-file .acos/plan-summary.json --preflight-provider local_ornith --supervise-after-planning --supervise-max-cycles 10 --supervise-steps-per-cycle 1 --supervise-summary-file .acos/final-summary.json --supervise-summary-dir .acos/cycles --supervise-preflight-provider local_ornith
acos run-autonomous --file job.yaml --jobs-dir .acos/jobs --max-steps 3
acos run-supervised --file job.yaml --jobs-dir .acos/jobs --max-cycles 10 --steps-per-cycle 1 --max-stalled-cycles 3 --max-runtime-seconds 3600 --summary-file .acos/final-summary.json --summary-dir .acos/cycles
acos run-supervised --request "Build a project tracker with auth and tests" --repo-path . --jobs-dir .acos/jobs --plan-first --max-cycles 10 --steps-per-cycle 1 --summary-file .acos/final-summary.json --summary-dir .acos/cycles --preflight-provider local_ornith
acos run-supervised --request "Build a project tracker with auth and tests" --repo-path . --jobs-dir .acos/jobs --max-cycles 10
acos run-supervised --request "Build the app from this PRD" --repo-path . --jobs-dir .acos/jobs --autonomous-until-done --summary-file .acos/final-summary.json --summary-dir .acos/cycles --preflight-provider local_ornith
acos run-supervised --request "Build a project tracker" --repo-path . --preflight-provider local_ornith
acos run-job --file job.yaml --jobs-dir .acos/jobs --large-autonomous
acos job-status --job-id <job-id> --jobs-dir .acos/jobs
acos continue-job --job-id <job-id> --jobs-dir .acos/jobs --max-steps 3
acos continue-job --job-id <job-id> --jobs-dir .acos/jobs --max-steps 3 --json-summary
acos continue-job --job-id <job-id> --jobs-dir .acos/jobs --max-steps 3 --json-summary --summary-file .acos/last-summary.json
acos supervise-job --job-id <job-id> --jobs-dir .acos/jobs --max-cycles 10 --steps-per-cycle 1 --max-stalled-cycles 3 --summary-file .acos/final-summary.json --summary-dir .acos/cycles
acos supervise-job --job-id <job-id> --jobs-dir .acos/jobs --autonomous-until-done --summary-file .acos/final-summary.json --summary-dir .acos/cycles
acos job-status --job-id <job-id> --jobs-dir .acos/jobs --next-supervise-command --supervise-max-cycles 10 --supervise-steps-per-cycle 1 --supervise-summary-file .acos/final-summary.json --supervise-preflight-provider local_ornith
acos job-status --job-id <job-id> --jobs-dir .acos/jobs --next-continue-command --continue-max-steps 3 --continue-json-summary
acos job-status --job-id <job-id> --jobs-dir .acos/jobs --next-command
acos resume-job --job-id <job-id> --jobs-dir .acos/jobs --bump-stage-limit
```

`--large-autonomous` keeps those strict gates enabled and adds a conservative
one-stage execution limit for larger autonomous jobs.
The PRD quality gate also requires enough acceptance tests to cover every
declared `small_parts` item, so large requests are decomposed into verifiable
units before coding begins. Task graph validation records `small_part_coverage`
and `acceptance_test_coverage` so supervisors can audit which implementation
task is responsible for each requirement slice and acceptance check.

`--autonomous-until-done` changes supervision from operator-assisted progress to
PM-owned recovery. Repeated test failures, completion integrity failures, PRD
quality failures, invalid task graphs, stage limits, stalls, and runtime
limits are treated as recoverable strategy-change events. The job record stores
`autonomous_recovery_plan` and `pm_interventions` so the next agent context sees
why ACOS changed approach. Only `policy_hard_stop:*` style errors require human
inspection.
Recoverable errors are surfaced as `current_recovery_event` and
`last_recoverable_error`; `last_error` is reserved for hard-stop conditions.
Use `plan-job` when you want ACOS to spend a separate pass on PRD, architecture,
task graph validation, and planning evidence before any implementation patches
are applied. A successful planning run remains resumable through `continue-job`
or `supervise-job`; `job-status --json` exposes `planning_summary` with
planning completion, implementation readiness, and small-part coverage. Add
`--preflight-provider local_ornith` when planning should first verify that the
model endpoint is reachable. Add `--supervise-after-planning` when a successful
planning pass should return a ready-to-run `supervise-job` command for the long
implementation loop.
Use `run-autonomous` to start a new job and immediately continue through several
guarded stages. Use `run-supervised` to start a new large autonomous job and let
ACOS repeatedly call the continuation logic itself for a fixed number of cycles,
writing per-cycle summaries to `--summary-dir` and a final rollup to
`--summary-file`. Add `--plan-first` when the same command should complete the
planning-only PRD, architecture, and task validation gate before entering the
supervised implementation loop. Pass either `--file job.yaml` or a direct `--request` with
`--repo-path` when no YAML file is needed. `--max-stalled-cycles` stops
supervision when repeated cycles produce the same progress marker, avoiding
unbounded retry loops that are not changing task completion or patch progress.
When this happens, the final JSON includes `stall_analysis`, and
`operator_decision` returns a supervision recovery path. With
`--autonomous-until-done`, ACOS records a PM strategy change and continues
instead of returning to the user.
`--max-runtime-seconds` stops supervision after the current cycle once the
runtime budget is reached and returns `terminal_reason: runtime_limit`.
Use `--preflight-provider local_ornith` on `run-supervised` or `supervise-job`
to probe the OpenAI-compatible `/models` endpoint before starting or resuming a
long run and before each supervised cycle; unhealthy providers return
`terminal_reason: provider_unhealthy` without entering the implementation loop
for that cycle. The same summary includes `operator_decision` with
`resume_action: check_provider`, the latest preflight result, and
`provider_events` history for pre-start and per-cycle probes. If the stopped
command was supervising an existing persisted job, the summary also includes
`next_supervise_command` so automation can retry the same guarded loop after the
provider recovers.
Use `job-status --json` when another script needs to inspect progress and decide
whether to resume, raise the stage limit, or retry a failed stage; the JSON
includes a `supervision` section with `next_supervise_command` for incomplete
jobs. When a repeated or classified failure requires guarded recovery,
`operator_decision` includes `failure_classification` and `recommended_recovery`
so automation can see the intended recovery strategy and constraints before
using the explicit recovery override. The same JSON includes `operator_summary`,
a compact control view with the chosen action, command source, override
requirement, and continuation/supervision availability. Use
`job-status --next-supervise-command` to print a ready-to-run supervision
command for an existing incomplete job, or `job-status --next-continue-command`
to print a lower-level continuation command, with `--continue-max-steps` and
`--continue-json-summary` when a supervising script wants a ready-to-run
multi-step command. Add `--supervise-summary-file` when the generated
supervision command should keep updating the same final JSON file, and
`--supervise-preflight-provider` when it should keep probing the model endpoint
before each resumed run. Use `continue-job` when ACOS should make that resume
decision from an existing persisted job state.
The legacy `job-status --next-command` still prints the older `resume-job`
command when available, and falls back to `operator_decision.command` for
guarded recovery cases such as repeated failures or completion integrity
failures.
Add `--json-summary` to `run-autonomous` or `continue-job` when automation needs
the final progress summary and the number of continuation steps that actually
ran. Add `--summary-file` with `--json-summary` to persist that same
machine-readable summary for a supervisor, CI job, or later resume audit. Use
`supervise-job` when the job already exists and ACOS should supervise its
continuation cycles. The JSON summary includes `terminal_reason`,
`next_action`, `can_continue`, `step_events`, `next_continue_cli_args`,
`next_continue_command`, and `operator_decision` so supervisors can distinguish
completed jobs from jobs that simply reached the current step limit, audit each
continuation step, and launch the next continuation without inspecting nested
summary fields.
Supervised summaries also include `can_supervise_continue`,
`next_supervise_cli_args`, and `next_supervise_command` when a stopped
supervision run can be resumed as another guarded supervision run. Per-cycle
summaries include the cycle `operator_decision`, and stalled stopping cycles
also include `stall_analysis`, so a supervisor can audit which exact cycle
changed from continuation to inspection. Runtime-limited stopping cycles include
`runtime_analysis` and an inspection decision for the same reason; the final
summary also carries the same `runtime_analysis` alongside the supervise retry
command. Final supervised and provider-preflight stop summaries also include
`stop_summary`, a compact aggregation of the terminal reason, operator action,
next operator command, and any stall, runtime, or provider analysis needed by an
external supervisor.

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

## Runtime Recovery

ACOS writes recovery plans to job runtime state when tests, quality gates,
reviews, or required artifacts fail. The next worker cycle uses that plan as
agent context and changes strategy instead of repeating the same fixer loop.
Recoverable failures are tracked in recovery events, while `last_error` is kept
for policy hard stops and other hard terminal failures.
See [docs/RUNTIME_RECOVERY.md](docs/RUNTIME_RECOVERY.md).

Hard terminal statuses are only `DONE`, `CANCELLED`, and
`POLICY_HARD_STOP`. `BLOCKED`, `STUCK`, and `FAILED` are recoverable signals:
the worker moves them through `RECOVERING`, asks `RecoveryGovernor` for a plan,
and `RecoveryExecutor` consumes the plan before normal execution resumes.

Durable execution can use `SQLiteJobStore`:

```bash
acos-worker --repo . --sqlite-path .acos/acos.sqlite3 --forever
```

Provider outages move jobs to `WAITING_RUNTIME`; once the provider health check
passes, the runtime manager changes the job to `RESUMING` for automatic worker
pickup.

## Current Limitations

- MCP server implementations are still local skeletons and fake adapters for MVP
- the Docker sandbox runner is structural only
- OpenAI-compatible integration is implemented, but tests do not call real APIs
- branch and patch handling are safe MVP abstractions, not full git-native automation
- API mutation auth is enabled when `ACOS_API_TOKEN` is set; local dev can opt
  out with `ACOS_LOCAL_DEV_AUTH_DISABLED=1`

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for setup details.
