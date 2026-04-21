# Daemon Mode

## Why

Durable runtime moves execution ownership from an interactive Codex session to
an ACOS worker daemon. A submitted job is persisted in SQLite, then advanced by
the worker even if the original terminal or browser disconnects.

## Core Pieces

- `SQLiteJobStore`: persists jobs, tasks, approvals, checkpoints, leases,
  heartbeats, runtime issues, and notifications in `.acos/acos.sqlite3`
- `WorkerDaemon`: polls runnable jobs, acquires leases, renews heartbeats, and
  resumes from checkpoints
- `RuntimeManager`: turns provider outages into `waiting_runtime` instead of
  immediate failure
- `ApprovalGateway`: stores approval requests and pauses in `waiting_approval`
- `CheckpointStore`: records step start/completion markers for idempotent resume

## Connection Loss

If your terminal, browser, SSH session, or Codex UI disconnects, the worker can
continue because it is a separate process and the durable job state lives in
SQLite rather than memory.

## Waiting States

- `waiting_approval`: high-risk operation paused until a human approves or
  rejects it
- `waiting_runtime` / `provider_unavailable`: model provider is unavailable,
  ACOS saved progress and will retry health checks
- `recovering`: the worker found stale heartbeat or stale lease and is
  preparing to resume from checkpoints
- `resuming`: recovery passed and the next poll can continue execution

## Commands

```bash
acos validate-config
acos daemon start --foreground --workspace .
acos jobs submit --file job.yaml --workspace .
acos jobs watch <job_id> --workspace .
acos runtime status --workspace .
acos approvals list --workspace .
```

## macOS launchd

`acos daemon install-launchd --workspace .` writes
`~/Library/LaunchAgents/com.acos.worker.plist`.

The plist:

- runs `acos worker run --forever`
- uses the repo as `WorkingDirectory`
- writes stdout/stderr to `.acos/logs/worker.out.log` and `worker.err.log`
- does not embed API keys, approval tokens, or other secrets

## Crash Recovery

On worker startup ACOS can:

- detect stale leases
- mark stale running jobs as `recovering`
- keep `waiting_approval` and `waiting_runtime` jobs paused
- move recovered runtime waits to `resuming` if the provider is healthy again

## Limitations

- If the host machine sleeps or loses power, execution stops while the machine
  is unavailable.
- Recovery resumes from the last completed checkpoint; it does not preserve an
  in-flight Python stack frame.
