# Daemon Mode

ACOS daemon mode is for durable autonomous runs. Use SQLite persistence when a
job should survive process restarts:

```bash
acos-worker --repo . --sqlite-path .acos/acos.sqlite3 --forever
```

The worker polls runnable jobs, acquires a lease, records heartbeats, and runs
until the job reaches `DONE`, `CANCELLED`, `POLICY_HARD_STOP`,
`WAITING_APPROVAL`, or `WAITING_RUNTIME`. `BLOCKED`, `STUCK`, and `FAILED` are
not settled states; the daemon normalizes them to `RECOVERING` and invokes
recovery.

For a single durable job:

```bash
acos-worker --repo . --sqlite-path .acos/acos.sqlite3 --request "Build the app"
```

For an existing job:

```bash
acos-worker --repo . --sqlite-path .acos/acos.sqlite3 --job-id JOB_ID
```

Provider outages are recorded as runtime issues. When the provider becomes
healthy again, `RuntimeManager` marks the job `RESUMING` so the daemon can pick
it up automatically.
