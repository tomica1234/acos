# Runtime Recovery

## Provider Unavailable

When an OpenAI-compatible provider times out or becomes unreachable, ACOS does
not immediately fail the job. It stores a `RuntimeIssue`, records the reason,
notifies the user, and moves the job to `waiting_runtime` or
`provider_unavailable`.

## Recovery Flow

1. Provider call fails with timeout / connection error / provider unavailable.
2. `RuntimeManager.handle_provider_error()` stores the issue in SQLite.
3. The worker stops advancing that job.
4. `ProviderHealthChecker` periodically checks `/models` and optionally a short
   chat completion.
5. When the provider recovers:
   - auto-resume mode: the job becomes `resuming`
   - manual mode: the job becomes `paused` and waits for `acos jobs resume`

## Auth Error And Model Not Found

- `auth_error`: blocked by default
- `model_not_found`: blocked by default

These are configuration/runtime mismatches, not transient outages.

## Commands

```bash
acos runtime status --workspace .
acos runtime check --workspace .
acos check-provider --provider local_qwen
acos check-model --model qwen_35b
acos jobs resume <job_id> --workspace .
```

## Backoff

Provider retry uses a bounded backoff based on `configs/runtime.yaml`:

- `check_interval_seconds`
- `max_backoff_seconds`

## Secret Handling

Runtime issues and notifications do not store raw API keys, full prompts, or
approval tokens.
