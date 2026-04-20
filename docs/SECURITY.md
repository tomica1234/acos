# Security

## Goal

ACOS MVP treats the orchestrator and MCP boundary as the primary security
control plane. LLM roles do not get arbitrary shell access, direct git
authority, or unrestricted filesystem access.

## Implemented Controls

### Workspace Auto Mode

`configs/policies.yaml` defaults to `autonomy.mode = workspace_auto`.

- Low risk workspace-local development actions are auto-allowed.
- Medium risk actions are allowed with explicit audit.
- High risk actions require approval.
- Critical actions are denied.

### Filesystem and Workspace Boundaries

- `RepoServer` resolves all paths against `WORKSPACE_ROOT` and rejects
  absolute paths, `..` traversal, and symlink escapes.
- Hidden directories and sensitive paths are blocked, including `.git`,
  `.env*`, `.ssh`, `.aws`, private key suffixes, and secret-like filenames
  such as `*secret*`, `*credential*`, and `id_rsa*`.
- `repo_tree()` filters forbidden files before exposing them to agents.
- `read_file()` rejects binary-like files and caps `max_chars`.
- `apply_patch()` only accepts `create` and `update`, validates the path
  first, and caps patch size.
- Large patches and mass-delete style operations are classified as
  `require_approval` rather than auto-allowed.

### Tool and Role Policy

- Tool access is deny-by-default via `PolicyEngine`.
- `AgentRunner` verifies that any runtime `allowed_tools` override is a subset
  of the role's configured `allowed_tools`.
- `AgentRunner` and `JobRunner` both enforce patch target policy before any
  `repo_server.apply_patch` call.
- Every tool request is classified before execution:
  - `allow`
  - `allow_and_audit`
  - `require_approval`
  - `deny`
- `release_manager` is the only role allowed to commit.
- Branch creation and commit targets must use the `acos/` prefix and cannot
  target `main`, `master`, or `develop`.

### Test Integrity

- `implementer` and `fixer` may not modify files under `tests/`.
- `test_writer` may modify tests, but obvious weakening patterns are blocked:
  `xfail`, `skip`, `skipif`, `mark.skip`, `mark.xfail`, and `assert True`.
- `fixer` patches are checked again by quality gates to prevent test
  weakening.
- `test_server.run_test` accepts only allowlisted command names and requires a
  bounded timeout. Arbitrary shell strings are never executed.

### Dependency and Release Safety

- `implementer`, `test_writer`, and `fixer` may not modify dependency
  manifests such as `pyproject.toml`, `requirements*.txt`, `package.json`,
  `poetry.lock`, or `uv.lock`.
- Direct mainline writes, force push, production deploy, destructive DB
  migration, and arbitrary shell remain blocked by policy.
- High-risk release-like actions, external data send, mass delete, and large
  patch application require approval.

### Secret Hygiene

- Shared redaction covers private key blocks, OpenAI-style keys, GitHub tokens,
  Slack tokens, AWS access keys, bearer tokens, and common `secret=` /
  `password=` / `token=` forms.
- Redaction is applied before:
  - context packet construction
  - memory writes
  - notification sends
  - audit metadata hashing
  - surfaced tool errors
- `ContextBuilder` redacts request text, diffs, logs, memory summaries, and
  task fields before prompt rendering.
- Approval notifications and audit records store redacted summaries only.
  They do not store raw approval tokens, raw secrets, or full file contents.

### Approval Gateway

- Approval requests are stored in SQLite, not only in memory.
- Raw approval tokens are never stored. Only a hash is persisted.
- Approval tokens are one-time: once approved or rejected, the stored hash is
  cleared and the request no longer accepts the old token.
- Jobs move to `waiting_approval` until approval is resolved.
- Approved operations are resumed exactly once through a one-shot override so
  the same action does not immediately re-trigger approval.

### Notification Link Risk

- One-time approve/reject GET links are enabled only for local/dev ergonomics.
- The links do not embed secrets or operation payloads beyond the approval id
  and one-time token.
- For production deployment, use HTTPS, authenticated POST endpoints, and CSRF
  protection. GET approval links should be disabled outside trusted local/dev
  environments.

### Audit and Failure Containment

- Audit records include role, selected model, provider, routing reason, token
  estimates, and hashed payload references.
- Audit metadata is sanitized; raw secrets are not retained.
- Job failures are redacted before being stored in `JobRecord.last_error`.
- Infinite fix loops are bounded by `max_attempts_per_task` and
  `max_same_failure_repeats`.

## Review Outcome

No critical or high findings remain in the current MVP implementation after
hardening the fake MCP boundary, patch policy enforcement, redaction, and
secret-safe error handling.

## Remaining Medium / Low Findings

### Medium

- `apps/api/main.py` exposes `/models` and `/agents` from config loads without
  running the full policy-aware validation path used by `/config/validate`.
- `mcp_servers/*/main.py` are skeleton wrappers around the in-process fake
  servers. The safety checks are enforced in the fake implementations today,
  but a production MCP transport layer still needs equivalent enforcement.
- Approval pause is job-level in the MVP. If one task requires approval, the
  whole job waits rather than allowing independent tasks to continue.

### Low

- The Docker sandbox runner is still a skeleton. Network and resource policy
  are defined, but containerized execution enforcement is not implemented yet.
- `test_writer` can still author logically weak but syntactically valid tests.
  The MVP blocks common weakening tokens, but semantic test quality still
  depends on review plus deterministic test execution.

## Tests Covering Security Controls

- `tests/test_security_controls.py`
- `tests/test_policy.py`
- `tests/test_context_builder.py`
- `tests/test_agent_runner.py`
- `tests/test_job_runner.py`

These tests verify path restrictions, symlink escape blocking, secret
redaction, tool policy enforcement, patch restrictions, test command
allowlisting, approval storage, approval API/CLI behavior, and secret-safe
notification and memory behavior.
