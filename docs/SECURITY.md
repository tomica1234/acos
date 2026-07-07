# Security

## Non-Negotiable Constraints

- Secrets must be redacted before prompts, memory writes, logs, and
  notifications.
- MCP tool access is deny-by-default and constrained by role.
- Arbitrary shell execution is blocked.
- Direct writes to `main` or `master` are blocked.
- Force push is blocked.
- Production deploy and destructive migrations are blocked.

## Prompt Hygiene

Context packets contain only the minimum required files, diffs, logs, and
memory summaries. Secret patterns are redacted before model submission and
before audit persistence.

## Execution Hygiene

The deterministic test runner and Docker sandbox skeleton exist outside the LLM
role layer. Agents can request changes only through validated structured
outputs.

## API Hygiene

Set `ACOS_API_TOKEN` to require `Authorization: Bearer ...` or
`X-ACOS-API-Token` for mutating API calls. Local development can explicitly opt
out with `ACOS_LOCAL_DEV_AUTH_DISABLED=1`.

Use `ACOS_REPO_ALLOWLIST` to restrict job repo paths to approved workspace
roots. Use `ACOS_CORS_ALLOW_ORIGINS` to avoid wildcard CORS outside local
development.

Approval approve/reject flows should use signed approval tokens. Rejected
approval is only a `POLICY_HARD_STOP` for critical operations; otherwise ACOS
records the rejection and replans around that operation.

Recovery never weakens policy. Secret access, workspace escape, direct
main/master writes, force push, production deploy, sudo, arbitrary shell, and
credential access remain hard stops.
