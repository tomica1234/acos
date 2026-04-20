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
