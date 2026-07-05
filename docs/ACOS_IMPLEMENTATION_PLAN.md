# ACOS Implementation Plan

## Goal

Build ACOS as an explicit orchestration system that turns natural-language
product requirements into a controlled multi-role delivery pipeline:
requirements, architecture, planning, implementation, test authoring, review,
deterministic test execution, fixing, release finalization, memory
summarization, and notification.

## Delivery Principles

- Orchestration logic stays in code, not in prompts.
- All role outputs are structured and schema-validated.
- Model selection stays configurable and provider-agnostic.
- Tool access is deny-by-default and role-scoped.
- Security constraints apply to prompts, memory, logs, notifications, and file
  operations.

## MVP Scope

- Config-driven provider and model registry
- Dynamic model routing with fallback and escalation
- Context budgeting and truncation
- In-memory job store and SQLite memory store
- Fake MCP tools for deterministic testing
- FastAPI API, CLI, and worker entrypoints
- MCP server skeletons
- End-to-end orchestration vertical slice

## Post-MVP Expansion

- Real MCP transport wiring
- Rich diff-aware context selection
- Multiple concurrent work trees
- Sandbox execution with stronger isolation
- Human approval checkpoints for high-risk actions
