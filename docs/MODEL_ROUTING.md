# Model Routing

## Core Types

- `ModelProviderConfig`
  Defines provider type, base URL, API key environment variable, timeout,
  capability flags, and provider-wide token hints.
- `ModelConfig`
  Defines a concrete model record, its provider, token window, tool support,
  structured output support, JSON repair capability, and tags.
- `AgentModelConfig`
  Binds a role to a primary model, ordered fallback models, sampling settings,
  output budget, context budget, tool permissions, and output schema name.

## ModelRegistry

`packages.llm.registry.ModelRegistry` loads YAML from:

- `configs/model_providers.yaml`
- `configs/agents.yaml`
- `configs/model_routing.yaml`

It validates:

- unknown providers
- unknown models
- unknown roles in routing escalation
- primary and fallback model context/output budgets
- tool-required roles pointing at tool-incompatible models
- strict JSON roles pointing at models that support neither structured output nor
  JSON repair
- policy mismatches between `AgentModelConfig.allowed_tools` and
  `configs/policies.yaml`

## ModelAdapter

`packages.llm.adapters.base.ModelAdapter` is the common interface.

- `OpenAICompatibleAdapter`
  Uses the OpenAI Python SDK with provider `base_url` and `api_key_env`
  settings. It supports tool calls, returns normalized `ModelResult`, and raises
  explicit `invalid_json` errors when a provider running in JSON mode returns
  malformed JSON.
- `MockAdapter`
  Is deterministic and test-only. It accepts response sequences and can emit
  tool calls, invalid JSON, or fully structured fake responses without calling
  external APIs.

## ModelRouter

`packages.llm.routing.ModelRouter` selects a model for each role invocation.

Inputs:

- role
- task complexity
- repeated failures
- repeated same-test failures
- changed file count
- security-sensitive flag
- context token estimate
- last error
- ordered fallback attempt index

Output:

- selected role
- selected model key
- selected provider key
- routing reason
- routing details
- sampling settings and capped output budget

1. Start with the role's primary model.
2. Apply escalation when configured thresholds match.
3. Apply ordered fallback when the last error is in the configured fallback set.
4. Drop any candidate that violates capability requirements.
5. If no candidate can fit the context budget, raise a compaction-required
   error.

`packages.agents.runner.AgentRunner` uses this selection directly. It does not
hard-code provider names or model ids. The runner:

1. loads the role config from `ModelRegistry`
2. asks `ModelRouter` for a `ModelSelection`
3. resolves the concrete adapter from `ModelRegistry`
4. converts allowed MCP tools into OpenAI-compatible tool definitions
5. executes tool calls only when `PolicyEngine` allows them
6. retries invalid JSON once with a repair prompt on the same model
7. switches to fallback routing only after repair fails

## Fallback

Fallback is configured in `configs/model_routing.yaml`.

- timeouts
- rate limits
- invalid JSON
- tool calling unsupported
- context overflow

Fallbacks are ordered. ACOS tries them in sequence rather than jumping directly
to an arbitrary model.

## Escalation

Escalation is also configured in `configs/model_routing.yaml`.

Examples:

- Implementer escalates when failures repeat or the task is high complexity
- Fixer escalates when the same failure repeats
- Reviewer or Security Reviewer can escalate for security-sensitive work

## Context Budget

Context budgeting combines:

- role-level budget from `AgentModelConfig.context_budget_tokens`
- selected model maximum context window
- selected model output reservation
- a safety overhead for prompt framing

`ContextBuilder` truncates or summarizes:

- relevant files
- repo diff
- logs
- memory summaries
- raw request text

It also redacts secrets before prompt construction and writes
`model_context_budget` plus `selected_model_hint` into the `ContextPacket`.

## Capability Requirements

Roles that require tools may only use models that support tool calling and whose
provider also allows tools.

Roles that require strict JSON may only use models that support:

- structured output
- or JSON repair/retry

## Role Examples

- PM and Architect use `ornith_35b_q4` for long-context planning.
- Implementer, Test Writer, Reviewer, Fixer, Release Manager, and Summarizer all
  route to the same local Ornith model by default.
- Repeated-failure escalation still records an escalation decision, even when
  the configured escalated model is the same local Ornith model.

## Audit Expectations

Every selection records the role, chosen model, provider, routing reason,
estimated tokens, routing details, and redacted input/output hashes.
