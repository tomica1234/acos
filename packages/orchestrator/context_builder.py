"""Context packet builder with budget-aware truncation."""

from __future__ import annotations

from packages.llm.budget import TokenBudgetManager, truncate_to_budget
from packages.memory.redaction import redact_text
from packages.schemas.context import ContextPacket
from packages.schemas.models import AgentModelConfig, ModelConfig
from packages.schemas.tasks import PlannedTask


class ContextBuilder:
    """Construct role-specific context packets."""

    def __init__(self) -> None:
        self.token_manager = TokenBudgetManager()

    def build(
        self,
        job_id: str,
        role: str,
        objective: str,
        repo_path: str,
        request_text: str,
        constraints: list[str],
        relevant_files: dict[str, str],
        diff: str,
        memory_summaries: list[str],
        logs: list[str],
        token_budget: int,
        agent_config: AgentModelConfig | None = None,
        selected_model: ModelConfig | None = None,
        task: PlannedTask | None = None,
        metadata: dict | None = None,
    ) -> ContextPacket:
        requested_budget = token_budget
        if agent_config is not None:
            requested_budget = agent_config.context_budget_tokens
        model_context_budget = requested_budget
        selected_model_hint = None
        if selected_model is not None:
            model_context_budget = self.token_manager.fit_context_budget(
                requested_budget=requested_budget,
                model_max_context_tokens=selected_model.max_context_tokens,
                model_max_output_tokens=selected_model.max_output_tokens,
            )
            selected_model_hint = selected_model.model_id
        allocation = self.token_manager.allocate(model_context_budget)
        trimmed_files: dict[str, str] = {}
        if relevant_files:
            per_file_budget = max(64, allocation.files // len(relevant_files))
            for path, content in relevant_files.items():
                trimmed_files[path] = truncate_to_budget(redact_text(content), per_file_budget)
        per_memory_budget = max(32, allocation.memory // max(1, len(memory_summaries)))
        trimmed_memory = [
            truncate_to_budget(redact_text(item), per_memory_budget) for item in memory_summaries
        ]
        per_log_budget = max(32, allocation.logs // max(1, len(logs)))
        trimmed_logs = [truncate_to_budget(redact_text(item), per_log_budget) for item in logs]
        return ContextPacket(
            job_id=job_id,
            role=role,
            objective=objective,
            repo_path=repo_path,
            request_text=truncate_to_budget(redact_text(request_text), allocation.request),
            constraints=[redact_text(item) for item in constraints],
            relevant_files=trimmed_files,
            diff=truncate_to_budget(redact_text(diff), allocation.diff),
            memory_summaries=trimmed_memory,
            logs=trimmed_logs,
            task=task,
            token_budget=requested_budget,
            model_context_budget=model_context_budget,
            selected_model_hint=selected_model_hint,
            metadata=metadata or {},
        )
