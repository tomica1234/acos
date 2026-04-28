"""Context packet builder with budget-aware truncation."""

from __future__ import annotations

from typing import Any

from packages.llm.budget import TokenBudgetManager, TokenBudgetPolicy, estimate_tokens, truncate_to_budget
from packages.memory.redaction import redact_text
from packages.schemas.context import ContextPacket
from packages.schemas.models import AgentModelConfig, ModelConfig
from packages.schemas.tasks import PlannedTask


class ContextBuilder:
    """Construct role-specific context packets."""

    def __init__(self, token_budget_policy: TokenBudgetPolicy | None = None) -> None:
        self.token_budget_policy = token_budget_policy or TokenBudgetPolicy()
        self.token_manager = TokenBudgetManager(self.token_budget_policy)

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
        requested_budget = (
            agent_config.context_budget_tokens if agent_config is not None else token_budget
        )
        model_context_budget = requested_budget
        selected_model_hint = None
        if selected_model is not None:
            model_context_budget = self.token_manager.fit_context_budget(
                requested_budget=requested_budget,
                model_max_context_tokens=selected_model.max_context_tokens,
            )
            selected_model_hint = selected_model.model_id
        allocation = self.token_manager.allocate(model_context_budget)
        scale = 1.0
        base_metadata = dict(metadata or {})
        packet: ContextPacket | None = None
        truncation_notes: list[dict[str, Any]] = []
        estimated_input_tokens = 0

        for _ in range(4):
            notes: list[dict[str, Any]] = []
            scaled = self._scale_allocation(allocation, scale)
            trimmed_files: dict[str, str] = {}
            if relevant_files:
                per_file_budget = max(64, scaled.files // len(relevant_files))
                for path, content in relevant_files.items():
                    trimmed_files[path] = self._truncate_with_note(
                        redact_text(content),
                        per_file_budget,
                        f"file:{path}",
                        notes,
                    )
            per_memory_budget = max(32, scaled.memory // max(1, len(memory_summaries)))
            trimmed_memory = [
                self._truncate_with_note(
                    redact_text(item),
                    per_memory_budget,
                    "memory",
                    notes,
                )
                for item in memory_summaries
            ]
            per_log_budget = max(32, scaled.logs // max(1, len(logs)))
            trimmed_logs = [
                self._truncate_with_note(
                    redact_text(item),
                    per_log_budget,
                    "log",
                    notes,
                )
                for item in logs
            ]
            trimmed_task = None
            if task is not None:
                trimmed_task = PlannedTask.model_validate(
                    {
                        **task.model_dump(),
                        "title": self._truncate_with_note(
                            redact_text(task.title),
                            128,
                            "task:title",
                            notes,
                        ),
                        "description": self._truncate_with_note(
                            redact_text(task.description),
                            256,
                            "task:description",
                            notes,
                        ),
                    }
                )
            packet = ContextPacket(
                job_id=job_id,
                role=role,
                objective=objective,
                repo_path=repo_path,
                request_text=self._truncate_with_note(
                    redact_text(request_text),
                    scaled.request,
                    "request",
                    notes,
                ),
                constraints=[redact_text(item) for item in constraints],
                relevant_files=trimmed_files,
                diff=self._truncate_with_note(redact_text(diff), scaled.diff, "diff", notes),
                memory_summaries=trimmed_memory,
                logs=trimmed_logs,
                task=trimmed_task,
                token_budget=requested_budget,
                model_context_budget=model_context_budget,
                selected_model_hint=selected_model_hint,
                metadata=base_metadata,
            )
            estimated_input_tokens = estimate_tokens(
                packet.render_text(),
                chars_per_token=self.token_budget_policy.estimate_chars_per_token,
            )
            truncation_notes = notes
            if selected_model is None or estimated_input_tokens <= model_context_budget:
                break
            scale *= max(0.35, (model_context_budget / max(estimated_input_tokens, 1)) * 0.95)

        assert packet is not None
        packet = packet.model_copy(
            update={
                "metadata": {
                    **base_metadata,
                    "context_truncated": bool(truncation_notes),
                    "context_truncation_notes": truncation_notes,
                    "estimated_input_tokens": estimated_input_tokens,
                    "context_budget_tokens": requested_budget,
                    "effective_context_budget_tokens": model_context_budget,
                    "safety_margin_tokens": self.token_budget_policy.safety_margin_tokens,
                }
            }
        )
        return packet

    def _truncate_with_note(
        self,
        text: str,
        token_budget: int,
        section: str,
        notes: list[dict[str, Any]],
    ) -> str:
        truncated = truncate_to_budget(
            text,
            token_budget,
            chars_per_token=self.token_budget_policy.estimate_chars_per_token,
        )
        if truncated != text:
            notes.append(
                {
                    "section": section,
                    "original_tokens": estimate_tokens(
                        text,
                        chars_per_token=self.token_budget_policy.estimate_chars_per_token,
                    ),
                    "truncated_tokens": estimate_tokens(
                        truncated,
                        chars_per_token=self.token_budget_policy.estimate_chars_per_token,
                    ),
                }
            )
        return truncated

    @staticmethod
    def _scale_allocation(allocation: Any, scale: float) -> Any:
        scaled = max(0.1, min(scale, 1.0))
        return type(allocation)(
            files=max(64, int(allocation.files * scaled)),
            diff=max(64, int(allocation.diff * scaled)),
            memory=max(32, int(allocation.memory * scaled)),
            logs=max(32, int(allocation.logs * scaled)),
            request=max(64, int(allocation.request * scaled)),
        )
