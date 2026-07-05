"""Agent prompt registry."""

from packages.llm.errors import UnknownRoleError
from packages.agents import (
    architect,
    diagnoser,
    fixer,
    implementer,
    planner,
    pm,
    release_manager,
    reviewer,
    security_reviewer,
    summarizer,
    test_writer,
)

ROLE_PROMPTS = {
    "pm": pm.SYSTEM_PROMPT,
    "architect": architect.SYSTEM_PROMPT,
    "planner": planner.SYSTEM_PROMPT,
    "implementer": implementer.SYSTEM_PROMPT,
    "test_writer": test_writer.SYSTEM_PROMPT,
    "diagnoser": diagnoser.SYSTEM_PROMPT,
    "reviewer": reviewer.SYSTEM_PROMPT,
    "security_reviewer": security_reviewer.SYSTEM_PROMPT,
    "fixer": fixer.SYSTEM_PROMPT,
    "release_manager": release_manager.SYSTEM_PROMPT,
    "summarizer": summarizer.SYSTEM_PROMPT,
}


def get_role_prompt(role: str) -> str:
    try:
        return ROLE_PROMPTS[role]
    except KeyError as exc:
        raise UnknownRoleError(role) from exc
