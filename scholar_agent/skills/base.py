"""Skill data model — a skill extends agent behavior with prompts and routing hints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Skill:
    """A skill that extends agent behavior.

    Skills are presented to the LLM as ``{name, description}`` metadata
    (like tools).  The LLM autonomously decides which skill(s) to activate.
    When activated, the skill's *system_prompt_append* is injected into the
    supervisor's context.
    """

    name: str
    description: str = ""
    system_prompt_append: str = ""
    expert_overrides: Dict[str, dict] = field(default_factory=dict)
    routing_hints: Optional[List[str]] = None
    requires_tools: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        """Parse a Skill from a dictionary (YAML/JSON loaded)."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            system_prompt_append=data.get("system_prompt_append", ""),
            expert_overrides=data.get("expert_overrides", {}),
            routing_hints=data.get("routing_hints"),
            requires_tools=data.get("requires_tools", []),
        )

    @classmethod
    def from_skill_md(cls, frontmatter: dict, body: str) -> "Skill":
        """Create a Skill from Claude Code-style SKILL.md frontmatter + body.

        Optional ScholarAgent extensions (*expert_overrides*, *routing_hints*)
        are read from the frontmatter if present.
        """
        return cls(
            name=frontmatter.get("name", ""),
            description=frontmatter.get("description", ""),
            system_prompt_append=body.strip(),
            expert_overrides=frontmatter.get("expert_overrides", {}),
            routing_hints=frontmatter.get("routing_hints"),
            requires_tools=frontmatter.get("requires_tools", []),
        )
