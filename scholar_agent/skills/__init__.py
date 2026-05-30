"""Skill system for extending agent behavior via pluggable skill definitions."""

from scholar_agent.skills.base import Skill
from scholar_agent.skills.loader import SkillLoader, get_skill_loader

__all__ = ["Skill", "SkillLoader", "get_skill_loader"]
