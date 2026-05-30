from pathlib import Path

from scholar_agent.skills.loader import SkillLoader


def test_skill_loader_uses_root_manifest_with_skills_source():
    root = Path(__file__).resolve().parents[1]
    loader = SkillLoader(str(root / "skills"))

    skill = loader.get("example")

    assert skill is not None
    assert skill.name == "example"
    assert "root-level skill.json" in skill.description
    assert skill.routing_hints == ["code_builder"]
    assert skill.expert_overrides["code_builder"]["language_hint"] == "Python"
    assert "skills/example/SKILL.md" in skill.system_prompt_append
