"""Skill loader - scans a directory for skill definition files.

Supports two modes:

1. **Manifest mode** - ``skill.json`` references skill directories that each
   contain a ``SKILL.md`` (Claude Code format with YAML frontmatter).
2. **Direct mode** - individual ``.yaml`` / ``.yml`` / ``.json`` files, each
   a single skill in the custom ScholarAgent format.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from scholar_agent.core.logging import get_logger
from scholar_agent.skills.base import Skill

logger = get_logger(__name__)

# Matches YAML frontmatter in SKILL.md files
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class SkillLoader:
    """Scans a directory for skills.

    If ``skill.json`` exists at the root, it is treated as a **manifest**
    that references skill directories (Claude Code format).  Otherwise,
    every ``.yaml`` / ``.yml`` / ``.json`` file is parsed as a single skill.

    Usage::

        loader = SkillLoader("skills/")
        matched = loader.match("create an Excel spreadsheet")
        for skill in matched:
            print(skill.system_prompt_append)
    """

    def __init__(self, skills_dir: str):
        self._skills_dir = Path(skills_dir)
        self._project_root = self._find_project_root(self._skills_dir)
        self._skills: Dict[str, Skill] = {}
        self._loaded = False

    # ---- loading ------------------------------------------------------------

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        if not self._skills_dir.exists():
            logger.info(f"[skills] directory not found: {self._skills_dir}")
            return

        # Manifest mode: prefer repository-root skill.json, keep skills/skill.json
        # as a backward-compatible fallback for older checkouts.
        manifest_path = self._resolve_manifest_path()
        if manifest_path.exists():
            self._load_manifest(manifest_path)
        # Always also scan for direct-format files (supports hybrid mode)
        self._load_direct()
        # Auto-discover any */SKILL.md directories not already loaded
        self._scan_skill_directories()

        logger.info(f"[skills] loaded {len(self._skills)} skill(s) from {self._skills_dir}")

    @staticmethod
    def _find_project_root(start: Path) -> Path:
        """Return the repository root for resolving root-level skill manifests."""
        try:
            current = start.resolve()
        except Exception:
            current = start
        if current.is_file():
            current = current.parent
        for candidate in (current, *current.parents):
            if (candidate / "pyproject.toml").exists():
                return candidate
        return current.parent if current.name == "skills" else current

    def _resolve_manifest_path(self) -> Path:
        root_manifest = self._project_root / "skill.json"
        if root_manifest.exists():
            return root_manifest
        return self._skills_dir / "skill.json"

    # ---- manifest mode (Claude Code / skill.json) ---------------------------

    def _load_manifest(self, manifest_path: Path):
        """Parse ``skill.json`` and load each referenced skill directory."""
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except Exception as exc:
            logger.info(f"[skills] failed to parse manifest: {exc}")
            return

        plugins = manifest.get("plugins", [])
        for plugin in plugins:
            plugin_source = plugin.get("source", "./")
            plugin_dir = (manifest_path.parent / plugin_source).resolve()
            for skill_rel in plugin.get("skills", []):
                skill_dir = (plugin_dir / skill_rel).resolve()
                # Security: only load skills within the project tree
                try:
                    skill_dir.relative_to(self._project_root)
                except ValueError:
                    logger.info(f"[skills] skipping external path: {skill_dir}")
                    continue
                self._load_skill_directory(skill_dir)

    def _load_skill_directory(self, skill_dir: Path):
        """Load a single skill from a directory containing SKILL.md.

        If *skill_dir* does not exist, tries a fallback: when the manifest
        is inside the skills directory itself, paths like ``./skills/xlsx``
        resolve to ``skills/skills/xlsx/`` — strip the redundant segment.
        """
        if not skill_dir.exists():
            # Fallback: manifest inside skills/ dir causes doubled path segment
            parts = skill_dir.parts
            for i, part in enumerate(parts):
                if part == "skills" and i + 1 < len(parts) and parts[i + 1] == "skills":
                    fallback = Path(*parts[:i + 1], *parts[i + 2:])
                    if fallback.exists():
                        logger.info(f"[skills] path fallback: {skill_dir} → {fallback}")
                        skill_dir = fallback
                        break

        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            logger.info(f"[skills] SKILL.md not found: {skill_dir}")
            return

        try:
            with open(skill_md, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except Exception as exc:
            logger.info(f"[skills] failed to read {skill_md}: {exc}")
            return

        frontmatter, body = self._parse_frontmatter(raw)
        if not frontmatter:
            logger.info(f"[skills] no frontmatter in {skill_md}")
            return

        skill = Skill.from_skill_md(frontmatter, body)
        if not skill.name:
            logger.info(f"[skills] missing 'name' in {skill_md}")
            return

        self._skills[skill.name] = skill
        logger.info(f"[skills] loaded SKILL.md: {skill.name}")

    @staticmethod
    def _parse_frontmatter(raw: str) -> tuple:
        """Extract YAML frontmatter and markdown body from SKILL.md content.

        Returns ``(frontmatter_dict, body_text)``.  If no frontmatter is
        found, returns ``({}, raw)``.
        """
        m = _FRONTMATTER_RE.match(raw)
        if not m:
            return {}, raw

        try:
            import yaml
            frontmatter = yaml.safe_load(m.group(1)) or {}
        except Exception:
            frontmatter = {}
        body = raw[m.end():].strip()
        return frontmatter, body

    def _scan_skill_directories(self):
        """Auto-discover any ``*/SKILL.md`` directories not loaded via manifest.

        This allows adding a new skill by simply creating a subdirectory
        with a ``SKILL.md`` file — no manifest edit required.
        """
        for skill_dir in sorted(self._skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            # Check if already loaded (by name or by path)
            try:
                with open(skill_md, "r", encoding="utf-8") as fh:
                    raw = fh.read()
            except Exception:
                continue
            frontmatter, body = self._parse_frontmatter(raw)
            name = frontmatter.get("name", "")
            if name and name not in self._skills:
                skill = Skill.from_skill_md(frontmatter, body)
                self._skills[name] = skill
                logger.info(f"[skills] auto-discovered SKILL.md: {name}")

    # ---- direct mode (custom YAML / JSON files) -----------------------------

    def _load_direct(self):
        """Load individual ``.yaml`` / ``.yml`` / ``.json`` files as skills.

        Skips ``skill.json`` (the manifest itself) and any file whose name
        was already loaded via the manifest.
        """
        for path in sorted(self._skills_dir.glob("*")):
            if not path.is_file():
                continue
            if path.name == "skill.json":
                continue  # manifest, not a skill
            if path.suffix.lower() in (".yaml", ".yml"):
                self._load_yaml(path)
            elif path.suffix.lower() == ".json":
                self._load_json(path)

    def _load_yaml(self, path: Path):
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except Exception as exc:
            logger.info(f"[skills] failed to parse {path.name}: {exc}")
            return
        self._register(data, path.name)

    def _load_json(self, path: Path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.info(f"[skills] failed to parse {path.name}: {exc}")
            return
        self._register(data, path.name)

    def _register(self, data: dict, source: str):
        skill = Skill.from_dict(data)
        if not skill.name:
            logger.info(f"[skills] skipping {source}: missing 'name'")
            return
        self._skills[skill.name] = skill

    # ---- querying -----------------------------------------------------------

    def list_all(self) -> List[Skill]:
        """Return all loaded skills."""
        self._ensure_loaded()
        return list(self._skills.values())

    def get(self, name: str) -> Optional[Skill]:
        """Look up a skill by name."""
        self._ensure_loaded()
        return self._skills.get(name)

    def list_metadata(self) -> List[dict]:
        """Return ``[{name, description}]`` for all loaded skills.

        This compact metadata (~100 tokens total) is injected into the
        supervisor's system prompt so the LLM can autonomously select
        which skill(s) to activate — exactly like how tools work.
        """
        self._ensure_loaded()
        return [
            {"name": s.name, "description": s.description}
            for s in self._skills.values()
        ]


@lru_cache(maxsize=1)
def get_skill_loader(skills_dir: str) -> SkillLoader:
    """Return a cached ``SkillLoader`` for the given directory."""
    return SkillLoader(skills_dir)
