---
name: example
description: Example ScholarAgent skill for validating root-level skill.json registration and skill prompt injection.
routing_hints:
  - code_builder
expert_overrides:
  code_builder:
    language_hint: Python
requires_tools:
  - code_workspace_write_file
  - code_workspace_run_python
---

Use this skill as a minimal reference implementation for ScholarAgent skills.

When this skill is active:

- Prefer a small, runnable Python example when the user asks for demonstration code.
- Keep generated files inside the selected Code Builder workspace and use relative paths only.
- Include a concise validation step, such as running the main script or a smoke test.

Registration notes:

- The manifest is stored at the repository root as `skill.json`.
- The plugin `source` points to `./skills`.
- The skill entry `example` resolves to `skills/example/SKILL.md`.
