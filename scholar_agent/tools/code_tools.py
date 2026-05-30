"""Controlled coding tools for the Code Builder expert.

These tools intentionally operate only inside an explicit Code Builder
workspace and expose a narrow command surface. They are meant for bounded
autonomous coding loops, not arbitrary shell access.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from scholar_agent.config import conf
from scholar_agent.core.logging import get_logger
from scholar_agent.tools.base import BaseTool, ToolResult

logger = get_logger(__name__)

CODE_WORKSPACE_DIR = conf.CODE_BUILDER_WORKSPACE_DIR
_ACTIVE_CODE_WORKSPACE_DIR: ContextVar[str] = ContextVar("active_code_workspace_dir", default="")
_ACTIVE_CODE_PYTHON_EXECUTABLE: ContextVar[str] = ContextVar("active_code_python_executable", default="")
_ACTIVE_CODE_VALIDATION_RECORD: ContextVar[dict | None] = ContextVar(
    "active_code_validation_record",
    default=None,
)
_ACTIVE_CODE_VALIDATION_PLAN: ContextVar[list[str]] = ContextVar(
    "active_code_validation_plan",
    default=[],
)
_ACTIVE_CODE_VALIDATION_RECORDS: ContextVar[dict[str, dict]] = ContextVar(
    "active_code_validation_records",
    default={},
)


def _workspace_root() -> Path:
    root = Path(_ACTIVE_CODE_WORKSPACE_DIR.get() or CODE_WORKSPACE_DIR).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_code_workspace_dir() -> str:
    """Return the active Code Builder workspace directory."""
    return str(_workspace_root())


def set_code_workspace_dir(path: str) -> str:
    """Set the active Code Builder workspace directory and ensure it exists."""
    value = str(path or "").strip() or conf.CODE_BUILDER_WORKSPACE_DIR
    _ACTIVE_CODE_WORKSPACE_DIR.set(value)
    return str(_workspace_root())


def reset_code_workspace_dir(token: Token[str]) -> None:
    """Restore the previous active Code Builder workspace directory."""
    _ACTIVE_CODE_WORKSPACE_DIR.reset(token)


@contextmanager
def use_code_workspace_dir(path: str) -> Iterator[str]:
    """Temporarily bind code tools to a workspace for one tool-calling loop."""
    value = str(path or "").strip() or conf.CODE_BUILDER_WORKSPACE_DIR
    token = _ACTIVE_CODE_WORKSPACE_DIR.set(value)
    try:
        yield str(_workspace_root())
    finally:
        reset_code_workspace_dir(token)


def get_code_python_executable() -> str:
    """Return the active Python executable used by Code Builder tools."""
    return _python_executable()


def set_code_python_executable(path: str) -> str:
    """Set the active Python executable for the current context."""
    value = str(path or "").strip() or conf.CODE_BUILDER_PYTHON_EXECUTABLE
    _ACTIVE_CODE_PYTHON_EXECUTABLE.set(value)
    return _python_executable()


def reset_code_python_executable(token: Token[str]) -> None:
    """Restore the previous active Python executable."""
    _ACTIVE_CODE_PYTHON_EXECUTABLE.reset(token)


@contextmanager
def use_code_python_executable(path: str) -> Iterator[str]:
    """Temporarily bind Code Builder Python tools to a selected interpreter."""
    value = str(path or "").strip() or conf.CODE_BUILDER_PYTHON_EXECUTABLE
    token = _ACTIVE_CODE_PYTHON_EXECUTABLE.set(value)
    try:
        yield _python_executable()
    finally:
        reset_code_python_executable(token)


def clear_code_validation_record() -> None:
    """Clear validation evidence after a project file changes."""
    _ACTIVE_CODE_VALIDATION_RECORD.set(None)
    _ACTIVE_CODE_VALIDATION_RECORDS.set({})


def clear_code_validation_state() -> None:
    """Clear the active Code Builder validation plan and evidence."""
    clear_code_validation_record()
    _ACTIVE_CODE_VALIDATION_PLAN.set([])


def get_code_validation_record() -> dict | None:
    """Return the latest explicit validation record for the active coding loop."""
    record = _ACTIVE_CODE_VALIDATION_RECORD.get()
    return dict(record) if isinstance(record, dict) else None


def _normalise_validation_target(value: str) -> str:
    return str(value or "").strip().strip("'\"").replace("\\", "/")


def _parse_validation_targets(raw_targets: str) -> list[str]:
    raw = str(raw_targets or "").strip()
    if not raw:
        return []
    targets: list[str] = []
    for part in re.split(r"[\n,;]+", raw):
        target = _normalise_validation_target(part)
        if target and target not in targets:
            targets.append(target)
    return targets


def set_code_validation_plan(targets: str, rationale: str = "") -> dict:
    """Set the validation targets that must pass before final delivery."""
    parsed = _parse_validation_targets(targets)
    _ACTIVE_CODE_VALIDATION_PLAN.set(parsed)
    _ACTIVE_CODE_VALIDATION_RECORDS.set({})
    _ACTIVE_CODE_VALIDATION_RECORD.set(None)
    return {
        "targets": parsed,
        "rationale": str(rationale or "").strip()[:1200],
    }


def get_code_validation_plan() -> list[str]:
    """Return the active validation targets for the current coding loop."""
    return list(_ACTIVE_CODE_VALIDATION_PLAN.get() or [])


def get_code_validation_records() -> dict[str, dict]:
    """Return validation evidence keyed by target."""
    records = _ACTIVE_CODE_VALIDATION_RECORDS.get()
    return dict(records) if isinstance(records, dict) else {}


def set_code_validation_record(passed: bool, tool: str, target: str, evidence: str) -> dict:
    """Set the current Code Builder validation record."""
    normalised_target = _normalise_validation_target(target)
    record = {
        "passed": bool(passed),
        "tool": str(tool or "").strip(),
        "target": normalised_target,
        "evidence": str(evidence or "").strip()[:1200],
    }
    _ACTIVE_CODE_VALIDATION_RECORD.set(record)
    if normalised_target:
        records = get_code_validation_records()
        records[normalised_target] = record
        _ACTIVE_CODE_VALIDATION_RECORDS.set(records)
    return record


def code_validation_ready() -> bool:
    """Return whether the active validation evidence satisfies the active plan."""
    plan = get_code_validation_plan()
    records = get_code_validation_records()
    if not plan:
        latest = get_code_validation_record()
        return bool(latest and latest.get("passed") is True)
    for target in plan:
        record = records.get(_normalise_validation_target(target))
        if not record or record.get("passed") is not True:
            return False
    return True


def _resolve_workspace_path(relative_path: str) -> Path:
    root = _workspace_root()
    rel = Path(str(relative_path or "").replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("path must be relative to the selected code workspace and must not contain '..'")
    target = (root / rel).resolve()
    if root != target and root not in target.parents:
        raise ValueError("resolved path escapes the selected code workspace")
    return target


def _run_command(
    args: list[str],
    cwd: Optional[Path] = None,
    timeout_s: int = 30,
    env: Optional[dict[str, str]] = None,
) -> ToolResult:
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd or _workspace_root()),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, min(int(timeout_s), 120)),
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        return ToolResult("fail", f"Command timed out.\n{output[-4000:]}")
    except Exception as exc:
        return ToolResult("fail", f"Command failed to start: {exc}")

    output = (completed.stdout or "") + (completed.stderr or "")
    status = "success" if completed.returncode == 0 else "fail"
    return ToolResult(status, f"exit_code={completed.returncode}\n{output[-6000:]}")


def _python_executable() -> str:
    configured = str(
        _ACTIVE_CODE_PYTHON_EXECUTABLE.get()
        or getattr(conf, "CODE_BUILDER_PYTHON_EXECUTABLE", "")
        or ""
    ).strip()
    return configured or sys.executable


def _python_shell_env() -> dict[str, str]:
    env = os.environ.copy()
    python_exe = str(Path(_python_executable()).expanduser().resolve())
    python_dir_path = Path(python_exe).parent
    python_dir = str(python_dir_path)
    scripts_dir = str(python_dir_path / ("Scripts" if os.name == "nt" else "bin"))
    env["PYTHON"] = python_exe
    env["PYTHON_EXECUTABLE"] = python_exe
    current_paths = []
    path_key = "PATH"
    for key in list(env):
        if key.lower() == "path":
            path_key = key
            value = env.pop(key)
            if value:
                current_paths.append(value)
    path_parts = [python_dir]
    if Path(scripts_dir).exists():
        path_parts.append(scripts_dir)
    path_parts.extend(current_paths)
    env[path_key] = os.pathsep.join(path_parts)
    return env


def _resolve_command_cwd(cwd: str = ".") -> Path:
    target = _resolve_workspace_path(cwd or ".")
    if not target.exists():
        raise ValueError(f"cwd does not exist: {cwd}")
    if not target.is_dir():
        raise ValueError(f"cwd must be a directory: {cwd}")
    return target


def _looks_unsafe_shell_command(command: str) -> str:
    text = str(command or "").strip()
    if not text:
        return "command is empty"
    lowered = text.lower()
    forbidden = [
        "rm -rf",
        "del /",
        "format ",
        "shutdown",
        "restart-computer",
        "reg delete",
        "mkfs",
    ]
    for pattern in forbidden:
        if pattern in lowered:
            return f"blocked unsafe command pattern: {pattern}"
    if re.search(r"(^|\s)(sudo|su)\b", lowered):
        return "privilege escalation commands are not allowed"
    return ""


def _shell_invocation(command: str) -> list[str]:
    if os.name == "nt":
        return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command]
    return ["bash", "-lc", command]


def _make_unified_diff(path: str, old_text: str, new_text: str) -> str:
    import difflib

    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="\n",
        )
    )


@dataclass
class _PatchHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _parse_unified_patch(patch_text: str) -> tuple[str, list[_PatchHunk]]:
    lines = str(patch_text or "").splitlines()
    target_path = ""
    hunks: list[_PatchHunk] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            target_path = path
            i += 1
            continue
        match = _HUNK_RE.match(line)
        if not match:
            i += 1
            continue
        old_start = int(match.group(1))
        old_count = int(match.group(2) or "1")
        new_start = int(match.group(3))
        new_count = int(match.group(4) or "1")
        i += 1
        hunk_lines = []
        while i < len(lines) and not lines[i].startswith("@@ "):
            if lines[i].startswith((" ", "+", "-", "\\")):
                hunk_lines.append(lines[i])
            i += 1
        hunks.append(_PatchHunk(old_start, old_count, new_start, new_count, hunk_lines))
    if not target_path:
        raise ValueError("patch must include a '+++ b/<path>' header")
    if not hunks:
        raise ValueError("patch contains no hunks")
    return target_path, hunks


def _apply_unified_patch_to_text(original: str, hunks: list[_PatchHunk]) -> str:
    source = original.splitlines()
    result: list[str] = []
    cursor = 0
    for hunk in hunks:
        start = max(hunk.old_start - 1, 0)
        if start < cursor:
            raise ValueError("overlapping or out-of-order hunks")
        result.extend(source[cursor:start])
        cursor = start
        for line in hunk.lines:
            if not line or line.startswith("\\"):
                continue
            marker = line[0]
            text = line[1:]
            if marker == " ":
                if cursor >= len(source) or source[cursor] != text:
                    raise ValueError(f"hunk context mismatch near line {cursor + 1}")
                result.append(source[cursor])
                cursor += 1
            elif marker == "-":
                if cursor >= len(source) or source[cursor] != text:
                    raise ValueError(f"hunk removal mismatch near line {cursor + 1}")
                cursor += 1
            elif marker == "+":
                result.append(text)
            else:
                raise ValueError(f"unsupported patch line: {line!r}")
    result.extend(source[cursor:])
    trailing_newline = original.endswith("\n") or any(
        line.startswith("+") and line[1:] == "" for hunk in hunks for line in hunk.lines
    )
    text = "\n".join(result)
    return text + ("\n" if trailing_newline else "")


class CodeWorkspaceWriteTool(BaseTool):
    name = "code_workspace_write_file"
    description = (
        "Write a source/test/documentation file inside the selected Code Builder workspace. "
        "Use this to materialize generated code before running tests."
    )
    params = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path under the selected code workspace, e.g. 'scma_demo/main.py'.",
            },
            "content": {
                "type": "string",
                "description": "Complete file content to write. Overwrites existing file.",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def execute(self, path: str, content: str) -> ToolResult:
        try:
            target = _resolve_workspace_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content or ""), encoding="utf-8")
            return ToolResult("success", f"Wrote {target}")
        except Exception as exc:
            return ToolResult("fail", f"Write failed: {exc}")


class CodeWorkspaceReadTool(BaseTool):
    name = "code_workspace_read_file"
    description = "Read a file from the selected Code Builder workspace for inspection or revision."
    params = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path under the selected code workspace."},
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to return (default 6000).",
                "minimum": 1,
                "maximum": 20000,
                "default": 6000,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def execute(self, path: str, max_chars: int = 6000) -> ToolResult:
        try:
            target = _resolve_workspace_path(path)
            text = target.read_text(encoding="utf-8")
            limit = max(1, min(int(max_chars), 20000))
            return ToolResult("success", text[:limit])
        except Exception as exc:
            return ToolResult("fail", f"Read failed: {exc}")


class CodeWorkspaceMakePatchTool(BaseTool):
    name = "code_workspace_make_patch"
    description = (
        "Create a unified diff for replacing one file inside the selected Code Builder workspace. "
        "Use this to preview project-level edits before applying them. This does not modify files; "
        "call code_workspace_apply_patch with the returned diff to make the change real."
    )
    params = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative file path under the selected code workspace."},
            "new_content": {"type": "string", "description": "Complete desired new file content."},
        },
        "required": ["path", "new_content"],
        "additionalProperties": False,
    }

    def execute(self, path: str, new_content: str) -> ToolResult:
        try:
            target = _resolve_workspace_path(path)
            old_text = target.read_text(encoding="utf-8") if target.exists() else ""
            patch = _make_unified_diff(path, old_text, str(new_content or ""))
            return ToolResult("success", patch or "No changes.")
        except Exception as exc:
            return ToolResult("fail", f"Make patch failed: {exc}")


class CodeWorkspaceApplyPatchTool(BaseTool):
    name = "code_workspace_apply_patch"
    description = (
        "Apply a single-file unified diff inside the selected Code Builder workspace. "
        "Patch headers must use '--- a/<path>' and '+++ b/<path>'; paths cannot escape the workspace."
    )
    params = {
        "type": "object",
        "properties": {
            "patch": {"type": "string", "description": "Unified diff text for one file."},
        },
        "required": ["patch"],
        "additionalProperties": False,
    }

    def execute(self, patch: str) -> ToolResult:
        try:
            path, hunks = _parse_unified_patch(patch)
            target = _resolve_workspace_path(path)
            original = target.read_text(encoding="utf-8") if target.exists() else ""
            updated = _apply_unified_patch_to_text(original, hunks)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(updated, encoding="utf-8")
            return ToolResult("success", f"Applied patch to {target}")
        except Exception as exc:
            return ToolResult("fail", f"Apply patch failed: {exc}")


class CodeWorkspaceSetValidationPlanTool(BaseTool):
    name = "code_workspace_set_validation_plan"
    description = (
        "Declare the concrete validation targets that must all pass before Code Builder may claim completion. "
        "Use this after inspecting the task, project files, and available tools. Put one required script/test/entry "
        "target per line, for example 'test_components.m\\nrun_simulation.m'. If the task does not name targets, "
        "choose the smallest meaningful project validation targets yourself."
    )
    params = {
        "type": "object",
        "properties": {
            "targets": {
                "type": "string",
                "description": "Newline-, comma-, or semicolon-separated validation targets that must all pass.",
            },
            "rationale": {
                "type": "string",
                "description": "Short reason these targets prove the project is runnable.",
                "default": "",
            },
        },
        "required": ["targets"],
        "additionalProperties": False,
    }

    def execute(self, targets: str, rationale: str = "") -> ToolResult:
        plan = set_code_validation_plan(targets=targets, rationale=rationale)
        if not plan["targets"]:
            return ToolResult("fail", "Validation plan is empty.", data=plan)
        return ToolResult("success", f"Validation plan set: {plan}", data=plan)


class CodeWorkspaceListTool(BaseTool):
    name = "code_workspace_list_files"
    description = "List files under the selected Code Builder workspace, optionally within a relative subdirectory."
    params = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative subdirectory under the selected code workspace (default '.').",
                "default": ".",
            }
        },
        "additionalProperties": False,
    }

    def execute(self, path: str = ".") -> ToolResult:
        try:
            root = _workspace_root()
            target = _resolve_workspace_path(path)
            if not target.exists():
                return ToolResult("success", "No files found.", data=[])
            files = []
            for item in sorted(target.rglob("*")):
                if item.is_file():
                    files.append(str(item.relative_to(root)))
            return ToolResult("success", "\n".join(files) or "No files found.", data=files)
        except Exception as exc:
            return ToolResult("fail", f"List failed: {exc}", data=[])


class CodeWorkspaceRunPythonTool(BaseTool):
    name = "code_workspace_run_python"
    description = (
        "Run a Python script inside the selected Code Builder workspace. "
        "Only relative script paths are accepted; use after writing code files."
    )
    params = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative Python script path, e.g. 'demo/main.py'."},
            "timeout_s": {
                "type": "integer",
                "description": "Timeout in seconds (1-120, default 30).",
                "minimum": 1,
                "maximum": 120,
                "default": 30,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def execute(self, path: str, timeout_s: int = 30) -> ToolResult:
        try:
            script = _resolve_workspace_path(path)
            if script.suffix.lower() != ".py":
                return ToolResult("fail", "Only .py scripts can be executed by this tool.")
            if not script.exists():
                return ToolResult("fail", f"Script not found: {path}")
            result = _run_command([_python_executable(), str(script)], cwd=_workspace_root(), timeout_s=timeout_s)
            if result.status == "success":
                set_code_validation_record(
                    passed=True,
                    tool=self.name,
                    target=path,
                    evidence=result.result,
                )
            return result
        except Exception as exc:
            return ToolResult("fail", f"Run failed: {exc}")


class CodeWorkspaceRunPytestTool(BaseTool):
    name = "code_workspace_run_pytest"
    description = "Run pytest inside the selected Code Builder workspace for a relative test path or directory."
    params = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative test file or directory under the selected code workspace (default '.').",
                "default": ".",
            },
            "timeout_s": {
                "type": "integer",
                "description": "Timeout in seconds (1-120, default 60).",
                "minimum": 1,
                "maximum": 120,
                "default": 60,
            },
        },
        "additionalProperties": False,
    }

    def execute(self, path: str = ".", timeout_s: int = 60) -> ToolResult:
        try:
            target = _resolve_workspace_path(path)
            if not target.exists():
                return ToolResult("fail", f"Test path not found: {path}")
            result = _run_command([_python_executable(), "-m", "pytest", str(target)], cwd=_workspace_root(), timeout_s=timeout_s)
            if result.status == "success":
                set_code_validation_record(
                    passed=True,
                    tool=self.name,
                    target=path,
                    evidence=result.result,
                )
            return result
        except Exception as exc:
            return ToolResult("fail", f"Pytest failed: {exc}")


class CodeWorkspaceRunShellTool(BaseTool):
    name = "code_workspace_run_shell"
    description = (
        "Run a bounded shell command inside the selected Code Builder project/workspace. "
        "Use for project commands such as installing local requirements, running package scripts, "
        "or invoking non-Python CLIs. The command runs with cwd restricted to the workspace; "
        "do not use it for destructive filesystem operations or interactive long-running services."
    )
    params = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "Shell command to run in the selected code workspace, e.g. "
                    "'python -m pip install -r requirements.txt' or 'python main.py'."
                ),
            },
            "cwd": {
                "type": "string",
                "description": "Relative working directory under the selected code workspace (default '.').",
                "default": ".",
            },
            "timeout_s": {
                "type": "integer",
                "description": "Timeout in seconds (1-180, default 60).",
                "minimum": 1,
                "maximum": 180,
                "default": 60,
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def execute(self, command: str, cwd: str = ".", timeout_s: int = 60) -> ToolResult:
        try:
            issue = _looks_unsafe_shell_command(command)
            if issue:
                return ToolResult("fail", f"Shell command blocked: {issue}")
            command_cwd = _resolve_command_cwd(cwd)
            timeout = max(1, min(int(timeout_s or 60), 180))
            result = _run_command(
                _shell_invocation(str(command)),
                cwd=command_cwd,
                timeout_s=timeout,
                env=_python_shell_env(),
            )
            if result.status == "success":
                set_code_validation_record(
                    passed=True,
                    tool=self.name,
                    target=str(command)[:240],
                    evidence=result.result,
                )
            return result
        except Exception as exc:
            return ToolResult("fail", f"Shell command failed: {exc}")


class CodeWorkspaceRecordValidationTool(BaseTool):
    name = "code_workspace_record_validation"
    description = (
        "Record structured validation evidence after you run or inspect code with any available tool, "
        "including internal Python/pytest tools or external MCP tools such as MATLAB/Jupyter. "
        "Use only after an actual verification attempt; set passed=false when the attempt failed."
    )
    params = {
        "type": "object",
        "properties": {
            "passed": {
                "type": "boolean",
                "description": "Whether the latest verification attempt passed.",
            },
            "tool": {
                "type": "string",
                "description": "Tool or command used for verification, e.g. code_workspace_run_pytest or matlab__evaluate_matlab_code.",
            },
            "target": {
                "type": "string",
                "description": "Relative file, test path, script, or project target that was verified.",
            },
            "evidence": {
                "type": "string",
                "description": "Short factual evidence from the tool output, including exit code or error summary.",
            },
        },
        "required": ["passed", "tool", "target", "evidence"],
        "additionalProperties": False,
    }

    def execute(self, passed: bool, tool: str, target: str, evidence: str) -> ToolResult:
        record = set_code_validation_record(
            passed=passed,
            tool=tool,
            target=target,
            evidence=evidence,
        )
        status = "success" if record["passed"] else "fail"
        return ToolResult(status, f"Validation recorded: {record}", data=record)
