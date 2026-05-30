"""Shared utilities for expert agent nodes.

Provides helpers that were previously duplicated across multiple expert
modules: user-query extraction, paper-chunk grouping, and empty-result
responses.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from scholar_agent.agents.state import MultiAgentState
from scholar_agent.storage.memory_store import AgentMemoryStore


def extract_user_query(state: MultiAgentState) -> str:
    """Return the original user message from the conversation history.

    Looks for the latest ``HumanMessage`` where ``name is None`` (i.e. a
    message sent by the actual user, not an expert or the supervisor).

    Args:
        state: The current multi-agent state.

    Returns:
        The user's original query string, or an empty string if not found.
    """
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage) and msg.name is None:
            return str(msg.content)
    return ""


def build_paper_context(
    chunks: List[Dict[str, Any]],
    max_chars_per_paper: int = 8000,
) -> Dict[str, Any]:
    """Group knowledge-base chunks by paper_id and build a text context block.

    Args:
        chunks: Raw chunks from ``PaperManager.search_knowledge()``.  Each
            dict must have ``paper_id``, ``title``, and ``content`` keys.
        max_chars_per_paper: Per-paper content character limit before
            truncation (default 8000).

    Returns:
        A dict with:
        - ``context``: formatted text block ready to insert into an LLM prompt.
        - ``paper_ids``: list of unique paper IDs found.
        - ``title``: title of the first paper (or ``"Unknown"``).
    """
    papers: Dict[str, Dict[str, Any]] = {}
    for ch in chunks:
        pid = ch["paper_id"]
        if pid not in papers:
            papers[pid] = {"title": ch["title"], "chunks": []}
        papers[pid]["chunks"].append(ch["content"])

    context_lines: List[str] = []
    paper_ids: List[str] = []
    for pid, info in papers.items():
        paper_ids.append(pid)
        joined = "\n---\n".join(info["chunks"])
        if len(joined) > max_chars_per_paper:
            joined = joined[:max_chars_per_paper] + "..."
        context_lines.append(
            f"论文ID: {pid}\n标题: {info['title']}\n\n内容:\n{joined}"
        )

    context = "\n\n" + "=" * 60 + "\n\n".join(context_lines)
    title = papers[paper_ids[0]]["title"] if paper_ids else "Unknown"
    return {"context": context, "paper_ids": paper_ids, "title": title}


def collect_paper_ids_from_outputs(state: MultiAgentState, preferred_experts: List[str] | None = None) -> List[str]:
    """Return de-duplicated paper IDs from prior expert metadata."""
    preferred = set(preferred_experts or [])
    outputs = state.get("expert_outputs", []) or []
    if preferred:
        outputs = [eo for eo in outputs if eo.get("expert_name") in preferred]

    paper_ids: List[str] = []
    seen = set()
    for output in outputs:
        metadata = output.get("metadata", {}) or {}
        for pid in metadata.get("paper_ids", []) or []:
            value = str(pid or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            paper_ids.append(value)
    return paper_ids


def collect_context_paper_ids(
    state: MultiAgentState,
    preferred_experts: List[str] | None = None,
) -> List[str]:
    """Return paper IDs from prior outputs first, then conversation memory."""
    paper_ids = collect_paper_ids_from_outputs(state, preferred_experts=preferred_experts)
    seen = set(paper_ids)
    memory = state.get("memory", {})
    if isinstance(memory, dict):
        for pid in memory.get("recent_paper_ids", []) or []:
            value = str(pid or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            paper_ids.append(value)
    return paper_ids


def collect_retrieval_query_from_outputs(
    state: MultiAgentState,
    preferred_experts: List[str] | None = None,
) -> str:
    """Return the first concise retrieval query recorded by prior experts."""
    preferred = set(preferred_experts or [])
    outputs = state.get("expert_outputs", []) or []
    if preferred:
        outputs = [eo for eo in outputs if eo.get("expert_name") in preferred]

    for output in outputs:
        metadata = output.get("metadata", {}) or {}
        for key in ("local_query", "retrieval_query", "arxiv_query"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
    return ""


def filter_chunks_by_paper_ids(chunks: List[Dict[str, Any]], paper_ids: List[str]) -> List[Dict[str, Any]]:
    """Keep chunks whose paper_id is in the selected paper set."""
    if not paper_ids:
        return chunks
    allowed = set(paper_ids)
    filtered = [chunk for chunk in chunks if chunk.get("paper_id") in allowed]
    return filtered or chunks


def format_prior_expert_context(
    state: MultiAgentState,
    expert_names: List[str],
    max_chars_per_expert: int = 1800,
) -> str:
    """Format selected prior expert outputs for downstream modules."""
    blocks = []
    wanted = set(expert_names)
    for output in state.get("expert_outputs", []) or []:
        name = output.get("expert_name", "unknown")
        if name not in wanted:
            continue
        content = str(output.get("content", "") or "").strip()
        if not content:
            continue
        blocks.append(f"### {name}\n{content[:max_chars_per_expert]}")
    if not blocks:
        return ""
    return "\n\n## 前序专家输出\n" + "\n\n".join(blocks)


def format_memory_context(state: MultiAgentState, heading: str = "## 会话记忆") -> str:
    """Return a compact memory block for prompts, or an empty string."""
    memory_text = AgentMemoryStore.format_for_prompt(state.get("memory", {}))
    if not memory_text:
        return ""
    return f"\n\n{heading}\n{memory_text}"


def memory_augmented_query(
    state: MultiAgentState,
    primary_query: str,
    fallback_query: str,
) -> str:
    """Append compact memory context to retrieval prompts when available."""
    query = (primary_query or "").strip() or fallback_query
    memory = state.get("memory", {})
    topics = memory.get("recent_topics", []) if isinstance(memory, dict) else []
    paper_ids = memory.get("recent_paper_ids", []) if isinstance(memory, dict) else []

    hints = []
    if topics:
        hints.append(f"近期主题: {', '.join(topics[:3])}")
    if paper_ids:
        hints.append(f"近期论文ID: {', '.join(paper_ids[:6])}")
    if not hints:
        return query
    return f"{query}\n\n会话记忆提示: {'; '.join(hints)}"


def retrieval_query_from_context(
    state: MultiAgentState,
    *,
    prior_query: str = "",
    fallback_query: str = "",
) -> str:
    """Build a retrieval query only from retrieval context, not task prose."""
    return memory_augmented_query(
        state,
        primary_query=prior_query,
        fallback_query=fallback_query,
    )


def no_papers_response(expert_name: str) -> Dict[str, Any]:
    """Return a standardised response dict when no papers are found.

    Args:
        expert_name: The expert's name string (e.g. ``"summarizer"``).

    Returns:
        A dict with ``next_agent``, ``expert_outputs``, and ``messages`` keys
        suitable for returning from an expert node function.
    """
    return {
        "next_agent": "supervisor",
        "expert_outputs": [
            {
                "expert_name": expert_name,
                "content": "本地数据库中未找到相关论文。",
                "metadata": {"error": "no_papers_found", "paper_ids": []},
            }
        ],
        "messages": [
            HumanMessage(content=f"[{expert_name} 输出]\n本地数据库中未找到相关论文。", name=expert_name)
        ],
    }
