"""PDF-to-markdown parser using marker (deep learning) and PyMuPDF (metadata).

Features:
- PDF metadata extraction (title, author, DOI)
- Full PDF-to-markdown via marker (GPU-accelerated)
- Section detection and canonicalization (bilingual support)
- Formula extraction (block and inline)
- Chunking via RecursiveCharacterTextSplitter
- Drops references/bibliography/acknowledgments sections
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ is None:
    _proj = Path(__file__).resolve().parent
    while _proj and not (_proj / "pyproject.toml").exists():
        _proj = _proj.parent
    if _proj and str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from scholar_agent.config import conf
from scholar_agent.core.logging import get_logger

logger = get_logger(__name__)


def _resolve_marker_device(marker_device: str) -> str:
    requested = str(marker_device or "auto").strip().lower()
    if requested in {"cuda", "cpu", "mps"}:
        return requested
    if requested not in {"", "auto"}:
        logger.info("[paper_parser] unknown PAPER_PARSER_DEVICE=%r; falling back to auto", marker_device)
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception as exc:
        logger.info("[paper_parser] torch device detection failed, using CPU: %s", exc)
    return "cpu"


class PaperParser:
    """Markdown-based PDF parser with section detection, formula extraction, and smart chunking."""

    MD_HEADER_PAT = re.compile(r"^(#{2,6})\s+(.+?)\s*$")
    BLOCK_EQ_PAT = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
    INLINE_EQ_PAT = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)
    EQ_NO_PAT = re.compile(r"(?:Eq(?:uation)?\.?\s*)?\(\s*(\d{1,4})\s*\)", re.IGNORECASE)

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        context_window_chars: int = 300,
        marker_device: str = "auto",
        drop_sections: Optional[List[str]] = None,
    ):
        self.context_window_chars = context_window_chars
        self.marker_device = _resolve_marker_device(marker_device)
        self.drop_sections = set(
            [s.lower() for s in (drop_sections or ["references", "bibliography", "acknowledgments", "致谢", "参考文献"])]
        )

        logger.info("[paper_parser] marker device=%s", self.marker_device)
        self.converter = PdfConverter(artifact_dict=create_model_dict(device=self.marker_device))

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def close(self):
        """Best-effort release of marker/PyTorch resources."""
        try:
            self.converter = None
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return None

    # -------------------------
    # Step 1) PDF → Markdown + Meta
    # -------------------------
    def _pdf_to_markdown(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        raw = doc.metadata or {}
        page_count = len(doc)
        title = (raw.get("title") or "").strip()
        author = (raw.get("author") or "").strip()

        probe = []
        for i in range(min(3, page_count)):
            probe.append(doc.load_page(i).get_text("text") or "")
        doc.close()

        probe_text = "\n".join(probe)
        doi = ""
        m = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", probe_text, flags=re.IGNORECASE)
        if m:
            doi = m.group(0)

        rendered = self.converter(pdf_path)
        markdown, _, _ = text_from_rendered(rendered)
        if not isinstance(markdown, str):
            markdown = str(markdown or "")
        markdown = self._normalize_text(markdown)

        if not author:
            author = self._extract_author_from_markdown(markdown, title=title)

        meta = {
            "title": title,
            "author": author,
            "doi": doi,
            "page_count": page_count,
            "source_path": pdf_path,
        }
        logger.info(f"[paper_parser] marker markdown chars={len(markdown)}")
        return markdown, meta

    # -------------------------
    # Step 2) Element detection (headings, formulas)
    # -------------------------
    @staticmethod
    def _extract_author_from_markdown(markdown: str, title: str = "") -> str:
        lines = [ln.strip() for ln in markdown.splitlines() if ln.strip()]
        if not lines:
            return ""

        title_idx = -1
        if title:
            low_t = title.lower()
            for i, ln in enumerate(lines[:80]):
                if low_t in ln.lower():
                    title_idx = i
                    break

        if title_idx < 0:
            for i, ln in enumerate(lines[:80]):
                if re.match(r"^#{2,6}\s+.+", ln):
                    title_idx = i
                    break

        if title_idx < 0:
            return ""

        abs_idx = -1
        for i in range(title_idx + 1, min(len(lines), title_idx + 80)):
            l = lines[i].lower()
            if l.startswith("abstract") or "abstract—" in l or l == "摘要" or l.startswith("摘要"):
                abs_idx = i
                break

        if abs_idx < 0:
            abs_idx = min(len(lines), title_idx + 15)

        candidates = []
        for ln in lines[title_idx + 1 : abs_idx]:
            low = ln.lower()
            if any(
                k in low
                for k in [
                    "@",
                    "university",
                    "institute",
                    "department",
                    "school",
                    "doi",
                    "digital object identifier",
                    "index terms",
                    "manuscript received",
                    "personal use is permitted",
                ]
            ):
                continue
            if len(ln) > 140:
                continue
            name_like = (
                "," in ln
                or " and " in low
                or bool(re.search(r"\b[A-Z]\.\s*[A-Z][a-z]+", ln))
                or bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", ln))
            )
            if name_like:
                clean = re.sub(r"<sup>.*?</sup>", "", ln, flags=re.IGNORECASE).strip(" ,;")
                if clean:
                    candidates.append(clean)

        return candidates[0] if candidates else ""

    def _split_sections(self, markdown: str) -> List[Dict[str, Any]]:
        lines = markdown.splitlines()
        sections: List[Dict[str, Any]] = []

        cur_title = "FullText"
        cur_level = 2
        buf: List[str] = []

        def flush():
            nonlocal buf, cur_title, cur_level
            content = self._normalize_text("\n".join(buf))
            if content:
                sname = self._canonical_section_name(cur_title)
                if sname.lower() not in self.drop_sections:
                    sections.append(
                        {
                            "section_title": cur_title,
                            "section_level": cur_level,
                            "section_name": sname,
                            "content": content,
                        }
                    )
            buf.clear()

        for line in lines:
            m = self.MD_HEADER_PAT.match(line.strip())
            if m:
                flush()
                cur_level = len(m.group(1))
                cur_title = m.group(2).strip()
            else:
                buf.append(line)

        flush()

        if not sections:
            sections = [{"section_title": "FullText", "section_level": 2, "section_name": "full_text", "content": markdown}]
        return sections

    def _extract_elements_in_section(self, section: Dict[str, Any]) -> Dict[str, Any]:
        text = section["content"]

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        formulas = []
        for i, m in enumerate(self.BLOCK_EQ_PAT.finditer(text)):
            raw = f"$${m.group(1).strip()}$$"
            eq_no = self._find_equation_no(text, m.start(), m.end())
            eq_id = f"eq_{eq_no}" if eq_no else f"eq_block_{i}"
            formulas.append(self._build_formula_item(text, m.start(), m.end(), raw, eq_id, eq_no, "block"))

        for i, m in enumerate(self.INLINE_EQ_PAT.finditer(text)):
            raw = f"${m.group(1).strip()}$"
            eq_no = self._find_equation_no(text, m.start(), m.end())
            eq_id = f"eq_{eq_no}" if eq_no else f"eq_inline_{i}"
            formulas.append(self._build_formula_item(text, m.start(), m.end(), raw, eq_id, eq_no, "inline"))

        uniq = {}
        for f in formulas:
            uniq[(f["start"], f["end"], f["formula_raw"])] = f
        formulas = sorted(list(uniq.values()), key=lambda x: x["start"])

        section["paragraphs"] = paragraphs
        section["formulas"] = formulas
        return section

    def _build_formula_item(
        self, text, start, end, raw, eq_id, eq_no, formula_type
    ) -> Dict[str, Any]:
        l = max(0, start - self.context_window_chars)
        r = min(len(text), end + self.context_window_chars)
        ctx = text[l:r].strip()
        return {
            "start": start,
            "end": end,
            "formula_type": formula_type,
            "formula_raw": raw,
            "equation_id": eq_id,
            "equation_no": eq_no or "",
            "context_text": ctx,
            "bound_text": f"{ctx}\n\n[FORMULA {eq_id}]\n{raw}\n[/FORMULA]",
        }

    def _find_equation_no(self, text: str, start: int, end: int) -> Optional[str]:
        around = text[max(0, end - 40) : min(len(text), end + 140)]
        m = self.EQ_NO_PAT.search(around)
        return m.group(1) if m else None

    # -------------------------
    # Step 3) Chunking
    # -------------------------
    def _build_chunks(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for sec in sections:
            sname = sec["section_name"]
            stitle = sec["section_title"]

            para_text = "\n\n".join(sec.get("paragraphs", []))
            if para_text.strip():
                for ch in self.splitter.split_text(para_text):
                    c = ch.strip()
                    if not c:
                        continue
                    chunks.append(
                        {
                            "chunk_type": "text_chunk",
                            "section_name": sname,
                            "section_title": stitle,
                            "content": c,
                            "equation_id": "",
                            "equation_no": "",
                        }
                    )

            for f in sec.get("formulas", []):
                for ch in self.splitter.split_text(f["bound_text"]):
                    c = ch.strip()
                    if not c:
                        continue
                    chunks.append(
                        {
                            "chunk_type": "formula_chunk",
                            "section_name": sname,
                            "section_title": stitle,
                            "content": c,
                            "equation_id": f["equation_id"],
                            "equation_no": f["equation_no"],
                        }
                    )

        return chunks

    # -------------------------
    # Main pipeline
    # -------------------------
    def process_paper(self, pdf_path: str) -> Dict[str, Any]:
        logger.info(f"[paper_parser] processing: {pdf_path}")
        markdown, meta = self._pdf_to_markdown(pdf_path)

        sections = self._split_sections(markdown)
        sections = [self._extract_elements_in_section(s) for s in sections]
        chunks = self._build_chunks(sections)

        return {
            "meta": meta,
            "markdown": markdown,
            "sections": sections,
            "chunks": chunks,
            "chunk_count": len(chunks),
        }

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\u00a0", " ")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _canonical_section_name(title: str) -> str:
        t = title.strip().lower()
        mapping = {
            "abstract": "abstract", "摘要": "abstract",
            "introduction": "introduction", "引言": "introduction", "绪论": "introduction",
            "related work": "related_work", "相关工作": "related_work", "背景": "background",
            "system model": "system_model",
            "proposed": "method", "method": "method", "methods": "method", "approach": "method", "方法": "method",
            "simulations": "experiment", "results": "experiment", "evaluation": "experiment", "实验": "experiment",
            "discussion": "discussion", "讨论": "discussion",
            "conclusion": "conclusion", "conclusions": "conclusion", "结论": "conclusion", "总结": "conclusion",
            "appendix": "appendix",
            "acknowledgments": "acknowledgments", "acknowledgements": "acknowledgments", "致谢": "acknowledgments",
            "references": "references", "bibliography": "references", "参考文献": "references",
        }
        for k, v in mapping.items():
            if k in t:
                return v
        return re.sub(r"\s+", "_", t)[:60]


if __name__ == "__main__":
    conf.check_config()
    pdf_path = os.path.join(conf.PAPERS_DIR, "Automatic Ontology Construction Using LLMs as an External Layer of Memory, Verification, and Planning for Hybrid Intelligent Systems.pdf")

    parser = PaperParser(chunk_size=1200, chunk_overlap=150, context_window_chars=300, marker_device="cuda")
    out = parser.process_paper(pdf_path)

    print("\nParsing complete!")
    print(f"Title: {out['meta'].get('title', '')}")
    print(f"Author: {out['meta'].get('author', '')}")
    print(f"DOI: {out['meta'].get('doi', '')}")
    print(f"Pages: {out['meta'].get('page_count', 0)}")
    print(f"Sections: {len(out['sections'])}")
    print(f"Chunks: {out['chunk_count']}")
