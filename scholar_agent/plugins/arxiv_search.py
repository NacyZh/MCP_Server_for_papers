"""arXiv API client with caching, rate limiting, and advanced query building."""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List

if __package__ is None:
    _proj = Path(__file__).resolve().parent
    while _proj and not (_proj / "pyproject.toml").exists():
        _proj = _proj.parent
    if _proj and str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))

import arxiv
import certifi
from arxiv import Result

from scholar_agent.config import conf
from scholar_agent.core.logging import get_logger

logger = get_logger(__name__)

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


class ArxivManager:
    """Wraps the arXiv API client with caching, rate limiting, and relevance filtering."""

    def __init__(self):
        logger.info("[arxiv] init search client")
        self.client = arxiv.Client(page_size=5, delay_seconds=5, num_retries=3)
        self.last_error = ""
        self._cache: Dict[str, tuple[float, List[Dict]]] = {}
        self._cache_ttl_seconds = 180
        self._rate_limit_retry = 0
        self._rate_limit_backoff_base = 2

    @staticmethod
    def _map_sort_by(sort_by: str):
        mapping = {
            "relevance": arxiv.SortCriterion.Relevance,
            "submitted_date": arxiv.SortCriterion.SubmittedDate,
            "last_updated_date": arxiv.SortCriterion.LastUpdatedDate,
        }
        return mapping.get((sort_by or "").lower(), arxiv.SortCriterion.SubmittedDate)

    @staticmethod
    def _map_sort_order(sort_order: str):
        mapping = {
            "descending": arxiv.SortOrder.Descending,
            "ascending": arxiv.SortOrder.Ascending,
        }
        return mapping.get((sort_order or "").lower(), arxiv.SortOrder.Descending)

    @staticmethod
    def _looks_like_advanced_query(query: str) -> bool:
        q = (query or "").lower()
        markers = ["all:", "ti:", "au:", "abs:", "cat:", "and", "or", "not", "(", ")"]
        return any(m in q for m in markers)

    @staticmethod
    def _normalize_terms(query: str) -> List[str]:
        tokens = []
        for part in (query or "").replace("/", " ").replace("-", " ").split():
            word = part.strip().lower()
            if len(word) < 2:
                continue
            tokens.append(word)
        return list(dict.fromkeys(tokens))

    def _build_candidate_queries(self, query: str) -> List[str]:
        raw = (query or "").strip()
        if not raw:
            return []

        if self._looks_like_advanced_query(raw):
            return [raw]

        terms = self._normalize_terms(raw)
        if not terms:
            return [raw]

        if len(terms) == 1:
            return [raw]

        and_query = " AND ".join([f'all:"{t}"' for t in terms])
        candidates = [and_query, raw]
        deduped = []
        seen = set()
        for candidate in candidates:
            key = candidate.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _is_rate_limited_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "http 429" in text or "too many requests" in text

    @staticmethod
    def _build_cache_key(query: str, max_results: int, sort_by: str, sort_order: str) -> str:
        return f"{query.strip().lower()}|{max_results}|{sort_by.lower()}|{sort_order.lower()}"

    def _get_cached(self, key: str) -> List[Dict]:
        item = self._cache.get(key)
        if not item:
            return []
        ts, data = item
        if time.time() - ts > self._cache_ttl_seconds:
            self._cache.pop(key, None)
            return []
        return list(data)

    def _set_cache(self, key: str, data: List[Dict]):
        self._cache[key] = (time.time(), list(data))

    def _execute_single_candidate(
        self, candidate_query, fetch_k, sort_criterion, order, candidate_idx
    ) -> List[Dict]:
        attempts = self._rate_limit_retry + 1
        for attempt in range(1, attempts + 1):
            search = arxiv.Search(
                query=candidate_query,
                max_results=fetch_k,
                sort_by=sort_criterion,
                sort_order=order,
            )
            try:
                results = []
                for paper in self.client.results(search):
                    results.append(
                        {
                            "id": paper.get_short_id(),
                            "title": paper.title,
                            "authors": [author.name for author in paper.authors],
                            "published": paper.published.strftime("%Y-%m-%d"),
                            "abstract": paper.summary.replace("\n", " "),
                            "pdf_url": paper.pdf_url,
                        }
                    )
                return results
            except Exception as e:
                if self._is_rate_limited_error(e):
                    if attempt >= attempts:
                        raise
                    wait_s = self._rate_limit_backoff_base ** (attempt - 1)
                    logger.info(
                        f"[arxiv] rate-limited candidate[{candidate_idx}] retry={attempt}/{attempts} wait={wait_s}s"
                    )
                    time.sleep(wait_s)
                    continue
                raise

        return []

    @staticmethod
    def _is_relevant(paper: Dict, terms: List[str]) -> bool:
        if not terms:
            return True
        haystack = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        return all(t in haystack for t in terms)

    def search_papers(
        self, query, max_results=5, sort_by="submitted_date", sort_order="descending"
    ) -> List[Dict]:
        """Search arXiv with caching, rate limiting, and relevance filtering."""
        self.last_error = ""
        logger.info(f"[arxiv] search query='{query}'")

        cache_key = self._build_cache_key(query, max_results, sort_by, sort_order)
        cached = self._get_cached(cache_key)
        if cached:
            logger.info(f"[arxiv] cache hit query='{query}' count={len(cached)}")
            return cached[:max_results]

        sort_criterion = self._map_sort_by(sort_by)
        order = self._map_sort_order(sort_order)

        terms = self._normalize_terms(query)
        candidate_queries = self._build_candidate_queries(query)
        fetch_k = max(max_results * 4, max_results)

        for idx, candidate_query in enumerate(candidate_queries, start=1):
            logger.info(f"[arxiv] search candidate[{idx}]='{candidate_query}'")
            try:
                results = self._execute_single_candidate(
                    candidate_query=candidate_query,
                    fetch_k=fetch_k,
                    sort_criterion=sort_criterion,
                    order=order,
                    candidate_idx=idx,
                )
                filtered = [p for p in results if self._is_relevant(p, terms)]
                if filtered:
                    final_results = filtered[:max_results]
                    logger.info(
                        f"[arxiv] search done candidate[{idx}] relevant={len(filtered)} returned={len(final_results)}"
                    )
                    self._set_cache(cache_key, final_results)
                    return final_results
            except Exception as e:
                logger.info(f"[arxiv] search failed candidate[{idx}]: {e}")
                if self._is_rate_limited_error(e):
                    self.last_error = "arXiv interface is rate-limited (HTTP 429). Please try again later."
                    break

        logger.info("[arxiv] search done no relevant results")
        return []

    def download_paper_by_id(self, arxiv_id: str, filename: str, dirpath: str = conf.PAPERS_DIR):
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(self.client.results(search), None)
            if result is None:
                return "failure", f"arXiv paper not found: {arxiv_id}"

            os.makedirs(dirpath, exist_ok=True)
            filename = filename + ".pdf"
            saved_path = result.download_pdf(dirpath=dirpath, filename=filename)
            return "success", f"Downloaded to {saved_path}"
        except Exception as e:
            return "failure", str(e)

    @staticmethod
    def format_for_llm(paper_json: List[Dict]) -> str:
        """Format search results into LLM-friendly text."""
        if not paper_json:
            return "No relevant papers found on arXiv."

        formatted_text = "=== arXiv Search Results ===\n\n"
        for i, p in enumerate(paper_json, 1):
            formatted_text += f"[{i}] Title: {p['title']}\n"
            formatted_text += f"    Authors: {', '.join(p['authors'])}\n"
            formatted_text += f"    Published: {p['published']}\n"
            formatted_text += f"    arXiv ID: {p['id']}\n"
            formatted_text += f"    PDF: {p['pdf_url']}\n"
            formatted_text += f"    Abstract: {p['abstract'][:500]}...\n\n"

        return formatted_text


# ==========================================
# 独立测试代码
# ==========================================
if __name__ == "__main__":
    searcher = ArxivManager()

    test_query = "Large Language Model RAG"
    papers = searcher.search_papers(test_query, max_results=1, sort_by="submitted_date", sort_order="descending")

    if papers:
        file_title = papers[0]["title"]
        arxiv_id = papers[0]["id"]
        status, msg = searcher.download_paper_by_id(arxiv_id=arxiv_id, dirpath=conf.PAPERS_DIR, filename=file_title)
        print(status)
        print(msg)
        prompt_text = searcher.format_for_llm(papers)
        print("\n" + "=" * 50)
        print("LLM context preview:")
        print("=" * 50)
        print(prompt_text)
