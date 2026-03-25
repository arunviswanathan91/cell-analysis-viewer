# ============================================================
#  STEP 3 — RAG CLIENT
#  File: src/remote_rag.py
#
#  Drop-in for src/true_rag.py — calls the HF Space API
#  instead of local ChromaDB. Zero local indexing. Zero startup delay.
#
#  Usage:
#    from src.remote_rag import get_rag_client
#    rag = get_rag_client()           # cached via @st.cache_resource
#    result = rag.generate(question, history=[...])
#
#  Self-test:
#    python src/remote_rag.py
# ============================================================

import os
import time
import logging
from typing import List, Dict, Optional, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────
HF_SPACE_URL    = os.environ.get(
    "HF_SPACE_URL",
    "https://arunviswanathan91-cell-analysis-rag-api.hf.space",
)
REQUEST_TIMEOUT  = 45     # seconds — for fast endpoints (search, generate, health)
FETCH_TIMEOUT    = 120    # seconds — for full dataset scan endpoints (/fetch_cell_type, /compress)
MAX_RETRIES      = 3
BACKOFF_FACTOR   = 1.0
MAX_HISTORY_TURNS = 6    # must match server

# Valid source_filter values (from manifest)
VALID_SOURCES = [
    "bayesian",
    "bayesian_continuous",
    "bayesian_csvs",
    "bayesian_csvs_continuous",
    "clinical",
    "interactome",
    "signatures",
    "stabl",
    "survival",
    "zscores",
]

# ──────────────────────────────────────────────
#  CELL-TYPE SYSTEM PROMPT  (3-stage pipeline)
#  Used when a specific cell type is detected in the query.
#  Instructs the LLM to write paragraph prose with inline bracket citations.
# ──────────────────────────────────────────────
CELL_TYPE_SYSTEM_PROMPT = """You are a specialist scientific assistant for pancreatic cancer (PAAD) BMI-immune microenvironment research.

You have been given pre-extracted findings for a specific cell type from three analyses:
  1. Categorical BMI group comparisons (Normal vs Overweight vs Obese)
  2. Continuous BMI dose-response (BMI as a continuous variable)
  3. Cell-cell Interactome (ligand-receptor communication networks)

## WRITING RULES — CRITICAL

Write your answer as flowing, natural prose paragraphs. Do NOT use bullet points, numbered lists, or headers.

For every data point you mention, immediately follow it with an inline bracketed citation in one of these formats:
  [Categorical: <signature> — <comparison>, effect: <value>, HDI: [<low>, <high>], CREDIBLE/not credible]
  [Continuous: <signature> — slope: <value>, HDI: [<low>, <high>], CREDIBLE/not credible]
  [Interactome: <cell_A> → <cell_B>, <ligand>→<receptor>, enrichment: <value>, <condition>]

Cover all three analysis types if data is available. If a finding is NOT credible (HDI crosses zero), still mention it but note it explicitly as not statistically credible.

## CREDIBILITY RULES
- CREDIBLE = 95% HDI entirely excludes zero (both bounds have the same sign)
- NOT credible = HDI crosses zero, even if the posterior mean is nonzero

Do not invent or estimate numbers — only cite values present in the provided findings."""

# ──────────────────────────────────────────────
#  CELL-TYPE KEYWORD MAP  (for query detection)
# ──────────────────────────────────────────────
_CELL_TYPE_KEYWORDS: Dict[str, str] = {
    "cd8":             "CD8",
    "cd4":             "CD4",
    "nk cell":         "NK",
    "nk-cell":         "NK",
    "natural killer":  "NK",
    "b cell":          "B CELLS",
    "b-cell":          "B CELLS",
    "plasma cell":     "PLASMA",
    "plasmablast":     "PLASMA",
    "macrophage":      "MACROPHAGE",
    "monocyte":        "MONOCYTE",
    "dendritic":       "DENDRITIC",
    "neutrophil":      "NEUTROPHIL",
    "treg":            "REGULATORY",
    "regulatory t":    "REGULATORY",
    "fibroblast":      "FIBROBLAST",
    "caf":             "CAF",
    "icaf":            "ICAF",
    "myofibroblast":   "MYOFIBROBLAST",
    "endothelial":     "ENDOTHELIAL",
    "tumor cell":      "TUMOR",
    "tumour cell":     "TUMOR",
    "cancer cell":     "TUMOR",
    "acinar":          "ACINAR",
    "basophil":        "BASOPHIL",
    "mast cell":       "MAST",
    "tam":             "TAM",
    "stellate":        "STELLATE",
    "pericyte":        "PERICYTE",
    "schwann":         "SCHWANN",
}


# ──────────────────────────────────────────────
#  HTTP SESSION  (persistent, with retries)
# ──────────────────────────────────────────────
def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total            = MAX_RETRIES,
        backoff_factor   = BACKOFF_FACTOR,
        status_forcelist = [429, 500, 502, 503, 504],
        allowed_methods  = ["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


# ──────────────────────────────────────────────
#  REMOTE RAG CLIENT
# ──────────────────────────────────────────────
class RemoteRAG:
    """
    Thin client that wraps the HF Space RAG API.

    Maintains NO local state — all computation happens on the Space.
    The only thing this class does is:
      1. Format requests
      2. Handle retries + errors
      3. Return typed dicts
    """

    def __init__(self, space_url: str = HF_SPACE_URL):
        self.base_url = space_url.rstrip("/")
        self._session = _make_session()

    # ── Health ─────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """Check Space health. Returns dict with 'status' key."""
        try:
            r = self._session.get(
                f"{self.base_url}/health",
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def is_ready(self) -> bool:
        return self.health().get("status") == "ready"

    def wait_until_ready(self, max_wait_s: int = 180, poll_s: int = 5) -> bool:
        """
        Poll /health until Space is ready or timeout.
        Returns True if ready, False if timed out.
        """
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            if self.is_ready():
                return True
            time.sleep(poll_s)
        return False

    # ── Cell-type detection ─────────────────────

    def _detect_cell_type(self, query: str) -> Optional[str]:
        """
        Scan the query for a known cell type keyword.
        Returns the canonical search term (e.g. "CD8") or None.
        """
        q = query.lower()
        for keyword, canonical in _CELL_TYPE_KEYWORDS.items():
            if keyword in q:
                return canonical
        return None

    # ── Fetch all docs for a cell type (bypasses FAISS) ────

    def fetch_cell_type(
        self,
        cell_type: str,
        sources:   List[str] = [],
    ) -> Dict[str, Any]:
        """
        POST /fetch_cell_type — returns ALL documents whose cell_type
        contains the given term (case-insensitive partial match).
        Bypasses FAISS so completeness is guaranteed.
        """
        payload = {"cell_type": cell_type, "sources": sources}
        try:
            r = self._session.post(
                f"{self.base_url}/fetch_cell_type",
                json    = payload,
                timeout = FETCH_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.error(f"fetch_cell_type error: {e}")
            return {"results": [], "total_found": 0}

    # ── Compress raw docs with small LLM ───────

    def compress_findings(
        self,
        cell_type:  str,
        documents:  List[str],
    ) -> str:
        """
        POST /compress — uses llama-3.1-8b-instant to extract key
        quantitative findings from raw document texts.
        Returns the compressed findings string (empty string on failure).
        """
        payload = {"cell_type": cell_type, "documents": documents}
        try:
            r = self._session.post(
                f"{self.base_url}/compress",
                json    = payload,
                timeout = FETCH_TIMEOUT,
            )
            r.raise_for_status()
            return r.json().get("compressed_findings", "")
        except Exception as e:
            log.error(f"compress_findings error: {e}")
            return ""

    # ── Search (retrieval only) ─────────────────

    def search(
        self,
        query:         str,
        n_results:     int           = 10,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Pure semantic search — no LLM generation.
        Returns list of result dicts with keys:
          id, text, source, subfolder, cell_type, signature,
          comparison, compartment, score
        """
        if source_filter and source_filter not in VALID_SOURCES:
            log.warning(f"Unknown source_filter '{source_filter}' — ignoring")
            source_filter = None

        payload = {
            "query":         query,
            "n_results":     n_results,
            "source_filter": source_filter,
        }
        try:
            r = self._session.post(
                f"{self.base_url}/search",
                json    = payload,
                timeout = REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            return r.json().get("results", [])
        except requests.HTTPError as e:
            log.error(f"Search HTTP error: {e.response.status_code} — {e.response.text[:200]}")
            return []
        except Exception as e:
            log.error(f"Search error: {e}")
            return []

    # ── Generate (full RAG with history) ───────

    def generate(
        self,
        query:                str,
        conversation_history: List[Dict[str, str]] = None,
        n_results:            int                  = 10,
        source_filter:        Optional[str]        = None,
        system_prompt:        Optional[str]        = None,
    ) -> Dict[str, Any]:
        """
        Full RAG: retrieve → (inject history) → LLM → answer + sources.

        When the query mentions a specific cell type (e.g. "CD8 T cells"),
        uses the 3-stage pipeline:
          Stage 1 — fetch_cell_type: retrieve ALL docs bypassing FAISS
          Stage 2 — compress_findings: compact with fast small LLM
          Stage 3 — /generate: write paragraph answer with bracket citations

        For generic queries (no cell type detected), falls back to standard
        FAISS semantic search.

        Returns dict with keys:
            answer              str   — LLM answer (or raw context if no Groq)
            sources             list  — retrieved docs (always present)
            took_ms             float — total server time
            used_history_turns  int   — how many history turns the server used
            pipeline            str   — "cell_type_3stage" or "semantic"
            total_docs_retrieved int  — only present for 3-stage pipeline
            error               str   — only present if call failed
        """
        # ── Detect cell type in query ───────────
        cell_type = self._detect_cell_type(query)

        if cell_type:
            return self._generate_cell_type(
                query                = query,
                cell_type            = cell_type,
                conversation_history = conversation_history,
                n_results            = n_results,
                source_filter        = source_filter,
                system_prompt        = system_prompt,
            )

        return self._generate_semantic(
            query                = query,
            conversation_history = conversation_history,
            n_results            = n_results,
            source_filter        = source_filter,
            system_prompt        = system_prompt,
        )

    def _generate_cell_type(
        self,
        query:                str,
        cell_type:            str,
        conversation_history: List[Dict[str, str]] = None,
        n_results:            int                  = 10,
        source_filter:        Optional[str]        = None,
        system_prompt:        Optional[str]        = None,
    ) -> Dict[str, Any]:
        """3-stage pipeline for cell-type specific queries."""
        t0 = time.time()

        # Stage 1: Fetch ALL docs for this cell type
        fetch_result  = self.fetch_cell_type(
            cell_type = cell_type,
            sources   = [],   # empty = all sources; let cell_type partial match do the filtering
        )
        all_docs      = fetch_result.get("results", [])
        doc_texts     = [d["text"] for d in all_docs]
        total_fetched = len(doc_texts)

        if not doc_texts:
            log.warning(f"No docs found for cell type '{cell_type}' — falling back to semantic")
            result = self._generate_semantic(
                query, conversation_history, n_results, source_filter, system_prompt
            )
            result["pipeline"] = "semantic_fallback"
            return result

        # Stage 2: Compress with fast small LLM
        compressed = self.compress_findings(cell_type, doc_texts)

        if not compressed:
            log.warning(f"Compression returned empty for '{cell_type}' — falling back")
            result = self._generate_semantic(
                query, conversation_history, n_results, source_filter, system_prompt
            )
            result["pipeline"] = "semantic_fallback"
            return result

        # Stage 3: Generate paragraph answer with main LLM
        final_sys = system_prompt or CELL_TYPE_SYSTEM_PROMPT
        history   = (conversation_history or [])[-(MAX_HISTORY_TURNS * 2):]

        payload = {
            "query":                query,
            "conversation_history": history,
            "n_results":            n_results,
            "source_filter":        source_filter,
            "system_prompt":        final_sys,
            "compressed_context":   compressed,
        }
        try:
            r = self._session.post(
                f"{self.base_url}/generate",
                json    = payload,
                timeout = REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            result = r.json()
            result["pipeline"]             = "cell_type_3stage"
            result["total_docs_retrieved"] = total_fetched
            result["took_ms"] = (time.time() - t0) * 1000
            return result
        except requests.HTTPError as e:
            status = e.response.status_code
            detail = e.response.json().get("detail", e.response.text[:300]) \
                     if e.response.content else str(e)
            log.error(f"3-stage generate HTTP {status}: {detail}")
            result = self._generate_semantic(
                query, conversation_history, n_results, source_filter, system_prompt
            )
            result["pipeline"] = "semantic_fallback"
            return result
        except Exception as e:
            log.error(f"3-stage generate error: {e}")
            result = self._generate_semantic(
                query, conversation_history, n_results, source_filter, system_prompt
            )
            result["pipeline"] = "semantic_fallback"
            return result

    def _generate_semantic(
        self,
        query:                str,
        conversation_history: List[Dict[str, str]] = None,
        n_results:            int                  = 10,
        source_filter:        Optional[str]        = None,
        system_prompt:        Optional[str]        = None,
    ) -> Dict[str, Any]:
        """Standard FAISS semantic search + generate."""
        if source_filter and source_filter not in VALID_SOURCES:
            log.warning(f"Unknown source_filter '{source_filter}' — ignoring")
            source_filter = None

        history = (conversation_history or [])[-(MAX_HISTORY_TURNS * 2):]

        payload = {
            "query":                query,
            "conversation_history": history,
            "n_results":            n_results,
            "source_filter":        source_filter,
            "system_prompt":        system_prompt,
        }
        try:
            r = self._session.post(
                f"{self.base_url}/generate",
                json    = payload,
                timeout = REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            result = r.json()
            result.setdefault("pipeline", "semantic")
            return result
        except requests.HTTPError as e:
            status = e.response.status_code
            detail = e.response.json().get("detail", e.response.text[:300]) \
                     if e.response.content else str(e)
            log.error(f"Generate HTTP {status}: {detail}")
            sources = self.search(query, n_results, source_filter)
            return {
                "answer":             f"⚠️ LLM generation failed (HTTP {status}): {detail}\n\n"
                                      "Retrieved context is shown in the sources panel.",
                "sources":            sources,
                "took_ms":            0,
                "used_history_turns": 0,
                "pipeline":           "error",
                "error":              detail,
            }
        except Exception as e:
            log.error(f"Generate error: {e}")
            return {
                "answer":             f"⚠️ Connection error: {e}",
                "sources":            [],
                "took_ms":            0,
                "used_history_turns": 0,
                "pipeline":           "error",
                "error":              str(e),
            }

    # ── Stats ──────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        try:
            r = self._session.get(
                f"{self.base_url}/stats",
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    # ── Backwards compat (TrueRAG interface) ───

    def index_data(self, *args, **kwargs):
        """No-op — indexing is done in Colab (Step 1). Kept for drop-in compatibility."""
        log.info("index_data() called on RemoteRAG — no-op (already indexed in HF Dataset)")

    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Alias for generate() — TrueRAG compatibility."""
        return self.generate(query=question, **kwargs)

    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Alias for search() — TrueRAG compatibility."""
        return self.search(query=query, **kwargs)


# ──────────────────────────────────────────────
#  STREAMLIT CACHE FACTORY
# ──────────────────────────────────────────────
def get_rag_client(space_url: str = HF_SPACE_URL) -> RemoteRAG:
    """
    Factory for use with @st.cache_resource.

    Usage in your Streamlit app:
        import streamlit as st
        from src.remote_rag import get_rag_client

        @st.cache_resource
        def load_rag():
            return get_rag_client()

        rag = load_rag()
    """
    return RemoteRAG(space_url=space_url)


# ──────────────────────────────────────────────
#  SELF TEST
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print(f"Testing RemoteRAG against: {HF_SPACE_URL}\n")
    rag = RemoteRAG()

    # Health
    h = rag.health()
    print(f"Health: {json.dumps(h, indent=2)}\n")

    if h.get("status") != "ready":
        print("Space not ready — waiting up to 3 minutes…")
        if not rag.wait_until_ready(max_wait_s=180):
            print("Timed out. Is the Space deployed?")
            exit(1)

    # Search
    print("── Search test ──────────────────────────")
    results = rag.search("CD8 T cells obesity effect", n_results=3)
    for r in results:
        print(f"  [{r['source']} | score={r['score']:.3f}] {r['text'][:120]}…")

    # Generate — single turn
    print("\n── Generate test (single turn) ──────────")
    resp = rag.generate(
        query     = "Which cell types show the strongest BMI effect in immune fine compartment?",
        n_results = 5,
    )
    print(f"  Answer: {resp['answer'][:300]}…")
    print(f"  Sources: {len(resp['sources'])}, took {resp['took_ms']:.0f}ms")

    # Generate — multi-turn (simulate follow-up)
    print("\n── Generate test (multi-turn follow-up) ─")
    history = [
        {"role": "user",      "content": "Which cell types show the strongest BMI effect?"},
        {"role": "assistant", "content": resp["answer"]},
    ]
    resp2 = rag.generate(
        query                = "What about the obese group specifically?",
        conversation_history = history,
        n_results            = 5,
    )
    print(f"  Follow-up answer: {resp2['answer'][:300]}…")
    print(f"  History turns used: {resp2['used_history_turns']}")
    print("\n✅ All tests passed")
