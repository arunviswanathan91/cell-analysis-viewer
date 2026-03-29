# ============================================================
#  STEP 3 — STREAMLIT INTEGRATION
#  File: streamlit_remote_rag.py
#
#  Drop this file into your repo root and call the functions
#  from your existing streamlit_app_with_explorer.py.
#
#  Quick integration (add to your main app file):
#
#    from streamlit_remote_rag import (
#        load_remote_rag,
#        render_rag_sidebar_status,
#        render_rag_chat,
#    )
#
#    # In your sidebar:
#    render_rag_sidebar_status()
#
#    # In an "Ask AI" tab or section:
#    render_rag_chat()
# ============================================================

import streamlit as st
from typing import Optional, List, Dict

from src.remote_rag import RemoteRAG, get_rag_client, VALID_SOURCES

# ──────────────────────────────────────────────
#  HF SPACE URL RESOLUTION
#  Priority: env var → st.secrets → hardcoded
# ──────────────────────────────────────────────
import os

def _resolve_space_url() -> str:
    url = os.environ.get("HF_SPACE_URL", "")
    if url:
        return url
    try:
        return st.secrets["HF_SPACE_URL"]
    except Exception:
        pass
    return "https://arunviswanathan91-cell-analysis-rag-api.hf.space"


# ──────────────────────────────────────────────
#  CACHED CLIENT
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_remote_rag() -> RemoteRAG:
    """
    Load and cache the RAG client for the app session.
    On first load, wakes the HuggingFace Space if it is sleeping.
    """
    rag = get_rag_client(_resolve_space_url())
    # Attempt a quick health check; if not ready, wait up to 3 minutes
    if not rag.is_ready():
        rag.wait_until_ready(max_wait_s=180, poll_s=5)
    return rag


# ──────────────────────────────────────────────
#  SIDEBAR STATUS WIDGET
# ──────────────────────────────────────────────
def render_rag_sidebar_status(rag: Optional[RemoteRAG] = None):
    """
    Show a compact RAG status badge in the sidebar.
    Call once from your main sidebar block.
    """
    if rag is None:
        rag = load_remote_rag()

    h = rag.health()
    status = h.get("status", "unknown")

    with st.sidebar:
        st.markdown("---")
        if status == "ready":
            docs = h.get("documents", 0)
            groq = "✅" if h.get("groq_enabled") else "⚠️ no LLM"
            st.success(f"🧠 RAG ready — {docs:,} docs  |  Groq {groq}")
        elif status == "initializing":
            st.warning("⏳ RAG Space is initializing…")
            if st.button("Check again", key="rag_retry_init"):
                st.cache_resource.clear()
                st.rerun()
        else:
            err = h.get("error", "unknown error")
            st.warning("⏳ RAG Space is waking up — this may take up to 60 seconds on first use.")
            if st.button("Wake up Space", key="rag_retry"):
                with st.spinner("Waking up the AI Space…"):
                    ready = rag.wait_until_ready(max_wait_s=120, poll_s=5)
                if ready:
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error(f"Space did not respond in time. Last error: {err}")


# ──────────────────────────────────────────────
#  PER-TAB SYSTEM PROMPTS  (Gap 1 fix)
#  Pass one of these to render_rag_chat() to get
#  a domain-specific LLM framing without redeploying.
# ──────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "default": None,   # uses server default (full scientific prompt)

    "interactome": (
        "You are analysing cell-cell interaction data from CPTAC PAAD tumours. "
        "Focus on ligand-receptor pairs, enrichment ratios, and how interactions "
        "differ between Normal Weight and Overweight conditions. "
        "Compare Bindea, Zheng, and Newman datasets when relevant. "
        "Be concise and cite specific cell type pairs and gene interactions."
    ),

    "survival": (
        "You are interpreting survival analysis results from CPTAC PAAD. "
        "Focus on hazard ratios, 95% confidence intervals, and p-values. "
        "Distinguish between prognostic (HR>1 = worse survival) and protective "
        "(HR<1 = better survival) features. Cite specific signatures and cell types."
    ),

    "bayesian": (
        "You are interpreting Bayesian hierarchical model results for CPTAC PAAD.\n\n"
        "DOCUMENT TYPES:\n"
        "[CATEGORICAL BMI GROUP COMPARISON] = discrete group comparisons "
        "(Normal vs Overweight vs Obese). Fields: effect size (posterior mean), "
        "95% HDI: [low, high], Credibility, Direction, P(effect>0).\n"
        "[CONTINUOUS BMI DOSE-RESPONSE] = BMI slope per 1 SD increase. "
        "A slope near zero does NOT mean no group difference.\n\n"
        "RULES:\n"
        "1. For 'overweight vs normal' or 'obese vs normal' questions → use CATEGORICAL docs only\n"
        "2. Always report the exact HDI numbers from the document e.g. [0.0312, 0.2187]\n"
        "3. CREDIBLE = HDI fully excludes zero. NOT credible = HDI crosses zero\n"
        "4. Never say HDI: ? — values are always present in the retrieved documents\n"
        "5. Never use continuous slope to answer a group comparison question\n"
        "6. Cite cell type, signature, compartment, and comparison for every finding\n"
        "7. Write in flowing prose paragraphs — no bullet points or numbered lists. "
        "For every data point, include an inline bracketed citation immediately after it: "
        "[Categorical: <signature> — <comparison>, effect: <value>, HDI: [<low>, <high>], CREDIBLE/not credible] "
        "or [Continuous: <signature> — slope: <value>, HDI: [<low>, <high>], CREDIBLE/not credible]"
    ),

    "signatures": (
        "You are explaining gene signatures used in CPTAC PAAD cell analysis. "
        "Describe what genes are in the signature, what biological process it "
        "represents, and which cell types it applies to. Be educational and precise."
    ),
}


# ──────────────────────────────────────────────
#  MAIN CHAT UI  (Gap 2 + Gap 3 fix)
# ──────────────────────────────────────────────
def render_rag_chat(
    rag:            Optional[RemoteRAG] = None,
    context_prefix: str                 = "",
    system_prompt_key: str              = "default",
    session_key:    str                 = "rag_chat",
    title:          str                 = "🧠 Ask the AI",
    n_results:      int                 = 10,
    source_filter:  Optional[str]       = None,
):
    """
    Full multi-turn chat UI with:
      - Conversation history (Gap 2)
      - Sources shown separately in expander below each answer (Gap 3)
      - Per-tab system prompt (Gap 1)
      - Source filter dropdown
      - Clear conversation button

    Args:
        rag:               RemoteRAG client (loads if None)
        context_prefix:    Prepended to every query (e.g. "Immune Fine compartment:")
                           Lets the app pass the current sidebar selection automatically.
        system_prompt_key: Key from SYSTEM_PROMPTS dict, or pass a full prompt string.
        session_key:       Unique key for st.session_state — use different keys for
                           different tabs so each has its own history.
        title:             Section header text.
        n_results:         Number of docs to retrieve per query.
        source_filter:     Pre-set source filter (e.g. "interactome").
                           If None, the user sees a dropdown to pick.
    """
    if rag is None:
        rag = load_remote_rag()

    # ── Session state init ────────────────────
    history_key  = f"{session_key}_history"   # list of {role, content}
    messages_key = f"{session_key}_messages"  # rendered messages (include sources)

    if history_key  not in st.session_state:
        st.session_state[history_key]  = []   # [{role, content}, ...]
    if messages_key not in st.session_state:
        st.session_state[messages_key] = []   # [{role, content, sources?}, ...]

    # ── Header + controls ─────────────────────
    st.markdown(f"### {title}")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1])

    with ctrl_col1:
        # Source filter — override if caller pre-set it
        if source_filter is None:
            chosen_source = st.selectbox(
                "Filter source",
                ["All"] + VALID_SOURCES,
                key=f"{session_key}_src_filter",
            )
            active_source = None if chosen_source == "All" else chosen_source
        else:
            active_source = source_filter
            st.caption(f"Source: `{source_filter}`")

    with ctrl_col2:
        st.caption("Retrieval is automatic — cell-type queries load all available data.")

    with ctrl_col3:
        if st.button("🗑 Clear chat", key=f"{session_key}_clear"):
            st.session_state[history_key]  = []
            st.session_state[messages_key] = []
            st.rerun()

    # ── Example questions ─────────────────────
    examples = [
        "Which cell types show the strongest BMI effect in the immune fine compartment?",
        "Are there any obese vs overweight differences in CD8 T cell signatures?",
        "Which ligand-receptor pairs are enriched in overweight tumours?",
        "What signatures are associated with worse survival outcomes?",
        "Show me STABL-selected features for non-immune compartment.",
    ]
    with st.expander("💡 Example questions", expanded=False):
        for ex in examples:
            if st.button(ex, key=f"{session_key}_ex_{hash(ex)}"):
                st.session_state[f"{session_key}_prefill"] = ex

    # ── Render past messages ──────────────────
    for msg in st.session_state[messages_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Gap 3: sources rendered separately below each assistant answer
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"], key=f"{session_key}_{msg.get('idx',0)}")

    # ── Chat input ────────────────────────────
    prefill = st.session_state.pop(f"{session_key}_prefill", "")
    user_input = st.chat_input(
        "Ask about cell types, signatures, BMI effects, survival…",
        key=f"{session_key}_input",
    ) or prefill

    if not user_input:
        return

    # Prepend context from sidebar selection if provided
    full_query = f"{context_prefix} {user_input}".strip() if context_prefix else user_input

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save to rendered messages
    st.session_state[messages_key].append({
        "role": "user", "content": user_input
    })

    # ── Call RAG API ──────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Searching data + generating answer…"):

            # Resolve system prompt
            if system_prompt_key in SYSTEM_PROMPTS:
                sys_prompt = SYSTEM_PROMPTS[system_prompt_key]
            else:
                sys_prompt = system_prompt_key  # treat as literal string

            resp = rag.generate(
                query                = full_query,
                conversation_history = st.session_state[history_key],
                n_results            = n_results,
                source_filter        = active_source,
                system_prompt        = sys_prompt,
            )

        answer   = resp.get("answer",  "No answer returned.")
        sources  = resp.get("sources", [])
        took_ms  = resp.get("took_ms", 0)
        turns    = resp.get("used_history_turns", 0)
        pipeline = resp.get("pipeline", "semantic")
        n_fetched = resp.get("total_docs_retrieved", 0)

        st.markdown(answer)

        # Meta info
        meta_parts = [f"⏱ {took_ms:.0f}ms", f"📄 {len(sources)} sources"]
        if turns > 0:
            meta_parts.append(f"🔄 {turns} history turns")
        if pipeline == "cell_type_3stage" and n_fetched > 0:
            meta_parts.append(f"🔬 3-stage · {n_fetched} docs scanned")
        st.caption(" · ".join(meta_parts))

        # Gap 3: sources in expander
        idx = len(st.session_state[messages_key])
        _render_sources(sources, key=f"{session_key}_{idx}")

    # ── Update history (Gap 2) ────────────────
    # Only push the raw question (not context_prefix) so history
    # reads naturally in follow-up turns.
    st.session_state[history_key].append(
        {"role": "user",      "content": user_input}
    )
    st.session_state[history_key].append(
        {"role": "assistant", "content": answer}
    )

    # Save rendered message (with sources for re-render)
    st.session_state[messages_key].append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
        "idx":     idx,
    })


# ──────────────────────────────────────────────
#  SOURCE RENDERING  (Gap 3 helper)
# ──────────────────────────────────────────────
def _render_sources(sources: List[Dict], key: str = "src"):
    """Render retrieved sources in a collapsed expander."""
    if not sources:
        return
    with st.expander(f"📚 {len(sources)} retrieved sources", expanded=False):
        for i, src in enumerate(sources, 1):
            score     = src.get("score", 0)
            source    = src.get("source", "")
            subfolder = src.get("subfolder", "")
            cell_type = src.get("cell_type", "")
            signature = src.get("signature", "")
            comp      = src.get("compartment", "")
            text      = src.get("text", "")

            # Colour-code score
            if score >= 0.7:
                score_badge = f"🟢 {score:.2f}"
            elif score >= 0.5:
                score_badge = f"🟡 {score:.2f}"
            else:
                score_badge = f"🔴 {score:.2f}"

            meta = " · ".join(filter(None, [source, comp, cell_type, signature]))
            st.markdown(
                f"**{i}. {score_badge}** — `{subfolder}`  \n"
                f"*{meta}*"
            )
            st.text(text[:400] + ("…" if len(text) > 400 else ""))
            if i < len(sources):
                st.divider()


# ──────────────────────────────────────────────
#  CONVENIENCE: SCOPED PROMPTS PER ANALYSIS TAB
# ──────────────────────────────────────────────
def render_interactome_rag(rag=None, compartment=None):
    """Pre-configured chat for the Interactome tab."""
    ctx = f"Interactome analysis context:" if not compartment else \
          f"Interactome analysis ({compartment}):"
    render_rag_chat(
        rag               = rag,
        context_prefix    = ctx,
        system_prompt_key = "interactome",
        session_key       = "rag_interactome",
        title             = "🔗 Ask about Cell-Cell Interactions",
        source_filter     = "interactome",
    )


def render_survival_rag(rag=None):
    """Pre-configured chat for the Survival tab."""
    render_rag_chat(
        rag               = rag,
        system_prompt_key = "survival",
        session_key       = "rag_survival",
        title             = "📈 Ask about Survival Analysis",
        source_filter     = "survival",
    )


def render_bayesian_rag(rag=None, compartment=None):
    """Pre-configured chat for the Bayesian/categorical/continuous tabs."""
    ctx = f"Bayesian analysis ({compartment}):" if compartment else ""
    render_rag_chat(
        rag               = rag,
        context_prefix    = ctx,
        system_prompt_key = "bayesian",
        session_key       = f"rag_bayesian_{compartment or 'all'}",
        title             = "📊 Ask about BMI Effects",
    )


def render_signature_rag(rag=None, cell_type=None, signature=None):
    """Pre-configured chat for exploring signatures."""
    ctx_parts = []
    if cell_type:  ctx_parts.append(f"Cell type: {cell_type}")
    if signature:  ctx_parts.append(f"Signature: {signature}")
    ctx = " | ".join(ctx_parts)
    render_rag_chat(
        rag               = rag,
        context_prefix    = ctx,
        system_prompt_key = "signatures",
        session_key       = "rag_signatures",
        title             = "🧬 Ask about Gene Signatures",
        source_filter     = "signatures",
    )
