"""Streamlit chat interface for Ask the Model."""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import streamlit as st

from . import tools as T
from .agent import run_agent
from .data import AskModelDB
from .llm import PROVIDERS, LLMError, Provider, list_models, resolve_model

STARTERS = [
    "Which cell types show credible BMI effects when comparing obese with normal weight?",
    "Which signatures in CD8 T cells are credibly changed in obese patients?",
    "What are the most enriched cell-cell interactions in overweight patients in the Zheng dataset?",
    "Which signatures are linked to worse survival, and how strong is the evidence?",
    "Which features did STABL select in the non-immune compartment?",
    "Did the MCMC models converge for every cell type?",
    "How many patients are in each BMI group, and what is their median age?",
    "What does credible mean in this study?",
]

SS = st.session_state


# ----------------------------------------------------------------------------------
#  cached resources
# ----------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading study tables...")
def _load_backend():
    db = AskModelDB()
    return db, T.EntityIndex(db)


@st.cache_data(ttl=3600, show_spinner=False)
def _models(provider_key: str, api_key: str) -> List[str]:
    return list_models(PROVIDERS[provider_key], api_key)


def _secret(names: tuple) -> str:
    for n in names:
        try:
            v = st.secrets.get(n, "")
        except Exception:  # noqa: BLE001 - no secrets file
            v = ""
        v = v or os.environ.get(n, "")
        if v:
            return str(v).strip()
    return ""


def _available_keys() -> Dict[str, str]:
    keys = {}
    for pid, p in PROVIDERS.items():
        k = SS.get("ask_keys", {}).get(pid) or _secret(p.secret_names)
        if k:
            keys[pid] = k
    return keys


# ----------------------------------------------------------------------------------
#  sidebar: model settings
# ----------------------------------------------------------------------------------
def _model_settings() -> Optional[tuple]:
    """Returns (provider, api_key, model) or None when no usable provider is configured."""
    SS.setdefault("ask_keys", {})
    with st.sidebar.expander("Model settings", expanded=not _available_keys()):
        pid_new = st.selectbox("Provider", list(PROVIDERS), format_func=lambda k: PROVIDERS[k].label, key="ask_provider_add")
        typed = st.text_input(f"{PROVIDERS[pid_new].label} API key (kept in this session only)", type="password", key=f"ask_key_{pid_new}")
        if typed:
            SS["ask_keys"][pid_new] = typed.strip()
        st.caption(f"Create a key at {PROVIDERS[pid_new].signup}. To make it permanent, add "
                   f"{PROVIDERS[pid_new].secret_names[0]} to the app secrets.")
        keys = _available_keys()
        if not keys:
            return None
        pid = st.selectbox("Use provider", list(keys), format_func=lambda k: PROVIDERS[k].label, key="ask_provider")
        provider, api_key = PROVIDERS[pid], keys[pid]
        try:
            available = _models(pid, api_key)
        except LLMError as e:
            st.error(str(e))
            return None
        default = resolve_model(provider, available)
        model = st.selectbox("Model", available, index=available.index(default) if default in available else 0, key=f"ask_model_{pid}")
        st.caption("Needs a model that supports tool calling. The default is chosen automatically from what your key can use.")
    return provider, api_key, model


# ----------------------------------------------------------------------------------
#  rendering helpers
# ----------------------------------------------------------------------------------
def _evidence_panel(evidence: list, key: str) -> None:
    if not evidence:
        return
    n_q = sum(1 for e in evidence if e.tool == "run_sql")
    with st.expander(f"Evidence - {n_q} {'query' if n_q == 1 else 'queries'}", expanded=False):
        for e in evidence:
            if e.tool == "find_entities":
                st.markdown(f"**{e.purpose}**")
                st.code(e.sql, language="text")
                continue
            st.markdown(f"**{e.id}** - {e.purpose}")
            st.code(e.sql, language="sql")
            if e.error:
                st.error(e.error)
            elif e.df is not None and not e.df.empty:
                st.dataframe(e.df, hide_index=True, use_container_width=True, height=min(300, 38 + 35 * len(e.df)))
                tail = "" if e.total_rows <= len(e.df) else f" (first {len(e.df)} kept)"
                st.caption(f"{e.total_rows} rows{tail}")
                st.download_button("Download CSV", e.df.to_csv(index=False).encode("utf8"),
                                   file_name=f"{e.id}.csv", mime="text/csv", key=f"{key}_{e.id}")
            else:
                st.caption("0 rows")


def _verification_note(report, error: str = "") -> None:
    if error:
        return
    if report.unverified or report.bad_citations:
        bits = []
        if report.unverified:
            bits.append("figures not found in any query result: " + ", ".join(report.unverified))
        if report.bad_citations:
            bits.append("citations to missing queries: " + ", ".join(sorted(set(report.bad_citations))))
        st.warning("Treat parts of this answer as unverified - " + "; ".join(bits) + ".")
    elif report.checked:
        st.caption(f"{report.checked} figures checked - all match the query results shown under Evidence.")


def _render_message(m: dict, idx: int) -> None:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            _evidence_panel(m.get("evidence", []), key=f"ev{idx}")
            if m.get("report") is not None:
                _verification_note(m["report"])
            if m.get("model"):
                st.caption(f"Model: {m['model']}")


def _transcript() -> str:
    out = []
    for m in SS.get("ask_msgs", []):
        out.append(f"**{'You' if m['role'] == 'user' else 'Assistant'}**\n\n{m['content']}\n")
        for e in m.get("evidence", []):
            if e.tool == "run_sql":
                out.append(f"_{e.id}: {e.purpose}_\n\n```sql\n{e.sql}\n```\n")
    return "\n".join(out)


def _chunks(text: str):
    for tok in text.split(" "):
        yield tok + " "
        time.sleep(0.008)


# ----------------------------------------------------------------------------------
#  main entry
# ----------------------------------------------------------------------------------
def render_ask_model() -> None:
    SS.setdefault("ask_msgs", [])
    db, entities = _load_backend()

    st.markdown('<div class="sub-header">Ask the Model</div>', unsafe_allow_html=True)
    st.caption(
        "Ask anything about this study's results: cell types, signatures, BMI effects, STABL selection, interactions, "
        "survival, patients, or the methods. Answers are built from live queries on the study tables, every figure is "
        "checked against those queries, and questions outside the study are declined."
    )

    cfg = _model_settings()
    if SS["ask_msgs"]:
        c1, c2 = st.sidebar.columns(2)
        if c1.button("New chat", use_container_width=True):
            SS["ask_msgs"] = []
            st.rerun()
        c2.download_button("Save chat", _transcript().encode("utf8"), file_name="ask_the_model.md",
                           mime="text/markdown", use_container_width=True)

    if cfg is None:
        st.info("No language model is connected yet. Open Model settings in the sidebar and paste an API key "
                "(Groq, Hugging Face or OpenAI), or add GROQ_API_KEY / HF_TOKEN to the app secrets.")

    if not SS["ask_msgs"]:
        with st.expander("What data can I ask about?", expanded=False):
            st.dataframe(
                [{"table": t.name, "rows": t.rows, "contents": t.description.split(". ")[0]} for t in db.tables.values()],
                hide_index=True, use_container_width=True)
        st.markdown("**Try one of these**")
        cols = st.columns(2)
        for i, q in enumerate(STARTERS):
            if cols[i % 2].button(q, key=f"starter{i}", use_container_width=True, disabled=cfg is None):
                SS["ask_pending"] = q
                st.rerun()

    for i, m in enumerate(SS["ask_msgs"]):
        _render_message(m, i)

    typed = st.chat_input("Ask about the study results...", disabled=cfg is None)
    question = typed or SS.pop("ask_pending", None)
    if not question or cfg is None:
        return

    provider, api_key, model = cfg
    with st.chat_message("user"):
        st.markdown(question)
    history = [{"role": m["role"], "content": m["content"]} for m in SS["ask_msgs"]]
    first_q = 1 + sum(1 for m in SS["ask_msgs"] for e in m.get("evidence", []) if e.tool == "run_sql")

    with st.chat_message("assistant"):
        status = st.status("Working on it...", expanded=False)
        result = run_agent(question, history, provider, api_key, model, db, entities,
                           first_query_number=first_q, on_event=lambda s: status.update(label=s))
        if result.error:
            status.update(label="Could not complete", state="error")
            st.error(result.error)
            return
        status.update(label=f"Done - {sum(1 for e in result.evidence if e.tool == 'run_sql')} queries", state="complete")
        st.write_stream(_chunks(result.answer))
        _evidence_panel(result.evidence, key=f"ev{len(SS['ask_msgs'])}")
        _verification_note(result.report)
        st.caption(f"Model: {result.model}")

    SS["ask_msgs"].append({"role": "user", "content": question})
    SS["ask_msgs"].append({"role": "assistant", "content": result.answer, "evidence": result.evidence,
                           "report": result.report, "model": result.model})
