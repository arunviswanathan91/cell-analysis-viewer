"""Tools the language model may call. Every tool returns data the model can quote and
the UI can show as evidence; nothing here calls an LLM."""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd
from rapidfuzz import fuzz, process

from .data import AskModelDB, DATA
from .notes import STUDY_NOTES

MAX_ROWS_TO_MODEL = 40        # rows the model gets to see per query
MAX_ROWS_KEPT = 2000          # rows kept for the evidence panel
MAX_CELL_CHARS = 160          # long strings (gene-pair lists) are clipped for the model
QUERY_TIMEOUT_S = 15


@dataclass
class Evidence:
    """One tool call, kept for the evidence panel and for number verification."""
    id: str
    tool: str
    purpose: str = ""
    sql: str = ""
    df: Optional[pd.DataFrame] = None
    total_rows: int = 0
    error: str = ""
    shown_numbers: List[float] = field(default_factory=list)   # numbers the model actually saw


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        return float(f"{v:.6g}")
    if isinstance(v, str) and len(v) > MAX_CELL_CHARS:
        return v[:MAX_CELL_CHARS] + f"... [{len(v)} chars]"
    return v


def _numbers_in(df: pd.DataFrame) -> List[float]:
    out: List[float] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            out.extend(float(x) for x in df[col].dropna().tolist())
        elif df[col].dtype == object:
            # numbers embedded in text cells (e.g. 'HR=2.37')
            for s in df[col].dropna().astype(str).head(MAX_ROWS_TO_MODEL):
                out.extend(float(m) for m in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s))
    return out


# ----------------------------------------------------------------------------------
#  run_sql
# ----------------------------------------------------------------------------------
def run_sql(db: AskModelDB, sql: str, purpose: str, ev_id: str) -> tuple[str, Evidence]:
    ev = Evidence(id=ev_id, tool="run_sql", purpose=purpose, sql=sql.strip().rstrip(";"))
    cur = db.con.cursor()
    try:
        stmts = cur.extract_statements(ev.sql)
        if len(stmts) != 1 or stmts[0].type != duckdb.StatementType.SELECT:
            raise ValueError("Only a single SELECT statement is allowed.")
        timer = threading.Timer(QUERY_TIMEOUT_S, cur.interrupt)
        timer.start()
        try:
            df = cur.execute(ev.sql).fetchdf()
        finally:
            timer.cancel()
    except Exception as e:  # noqa: BLE001 - any failure is reported back to the model
        ev.error = str(e).splitlines()[0][:300]
        return f"[{ev_id}] ERROR: {ev.error}\nFix the query (check column names against the schema) and try again.", ev
    finally:
        cur.close()

    ev.total_rows = len(df)
    ev.df = df.head(MAX_ROWS_KEPT)
    shown = df.head(MAX_ROWS_TO_MODEL)
    ev.shown_numbers = _numbers_in(shown) + [float(len(df)), float(len(shown))]

    if df.empty:
        return f"[{ev_id}] 0 rows returned.", ev
    rows = [[_fmt(v) for v in r] for r in shown.itertuples(index=False, name=None)]
    note = "" if len(df) <= MAX_ROWS_TO_MODEL else (
        f"\nShowing the first {MAX_ROWS_TO_MODEL} of {len(df)} rows. Use ORDER BY / LIMIT / aggregates "
        "to get exactly the rows you need.")
    body = "\n".join(" | ".join("" if v is None else str(v) for v in r) for r in rows)
    return f"[{ev_id}] {len(df)} rows. columns: {' | '.join(df.columns)}\n{body}{note}", ev


# ----------------------------------------------------------------------------------
#  find_entities
# ----------------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_|+/\-]", " ", str(s))).strip().lower()


_GENERIC = {"cell", "cells"}

# plain-language names -> the names this dataset uses
SYNONYMS = {
    "natural killer": ["NK cells"],
    "macrophage": ["TAM", "M1", "M2"],
    "dendritic": ["DC", "dendritic cells"],
    "regulatory t": ["Treg", "CD4 T regulatory"],
    "cancer cell": ["tumor"],
    "tumour": ["tumor"],
    "stroma": ["fibroblasts", "pericytes"],
    "cytotoxic": ["CD8 T", "NK cells"],
    "myeloid": ["monocytes", "macrophages", "dendritic cells"],
}


def _core(s: str) -> str:
    """Normalised text without the uninformative word 'cell(s)'."""
    return " ".join(w for w in _norm(s).split() if w not in _GENERIC)


class EntityIndex:
    """Fuzzy lookup of cell types, signatures and genes across all tables."""

    def __init__(self, db: AskModelDB):
        con = db.con
        q = lambda sql: [r[0] for r in con.execute(sql).fetchall()]  # noqa: E731
        self.vocab: Dict[str, List[str]] = {
            "cell_type (result tables)": sorted(set(
                q("SELECT DISTINCT cell_type FROM sig_categorical") + q("SELECT DISTINCT cell_type FROM cell_categorical"))),
            "cell name (interactions table)": sorted(set(
                q("SELECT DISTINCT favorable_cell FROM interactions") + q("SELECT DISTINCT unfavorable_cell FROM interactions"))),
            "signature": q("SELECT DISTINCT signature FROM sig_categorical ORDER BY 1"),
            "gene": sorted(set(
                q("SELECT DISTINCT gene FROM signature_genes") + q("SELECT DISTINCT gene1 FROM gene_pairs") + q("SELECT DISTINCT gene2 FROM gene_pairs"))),
            "cell group (cell_groups.group_name)": q("SELECT DISTINCT group_name FROM cell_groups ORDER BY 1"),
        }
        self.norm = {k: [(v, _norm(v), _core(v)) for v in vs] for k, vs in self.vocab.items()}
        self.db = db

    def find(self, text: str, top: int = 6) -> str:
        """Resolve `text`, also trying common synonyms (e.g. 'natural killer' -> NK)."""
        outs = [self._find(text, top)]
        low = _norm(text)
        for phrase, alts in SYNONYMS.items():
            if phrase in low:
                outs += [self._find(a, top) for a in alts]
        seen, lines = set(), []
        for o in outs:
            for ln in o.splitlines():
                if ln not in seen and not ln.startswith("No cell type"):
                    seen.add(ln)
                    lines.append(ln)
        if lines:
            return chr(10).join(lines)
        return outs[0]

    def _find(self, text: str, top: int = 6) -> str:
        t = _norm(text)
        if not t:
            return "Empty query."
        lines: List[str] = []
        for kind, items in self.norm.items():
            cores = [c for _, _, c in items]
            tc = _core(text)
            hits = []
            if kind == "gene":
                if text.strip().upper() in self.vocab["gene"]:
                    hits.append((text.strip().upper(), 100))
            else:
                for (orig, n, c) in items:
                    if t == n:
                        hits.append((orig, 100))
                    elif len(tc) >= 3 and (tc in c or (c in tc and len(c) >= 4)):
                        hits.append((orig, 92))
                if len(hits) < top and len(tc) >= 3:
                    for _, score, idx in process.extract(tc, cores, scorer=fuzz.token_set_ratio, limit=top):
                        if score >= 88 and len(cores[idx]) >= 3 and items[idx][0] not in [h[0] for h in hits]:
                            hits.append((items[idx][0], int(score)))
            hits = sorted(hits, key=lambda h: -h[1])[:top]
            if hits:
                lines.append(f"{kind}: " + ", ".join(f"{h} ({s})" for h, s in hits))
        # members of any matching cell group, per naming scheme
        grp = self.db.con.execute(
            "SELECT group_name, naming_scheme, list(member_name) FROM cell_groups GROUP BY 1,2 ORDER BY 1,2").fetchall()
        for g, scheme, members in grp:
            if _norm(g) == t or (len(_core(text)) >= 3 and fuzz.token_set_ratio(_core(text), _core(g)) >= 95):
                lines.append(f"group '{g}' -> {scheme} names: {', '.join(members[:40])}{' ...' if len(members) > 40 else ''}")
        if not lines:
            return (f"No cell type, signature, gene or cell group in this dataset matches '{text}'. "
                    "Report that it is not in the data rather than guessing.")
        return "\n".join(lines)


# ----------------------------------------------------------------------------------
#  get_study_notes
# ----------------------------------------------------------------------------------
def _load_notes() -> Dict[str, str]:
    notes = dict(STUDY_NOTES)
    p = DATA / "README_MODEL_CONTEXT.md"
    try:
        text = p.read_text(encoding="utf8")
        for block in re.split(r"\n(?=##+ )", text):
            head = block.splitlines()[0].lstrip("# ").strip()
            if head and len(block) > 80:
                notes[f"Limitations - {head}"] = block.strip()
    except OSError:
        pass
    return notes


_NOTES = _load_notes()


_STOP = {"what", "does", "mean", "means", "this", "that", "with", "about", "which", "how", "the", "and", "for", "are",
         "was", "were", "explain", "tell", "study", "model", "data", "analysis"}


def _stems(text: str) -> set:
    return {w[:6] for w in _norm(text).split() if len(w) > 2 and w not in _STOP}


def get_study_notes(topic: str) -> tuple[str, Optional[str]]:
    words = _stems(topic)
    scored = []
    for title, body in _NOTES.items():
        score = 3 * len(words & _stems(title)) + len(words & _stems(body[:2500]))
        if score and not title.startswith("Limitations - "):
            score += 2
        if score:
            scored.append((score, title, body))
    scored.sort(reverse=True)
    if not scored:
        return "No study note matches that topic.", None
    out = "\n\n".join(f"## {t}\n{b}" for _, t, b in scored[:2])
    return out[:5000], out


def all_note_text() -> str:
    return "\n".join(_NOTES.values())


# ----------------------------------------------------------------------------------
#  tool schemas (OpenAI function-calling format)
# ----------------------------------------------------------------------------------
TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "find_entities",
        "description": ("Resolve a plain-language name to the exact values used in the data: cell types, signatures, genes "
                        "or cell groups (e.g. 'CD8 T cells', 'macrophage', 'glycolysis', 'GZMB'). Call this BEFORE writing "
                        "SQL whenever the user names a cell type, signature or gene, because naming differs between tables. "
                        "Pass one short name per call."),
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "run_sql",
        "description": ("Run ONE read-only SELECT (DuckDB SQL) against the study tables and get the rows back. This is the "
                        "only way to obtain numbers. Do all filtering, ranking, counting and arithmetic in SQL."),
        "parameters": {"type": "object", "properties": {
            "sql": {"type": "string", "description": "A single SELECT statement."},
            "purpose": {"type": "string", "description": "One short sentence: what this query is meant to answer."}},
            "required": ["sql", "purpose"]}}},
    {"type": "function", "function": {
        "name": "get_study_notes",
        "description": ("Look up how the study was done: cohort, deconvolution, signatures, STABL, Bayesian model, credibility "
                        "and ROPE definitions, survival analysis, limitations. Use for method or interpretation questions."),
        "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}}},
]
