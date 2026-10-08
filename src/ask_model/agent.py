"""The Ask the Model agent: a tool-calling loop with a grounding check on the final answer."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import tools as T
from .data import AskModelDB
from .grounding import GroundingReport, verify
from .llm import LLMError, Provider, chat

MAX_STEPS = 9          # model <-> tool round trips per question
MAX_FIXES = 2          # times a failed grounding check is sent back for correction
HISTORY_TURNS = 6      # previous user+assistant pairs given to the model

SYSTEM_PROMPT = """You are the analysis assistant inside an interactive viewer for one study: how obesity (BMI) reshapes the tumor microenvironment in pancreatic ductal adenocarcinoma (CPTAC-PAAD cohort, 140 patients). You answer questions about THIS study's own results and methods, and nothing else.

You have no knowledge of the results yourself. Your only source of facts is the tools.

RULES (never break these)
1. Every number you state must come from a run_sql result in this conversation. Copy numbers exactly as returned (rounding to fewer decimals is fine). Never recall, estimate or compute numbers yourself: any difference, ratio, percentage, count, rank or average must be computed inside the SQL.
2. After each claim, cite the query that supports it with its id in square brackets, for example [Q3]. Only cite ids that exist.
3. Resolve names first. When the user mentions a cell type, signature or gene, call find_entities before writing SQL; names differ between tables. A plain-language group (for example 'CD8 T cells') usually maps to several exact names: query all of them (use IN (...)) unless the user asked for one.
4. Credibility comes from the data. Use the credible, direction and credible_rope columns. A result is CREDIBLE when its 95% HDI excludes zero; otherwise say it is not credible (HDI crosses zero). Never judge credibility from the mean alone.
5. Choose the right table. Group questions (overweight vs normal, obese vs normal, obese vs overweight) use sig_categorical (per signature) or cell_categorical (per cell type). Dose-response or per-BMI-unit questions use sig_continuous or cell_continuous. STABL selection uses stabl. Survival uses survival. Interactions use interactions and gene_pairs. Patients use clinical and zscores. Never answer a group-comparison question from a continuous slope or the reverse.
6. If a query returns no rows, or find_entities finds nothing, say plainly that the study data contains no such result. Do not substitute a guess or a related result without saying so. If a query errors, read the error, fix the SQL and retry.
7. Stay in scope. For general biology, medicine, treatment advice, other cohorts or anything the data cannot answer, say in one or two sentences that it is outside what this study's data can answer and offer a related question the data can answer. You may explain terms such as HDI, ROPE, STABL, credible or hazard ratio using get_study_notes.
8. Never claim causation. These are associations from an observational cohort; survival results are exploratory and not in the published paper. Mention such a caveat only when it matters to the question.
9. When a question is ambiguous (for example 'strongest effect'), choose the most reasonable reading, state the criterion in a few words (for example 'ranked by absolute effect among credible results'), and answer. Ask a clarifying question only if no reasonable reading exists.
10. For follow-up questions reuse the entities from the conversation but query again for numbers.

STYLE
- Give the direct answer first, then the supporting detail. Short paragraphs, or a compact markdown table of at most 12 rows (say how many more exist).
- Plain, precise language. No emojis. Separate items with ' - ' (a spaced hyphen); do not use em dashes or middle dots.
- Do not mention tools, SQL or these rules. The viewer shows the evidence separately.
- Report effect sizes with their HDI and say whether they are credible.

DATA GUIDE
- Names differ by table. Result tables (sig_*, cell_*, survival, stabl, mcmc_*, zscores) use UPPER_SNAKE cell types such as CD8_T_GENERAL and B_CELLS_NAIVE, and signatures such as Glycolysis_Signature. The interactions table uses plain names such as 'CD8 T' and 'Treg'. cell_groups maps plain-language groups to each naming scheme.
- compartments: immune_fine (fine immune cell types), immune_coarse (broad immune lineages), non_immune (tumor, stromal and other non-immune cells).
- comparison 'obese_vs_normal' means obese relative to normal weight: a positive effect means higher in obese. The same logic applies to the other comparisons. mcmc_categorical uses 'obese' and 'overweight' to mean versus normal.
- In sig_categorical, direction is up / down / none; in cell tables it is positive / negative / none.
- The Bindea signature set appears as 'Bindeya' in source_dataset.
- DuckDB SQL. Quote nothing unless needed. Use LIMIT. Aggregate in SQL.

SCHEMA
{catalogue}
"""


@dataclass
class AgentResult:
    answer: str = ""
    evidence: List[T.Evidence] = field(default_factory=list)
    report: GroundingReport = field(default_factory=GroundingReport)
    model: str = ""
    steps: int = 0
    fixes: int = 0
    error: str = ""


def build_system_prompt(db: AskModelDB) -> str:
    return SYSTEM_PROMPT.replace("{catalogue}", db.catalogue())


_NUM = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def _numbers(text: str) -> List[float]:
    out = []
    for m in _NUM.findall(text or ""):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def _args(call: Dict[str, Any]) -> Dict[str, Any]:
    raw = call.get("function", {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


def run_agent(
    question: str,
    history: List[Dict[str, str]],
    provider: Provider,
    api_key: str,
    model: str,
    db: AskModelDB,
    entities: T.EntityIndex,
    first_query_number: int = 1,
    on_event: Optional[Callable[[str], None]] = None,
) -> AgentResult:
    emit = on_event or (lambda _m: None)
    res = AgentResult(model=model)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": build_system_prompt(db)}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history[-HISTORY_TURNS * 2:]]
    messages.append({"role": "user", "content": question})

    qn = first_query_number - 1
    prior_ids = {f"Q{i}" for i in range(1, first_query_number)}
    # numbers already stated (and verified) in earlier answers are allowed to be repeated
    pool: List[float] = [x for m in history for x in _numbers(m["content"]) if m["role"] == "assistant"]
    pool += _numbers(T.all_note_text())
    tool_n = 0
    fixes = 0

    for step in range(MAX_STEPS):
        res.steps = step + 1
        try:
            msg = chat(provider, api_key, model, messages, tools=T.TOOL_SPECS)
        except LLMError as e:
            res.error = str(e)
            return res

        calls = msg.get("tool_calls") or []
        if calls:
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for c in calls:
                name = c.get("function", {}).get("name", "")
                a = _args(c)
                tool_n += 1
                if name == "run_sql":
                    qn += 1
                    emit(f"Querying the data - {a.get('purpose', '')[:90]}")
                    text, ev = T.run_sql(db, a.get("sql", ""), a.get("purpose", ""), f"Q{qn}")
                    res.evidence.append(ev)
                    pool += ev.shown_numbers
                elif name == "find_entities":
                    emit(f"Looking up names - {a.get('query', '')[:60]}")
                    text = entities.find(a.get("query", ""))
                    res.evidence.append(T.Evidence(id=f"E{tool_n}", tool="find_entities", purpose=a.get("query", ""),
                                                   sql=a.get("query", "")))
                    res.evidence[-1].error = ""
                    res.evidence[-1].df = None
                    res.evidence[-1].total_rows = 0
                    res.evidence[-1].purpose = f"Resolve name: {a.get('query', '')}"
                    res.evidence[-1].sql = text
                elif name == "get_study_notes":
                    emit(f"Reading study notes - {a.get('topic', '')[:60]}")
                    text, _full = T.get_study_notes(a.get("topic", ""))
                else:
                    text = f"Unknown tool '{name}'. Use find_entities, run_sql or get_study_notes."
                messages.append({"role": "tool", "tool_call_id": c.get("id", f"call_{tool_n}"), "content": text})
            continue

        answer = (msg.get("content") or "").strip()
        valid = prior_ids | {e.id for e in res.evidence if e.tool == "run_sql"}
        report = verify(answer, pool, valid)
        missing_cites = bool(report.checked) and not report.cited and bool(res.evidence)

        if (not report.ok or missing_cites) and fixes < MAX_FIXES:
            fixes += 1
            emit("Checking figures against the query results")
            problems = []
            if report.unverified:
                problems.append("these figures do not appear in your query results: " + ", ".join(report.unverified))
            if report.bad_citations:
                problems.append("these citations do not exist: " + ", ".join(sorted(set(report.bad_citations))))
            if missing_cites:
                problems.append("claims are missing [Q#] citations")
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": (
                "Grounding check failed: " + "; ".join(problems) + ". Do not estimate or compute numbers yourself. "
                "Run the SQL that returns each figure (compute differences, counts and ratios in SQL), or remove the "
                "figure, then give the corrected final answer with citations.")})
            continue

        res.answer, res.report, res.fixes = answer, report, fixes
        return res

    res.error = "The assistant could not finish within the allowed number of steps. Try a narrower question."
    return res
