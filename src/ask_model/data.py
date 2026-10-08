"""Read-only analytical database behind the Ask the Model assistant.

Every table the viewer shows is loaded into an in-memory DuckDB database, then the
connection is locked (no file access, no configuration changes). The assistant can
only run SELECT statements against these tables, so every number it reports can be
traced back to a query result.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DATA2 = ROOT / "data2"

COMPARTMENTS = ["immune_fine", "immune_coarse", "non_immune"]
COMPARISONS = ["overweight_vs_normal", "obese_vs_normal", "obese_vs_overweight"]
HDI_LEVEL = 0.95


@dataclass
class TableInfo:
    name: str
    description: str
    rows: int = 0
    columns: Dict[str, str] = field(default_factory=dict)       # column -> duckdb type
    values: Dict[str, List[str]] = field(default_factory=dict)  # low-cardinality text columns


# ----------------------------------------------------------------------------------
#  helpers
# ----------------------------------------------------------------------------------
def _hdi(samples: np.ndarray, level: float = HDI_LEVEL):
    """Shortest interval containing `level` of the posterior samples."""
    x = np.sort(samples)
    n = len(x)
    k = int(np.floor(level * n))
    widths = x[k:] - x[: n - k]
    i = int(widths.argmin())
    return float(x[i]), float(x[i + k])


def _summarise_posterior(df: pd.DataFrame, id_cols: List[str], compartment: str) -> pd.DataFrame:
    """One row per (cell type, group) with mean, sd, HDI and probability of direction."""
    rows = []
    cell_cols = [c for c in df.columns if c not in id_cols + ["sample"]]
    groups = df.groupby("comparison") if "comparison" in df.columns else [(None, df)]
    for comp, g in groups:
        for c in cell_cols:
            a = g[c].to_numpy(dtype=float)
            lo, hi = _hdi(a)
            rows.append({
                "compartment": compartment,
                "cell_type": c,
                "comparison": comp,
                "effect_mean": round(float(a.mean()), 4),
                "effect_sd": round(float(a.std(ddof=1)), 4),
                "hdi_lower": round(lo, 4),
                "hdi_upper": round(hi, 4),
                "prob_positive": round(float((a > 0).mean()), 4),
                "credible": bool(lo > 0 or hi < 0),
            })
    out = pd.DataFrame(rows)
    out["direction"] = np.where(~out["credible"], "none", np.where(out["effect_mean"] > 0, "positive", "negative"))
    return out


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


# ----------------------------------------------------------------------------------
#  table builders
# ----------------------------------------------------------------------------------
def _build_tables() -> Dict[str, tuple]:
    """Return {table_name: (DataFrame, description)}."""
    t: Dict[str, tuple] = {}

    cat = _read(DATA2 / "categorical" / "results_all_comparisons.parquet")
    cat = cat.rename(columns={c: c.replace(".", "_") for c in cat.columns})
    t["sig_categorical"] = (
        cat.drop(columns=["feature"]),
        "Signature-level BMI GROUP comparisons (categorical Bayesian model). One row per compartment x cell_type x "
        "signature x comparison. effect_mean is the posterior mean effect (standardised z-score units); "
        "[hdi_lower, hdi_upper] is the 95% HDI; credible = HDI excludes zero; credible_rope = also clears the ROPE "
        "(practical-significance) check; prob_gt_X = posterior probability that |effect| exceeds X.",
    )

    cont = _read(DATA2 / "continuous" / "bmi_slope_results.parquet")
    cont = cont.rename(columns={c: c.replace(".", "_") for c in cont.columns}).drop(columns=["feature"])
    t["sig_continuous"] = (
        cont,
        "Signature-level CONTINUOUS BMI dose-response (slope of signature z-score per BMI unit, standardised). "
        "bmi_slope_mean/hdi_* are per 1 SD of BMI; bmi_slope_per_unit_* are per 1 BMI unit. credible = HDI excludes zero. "
        "tier1_large_credible / tier2_medium_credible / tier3_any_credible are the star tiers shown in the viewer "
        "(tier1 = two stars, tier2 = one star, tier3 = circle).",
    )

    # cell-type level effects, summarised from the posterior samples
    cell_cat, cell_cont = [], []
    for comp in COMPARTMENTS:
        pc = DATA2 / "posterior" / f"categorical_{comp}.parquet"
        if pc.exists():
            cell_cat.append(_summarise_posterior(_read(pc), ["comparison"], comp))
        pk = DATA2 / "posterior" / f"continuous_{comp}.parquet"
        if pk.exists():
            s = _summarise_posterior(_read(pk), [], comp).drop(columns=["comparison"])
            s = s.rename(columns={"effect_mean": "slope_mean", "effect_sd": "slope_sd"})
            cell_cont.append(s)
    if cell_cat:
        t["cell_categorical"] = (
            pd.concat(cell_cat, ignore_index=True),
            "Cell-TYPE level (pooled across signatures) BMI group effects, summarised from the MCMC posterior "
            "(8,000 draws) with a 95% shortest HDI. One row per compartment x cell_type x comparison.",
        )
    if cell_cont:
        t["cell_continuous"] = (
            pd.concat(cell_cont, ignore_index=True),
            "Cell-TYPE level (pooled) continuous BMI slope per 1 SD of BMI, summarised from the MCMC posterior with a "
            "95% shortest HDI. One row per compartment x cell_type.",
        )

    surv = _read(DATA2 / "survival" / "survival_features.parquet")
    t["survival"] = (
        surv.drop(columns=[c for c in ["plot", "plot_stats_txt", "cred_reason", "matched_col", "feature"] if c in surv.columns]),
        "Cox proportional-hazards survival results per signature feature, for patients in a BMI comparison. "
        "hr = hazard ratio of the signature z-score (hr > 1 = worse survival with higher score), [hr_ci_low, hr_ci_high] "
        "its 95% CI, hr_p its p-value, logrank_p the log-rank p-value (high vs low), n patients, events deaths, c_index. "
        "These are exploratory and not part of the published paper.",
    )

    stabl = []
    for comp in COMPARTMENTS:
        p = DATA / "stabl" / f"{comp}_selected.csv"
        if p.exists():
            d = pd.read_csv(p)
            parts = d["feature"].str.split(r"\|\|", n=1, expand=True)
            stabl.append(pd.DataFrame({"compartment": comp, "cell_type": parts[0], "signature": parts[1]}))
    if stabl:
        t["stabl"] = (
            pd.concat(stabl, ignore_index=True),
            "Features selected by STABL (stability selection) as robustly BMI-associated. One row per selected "
            "compartment x cell_type x signature. A feature absent from this table was NOT selected.",
        )

    inter = _read(DATA2 / "interactome" / "interactions_all_sources.parquet")
    inter = inter.drop(columns=[c for c in ["index", "total_imgp_list", "Unnamed: 0"] if c in inter.columns])
    t["interactions"] = (
        inter,
        "Cell-cell interaction enrichment (interactome) per source_dataset (Bindeya = the Bindea signature set, Newman, "
        "Zheng) and bmi_category (normal weight vs overweight). favorable_cell / unfavorable_cell are the interacting pair; "
        "enrichment_ratio, p_value, adj_p_value, fdr are the enrichment statistics; shared_imgp = number of shared "
        "interacting gene pairs out of total_imgp; shared_imgp_list is the '/'-separated gene pairs (GENE1_GENE2).",
    )

    gp = _read(DATA2 / "interactome" / "gene_pairs_exploded.parquet")
    t["gene_pairs"] = (
        gp,
        "The shared interacting gene pairs of each interaction in `interactions`, one row per pair (gene1 belongs to the "
        "favorable cell, gene2 to the unfavorable cell), with the enrichment statistics of the parent interaction.",
    )

    for kind in ("categorical", "continuous"):
        d = _read(DATA2 / "diagnostics" / f"mcmc_{kind}.parquet")
        d = d.rename(columns={"hdi_3%": "hdi_3pct", "hdi_97%": "hdi_97pct"})
        t[f"mcmc_{kind}"] = (
            d,
            f"MCMC convergence diagnostics for the {kind} cell-level model. r_hat should be < 1.01 and ess_bulk > 400; "
            "convergence_ok is the study's pass flag. hdi_3pct / hdi_97pct is a 94% HDI (ArviZ default).",
        )

    clin = _read(DATA2 / "reference" / "clinical_data.parquet")
    clin["bmi_group"] = pd.cut(clin["BMI"], [0, 25, 30, 1000], right=False, labels=["Normal", "Overweight", "Obese"]).astype(str)
    clin.columns = [c.lower() for c in clin.columns]
    keep = [c for c in clin.columns if c not in ("studyid", "oncotree_code", "cancer_type", "sample_count")]
    t["clinical"] = (
        clin[keep],
        "CPTAC-PAAD clinical metadata, one row per patient (140). bmi_group: Normal < 25, Overweight 25-30, Obese >= 30.",
    )

    z = _read(DATA2 / "zscores" / "zscores_long.parquet")
    t["zscores"] = (z, "Per-patient signature z-scores: one row per compartment x cell_type x signature x sample_id.")

    # signatures -> genes (with the canonical cell type name where one exists)
    sig_path = DATA / "signatures" / "ALL_CELL_SIGNATURES_FLAT.json"
    if sig_path.exists():
        raw = json.load(open(sig_path, encoding="utf8"))
        mapping = pd.read_csv(DATA2 / "reference" / "signature_to_celltype_mapping.csv")
        canon = dict(zip(mapping["signature_celltype_key"], mapping["canonical_celltype"]))
        rows = []
        for cell_key, sigs in raw.items():
            for sig, genes in (sigs.items() if isinstance(sigs, dict) else []):
                for g in (genes if isinstance(genes, list) else []):
                    rows.append((cell_key, canon.get(cell_key, cell_key), sig, str(g)))
        t["signature_genes"] = (
            pd.DataFrame(rows, columns=["signature_cell_key", "cell_type", "signature", "gene"]),
            "Gene membership of every signature, one row per cell type x signature x gene. cell_type is the canonical "
            "name used by the result tables where a mapping exists (signature_cell_key is the raw key).",
        )

    ug = _read(DATA2 / "reference" / "unified_cell_type_mapping.parquet")
    rows = []
    for _, r in ug.iterrows():
        for col, ns in (("categorical_names", "categorical"), ("interactome_granular", "interactome"), ("signature_names", "signature")):
            for name in str(r[col] or "").split(","):
                if name.strip():
                    rows.append((r["query_term"], r["description"], ns, name.strip()))
    t["cell_groups"] = (
        pd.DataFrame(rows, columns=["group_name", "description", "naming_scheme", "member_name"]),
        "Lookup that maps a plain-language cell group (e.g. 'CD8 T cells', 'Treg', 'macrophages') to the exact names used "
        "in each naming scheme: categorical (result tables), interactome (interactions.favorable_cell/unfavorable_cell) "
        "and signature (raw signature keys).",
    )
    return t


# ----------------------------------------------------------------------------------
#  database
# ----------------------------------------------------------------------------------
class AskModelDB:
    """In-memory, locked DuckDB plus the catalogue used to brief the language model."""

    def __init__(self) -> None:
        self.con = duckdb.connect(":memory:")
        self.tables: Dict[str, TableInfo] = {}
        for name, (df, desc) in _build_tables().items():
            self.con.register("_tmp_df", df)
            self.con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM _tmp_df')
            self.con.unregister("_tmp_df")
            info = TableInfo(name=name, description=desc, rows=len(df))
            for col, typ in self.con.execute(f'SELECT column_name, data_type FROM information_schema.columns '
                                             f"WHERE table_name = '{name}' ORDER BY ordinal_position").fetchall():
                info.columns[col] = typ
                if typ == "VARCHAR":
                    n = self.con.execute(f'SELECT COUNT(DISTINCT "{col}") FROM "{name}"').fetchone()[0]
                    if 0 < n <= 12:
                        info.values[col] = [r[0] for r in self.con.execute(
                            f'SELECT DISTINCT "{col}" FROM "{name}" WHERE "{col}" IS NOT NULL ORDER BY 1').fetchall()]
            self.tables[name] = info
        # lock down: user SQL can never read files or change settings
        self.con.execute("SET enable_external_access=false")
        self.con.execute("SET lock_configuration=true")

    def catalogue(self) -> str:
        """Compact schema text for the system prompt."""
        out = []
        for t in self.tables.values():
            cols = ", ".join(f"{c} {ty.split('(')[0].lower()}" for c, ty in t.columns.items())
            vals = "; ".join(f"{c} in {{{', '.join(v)}}}" for c, v in t.values.items())
            out.append(f"TABLE {t.name} ({t.rows} rows)\n  {t.description}\n  columns: {cols}\n  values: {vals or 'n/a'}")
        return "\n\n".join(out)
