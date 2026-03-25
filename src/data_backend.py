# src/data_backend.py
"""
Unified data backend that prefers DuckDB+Parquet and falls back to CSV.
Provides:
 - connect_duckdb()
 - query_sql(sql, params=None) -> pandas.DataFrame
 - get_table(table_name, filters=None, limit=None) -> pandas.DataFrame
"""
from pathlib import Path
import duckdb
import pandas as pd
import os

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA2 = REPO_ROOT / "data2"
DB_FILE = DATA2 / "agent.db"

# Ensure database views have correct paths for current environment (Windows vs Linux)
try:
    from src.ensure_database import ensure_database_ready
    ensure_database_ready()
except Exception as e:
    print(f"Warning: Could not ensure database ready: {e}")
    print("Will attempt to connect anyway...")

_duck_con = None

def connect_duckdb():
    """Connect to DuckDB database (or create in-memory if DB file doesn't exist)."""
    global _duck_con
    if _duck_con is None:
        if DB_FILE.exists():
            _duck_con = duckdb.connect(str(DB_FILE))
        else:
            # if DB doesn't exist, create an in-memory connection (duckdb can still read parquet files)
            _duck_con = duckdb.connect()
    return _duck_con

def query_sql(sql: str, params: dict = None, fetch: bool = True):
    """
    Executes SQL on DuckDB and returns a pandas DataFrame (if fetch True).

    Args:
        sql: SQL query string
        params: Optional dict of parameters for formatting (use with caution - internal only)
        fetch: Whether to return results as DataFrame

    Returns:
        pandas.DataFrame if fetch=True, None otherwise
    """
    con = connect_duckdb()
    if params:
        # Basic params support via Python formatting (careful about injection; only internal usage expected)
        sql = sql.format(**params)
    if fetch:
        return con.execute(sql).fetchdf()
    else:
        con.execute(sql)
        return None

def get_table(table_name: str, filters: str = None, limit: int = None):
    """
    Try DuckDB view first, fall back to CSV reading.

    Args:
        table_name: Name of the table/view to query
        filters: SQL WHERE clause conditions (without 'WHERE' keyword)
        limit: Maximum number of rows to return

    Example:
        get_table("categorical", "comparison='obese_vs_normal' AND effect_mean > 0.1", limit=100)

    Returns:
        pandas.DataFrame with query results
    """
    con = connect_duckdb()
    try:
        sql = f"SELECT * FROM {table_name}"
        if filters:
            sql += f" WHERE {filters}"
        if limit:
            sql += f" LIMIT {limit}"
        return con.execute(sql).fetchdf()
    except Exception as e:
        # Fallback: try to locate a CSV under data2 matching table_name
        csv_candidates = list(DATA2.rglob(f"*{table_name}*.csv"))
        if not csv_candidates:
            # Try more specific patterns based on common table names
            if table_name == "categorical":
                csv_candidates = list((DATA2 / "categorical").glob("results_all_comparisons.csv"))
            elif table_name == "continuous":
                csv_candidates = list((DATA2 / "continuous").glob("bmi_slope_results.csv"))
            elif table_name == "cell_type_mapping":
                csv_candidates = list((DATA2 / "reference").glob("celltype_mapping_all.csv"))
            elif table_name == "clinical":
                csv_candidates = list((DATA2 / "reference").glob("clinical_data.csv"))
            elif table_name == "zscores":
                csv_candidates = list((DATA2 / "zscores").glob("zscores_long.csv"))
            elif table_name == "interactome":
                csv_candidates = list((DATA2 / "interactome").glob("interactions_by_bmi.csv"))
            elif table_name == "survival":
                csv_candidates = list((DATA2 / "survival").glob("survival_features.csv"))
            elif table_name == "diagnostics":
                csv_candidates = list((DATA2 / "diagnostics").glob("*.csv"))

        if not csv_candidates:
            raise FileNotFoundError(f"No CSV/Parquet found for table {table_name} under data2/. Original error: {e}")

        csv_path = csv_candidates[0]
        df = pd.read_csv(csv_path, low_memory=False)

        if filters:
            # Naive filtering using pandas.query; translate SQL-ish filters to pandas query if needed
            try:
                df = df.query(filters)
            except Exception:
                # if query fails, return df and let caller filter
                pass
        if limit:
            df = df.head(limit)
        return df


# ======================================================================
# INTERACTOME-SPECIFIC QUERY FUNCTIONS
# ======================================================================

def resolve_interactome_cell_types(query_cell_type: str):
    """
    Resolve a query cell type to all matching interactome cell types.

    Handles:
    - Canonical name queries (e.g., "CD8_T_GENERAL" → all CD8 granular types)
    - Functional state queries (e.g., "exhausted" → all exhausted CD8 types)
    - Exact name queries (e.g., "CD8+terminal Tex" → exact match)

    Args:
        query_cell_type: Cell type name (canonical, functional state, or exact)

    Returns:
        List of matching original_name values from interactome_cell_mapping
    """
    con = connect_duckdb()
    query_upper = query_cell_type.upper()

    # Try exact match first
    sql = f"""
        SELECT original_name
        FROM interactome_cell_mapping
        WHERE UPPER(original_name) = '{query_upper}'
    """
    result = con.execute(sql).fetchdf()
    if not result.empty:
        return result['original_name'].tolist()

    # Try canonical parent match
    sql = f"""
        SELECT original_name
        FROM interactome_cell_mapping
        WHERE UPPER(canonical_parent) = '{query_upper}'
    """
    result = con.execute(sql).fetchdf()
    if not result.empty:
        return result['original_name'].tolist()

    # Try functional state match
    sql = f"""
        SELECT original_name
        FROM interactome_cell_mapping
        WHERE UPPER(functional_state) = '{query_upper}'
    """
    result = con.execute(sql).fetchdf()
    if not result.empty:
        return result['original_name'].tolist()

    # Try partial match (e.g., "CD8" matches all CD8+ types)
    sql = f"""
        SELECT original_name
        FROM interactome_cell_mapping
        WHERE UPPER(original_name) LIKE '%{query_upper}%'
           OR UPPER(canonical_parent) LIKE '%{query_upper}%'
    """
    result = con.execute(sql).fetchdf()
    if not result.empty:
        return result['original_name'].tolist()

    # No matches found
    return []


def query_interactome_interactions(
    favorable_cell=None,
    unfavorable_cell=None,
    bmi_category=None,
    source_dataset=None,
    min_enrichment=None,
    max_pvalue=None,
    limit=None
):
    """
    Query interactome cell-cell interactions with flexible cell type resolution.

    Args:
        favorable_cell: Cell type name (canonical, functional, or exact)
        unfavorable_cell: Cell type name (canonical, functional, or exact)
        bmi_category: 'normal' or 'overweight'
        source_dataset: 'Bindeya', 'Newman', or 'Zheng'
        min_enrichment: Minimum enrichment_ratio
        max_pvalue: Maximum p-value
        limit: Maximum number of rows

    Returns:
        pandas.DataFrame with matching interactions

    Example:
        # Get all CD8 T cell interactions in overweight
        query_interactome_interactions(favorable_cell="CD8_T_GENERAL", bmi_category="overweight")

        # Get exhausted T cell interactions
        query_interactome_interactions(favorable_cell="exhausted")

        # Get specific interaction
        query_interactome_interactions(
            favorable_cell="CD8+terminal Tex",
            unfavorable_cell="aDC",
            min_enrichment=5.0
        )
    """
    con = connect_duckdb()
    conditions = []

    # Resolve cell types to all matching granular types
    if favorable_cell:
        favorable_types = resolve_interactome_cell_types(favorable_cell)
        if favorable_types:
            favorable_list = "', '".join(favorable_types)
            conditions.append(f"favorable_cell IN ('{favorable_list}')")
        else:
            # No matches - return empty result
            return pd.DataFrame()

    if unfavorable_cell:
        unfavorable_types = resolve_interactome_cell_types(unfavorable_cell)
        if unfavorable_types:
            unfavorable_list = "', '".join(unfavorable_types)
            conditions.append(f"unfavorable_cell IN ('{unfavorable_list}')")
        else:
            # No matches - return empty result
            return pd.DataFrame()

    if bmi_category:
        conditions.append(f"bmi_category = '{bmi_category}'")

    if source_dataset:
        conditions.append(f"source_dataset = '{source_dataset}'")

    if min_enrichment is not None:
        conditions.append(f"enrichment_ratio >= {min_enrichment}")

    if max_pvalue is not None:
        conditions.append(f"p_value <= {max_pvalue}")

    # Build SQL query
    sql = "SELECT * FROM interactome_interactions"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY enrichment_ratio DESC"
    if limit:
        sql += f" LIMIT {limit}"

    return con.execute(sql).fetchdf()


def query_interactome_gene_pairs(
    favorable_cell=None,
    unfavorable_cell=None,
    gene1=None,
    gene2=None,
    bmi_category=None,
    source_dataset=None,
    min_enrichment=None,
    limit=None
):
    """
    Query interactome gene pairs with flexible cell type resolution.

    Args:
        favorable_cell: Cell type name (resolved to granular types)
        unfavorable_cell: Cell type name (resolved to granular types)
        gene1: Gene symbol for gene1
        gene2: Gene symbol for gene2
        bmi_category: 'normal' or 'overweight'
        source_dataset: 'Bindeya', 'Newman', or 'Zheng'
        min_enrichment: Minimum enrichment_ratio
        limit: Maximum number of rows

    Returns:
        pandas.DataFrame with matching gene pairs

    Example:
        # Get CD40LG gene pairs in T cell - DC interactions
        query_interactome_gene_pairs(
            favorable_cell="CD8 T",
            unfavorable_cell="DC",
            gene1="CD40LG"
        )

        # Get all gene pairs for exhausted T cells
        query_interactome_gene_pairs(favorable_cell="exhausted", limit=100)
    """
    con = connect_duckdb()
    conditions = []

    # Resolve cell types
    if favorable_cell:
        favorable_types = resolve_interactome_cell_types(favorable_cell)
        if favorable_types:
            favorable_list = "', '".join(favorable_types)
            conditions.append(f"favorable_cell IN ('{favorable_list}')")
        else:
            return pd.DataFrame()

    if unfavorable_cell:
        unfavorable_types = resolve_interactome_cell_types(unfavorable_cell)
        if unfavorable_types:
            unfavorable_list = "', '".join(unfavorable_types)
            conditions.append(f"unfavorable_cell IN ('{unfavorable_list}')")
        else:
            return pd.DataFrame()

    if gene1:
        conditions.append(f"gene1 = '{gene1}'")

    if gene2:
        conditions.append(f"gene2 = '{gene2}'")

    if bmi_category:
        conditions.append(f"bmi_category = '{bmi_category}'")

    if source_dataset:
        conditions.append(f"source_dataset = '{source_dataset}'")

    if min_enrichment is not None:
        conditions.append(f"enrichment_ratio >= {min_enrichment}")

    # Build SQL query
    sql = "SELECT * FROM interactome_gene_pairs"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY enrichment_ratio DESC"
    if limit:
        sql += f" LIMIT {limit}"

    return con.execute(sql).fetchdf()
