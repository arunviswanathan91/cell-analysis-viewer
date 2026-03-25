# data2: AI-Optimized Analysis Results

This directory contains semantically normalized analysis results in long-format CSV files, with an optional Parquet/DuckDB execution layer for fast querying.

## Structure

```
data2/
├── reference/           # Cell type mappings and clinical data
├── categorical/         # BMI category comparisons (Normal/Overweight/Obese)
├── continuous/          # BMI slope analysis
├── posterior/           # MCMC posterior samples (8,000 per analysis)
├── diagnostics/         # Model convergence diagnostics
├── zscores/            # Z-score data
├── interactome/        # Cell-cell interaction networks
├── survival/           # Survival analysis results
└── agent.db            # DuckDB database (execution layer)
```

## Data Formats

- **CSV files**: Canonical source of truth (human-readable, git-friendly)
- **Parquet files**: Binary mirrors of CSVs (fast columnar storage)
- **agent.db**: DuckDB database with SQL views over Parquet files

## Execution Layer

The execution layer provides fast SQL querying via DuckDB and Parquet:

### Build Commands

```bash
# 1. Create Parquet mirrors from CSV files
python scripts/parquet_mirror.py

# 2. Create DuckDB database with views
python scripts/build_duckdb.py

# 3. Validate the backend
python scripts/validate_backend.py
```

### Python API

```python
from src.data_backend import get_table, query_sql

# Load a table
df = get_table("categorical")

# Filtered query
df = get_table("categorical", filters="comparison='obese_vs_normal' AND effect_mean > 0.1", limit=100)

# Raw SQL
sql = """
SELECT c.cell_type, c.effect_mean, i.favorable_cell
FROM categorical c
JOIN interactome i ON c.cell_type = i.favorable_cell
WHERE c.comparison = 'obese_vs_normal'
LIMIT 50
"""
df = query_sql(sql)
```

## Available Views

- `categorical` - BMI category comparisons (4,338 rows)
- `continuous` - BMI slope results (723 rows)
- `zscores` - Z-score data (63,500 rows)
- `interactome` - Cell-cell interactions (924 rows)
- `survival` - Survival analysis (120 rows)
- `diagnostics` - MCMC diagnostics (107 rows)
- `cell_type_mapping` - Cell type mappings (42 rows)
- `clinical` - Clinical metadata (140 rows)
- `posterior_immune_fine` - Posterior samples (immune_fine)
- `posterior_immune_coarse` - Posterior samples (immune_coarse)
- `posterior_non_immune` - Posterior samples (non_immune)

## Key Features

1. **Performance**: Parquet + DuckDB = 10-100x faster than CSV
2. **SQL Queries**: Write complex multi-table joins
3. **Backward Compatible**: Falls back to CSV if Parquet/DuckDB unavailable
4. **Git-Friendly**: CSVs remain canonical; Parquet/DB are derivatives
5. **Type Safety**: DuckDB enforces types, reducing runtime errors

## Cell Type Naming

All cell types use `UPPERCASE_WITH_UNDERSCORES` format:
- `CD4_T_REGULATORY`
- `B_CELLS_NAIVE`
- `M1_MACROPHAGE`
- etc.

## Comparisons

- `overweight_vs_normal`
- `obese_vs_normal`
- `obese_vs_overweight`

## DO NOT

- Alter CSV files (they are canonical)
- Change column names or cell type canonicalization
- Commit large Parquet files to git (optional: add to .gitignore)
- Remove CSV fallback from agent code
