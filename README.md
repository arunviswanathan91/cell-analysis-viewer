# Obesity-Driven Pancreatic Cancer: Cell-Signature Analysis Viewer
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://obese-pdac-model.streamlit.app/)[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)]([https://huggingface.co/datasets](https://huggingface.co/datasets/arunviswanathan91/cell-analysis-vectors) [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/arunviswanathan91/cell-analysis-rag-api)

*Citation details will be provided upon publication of the associated manuscript. Interim DOI to cite the repo is* [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19386459.svg)](https://doi.org/10.5281/zenodo.19386459)

This repository contains an interactive Streamlit application that accompanies a manuscript investigating obesity-driven remodeling of the tumor microenvironment in pancreatic ductal adenocarcinoma (PDAC). The viewer allows readers to explore cell-type-resolved molecular signatures, BMI-associated effects, and survival-relevant patterns from the published analysis.

The statistical modeling code and upstream pipeline are maintained in a separate repository: [obese-model](https://github.com/arunviswanathan91/obese-model)

---

## Dataset

- **Source:** CPTAC Pancreatic Adenocarcinoma (PAAD) cohort
- **Samples:** 140 tumor samples with clinical annotation
- **Cell types:** Immune and non-immune populations inferred via BayesPrism deconvolution
- **Signatures:** 30+ metabolic and functional gene signatures per cell type

---
| | |
|:---:|:---:|
| <img src="images/Screenshot%202026-03-29%20230013.png" width="350" alt="Interactome chord diagram"><br><sub><strong>1. Interactome — Chord Diagram</strong></sub> | <img src="images/Screenshot%202026-03-29%20230102.png" width="350" alt="Interactome network"><br><sub><strong>2. Interactome — Interaction Network</strong></sub> |
| <img src="images/Screenshot%202026-03-29%20230240.png" width="350" alt="Signature explorer"><br><sub><strong>3. Signature Explorer</strong></sub> | <img src="images/Screenshot%202026-03-29%20232128.png" width="350" alt="Ask the model"><br><sub><strong>4. Ask the Model</strong></sub> |
---

## Analysis Modules

The application provides six analysis interfaces accessible from the sidebar.

**Signature Explorer** — Browse the gene signature database. View signature definitions and composition across cell types.

**Categorical Analysis** — Compare BMI groups (Normal < 25, Overweight 25-30, Obese >= 30) across cell types and signatures. Displays posterior effect sizes, 95% Highest Density Intervals (HDI), heatmaps, and ridge plots from the Bayesian hierarchical model.

**Continuous Analysis** — Treats BMI as a continuous variable. Displays the estimated slope (effect per 1 SD increase in BMI) from the dose-response model.

**Signature Survival Analysis** — Cox proportional hazards regression linking signature expression and BMI to clinical outcomes. Displays hazard ratios and confidence intervals.

**Interactome Analysis** — Cell-cell interaction network visualization based on ligand-receptor pair enrichment. Supports chord diagram rendering and comparison between BMI groups.

**Individual Interaction Explorer** — Gene-level drill-down for specific cell-cell pairs with detailed enrichment statistics.

---

## Ask the Model

In addition to the analysis interfaces, the application includes a conversational interface powered by a remote RAG (retrieval-augmented generation) system. It queries a pre-indexed document store of 73,000+ records derived from all analysis outputs. Queries are routed to the relevant analysis type and answered using a Groq-hosted LLM (Llama 3.3-70B).

---

## Computational Methods

**BayesPrism** — Bayesian cell-type deconvolution framework applied to bulk RNA-seq data to estimate cell-type proportions and cell-type-specific expression profiles per sample.

**STABL** — Stability-driven feature selection using repeated subsampling and bootstrapping to identify robust BMI-associated molecular features.

**Bayesian Hierarchical Modeling** — MCMC-based three-group model estimating cell-type-specific obesity effects on signature scores while accounting for between-sample variability.

**Convergence Diagnostics** — Model validity assessed via R-hat (target < 1.01), Effective Sample Size (> 400), and Hamiltonian Monte Carlo energy diagnostics.

**Cox Proportional Hazards** — Survival analysis linking BMI group and signature expression to patient outcomes.

---

## Local Setup

```bash
pip install -r requirements.txt
streamlit run streamlit_app_with_explorer.py
```

The app runs at `http://localhost:8501`.

A `.devcontainer/devcontainer.json` is included for VS Code dev container usage with Python 3.11.

### Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | Yes (for Ask the Model) | API key from console.groq.com |
| `HF_SPACE_URL` | No | Remote RAG endpoint. Defaults to the hosted HuggingFace Space. |
| `SEMANTIC_PROFILE` | No | Config preset: `default`, `conservative`, `aggressive`, or `dev` |

For Streamlit Cloud deployment, set these in `.streamlit/secrets.toml`.

---

## Project Structure

```text
cell-analysis-viewer/
├── streamlit_app_with_explorer.py   # Main application entry point
├── streamlit_remote_rag.py          # RAG UI components
├── config.py                        # Semantic config and LLM settings
├── requirements.txt
├── src/
│   ├── data_backend.py              # DuckDB and Parquet interface
│   ├── remote_rag.py                # Remote RAG client
│   ├── true_rag.py                  # Local RAG with ChromaDB
│   └── vocabulary.py                # Cell type vocabulary
├── data/                            # Raw analysis outputs
└── data2/                           # Parquet and DuckDB views for the app
    └── agent.db                     # DuckDB database
```

---

## Citation

Citation details will be provided upon publication of the associated manuscript.
