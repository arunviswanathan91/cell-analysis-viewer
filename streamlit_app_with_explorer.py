## -*- coding: utf-8 -*-
"""
Comprehensive Cell Analysis Viewer - Interactive Plots
=======================================================
Real-time interactive visualizations using Plotly and ArviZ.
All plots support zoom, pan, hover tooltips, and interactive legends.

FIXES APPLIED (v2.0):
- Fixed celltype name display: Now reads column 1 (celltype_name) instead of column 0 (celltype_idx)
- Improved error handling for missing z-score files
- Added debug messages for troubleshooting cell type loading issues
"""

import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')
import streamlit.components.v1 as components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.colors import to_rgba as mpl_to_rgba
try:
    import holoviews as hv
    from holoviews import opts
    from bokeh.embed import file_html
    from bokeh.resources import CDN
    hv.extension('bokeh')
    HV_AVAILABLE = True
except ImportError:
    HV_AVAILABLE = False
# Fix emoji encoding
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Try optional imports
try:
    from lifelines import CoxPHFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

# ── Remote RAG (HuggingFace Space) ──────────────────────────
# Pre-indexed 73,108 documents. No local ChromaDB. No startup indexing.
# Space URL: https://arunviswanathan91-cell-analysis-rag-api.hf.space
try:
    from streamlit_remote_rag import (
        load_remote_rag,
        render_rag_sidebar_status,
        render_rag_chat,
        render_interactome_rag,
        render_survival_rag,
        render_bayesian_rag,
        SYSTEM_PROMPTS,
    )
    REMOTE_RAG_AVAILABLE = True
except Exception as e:
    print(f"Warning: RemoteRAG not available: {e}")
    REMOTE_RAG_AVAILABLE = False

# Keep TRUE_RAG_AVAILABLE as False so old code paths don't activate
TRUE_RAG_AVAILABLE = False

# RAG/LLM Configuration
RAG_MODELS = {
    "qwen/qwen3-32b": "Qwen 3 32B (Default)",
    "groq/mini": "Groq Mini (Fast)"
}
DEFAULT_RAG_MODEL = "qwen/qwen3-32b"

# Analysis domain definitions for scoping
ANALYSIS_DOMAINS = {
    "Interactome": "Cell-cell interaction network analysis comparing normal vs overweight conditions",
    "Categorical": "Categorical BMI group comparisons (Normal vs Overweight vs Obese)",
    "Continuous": "Continuous BMI modeling (dose-response relationships)",
    "Survival": "Survival analysis stratified by BMI and signature expression"
}

# Author-provided limitations (from user request)
AUTHOR_LIMITATIONS = """
Limitations and future directions
Several limitations warrant consideration. The obese patient cohort (n=18) is the smallest group compared to normal-weight (n=51) and overweight (n=58) patients. While Bayesian modeling provides robust uncertainty quantification, the modest obese sample size necessitates validation in wider cohorts to confirm generalizability. This imbalance is particularly relevant for interaction analyses where network inference requires sufficient replicates per group which limited us from including obese group in the analysis. While all our methodology is appropriate for the smaller sample sizes, validation in larger independent cohorts with balanced group sizes is essential.
The cross-sectional design precludes assessment of longitudinal BMI trajectories and cannot establish causality. We observe associations between BMI and microenvironmental states but cannot determine whether weight gain drives these changes, whether pre-existing immune or metabolic states influence both BMI and tumor immunity, or whether reverse causation contributes to observed patterns. Prospective studies tracking patients through weight change interventions would be necessary to establish directionality.
Deconvolution relies on reference single-cell datasets that may not fully capture PDAC-specific cellular states or obesity-specific transcriptional programs. These references derive from mixed BMI cohorts and may not optimally represent extreme weight categories or obesity-induced cell states. Our signature database, while comprehensive (2,143 signatures across 65 cell types), may not include all obesity-relevant programs, which could explain the absence of certain bulk-level findings (such as detailed primary immunodeficiency pathway components) in the deconvolved analysis.
The apparent discrepancies between categorical and continuous modeling highlight an interpretational challenge that extends beyond statistical methodology to biological interpretation. Categorical models excel at identifying discrete state transitions and threshold effects but may lack power for gradual dose-response relationships, particularly when one group is smaller. Continuous models effectively capture linear trends across the BMI spectrum but may miss non-monotonic relationships, threshold effects, or categorical state switches. Both approaches provide complementary information, yet reconciling their conclusions requires careful consideration of underlying biological models and whether cellular responses follow linear, threshold, or biphasic patterns. The observation that some cell types (tumor classical, iCAF) show credible effects in continuous but not categorical models could reflect true gradual reprogramming, insufficient categorical power, or non-linear relationships that continuous linear models approximate but do not fully capture.
Our analysis focuses on transcriptional signatures and cannot directly assess protein-level changes, post-translational modifications, metabolite concentrations, or functional phenotypes that may diverge from mRNA expression patterns. Obesity-associated metabolic alterations operate through multiple post-transcriptional mechanisms including protein phosphorylation, lipid modifications, and metabolite-enzyme interactions that would not be captured in RNA-based analyses. Additionally, BMI serves as an imperfect proxy for adiposity, not accounting for body composition nuances such as visceral versus subcutaneous fat distribution, sarcopenic obesity, or adipose tissue dysfunction. More refined measures of obesity like metabolically healthy obesity, metabolically unhealthy obesity might be optimal for this research. How ever such elaborate dataset in pancreatic cancer is currently not available.
Future studies should address these limitations through: prospective longitudinal cohorts that track microenvironmental changes during weight interventions or disease progression; larger cohorts with balanced distribution enabling more robust subgroup analyses and interaction modeling; mechanistic studies to validate cell-cell interaction predictions and assess spatial organization of immune-stromal-tumor interfaces; functional validation of key signatures (especially gemcitabine resistance, NK cell exhaustion, CD4+ T cell collapse) in preclinical models with controlled dietary interventions; and integration of proteomics and metabolomics to connect transcriptional programs to functional protein activity and the metabolite landscape directly regulating cellular behavior.
"""

def load_model_context():
    """Load the model limitations and context file for RAG."""
    context_path = os.path.join("data", "README_MODEL_CONTEXT.md")
    try:
        if os.path.exists(context_path):
            with open(context_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                # Combine with author-provided limitations
                return file_content + "\n\n## Additional Author-Provided Limitations\n" + AUTHOR_LIMITATIONS
    except Exception as e:
        pass
    return """
    ## Default Limitations
    - This analysis is observational and cannot establish causality.
    - Effect sizes are relative and standardized.
    - Results require R-hat < 1.01 and ESS > 400 for reliability.
    - Do not make clinical recommendations based on these results.
    """ + "\n\n## Additional Author-Provided Limitations\n" + AUTHOR_LIMITATIONS

def get_current_analysis_context(mode, compartment=None, cell_type=None, signature=None):
    """Build context string from current sidebar selection."""
    context = f"Analysis Mode: {mode}\n"
    if compartment:
        context += f"Compartment: {compartment}\n"
    if cell_type:
        context += f"Cell Type: {cell_type}\n"
    if signature:
        context += f"Signature: {signature}\n"
    return context

def infer_analyses_from_question(question):
    """
    Infer which analyses are relevant based on the user's natural language question.
    Returns a list of analysis domains that should be consulted.
    """
    question_lower = question.lower()
    inferred_analyses = []

    # Keywords for each analysis type
    interactome_keywords = [
        "interaction", "interactome", "cell-cell", "network", "communication",
        "signaling", "crosstalk", "ligand", "receptor", "paracrine"
    ]
    categorical_keywords = [
        "categorical", "group", "normal", "overweight", "obese", "bmi category",
        "compare", "comparison", "difference", "between groups", "vs", "versus"
    ]
    continuous_keywords = [
        "continuous", "dose-response", "slope", "trend", "linear", "per unit",
        "gradient", "incremental", "regression", "association strength"
    ]
    survival_keywords = [
        "survival", "prognosis", "outcome", "mortality", "death", "hazard",
        "kaplan", "meier", "cox", "time to event", "overall survival"
    ]

    # Cell type specific keywords that might indicate interest in a cell type
    cell_type_keywords = [
        "cd8", "cd4", "t cell", "b cell", "nk", "macrophage", "monocyte",
        "dendritic", "fibroblast", "caf", "icaf", "mcaf", "tregs", "treg",
        "neutrophil", "myeloid", "tumor", "epithelial", "endothelial"
    ]

    # Check each analysis type
    if any(kw in question_lower for kw in interactome_keywords):
        inferred_analyses.append("Interactome")
    if any(kw in question_lower for kw in categorical_keywords):
        inferred_analyses.append("Categorical")
    if any(kw in question_lower for kw in continuous_keywords):
        inferred_analyses.append("Continuous")
    if any(kw in question_lower for kw in survival_keywords):
        inferred_analyses.append("Survival")

    # If no specific analysis mentioned but cell type is mentioned,
    # default to both categorical and continuous
    if not inferred_analyses and any(kw in question_lower for kw in cell_type_keywords):
        inferred_analyses = ["Categorical", "Continuous"]

    # If still no analyses inferred, include all
    if not inferred_analyses:
        inferred_analyses = list(ANALYSIS_DOMAINS.keys())

    return inferred_analyses

def infer_cell_type_from_question(question):
    """
    Infer cell type from user's natural language question.
    Returns a list of potential cell types mentioned.
    """
    question_lower = question.lower()
    cell_type_mappings = {
        "cd8 t cell": ["cd8", "cd8 t", "cd8+ t", "cytotoxic t"],
        "cd4 t cell": ["cd4", "cd4 t", "cd4+ t", "helper t"],
        "b cell": ["b cell", "b-cell", "b cells"],
        "nk cell": ["nk", "natural killer", "nk cell"],
        "macrophage": ["macrophage", "m1", "m2"],
        "dendritic cell": ["dendritic", "dc", "cdc", "pdc"],
        "monocyte": ["monocyte"],
        "neutrophil": ["neutrophil"],
        "treg": ["treg", "regulatory t", "t regulatory"],
        "fibroblast": ["fibroblast", "caf", "icaf", "mcaf", "myofibro"],
        "tumor": ["tumor", "cancer cell", "tumor cell", "malignant"],
        "endothelial": ["endothelial", "blood vessel"],
        "mast cell": ["mast cell", "mast"],
        "plasma cell": ["plasma cell", "plasma"],
        "tfh": ["tfh", "follicular helper"]
    }

    inferred_cells = []
    for cell_name, keywords in cell_type_mappings.items():
        if any(kw in question_lower for kw in keywords):
            inferred_cells.append(cell_name)

    return inferred_cells

def build_rag_prompt(user_question, analysis_context, model_limitations,
                     inferred_analyses=None, inferred_cells=None, analysis_results=None):
    """Build the RAG prompt with grounding context and multi-analysis support."""

    # Build inferred context
    inferred_context = ""
    if inferred_analyses:
        inferred_context += f"\n## Inferred Relevant Analyses\n"
        for analysis in inferred_analyses:
            desc = ANALYSIS_DOMAINS.get(analysis, "")
            inferred_context += f"- **{analysis}**: {desc}\n"

    if inferred_cells:
        inferred_context += f"\n## Inferred Cell Types of Interest\n"
        inferred_context += f"- {', '.join(inferred_cells)}\n"

    prompt = f"""You are a scientific assistant for a Bayesian analysis of obesity effects on pancreatic cancer tumor microenvironment.

## Your Role
- Answer questions ONLY based on the provided context, data, and limitations
- Be cautious and conservative in interpretations
- Explicitly acknowledge limitations when relevant
- Refuse to make causal or clinical claims
- When consulting multiple analyses, clearly state which analyses were consulted

## Current Analysis Scope (User-selected)
{analysis_context}
{inferred_context}

## Model Limitations (MUST be respected at all times)
{model_limitations}

## User Question
{user_question}

## Critical Scientific Guardrails
1. Answer based ONLY on the current analysis context and available data
2. Do NOT invent findings or extrapolate beyond the data
3. If the question cannot be answered with the available information, say so clearly
4. Always remind the user of relevant limitations when applicable
5. Keep responses concise and scientific
6. NEVER make causal claims - this is observational data showing associations only
7. NEVER provide clinical recommendations or treatment advice
8. Acknowledge uncertainty and credible intervals when discussing effect sizes
9. When multiple analyses are consulted, explicitly state which ones and what each shows

Answer:"""
    return prompt


# ==================================================================================
# ====================== DATA-AWARE AI RESPONSE PIPELINE ===========================
# ==================================================================================

# Cell type keyword mappings for question parsing
CELL_TYPE_KEYWORDS = {
    # CD8 T cells
    "cd8 t cell": ["cd8", "cd8 t", "cd8+ t", "cytotoxic t", "cd8 t cell", "cd8+ t cell", "killer t"],
    "cd8 exhausted": ["cd8 exhaust", "exhausted cd8", "tex", "cd8+ tex"],
    "cd8 effector": ["cd8 effector", "cd8+ effector", "effector cd8"],
    "cd8 memory": ["cd8 memory", "cd8+ memory", "memory cd8", "cd8 tem", "cd8 tcm"],
    # CD4 T cells
    "cd4 t cell": ["cd4", "cd4 t", "cd4+ t", "helper t", "cd4 t cell", "cd4+ t cell"],
    "treg": ["treg", "regulatory t", "t regulatory", "foxp3", "cd4+ treg"],
    "th1": ["th1", "t helper 1", "ifn gamma"],
    "th2": ["th2", "t helper 2", "il-4", "il-13"],
    "th17": ["th17", "t helper 17", "il-17"],
    "tfh": ["tfh", "follicular helper", "t follicular"],
    # Other lymphocytes
    "b cell": ["b cell", "b-cell", "b cells", "b lymphocyte"],
    "plasma cell": ["plasma cell", "plasma", "antibody secreting"],
    "nk cell": ["nk", "natural killer", "nk cell", "cd56"],
    "gamma delta t": ["gamma delta", "tgd", "gd t cell"],
    # Myeloid cells
    "macrophage": ["macrophage", "m1", "m2", "tam", "tumor associated macrophage"],
    "monocyte": ["monocyte"],
    "dendritic cell": ["dendritic", "dc", "cdc", "pdc", "dendritic cell"],
    "neutrophil": ["neutrophil", "pmn", "granulocyte"],
    "mast cell": ["mast cell", "mast"],
    "myeloid": ["myeloid", "mdsc"],
    "basophil": ["basophil"],
    "eosinophil": ["eosinophil"],
    # Stromal cells
    "fibroblast": ["fibroblast", "caf", "icaf", "mcaf", "myofibro", "cancer associated fibroblast", "apCAF"],
    "endothelial": ["endothelial", "blood vessel", "vascular"],
    # Tumor cells
    "tumor": ["tumor", "cancer cell", "tumor cell", "malignant", "epithelial", "classical", "basal"],
}

# Comparison keyword mappings
COMPARISON_KEYWORDS = {
    "overweight_vs_normal": ["overweight vs normal", "overweight versus normal", "normal vs overweight",
                             "overweight compared to normal", "normal to overweight"],
    "obese_vs_normal": ["obese vs normal", "obese versus normal", "normal vs obese",
                        "obese compared to normal", "normal to obese"],
    "obese_vs_overweight": ["obese vs overweight", "obese versus overweight", "overweight vs obese",
                            "obese compared to overweight"],
    "continuous": ["dose-response", "continuous", "per unit", "bmi slope", "linear", "gradient", "incremental"]
}


























def _render_new_tab_page(url: str, title: str, subtitle: str,
                         hero_color: str, icon: str) -> None:
    """Auto-open a static HTML doc in a new browser tab.

    When the user clicks the radio button, this function fires.  It
    immediately tries to open `url` in a new tab via window.open().
    Most browsers allow this because the click on the radio button counts
    as a user gesture.  A prominent manual button is always shown as a
    fallback in case the browser blocks the popup.

    No iframe is rendered — the HTML page lives in its own tab where
    CSS position:sticky, anchor links, canvas, and scroll all work
    natively with zero polyfills.
    """
    # Auto-open the new tab immediately on page render.
    # The script runs inside a tiny 1 px components.html iframe so that
    # it can call window.open on the parent tab's behalf.
    components.html(
        f"""<script>
        (function () {{
            try {{
                window.open('{url}', '_blank');
            }} catch (_) {{}}
        }})();
        </script>""",
        height=1,
    )

    # Clean landing card shown in the main Streamlit area after the tab opens.
    st.markdown(
        f"""
        <div style="
            display:flex; flex-direction:column; align-items:center;
            justify-content:center; min-height:60vh; text-align:center;
            padding:2rem;
        ">
          <div style="font-size:3.5rem; margin-bottom:1rem;">{icon}</div>
          <h2 style="margin:0 0 0.5rem; font-size:1.6rem; color:#1a1a1a;">{title}</h2>
          <p style="color:#555; margin-bottom:2rem; max-width:480px;">{subtitle}</p>
          <a href="{url}" target="_blank"
             style="
               display:inline-block; padding:0.65rem 1.6rem;
               background:{hero_color}; color:#fff; border-radius:8px;
               text-decoration:none; font-size:1rem; font-weight:700;
               letter-spacing:0.02em; box-shadow:0 2px 8px rgba(0,0,0,0.18);
             ">
            Open page ↗
          </a>
          <p style="margin-top:1.25rem; font-size:0.8rem; color:#888;">
            The page opened in a new tab. Click the button above if it was blocked.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# jsDelivr CDN serves files directly from GitHub with the correct text/html
# MIME type so the browser renders them as proper HTML pages.
# URL format: https://cdn.jsdelivr.net/gh/{user}/{repo}@{branch}/{path}
_JSDELIVR_BASE = (
    "https://cdn.jsdelivr.net/gh/"
    "arunviswanathan91/cell-analysis-viewer@main/html_docs"
)


def render_study_methodology():
    """Open Study Methodology in a new browser tab (served via jsDelivr CDN)."""
    _render_new_tab_page(
        url=f"{_JSDELIVR_BASE}/methodology.html",
        title="Study Methodology",
        subtitle=(
            "A full walkthrough of every analytical step — "
            "cohort design, deconvolution, signatures, Bayesian modelling, and more."
        ),
        hero_color="#1a6b7a",
        icon="📖",
    )


def render_bayesian_explained():
    """Open Bayesian Model Explained in a new browser tab (served via jsDelivr CDN)."""
    _render_new_tab_page(
        url=f"{_JSDELIVR_BASE}/bayesian_model.html",
        title="The Bayesian Model — Explained",
        subtitle=(
            "Everything you need to understand the hierarchical Bayesian approach — "
            "with analogies, formulas, and figures."
        ),
        hero_color="#0b1f3a",
        icon="🧮",
    )



# ==================================================================================
# ============================= PAGE CONFIGURATION =================================
# ==================================================================================

st.set_page_config(
    page_title="Obesity-Driven Pancreatic Cancer Analysis",
    page_icon="DS",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS - Advanced Material Design 3 + Creative Modern UI
st.markdown("""
<style>
    /* ========== ADVANCED DESIGN SYSTEM ========== */
    
    /* Google Fonts - Material + Display */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Roboto+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    :root {
        /* Material Design 3 - Dynamic Color Palette */
        --md-primary-50: #E3F2FD;
        --md-primary-100: #BBDEFB;
        --md-primary-200: #90CAF9;
        --md-primary-300: #64B5F6;
        --md-primary-400: #42A5F5;
        --md-primary-500: #2196F3;
        --md-primary-600: #1E88E5;
        --md-primary-700: #1976D2;
        --md-primary-800: #1565C0;
        
        /* Accent Colors */
        --md-accent-teal: #00BCD4;
        --md-accent-purple: #9C27B0;
        --md-accent-orange: #FF9800;
        
        /* Success, Warning, Error */
        --md-success-50: #E8F5E9;
        --md-success-500: #4CAF50;
        --md-success-700: #388E3C;
        --md-warning-50: #FFF8E1;
        --md-warning-500: #FFC107;
        --md-error-50: #FFEBEE;
        --md-error-500: #F44336;
        
        /* Sophisticated Neutral Palette */
        --md-grey-0: #FFFFFF;
        --md-grey-50: #FAFAFA;
        --md-grey-100: #F5F5F5;
        --md-grey-200: #EEEEEE;
        --md-grey-300: #E0E0E0;
        --md-grey-400: #BDBDBD;
        --md-grey-500: #9E9E9E;
        --md-grey-600: #757575;
        --md-grey-700: #616161;
        --md-grey-800: #424242;
        --md-grey-900: #212121;
        
        /* Glassmorphism */
        --glass-bg: rgba(255, 255, 255, 0.7);
        --glass-border: rgba(255, 255, 255, 0.18);
        --glass-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        
        /* Modern Gradients */
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-success: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        --gradient-info: linear-gradient(135deg, #2196F3 0%, #21CBF3 100%);
        --gradient-cosmic: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        --gradient-ocean: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --gradient-sunset: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        
        /* Premium Shadows - Multi-layered */
        --shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        --shadow-2xl: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        --shadow-glow: 0 0 20px rgba(33, 150, 243, 0.3);
        --shadow-glow-hover: 0 0 30px rgba(33, 150, 243, 0.5);
        
        /* Smooth Animations */
        --ease-smooth: cubic-bezier(0.4, 0, 0.2, 1);
        --ease-bounce: cubic-bezier(0.68, -0.55, 0.265, 1.55);
        --ease-in-out-back: cubic-bezier(0.68, -0.6, 0.32, 1.6);
    }
    
    * {
        font-family: 'Inter', 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* ========== MAIN LAYOUT - Subtle Texture ========== */
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        background-attachment: fixed;
        position: relative;
    }
    
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(102, 126, 234, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(118, 75, 162, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* ========== BLOCK CONTAINER — Reduce default Streamlit whitespace ========== */
    /* Streamlit's block-container defaults to ~6 rem top, 10 rem bottom, and 1 rem
       sides — creating the large blank gutters visible around every page. */

    .main .block-container,
    section[data-testid="stMain"] .block-container {
        padding-top: 1.5rem !important;      /* Just enough to clear the fixed toolbar */
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 2rem !important;     /* Down from Streamlit's default 10 rem  */
        max-width: 100% !important;
    }

    /* ========== HEADERS - Gradient Text with Depth ========== */
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: var(--gradient-cosmic);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
        margin-bottom: 2rem;
        letter-spacing: -0.03em;
        line-height: 1.1;
        position: relative;
        animation: fadeInDown 0.8s var(--ease-smooth);
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        will-change: transform, font-size, padding;
    }
    
    /* Shrunk state when scrolled */
    .main-header.shrunk {
        font-size: 1.6rem !important;
        padding: 0.8rem 0 !important;
        margin-bottom: 0.5rem !important;
        position: fixed !important;
        top: 3.5rem !important;
        left: 0 !important;
        right: 0 !important;
        width: 100% !important;
        background: rgba(255, 255, 255, 0.98) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        z-index: 999 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
        border-bottom: 3px solid rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Info box transitions */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid var(--md-primary-600);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
        box-shadow: var(--shadow-md);
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        opacity: 1;
        max-height: 500px;
        overflow: hidden;
        will-change: opacity, max-height;
    }
    
    .info-box.hidden {
        opacity: 0 !important;
        max-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        pointer-events: none !important;
    }
    
    /* Spacer for when header becomes fixed */
    .header-spacer {
        height: 0;
        transition: height 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .header-spacer.active {
        height: 6rem;
    }
    
    /* ========== COMPACT CARDS - Scientific Usability ========== */

    .info-box {
        background: var(--glass-bg);
        backdrop-filter: blur(10px) saturate(150%);
        -webkit-backdrop-filter: blur(10px) saturate(150%);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        border: 1px solid var(--glass-border);
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }

    .info-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: var(--gradient-info);
    }

    .info-box h3 {
        font-size: 0.9rem !important;
        margin: 0 0 0.25rem 0 !important;
    }

    .info-box p {
        font-size: 0.8rem !important;
        margin: 0 !important;
        line-height: 1.4 !important;
    }

    .method-box {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%);
        backdrop-filter: blur(10px) saturate(150%);
        -webkit-backdrop-filter: blur(10px) saturate(150%);
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin: 0.75rem 0;
        border: 1px solid rgba(76, 175, 80, 0.2);
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }

    .method-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: var(--gradient-success);
    }

    .method-box b {
        font-size: 0.85rem !important;
    }

    .method-box li, .method-box p {
        font-size: 0.8rem !important;
        margin: 0.15rem 0 !important;
        line-height: 1.3 !important;
    }

    .warning-box {
        background: linear-gradient(135deg, rgba(255, 248, 225, 0.95) 0%, rgba(255, 248, 225, 0.85) 100%);
        backdrop-filter: blur(10px);
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid var(--md-warning-500);
        box-shadow: var(--shadow-xs);
        font-size: 0.8rem;
    }
    
    /* ========== MODERN BUTTONS - Gradient with Shine Effect ========== */
    
    .stButton>button {
        width: 100%;
        background: var(--gradient-primary);
        color: white;
        font-weight: 600;
        font-size: 0.9375rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 1rem 2rem;
        border: none;
        border-radius: 12px;
        box-shadow: var(--shadow-lg), var(--shadow-glow);
        transition: all 0.3s var(--ease-smooth);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s var(--ease-smooth);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-xl), var(--shadow-glow-hover);
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    /* ========== PREMIUM INPUTS - Floating Labels Effect ========== */
    
    .stSelectbox label, .stTextInput label, .stTextArea label {
        font-weight: 600 !important;
        color: var(--md-grey-700) !important;
        font-size: 0.875rem !important;
        letter-spacing: 0.02em !important;
        margin-bottom: 0.5rem !important;
        text-transform: uppercase;
    }
    
    .stSelectbox > div > div,
    .stTextInput > div > div,
    .stTextArea > div > div {
        border-radius: 12px !important;
        border: 2px solid var(--md-grey-200) !important;
        background: white !important;
        transition: all 0.3s var(--ease-smooth) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .stSelectbox > div > div:hover,
    .stTextInput > div > div:hover,
    .stTextArea > div > div:hover {
        border-color: var(--md-primary-300) !important;
        box-shadow: var(--shadow-md), 0 0 0 4px rgba(33, 150, 243, 0.1) !important;
    }
    
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div:focus-within,
    .stTextArea > div > div:focus-within {
        border-color: var(--md-primary-600) !important;
        box-shadow: var(--shadow-md), 0 0 0 4px rgba(33, 150, 243, 0.15) !important;
        transform: translateY(-2px);
    }
    
    /* ========== COMPACT METRICS - Scientific Display ========== */

    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        box-shadow: var(--shadow-xs);
        border: 1px solid var(--md-grey-200);
        position: relative;
        overflow: hidden;
    }

    .stMetric::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: var(--gradient-primary);
    }

    .stMetric label {
        font-weight: 600 !important;
        color: var(--md-grey-600) !important;
        font-size: 0.65rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stMetric [data-testid="stMetricValue"] {
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }
    
    /* ========== MODERN TABS - Pill Style with Glow ========== */
    
    .stTabs {
        background-color: transparent;
        margin-top: 2.5rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.75rem;
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        padding: 0.75rem;
        border-radius: 16px;
        box-shadow: var(--shadow-md);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 0.875rem 1.75rem;
        background: transparent;
        border-radius: 12px;
        color: var(--md-grey-700);
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border: 2px solid transparent;
        transition: all 0.3s var(--ease-smooth);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(33, 150, 243, 0.08);
        color: var(--md-primary-700);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary) !important;
        color: white !important;
        box-shadow: var(--shadow-md), var(--shadow-glow);
        border-color: transparent !important;
    }
    
    /* ========== PREMIUM DATAFRAMES ========== */
    
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
        border: 1px solid var(--md-grey-200);
        background: white;
    }
    
    .stDataFrame thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 0.8125rem !important;
        padding: 1.25rem !important;
        border: none !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .stDataFrame tbody tr {
        transition: all 0.2s var(--ease-smooth);
        border-bottom: 1px solid var(--md-grey-100);
    }
    
    .stDataFrame tbody tr:hover {
        background: linear-gradient(90deg, rgba(33, 150, 243, 0.05) 0%, rgba(33, 150, 243, 0.02) 100%) !important;
        transform: translateX(4px);
    }
    
    .stDataFrame tbody tr:last-child {
        border-bottom: none;
    }
    
    /* ========== GLASSMORPHISM SIDEBAR ========== */
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 249, 250, 0.95) 100%);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border-right: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: var(--shadow-xl);
    }
    
    /* ========== MODERN EXPANDER ========== */
    
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid var(--md-grey-200);
        border-radius: 12px;
        padding: 1.25rem 1.75rem;
        font-weight: 600;
        color: var(--md-grey-900);
        transition: all 0.3s var(--ease-smooth);
        box-shadow: var(--shadow-sm);
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
        border-color: var(--md-primary-300);
    }
    
    .streamlit-expanderContent {
        border: 1px solid var(--md-grey-200);
        border-top: none;
        border-radius: 0 0 12px 12px;
        padding: 2rem;
        background: white;
        box-shadow: var(--shadow-sm);
    }
    
    /* ========== STYLED ALERTS ========== */
    
    .stAlert {
        border-radius: 12px;
        padding: 1.25rem 1.75rem;
        box-shadow: var(--shadow-md);
        border: none;
        backdrop-filter: blur(10px);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.15) 0%, rgba(56, 142, 60, 0.1) 100%);
        color: var(--md-success-700);
        border-left: 4px solid var(--md-success-500);
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.15) 0%, rgba(25, 118, 210, 0.1) 100%);
        color: var(--md-primary-800);
        border-left: 4px solid var(--md-primary-500);
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 160, 0, 0.1) 100%);
        color: #f57c00;
        border-left: 4px solid var(--md-warning-500);
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.15) 0%, rgba(211, 47, 47, 0.1) 100%);
        color: #c62828;
        border-left: 4px solid var(--md-error-500);
    }
    
    /* ========== PREMIUM DOWNLOAD BUTTON ========== */
    
    .stDownloadButton button {
        background: var(--gradient-success);
        color: white;
        font-weight: 600;
        font-size: 0.9375rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        box-shadow: var(--shadow-md), 0 4px 20px rgba(76, 175, 80, 0.3);
        transition: all 0.3s var(--ease-smooth);
        position: relative;
        overflow: hidden;
    }
    
    .stDownloadButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s var(--ease-smooth);
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-lg), 0 8px 30px rgba(76, 175, 80, 0.4);
    }
    
    .stDownloadButton button:hover::before {
        left: 100%;
    }
    
    /* ========== PLOTLY CHARTS - Static Scientific Display ========== */

    .js-plotly-plot {
        border-radius: 8px;
        overflow: hidden;
        background: white;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--md-grey-200);
    }

    /* Disable hover animations on plots for scientific inspection */
    .js-plotly-plot:hover {
        box-shadow: var(--shadow-sm);
        transform: none;
    }
    
    /* ========== ELEGANT DIVIDER ========== */
    
    hr {
        margin: 3rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, var(--md-grey-300) 50%, transparent 100%);
    }
    
    /* ========== ENHANCED TEXT ========== */
    
    .stMarkdown p, .stMarkdown li {
        color: var(--md-grey-800) !important;
        line-height: 1.7;
        font-weight: 400;
    }
    
    .stMarkdown strong {
        color: var(--md-grey-900) !important;
        font-weight: 700;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: var(--md-grey-900) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    .stMarkdown code {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        color: var(--md-primary-700);
        font-size: 0.9em;
        font-family: 'Roboto Mono', monospace;
        font-weight: 500;
        border: 1px solid var(--md-grey-300);
        box-shadow: var(--shadow-xs);
    }
    
    /* ========== MODERN SCROLLBAR ========== */
    
    ::-webkit-scrollbar {
        width: 14px;
        height: 14px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--md-grey-100);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--md-grey-400) 0%, var(--md-grey-500) 100%);
        border-radius: 10px;
        border: 3px solid var(--md-grey-100);
        transition: background 0.3s var(--ease-smooth);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--md-primary-400) 0%, var(--md-primary-600) 100%);
    }
    
    /* ========== LOADING SPINNER ========== */

    .stSpinner > div > div {
        border-color: var(--md-primary-500) transparent transparent transparent !important;
    }
    
    /* ========== CHECKBOX & RADIO - Modern Toggle ========== */
    
    .stCheckbox, .stRadio {
        padding: 0.25rem;
        border-radius: 8px;
        transition: background 0.3s var(--ease-smooth);
    }
    
    .stCheckbox:hover, .stRadio:hover {
        background: rgba(33, 150, 243, 0.05);
    }
    
    /* Reduce spacing between radio options */
    .stRadio > div {
        gap: 0.1rem !important;
    }
    
    .stRadio label {
        margin-bottom: 0 !important;
        padding: 0.2rem 0.5rem !important;
    }
    
    div[role="radiogroup"] {
        gap: 0.1rem !important;
    }
    
    /* ========== PROFESSIONAL CAPTION ========== */
    
    .stCaption {
        color: var(--md-grey-600) !important;
        font-size: 0.875rem !important;
        font-weight: 400 !important;
    }
    
    /* ========== MODERN PROGRESS BAR ========== */
    
    .stProgress > div > div {
        background: var(--gradient-primary) !important;
        border-radius: 10px !important;
    }
    
    /* ========== FADE IN ANIMATION FOR ALL ELEMENTS ========== */
    
    .element-container {
        animation: fadeIn 0.5s var(--ease-smooth);
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
""", unsafe_allow_html=True)
# JavaScript for scroll header effect
components.html("""
<script>
(function() {
    let ticking = false;

    function getDoc() {
        try { return window.parent.document; } catch (e) { return document; }
    }

    function updateHeader() {
        try {
            const doc = getDoc();
            const scrollTop = window.parent.pageYOffset || doc.documentElement.scrollTop;
            const headers  = doc.querySelectorAll('.main-header');
            const infoBoxes = doc.querySelectorAll('.info-box');
            const spacers  = doc.querySelectorAll('.header-spacer');
            if (!ticking) {
                window.requestAnimationFrame(() => {
                    if (scrollTop > 150) {
                        headers.forEach(h => h.classList.add('shrunk'));
                        infoBoxes.forEach(b => b.classList.add('hidden'));
                        spacers.forEach(s => s.classList.add('active'));
                    } else {
                        headers.forEach(h => h.classList.remove('shrunk'));
                        infoBoxes.forEach(b => b.classList.remove('hidden'));
                        spacers.forEach(s => s.classList.remove('active'));
                    }
                    ticking = false;
                });
                ticking = true;
            }
        } catch(e) {}
    }

    window.parent.addEventListener('scroll', updateHeader, { passive: true });
    setTimeout(updateHeader, 100);
    setInterval(updateHeader, 1000);
})();
</script>
""", height=1)

# ==================================================================================
# ============================= CONFIGURATION ======================================
# ==================================================================================

# Data directories - data2 for analysis results, data for legacy signatures
DATA_DIR = "data"  # For legacy signatures
DATA2_DIR = "data2"  # For AI-optimized analysis results

# Import data backend for fast querying (optional)
try:
    from src.data_backend import get_table, query_sql
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

LOG_TRANSFORM = True
BMI_COLORS = {'Normal': '#2ECC71', 'Overweight': '#F39C12', 'Obese': '#E74C3C'}
COLOR_OVERWEIGHT = "#1f78b4"
COLOR_OBESE = "#e31a1c"
COLOR_OBO = "#33a02c"

# Plotly template
PLOTLY_TEMPLATE = "plotly_white"

# Survival analysis configuration
COLOR_POSITIVE_HR = '#E53935'
COLOR_NEGATIVE_HR = '#1E88E5'
BMI_CATEGORIES = {
    'Underweight': (0, 18.5),
    'Normal': (18.5, 25),
    'Overweight': (25, 30),
    'Obese': (30, 50)
}
BMI_COLORS_SURVIVAL = {
    'Underweight': '#4CAF50',
    'Normal': '#2196F3',
    'Overweight': '#FF9800',
    'Obese': '#F44336'
}
CONFIDENCE_THRESHOLD = 10


# Survival plot colors
COLOR_POSITIVE_HR = '#E53935'  # Red for increased risk (HR > 1)
COLOR_NEGATIVE_HR = '#1E88E5'  # Blue for protective (HR < 1)

# BMI Categories (WHO)
BMI_CATEGORIES = {
    'Underweight': (0, 18.5),
    'Normal': (18.5, 25),
    'Overweight': (25, 30),
    'Obese': (30, 50)
}

BMI_COLORS_SURVIVAL = {
    'Underweight': '#4CAF50',
    'Normal': '#2196F3',
    'Overweight': '#FF9800',
    'Obese': '#F44336'
}

# Confidence threshold for solid vs dashed lines
CONFIDENCE_THRESHOLD = 10


# ==================================================================================
# ============================= DATA LOADING =======================================
# ==================================================================================
 
def load_signatures():
    """Load signatures from JSON and normalize gene structure"""
    try:
        sig_file = os.path.join(DATA_DIR, "signatures", "ALL_CELL_SIGNATURES_FLAT.json")
        with open(sig_file, 'r') as f:
            data = json.load(f)

        normalized = []

        # Logic for New JSON Structure: { "CELL_TYPE": { "SIG_NAME": ["GENE1", ...], ... } }
        if isinstance(data, dict):
            for cell_type, signatures_dict in data.items():
                if isinstance(signatures_dict, dict):
                    for sig_name, gene_list in signatures_dict.items():
                        # Flatten into a list of dictionaries
                        normalized.append({
                            "cell_type": str(cell_type),
                            "signature": str(sig_name),
                            "genes": gene_list if isinstance(gene_list, list) else [],
                            "positive_genes": [], # Not in new JSON
                            "negative_genes": []  # Not in new JSON
                        })
                        
        return normalized

    except Exception as e:
        st.error(f"Error loading signatures: {e}")
        return []


 
def load_clinical_data():
    """Load clinical data"""
    try:
        clinical_file = os.path.join(DATA_DIR, "clinical", "cptac_complete_clinical.csv")
        clinical = pd.read_csv(clinical_file)
        
        col_mapping = {'sampleId': 'sample_id', 'SampleId': 'sample_id', 'SAMPLE_ID': 'sample_id'}
        for old_col, new_col in col_mapping.items():
            if old_col in clinical.columns:
                clinical = clinical.rename(columns={old_col: new_col})
        
        def categorize_bmi(bmi):
            try:
                val = float(bmi)
                if val < 25:
                    return 'Normal'
                elif val < 30:
                    return 'Overweight'
                else:
                    return 'Obese'
            except:
                return np.nan
        
        clinical['bmi_category'] = clinical['BMI'].apply(categorize_bmi)
        clinical['vital_status_binary'] = clinical['VITAL_STATUS'].apply(
            lambda x: 1 if str(x).strip().upper() == 'DECEASED' else 0
        )
        clinical['follow_up_months'] = pd.to_numeric(clinical['FOLLOW_UP_DAYS'], errors='coerce') / 30.44
        
        return clinical
    except Exception as e:
        st.error(f"Error loading clinical data: {e}")
        return None

 
def load_tpm_data():
    """Load TPM expression data"""
    try:
        tpm_file = os.path.join(DATA_DIR, "tpm_expression", "bulk_combined_with_symbols_cleaned.csv")
        tpm = pd.read_csv(tpm_file, index_col=0)
        
        if LOG_TRANSFORM:
            tpm = np.log2(tpm + 1)
        
        return tpm
    except Exception as e:
        st.error(f"Error loading TPM data: {e}")
        return None

 
def load_compartment_data(compartment):
    """Load all data for a compartment"""
    comp_map = {
        'Immune Fine': 'immune_fine',
        'Immune Coarse': 'immune_coarse',
        'Non-Immune': 'non_immune'
    }
    comp_key = comp_map[compartment]
    
    data = {}
    
    try:
        zscore_file = os.path.join(DATA_DIR, "zscores", f"{comp_key}_zscores.csv")
        data['zscores'] = pd.read_csv(zscore_file)
    except:
        data['zscores'] = None
    
    try:
        stabl_file = os.path.join(DATA_DIR, "stabl", f"{comp_key}_selected.csv")
        data['stabl'] = pd.read_csv(stabl_file)
    except:
        data['stabl'] = None
    
    try:
        bayes_file = os.path.join(DATA_DIR, "bayesian", f"{comp_key}_results.csv")
        data['bayesian'] = pd.read_csv(bayes_file)
    except:
        data['bayesian'] = None
    
    try:
        ctmap_file = os.path.join(DATA_DIR, "bayesian", f"{comp_key}_celltype_mapping.csv")
        data['celltype_map'] = pd.read_csv(ctmap_file)
    except:
        data['celltype_map'] = None
    
    # Load posterior CSVs instead of .nc file
    try:
        csv_dir = os.path.join(DATA_DIR, "bayesian_csvs", comp_key)
        
        # Try to load celltype_mapping from bayesian_csvs directory
        ct_map_csv = os.path.join(csv_dir, "celltype_mapping.csv")
        if os.path.exists(ct_map_csv):
            data['celltype_map'] = pd.read_csv(ct_map_csv)
        
        # Load posterior samples
        post_over_file = os.path.join(csv_dir, "posterior_overweight.csv")
        post_ob_file = os.path.join(csv_dir, "posterior_obese.csv")
        post_obo_file = os.path.join(csv_dir, "posterior_obese_vs_overweight.csv")
        
        if os.path.exists(post_over_file) and os.path.exists(post_ob_file):
            data['posterior_overweight'] = pd.read_csv(post_over_file)
            data['posterior_obese'] = pd.read_csv(post_ob_file)
            
            if os.path.exists(post_obo_file):
                data['posterior_obese_vs_overweight'] = pd.read_csv(post_obo_file)
            else:
                # Calculate if not present
                df_over = data['posterior_overweight']
                df_ob = data['posterior_obese']
                df_obo = df_over.copy()
                df_obo.iloc[:, 1:] = df_ob.iloc[:, 1:].values - df_over.iloc[:, 1:].values
                data['posterior_obese_vs_overweight'] = df_obo
        else:
            data['posterior_overweight'] = None
            data['posterior_obese'] = None
            data['posterior_obese_vs_overweight'] = None
        
        # Load diagnostics
        diag_file = os.path.join(csv_dir, "diagnostics_summary.csv")
        if os.path.exists(diag_file):
            diag_df = pd.read_csv(diag_file)
            # Set first column as index if it looks like parameter names
            if diag_df.columns[0] in ['Unnamed: 0', 'parameter', 'index']:
                diag_df = diag_df.set_index(diag_df.columns[0])
            data['diagnostics'] = diag_df
        else:
            data['diagnostics'] = None
        
        # Load energy
        energy_file = os.path.join(csv_dir, "energy.csv")
        if os.path.exists(energy_file):
            data['energy'] = pd.read_csv(energy_file)
        else:
            data['energy'] = None
        
        # Load credible intervals
        hdi_file = os.path.join(csv_dir, "credible_intervals.csv")
        if os.path.exists(hdi_file):
            data['credible_intervals'] = pd.read_csv(hdi_file)
        else:
            data['credible_intervals'] = None
            
    except Exception as e:
        data['posterior_overweight'] = None
        data['posterior_obese'] = None
        data['posterior_obese_vs_overweight'] = None
        data['diagnostics'] = None
        data['energy'] = None
        data['credible_intervals'] = None
    
    return data

def load_compartment_data_continuous(compartment):
    """
    Load continuous BMI analysis data for a compartment.
    
    Args:
        compartment (str): 'Immune Fine', 'Immune Coarse', or 'Non-Immune'
        
    Returns:
        dict: Dictionary with loaded continuous data
    """
    comp_map = {
        'Immune Fine': 'immune_fine',
        'Immune Coarse': 'immune_coarse',
        'Non-Immune': 'non_immune'
    }
    comp_key = comp_map[compartment]
    
    data = {}
    
    # Load main results file
    try:
        results_file = os.path.join(DATA_DIR, "bayesian_continuous", f"{comp_key}_continuous.csv")
        data['continuous_results'] = pd.read_csv(results_file)
    except Exception as e:
        st.warning(f"⚠️ Could not load continuous results: {e}")
        data['continuous_results'] = None
    
    # Load posterior samples and diagnostics from continuous-specific folder
    try:
        csv_dir = os.path.join(DATA_DIR, "bayesian_csvs_continuous", comp_key)
        
        # Load celltype mapping (FROM CONTINUOUS FOLDER - NEW!)
        ctmap_file = os.path.join(csv_dir, "celltype_mapping.csv")
        if os.path.exists(ctmap_file):
            data['celltype_map'] = pd.read_csv(ctmap_file)
        else:
            data['celltype_map'] = None
        
        # Load posterior BMI slope samples
        post_slope_file = os.path.join(csv_dir, "posterior_bmi_slope.csv")
        if os.path.exists(post_slope_file):
            data['posterior_bmi_slope'] = pd.read_csv(post_slope_file)
        else:
            data['posterior_bmi_slope'] = None
        
        # Load diagnostics summary
        diag_file = os.path.join(csv_dir, "diagnostics_summary.csv")
        if os.path.exists(diag_file):
            diag_df = pd.read_csv(diag_file)
            if diag_df.columns[0] in ['Unnamed: 0', 'parameter', 'index']:
                diag_df = diag_df.set_index(diag_df.columns[0])
            data['diagnostics'] = diag_df
        else:
            data['diagnostics'] = None
        
        # Load energy
        energy_file = os.path.join(csv_dir, "energy.csv")
        if os.path.exists(energy_file):
            data['energy'] = pd.read_csv(energy_file)
        else:
            data['energy'] = None
        
        # Load credible intervals
        hdi_file = os.path.join(csv_dir, "credible_intervals.csv")
        if os.path.exists(hdi_file):
            data['credible_intervals'] = pd.read_csv(hdi_file)
        else:
            data['credible_intervals'] = None
            
    except Exception as e:
        st.warning(f"⚠️ Could not load continuous diagnostics: {e}")
        data['posterior_bmi_slope'] = None
        data['diagnostics'] = None
        data['energy'] = None
        data['credible_intervals'] = None
        data['celltype_map'] = None
    
    return data

def load_significant_features():
    """Load significant survival features"""
    try:
        sig_file = os.path.join(DATA_DIR, "survival", "significant_features.csv")
        sig_df = pd.read_csv(sig_file)
        
        # Filter for significant features (p < 0.05)
        if 'hr_p' in sig_df.columns:
            sig_df = sig_df[sig_df['hr_p'] < 0.05].copy()
        
        return sig_df
    except Exception as e:
        return None
 
def extract_base_sample_id(sample_id):
    """Extract base patient ID from sample identifiers"""
    if pd.isna(sample_id):
        return None
    
    sample_str = str(sample_id).strip()
    
    # Remove common suffixes
    suffixes = ['-T', '-N', '-tumor', '-normal', '_T', '_N']
    for suffix in suffixes:
        if sample_str.endswith(suffix):
            sample_str = sample_str[:-len(suffix)]
    
    return sample_str

 
def load_zscore_data_survival():
    """
    Load Z-score matrices for survival analysis from zscores_complete.
    Returns a long-format DataFrame with:
    sample_id, feature (Cell||Signature), Z, compartment, base_sample_id
    """
    base = "data/zscores_complete"

    files = {
        "Immune Fine": "immune_fine_zcomplete.csv",
        "Immune Coarse": "immune_coarse_zcomplete.csv",
        "Non-Immune": "non_immune_zcomplete.csv",
    }

    dfs = []
    errors = []
    
    for comp, fname in files.items():
        path = os.path.join(base, fname)
        
        if not os.path.exists(path):
            errors.append(f"Missing: {path}")
            continue

        try:
            df = pd.read_csv(path, low_memory=False)
            
            # Expect first column = sample_id, others = features
            id_col = df.columns[0]
            df = df.rename(columns={id_col: "sample_id"})

            # Melt to long format
            long = df.melt(
                id_vars="sample_id",
                var_name="feature",
                value_name="Z"
            )
            long["compartment"] = comp
            
            # Add base_sample_id for matching with clinical data
            long['base_sample_id'] = long['sample_id'].apply(extract_base_sample_id)
            
            dfs.append(long)
            
        except Exception as e:
            errors.append(f"Error loading {fname}: {str(e)}")
            continue

    # Show any errors
    if errors:
        with st.expander("⚠️ Z-score loading issues", expanded=False):
            for err in errors:
                st.warning(err)
    
    if not dfs:
        st.error(f"❌ No z-score files loaded from {base}/")
        st.info("Expected files: " + ", ".join(files.values()))
        return None

    out = pd.concat(dfs, ignore_index=True)
    out["sample_id"] = out["sample_id"].astype(str)
    
    return out

def assign_bmi_category(bmi):
    """Assign BMI category using WHO standards"""
    if pd.isna(bmi):
        return None
    for cat, (low, high) in BMI_CATEGORIES.items():
        if low <= bmi < high:
            return cat
    return 'Obese'

 
def load_significant_features():
    """Load significant survival features (p < 0.05)"""
    sig_file = os.path.join(DATA_DIR, "survival", "significant_features.csv")
    
    # Check if file exists
    if not os.path.exists(sig_file):
        st.error(f"❌ File not found: {sig_file}")
        return None
    
    # Try multiple encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    sig_df = None
    
    for encoding in encodings:
        try:
            sig_df = pd.read_csv(sig_file, encoding=encoding)
            # Success - no need to notify user about encoding
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            continue
    
    if sig_df is None:
        st.error(f"❌ Could not read survival features file")
        return None
    
    # Look for p-value column (try different possible names)
    p_col = None
    for col_name in ['hr_p', 'p_value', 'pvalue', 'p', 'P_value', 'HR_p', 'p-value']:
        if col_name in sig_df.columns:
            p_col = col_name
            break
    
    if p_col is None:
        st.warning(f"⚠️ No p-value column found in survival data")
        return sig_df
    
    # Filter for significant features
    sig_df_filtered = sig_df[sig_df[p_col] < 0.05].copy()
    
    if len(sig_df_filtered) == 0:
        st.error(f"❌ No significant features found (all p-values ≥ 0.05)")
        return None
    
    # Success - return filtered data
    return sig_df_filtered

def extract_base_sample_id(sample_id):
    """Extract base patient ID from sample identifiers"""
    if pd.isna(sample_id):
        return None
    sample_str = str(sample_id).strip()
    for suffix in ['-T', '-N', '-tumor', '-normal', '_T', '_N']:
        if sample_str.endswith(suffix):
            sample_str = sample_str[:-len(suffix)]
    return sample_str

 
def clean_label_text(text):
    """Clean signature/feature names for display"""
    if pd.isna(text):
        return "Unknown"
    text = str(text).strip()
    text = text.replace('_Signature', '').replace('_signature', '')
    text = text.replace('_Score', '').replace('_score', '')
    text = text.replace('_', ' ')
    if len(text) > 60:
        text = text[:57] + '...'
    return text.title()


def get_available_cells(compartment):
    """Get cell types from Z-score data"""
    comp_data = load_compartment_data(compartment)
    if comp_data['zscores'] is not None:
        # Try different possible column names for cell type
        possible_cols = ['CellType', 'celltype', 'cell_type', 'Cell_Type', 'CELLTYPE']
        cells = []
        
        for col in possible_cols:
            if col in comp_data['zscores'].columns:
                try:
                    cells = sorted(comp_data['zscores'][col].unique().tolist())
                    if len(cells) > 0:
                        break
                except Exception as e:
                    continue
        
        if len(cells) == 0:
            # Debug: show what columns are available
            st.sidebar.warning(f"❌ No cell types found in z-score data. Available columns: {list(comp_data['zscores'].columns)}")
        
        return cells
    else:
        st.sidebar.error(f"❌ Z-score data not loaded for {compartment}. Check if file exists: data/zscores/{compartment.lower().replace(' ', '_').replace('-', '_')}_zscores.csv")
    return []


def get_available_cells_continuous(compartment):
    """
    Get ONLY cell types with posterior data for continuous analysis.

    Uses continuous_results as the primary source of truth for cell type names,
    since the celltype_mapping CSVs in the continuous directory only contain
    integer-to-integer mappings (no actual cell type name strings).

    Args:
        compartment: 'Immune Fine', 'Immune Coarse', or 'Non-Immune'

    Returns:
        List of cell type names that have posterior data available
    """
    comp_data = load_compartment_data_continuous(compartment)

    # Primary source: continuous_results already has a 'cell_type' column with
    # actual string names (e.g. 'BASOPHILS', 'B CELLS', 'CD4+ T CELLS')
    continuous_results = comp_data.get('continuous_results')
    if continuous_results is not None and len(continuous_results) > 0:
        # Use cell_type column if it contains string values
        if 'cell_type' in continuous_results.columns:
            ct_vals = continuous_results['cell_type'].dropna()
            string_mask = ct_vals.apply(lambda x: isinstance(x, str))
            if string_mask.any():
                return sorted(ct_vals[string_mask].unique().tolist())

        # Fallback: parse cell type names from the feature column (CELL_TYPE||gene format)
        if 'feature' in continuous_results.columns:
            cells = set()
            for feat in continuous_results['feature'].dropna():
                feat_str = str(feat)
                if '||' in feat_str:
                    # Normalize underscores → spaces to match the cell_type column format
                    cells.add(feat_str.split('||')[0].strip().replace('_', ' '))
            if cells:
                return sorted(cells)

    # Final fallback: use credible_intervals + celltype_mapping
    # (only works when celltype_mapping contains actual string names)
    credible = comp_data.get('credible_intervals')
    if credible is None or len(credible) == 0:
        return []

    available_indices = sorted(credible['celltype_index'].unique())
    celltype_map = comp_data.get('celltype_map')
    if celltype_map is None or len(celltype_map) == 0:
        return []

    if 'celltype_idx' in celltype_map.columns and 'celltype_name' in celltype_map.columns:
        idx_col = 'celltype_idx'
        name_col = 'celltype_name'
    else:
        numeric_cols = celltype_map.select_dtypes(include='number').columns.tolist()
        string_cols = celltype_map.select_dtypes(include='object').columns.tolist()
        if numeric_cols and string_cols:
            idx_col = numeric_cols[0]
            name_col = string_cols[0]
        elif len(celltype_map.columns) >= 2:
            idx_col = celltype_map.columns[0]
            name_col = celltype_map.columns[1]
        else:
            return []

    available_cells = []
    for idx in available_indices:
        row = celltype_map[celltype_map[idx_col] == idx]
        if not row.empty:
            cell_name = str(row[name_col].values[0])
            available_cells.append(cell_name)

    return sorted(available_cells)


def build_celltype_mappings(celltype_map):
    """
    Build bidirectional celltype mappings (single source of truth for filtering).

    Args:
        celltype_map: DataFrame with celltype_idx and celltype_name columns

    Returns:
        tuple: (idx_to_name dict, name_to_idx dict) with normalized names
    """
    idx_to_name = {}
    name_to_idx = {}

    if celltype_map is None or len(celltype_map) == 0:
        return idx_to_name, name_to_idx

    for _, row in celltype_map.iterrows():
        idx = int(row['celltype_idx'])
        name = str(row['celltype_name'])
        # Normalize: uppercase, replace underscores with spaces, strip whitespace
        name_norm = name.upper().replace('_', ' ').strip()

        idx_to_name[idx] = name_norm
        name_to_idx[name_norm] = idx

    return idx_to_name, name_to_idx


def get_allowed_indices_for_cell(selected_cell, celltype_map):
    """
    Convert sidebar cell selection to list of allowed celltype indices.

    Args:
        selected_cell: Cell type name selected in sidebar (may have underscores or spaces)
        celltype_map: DataFrame with celltype_idx and celltype_name columns

    Returns:
        list: List of celltype indices that match the selected cell
    """
    if celltype_map is None or len(celltype_map) == 0:
        return []

    _, name_to_idx = build_celltype_mappings(celltype_map)

    # Normalize the selected cell name
    selected_norm = selected_cell.upper().replace('_', ' ').strip()

    # Find matching index
    if selected_norm in name_to_idx:
        return [name_to_idx[selected_norm]]

    return []


def get_cell_signatures(cell_type):
    """Get signatures for this cell type"""
    entries = load_signatures()
    cell_sigs = [e for e in entries 
                if e['cell_type'].upper().replace('_', ' ') == cell_type.upper().replace('_', ' ')]
    return cell_sigs

def format_signature_name(sig_name, max_length=40):
    """Format signature name for display - remove _Signature suffix and truncate if needed"""
    # Remove common suffixes
    display_name = sig_name.replace('_Signature', '').replace('_signature', '')
    display_name = display_name.replace('_', ' ')
    
    # Truncate if too long
    if len(display_name) > max_length:
        display_name = display_name[:max_length-3] + '...'
    
    return display_name

# ==================================================================================
# ============================= INTERACTIVE PLOTTING ===============================
# ==================================================================================
 
def plot_stabl_heatmap_interactive(cell_type, sig_name, comp_data, clinical):
    """Generate interactive Stabl Z-score heatmap"""
    if comp_data['zscores'] is None or comp_data['stabl'] is None:
        st.warning("❌ Stabl data not available")
        return None
    
    zscores = comp_data['zscores']
    zscores = zscores[zscores['CellType'].str.upper() == cell_type.upper()].copy()
    
    if len(zscores) == 0:
        st.warning(f"❌ No Z-scores found for {cell_type}")
        return None
    
    zscores = zscores.merge(clinical[['sample_id', 'bmi_category']], 
                           left_on='Sample', right_on='sample_id', how='inner')
    zscores = zscores[zscores['bmi_category'].notna()]
    
    heatmap_data = zscores.groupby(['Signature', 'bmi_category'])['Z'].mean().unstack(fill_value=0)
    heatmap_data = heatmap_data[['Normal', 'Overweight', 'Obese']]
    
    heatmap_data['abs_mean'] = heatmap_data.abs().mean(axis=1)
    heatmap_data = heatmap_data.sort_values('abs_mean', ascending=False).drop('abs_mean', axis=1)
    heatmap_data = heatmap_data.head(30)
    
    stabl_features = comp_data['stabl']['feature'].tolist() if comp_data['stabl'] is not None else []
    
    # Add Stabl marker to signature names
    signatures = []
    for sig in heatmap_data.index:
        feature_name = f"{cell_type}||{sig}"
        if feature_name in stabl_features:
            signatures.append(f"{sig} ⭐")
        else:
            signatures.append(sig)
    
    # Create interactive heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=['Normal', 'Overweight', 'Obese'],
        y=signatures,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-2,
        zmax=2,
        text=heatmap_data.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Mean Z-score"),
        hovertemplate='<b>%{y}</b><br>BMI: %{x}<br>Z-score: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'{cell_type} - {sig_name}<br>Stabl Z-scores by BMI Group',
            font=dict(size=16, color='#2c3e50')
        ),
        xaxis_title='BMI Category',
        yaxis_title='Signatures (⭐= STABL-selected)',
        height=max(600, len(heatmap_data) * 25),
        template=PLOTLY_TEMPLATE,
        hovermode='closest'
    )
    
    return fig
 
def plot_bayesian_heatmap_interactive(cell_type, sig_name, comp_data):
    """Generate interactive Bayesian effect size heatmap"""
    if comp_data['bayesian'] is None:
        st.warning("❌ Bayesian data not available")
        return None
    
    bayes = comp_data['bayesian'].copy()
    
    def normalize_name(name):
        return str(name).upper().replace('_', ' ').replace('-', ' ').strip()
    
    bayes['cell_normalized'] = bayes['feature'].apply(
        lambda x: normalize_name(str(x).split('||')[0]) if '||' in str(x) else normalize_name(x)
    )
    
    cell_norm = normalize_name(cell_type)
    cell_bayes = bayes[bayes['cell_normalized'] == cell_norm].copy()
    
    if len(cell_bayes) == 0:
        st.warning(f"❌ No Bayesian results for {cell_type}")
        return None
    
    cell_bayes['signature'] = cell_bayes['feature'].apply(
        lambda x: x.split('||')[1] if '||' in str(x) else x
    )
    
    effect_data = []
    for col_prefix in ['overweight_vs_normal', 'obese_vs_normal', 'obese_vs_overweight']:
        for col_suffix in ['_mean', '']:
            col = col_prefix + col_suffix
            if col in cell_bayes.columns:
                effect_data.append(cell_bayes.set_index('signature')[col].rename(col_prefix))
                break
    
    if len(effect_data) == 0:
        st.warning("❌ No effect size columns found")
        return None
    
    heatmap_data = pd.concat(effect_data, axis=1).T
    col_order = heatmap_data.abs().sum(axis=0).sort_values(ascending=False).index
    heatmap_data = heatmap_data[col_order].iloc[:, :30]
    
    # Create interactive heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=['Overweight vs Normal', 'Obese vs Normal', 'Obese vs Overweight'],
        colorscale='RdBu_r',
        zmid=0,
        zmin=-0.4,
        zmax=0.4,
        text=heatmap_data.values,
        texttemplate='%{text:.3f}',
        textfont={"size": 9},
        colorbar=dict(title="Effect Size"),
        hovertemplate='<b>%{x}</b><br>%{y}<br>Effect: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'{cell_type} - Bayesian Effect Sizes<br>Posterior Mean by Comparison',
            font=dict(size=16, color='#2c3e50')
        ),
        xaxis_title='Signatures',
        yaxis_title='Comparison',
        height=500,
        width=max(800, len(heatmap_data.columns) * 30),
        template=PLOTLY_TEMPLATE,
        hovermode='closest',
        xaxis=dict(tickangle=-45)
    )
    
    return fig
 
def plot_overlapped_ridges_interactive(cell_type, comp_data):
    """Generate interactive overlapped ridge plot"""
    if comp_data['posterior_overweight'] is None or comp_data['posterior_obese'] is None:
        st.info("❌ Posterior data not available - ridge plot skipped")
        return None
    
    try:
        # Read from CSV DataFrames
        df_over = comp_data['posterior_overweight']
        df_ob = comp_data['posterior_obese']
        
        if comp_data['posterior_obese_vs_overweight'] is not None:
            df_obo = comp_data['posterior_obese_vs_overweight']
        else:
            # Calculate if not present
            df_obo = df_over.copy()
            df_obo.iloc[:, 1:] = df_ob.iloc[:, 1:].values - df_over.iloc[:, 1:].values
        
        # Convert to numpy arrays (skip first column which is 'sample')
        post_over = df_over.iloc[:, 1:].values
        post_ob = df_ob.iloc[:, 1:].values
        post_obo = df_obo.iloc[:, 1:].values
        
        ct_map = comp_data['celltype_map']
        n_cells = post_ob.shape[1]

        cell_names = []
        if ct_map is not None and len(ct_map) > 0 and 'celltype_name' in ct_map.columns and 'celltype_idx' in ct_map.columns:
            for _, row in ct_map.iterrows():
                name = str(row['celltype_name'])
                cell_names.append(name.replace('_', ' ').title())
        else:
            cell_names = [f"Cell {i}" for i in range(n_cells)]
        
        sorted_pairs = sorted(enumerate(cell_names), key=lambda x: x[1].lower())
        indices = [p[0] for p in sorted_pairs]
        names = [p[1] for p in sorted_pairs]
        
        if len(indices) > 14:
            means = post_ob.mean(axis=0)
            abs_order = np.argsort(np.abs(means))[::-1][:14]
            indices = [idx for idx in indices if idx in abs_order]
            names = [name for idx, name in zip(indices, names) if idx in abs_order]
        
        indices = indices[::-1]
        names = names[::-1]
        
        # Create figure with subplots (one per cell type)
        fig = go.Figure()
        
        KDE_POINTS = 200
        RIDGE_HEIGHT = 1.0
        SPACING = 1.5
        
        all_samples = np.hstack([
            post_over[:, indices].flatten(),
            post_ob[:, indices].flatten(),
            post_obo[:, indices].flatten()
        ])
        x_min, x_max = np.percentile(all_samples, [0.5, 99.5])
        x_span = max(1e-6, x_max - x_min)
        xgrid = np.linspace(x_min - 0.03*x_span, x_max + 0.03*x_span, KDE_POINTS)
        
        means_over = post_over.mean(axis=0)
        means_ob = post_ob.mean(axis=0)
        means_obo = post_obo.mean(axis=0)
        
        y_base = 0
        
        for i, (ct_idx, ct_name) in enumerate(zip(indices, names)):
            s_over = post_over[:, ct_idx]
            s_ob = post_ob[:, ct_idx]
            s_obo = post_obo[:, ct_idx]
            
            # Compute KDEs
            try:
                kde_over = gaussian_kde(s_over)
                d_over = kde_over(xgrid)
                d_over = (d_over / d_over.max()) * RIDGE_HEIGHT
            except:
                d_over = np.zeros_like(xgrid)
            
            try:
                kde_ob = gaussian_kde(s_ob)
                d_ob = kde_ob(xgrid)
                d_ob = (d_ob / d_ob.max()) * RIDGE_HEIGHT
            except:
                d_ob = np.zeros_like(xgrid)
            
            try:
                kde_obo = gaussian_kde(s_obo)
                d_obo = kde_obo(xgrid)
                d_obo = (d_obo / d_obo.max()) * RIDGE_HEIGHT
            except:
                d_obo = np.zeros_like(xgrid)
            
            y_offset = y_base + i * SPACING

            # static cell-type label on the left
            fig.add_trace(go.Scatter(
                x=[x_min - 0.02 * x_span],   # a bit left of the ridges
                y=[y_offset + RIDGE_HEIGHT * 0.5],
                mode='text',
                text=[ct_name],
                textposition='middle right',
                textfont=dict(size=12, color='#2c3e50'),
                showlegend=False,
                hoverinfo='skip'
            ))

            
            # Add traces for each comparison
            fig.add_trace(go.Scatter(
                x=xgrid, y=d_ob + y_offset,
                fill='tonexty' if i > 0 else 'tozeroy',
                fillcolor=f'rgba(227, 26, 28, 0.5)',
                line=dict(color='rgba(227, 26, 28, 0.8)', width=1.5),
                name=f'Obese',
                hovertemplate=f'<b>{ct_name}</b><br>Obese vs Normal<br>Effect: %{{x:.3f}}<extra></extra>',
                showlegend=(i == 0),
                legendgroup='obese'
            ))
            
            fig.add_trace(go.Scatter(
                x=xgrid, y=d_over + y_offset,
                fill='tonexty',
                fillcolor=f'rgba(31, 120, 180, 0.5)',
                line=dict(color='rgba(31, 120, 180, 0.8)', width=1.5),
                name=f'Overweight',
                hovertemplate=f'<b>{ct_name}</b><br>Overweight vs Normal<br>Effect: %{{x:.3f}}<extra></extra>',
                showlegend=(i == 0),
                legendgroup='overweight'
            ))
            
            fig.add_trace(go.Scatter(
                x=xgrid, y=d_obo + y_offset,
                fill='tonexty',
                fillcolor=f'rgba(51, 160, 44, 0.5)',
                line=dict(color='rgba(51, 160, 44, 0.8)', width=1.5),
                name=f'Obese vs Overweight',
                hovertemplate=f'<b>{ct_name}</b><br>Obese vs Overweight<br>Effect: %{{x:.3f}}<extra></extra>',
                showlegend=(i == 0),
                legendgroup='obo'
            ))
            
            # Add mean markers
            fig.add_trace(go.Scatter(
                x=[means_ob[ct_idx]], y=[y_offset + RIDGE_HEIGHT * 0.5],
                mode='markers',
                marker=dict(color='black', size=8, symbol='line-ns-open'),
                hovertemplate=f'<b>{ct_name}</b><br>Mean (Obese): %{{x:.3f}}<extra></extra>',
                showlegend=False
            ))
        
        # Add zero reference line
        fig.add_vline(x=0, line_dash="dash", line_color="darkred", line_width=2, opacity=0.7)
        
       
        fig.update_layout(
            title=dict(
                text='Overlapped Posterior Distributions by Cell Type',
                font=dict(size=16, color='#2c3e50')
            ),
            xaxis_title='Effect Size',
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False
            ),
            height=max(600, len(indices) * 80),
            width=850,
            template=PLOTLY_TEMPLATE,
            hovermode='closest',
        
            # ✅ Legend outside to the right
            legend=dict(
                orientation="v",
                x=1.02,
                y=0.5,
                xanchor="left",
                yanchor="middle"
            ),
        
            # ✅ Add right margin so legend has space
            margin=dict(l=60, r=180, t=80, b=60)
        )

        
        return fig
        
    except Exception as e:
        st.warning(f"❌ Error creating ridge plot: {e}")
        return None
 
def plot_gene_bmi_interactive(genes, clinical, tpm):
    """Generate interactive gene-level BMI analysis plots"""
    if tpm is None:
        st.warning("ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â TPM data not available")
        return None, None
    
    tpm_t = tpm.T
    tpm_t.index.name = 'sample_id'
    tpm_t = tpm_t.reset_index()
    
    merged = clinical.merge(tpm_t, on='sample_id', how='inner')
    merged = merged[merged['BMI'].notna()].copy()
    
    results = []
    for gene in genes:
        if gene not in merged.columns:
            continue
        
        gene_data = merged[['BMI', gene]].dropna()
        if len(gene_data) < 20:
            continue
        
        try:
            slope, intercept, r_val, p_val, std_err = stats.linregress(gene_data['BMI'], gene_data[gene])
            results.append({
                'gene': gene,
                'slope': slope,
                'r_squared': r_val**2,
                'p_value': p_val
            })
        except:
            continue
    
    if not results:
        st.warning("ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â No genes analyzed")
        return None, None
    
    results_df = pd.DataFrame(results).sort_values('p_value')
    
    # Plot 1: Interactive bar plot of slopes
    plot_df = results_df.sort_values('slope')
    
    colors = []
    for _, row in plot_df.iterrows():
        if row['p_value'] < 0.05:
            colors.append('#E74C3C' if row['slope'] > 0 else '#3498DB')
        else:
            colors.append('#FADBD8' if row['slope'] > 0 else '#D6EAF8')
    
    sig_markers = []
    for _, row in plot_df.iterrows():
        if row['p_value'] < 0.001:
            sig_markers.append('***')
        elif row['p_value'] < 0.01:
            sig_markers.append('**')
        elif row['p_value'] < 0.05:
            sig_markers.append('*')
        else:
            sig_markers.append('ns')
    
    fig1 = go.Figure()
    
    fig1.add_trace(go.Bar(
        y=plot_df['gene'],
        x=plot_df['slope'],
        orientation='h',
        marker=dict(color=colors, line=dict(color='black', width=1)),
        text=[f"{s:.4f} {m}" for s, m in zip(plot_df['slope'], sig_markers)],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Slope: %{x:.4f}<br>R²: %{customdata[0]:.3f}<br>p-value: %{customdata[1]:.3e}<extra></extra>',
        customdata=np.column_stack((plot_df['r_squared'], plot_df['p_value']))
    ))
    
    fig1.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
    
    fig1.update_layout(
        title='Gene-Level BMI Association<br> ∆ Expression per ∆ BMI',
        xaxis_title='Expression Change per 1 Unit BMI Increase',
        yaxis_title='Genes',
        height=max(500, len(plot_df) * 25),
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        hovermode='closest'
    )
    
    # Plot 2: Interactive scatter plots for top genes
    top_genes = results_df.head(min(9, len(results_df)))
    
    n_genes = len(top_genes)
    n_cols = min(3, n_genes)
    n_rows = int(np.ceil(n_genes / n_cols))
    
    fig2 = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"{row['gene']} (slope={row['slope']:.4f}, R²={row['r_squared']:.3f})" 
                       for _, row in top_genes.iterrows()],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    for idx, (_, row) in enumerate(top_genes.iterrows()):
        r = idx // n_cols + 1
        c = idx % n_cols + 1
        
        gene = row['gene']
        gene_data = merged[['BMI', gene, 'bmi_category']].dropna()
        
        # Add scatter points by BMI category
        for cat in ['Normal', 'Overweight', 'Obese']:
            cat_data = gene_data[gene_data['bmi_category'] == cat]
            if len(cat_data) > 0:
                fig2.add_trace(
                    go.Scatter(
                        x=cat_data['BMI'],
                        y=cat_data[gene],
                        mode='markers',
                        name=cat,
                        marker=dict(color=BMI_COLORS[cat], size=6, opacity=0.6,
                                  line=dict(color='black', width=0.5)),
                        hovertemplate=f'<b>{cat}</b><br>BMI: %{{x:.1f}}<br>Expression: %{{y:.3f}}<extra></extra>',
                        showlegend=(idx == 0),
                        legendgroup=cat
                    ),
                    row=r, col=c
                )
        
        # Add regression line
        bmi_range = np.linspace(gene_data['BMI'].min(), gene_data['BMI'].max(), 100)
        pred = row['slope'] * bmi_range + (gene_data[gene].mean() - row['slope'] * gene_data['BMI'].mean())
        
        fig2.add_trace(
            go.Scatter(
                x=bmi_range,
                y=pred,
                mode='lines',
                line=dict(color='black', width=2.5, dash='dash'),
                name='Regression',
                showlegend=(idx == 0),
                hovertemplate='Predicted: %{y:.3f}<extra></extra>'
            ),
            row=r, col=c
        )
        
        fig2.update_xaxes(title_text='BMI' if r == n_rows else '', row=r, col=c)
        fig2.update_yaxes(title_text='Expression' if c == 1 else '', row=r, col=c)
    
    fig2.update_layout(
        title_text='Gene-Level BMI vs Expression (Top Genes)',
        height=n_rows * 350,
        template=PLOTLY_TEMPLATE,
        hovermode='closest'
    )
    
    return fig1, fig2
 
def plot_energy_diagnostic(comp_data):
    """Generate interactive energy diagnostic plot"""
    if comp_data['energy'] is None:
        st.info("❌ Energy data not available")
        return None
    
    energy = comp_data['energy']
    
    fig = go.Figure()
    
    # Plot each chain separately
    for chain in sorted(energy['chain'].unique()):
        chain_data = energy[energy['chain'] == chain]
        fig.add_trace(go.Scatter(
            x=chain_data['draw'],
            y=chain_data['energy'],
            mode='lines',
            name=f'Chain {chain}',
            line=dict(width=1),
            opacity=0.7,
            hovertemplate=f'Chain {chain}<br>Iteration: %{{x}}<br>Energy: %{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title='MCMC Energy Diagnostic',
        xaxis_title='Iteration',
        yaxis_title='Energy',
        height=500,
        template=PLOTLY_TEMPLATE,
        hovermode='closest',
        showlegend=True
    )
    
    return fig
 
def plot_trace_diagnostic(comp_data, selected_cell=None, n_celltypes=6):
    """
    Generate trace plots for cell types (index-based filtering).

    Uses posterior columns directly: celltype_0, celltype_1, etc.

    Args:
        comp_data: Dictionary containing posterior_overweight and celltype_map
        selected_cell: Cell type name to filter (if None, shows first n_celltypes)
        n_celltypes: Number of cell types to show if no selection
    """
    if comp_data['posterior_overweight'] is None:
        st.info("❌ Posterior data not available")
        return None
    
    # Get posterior data
    df_over = comp_data['posterior_overweight']
    
    # Assume 4 chains based on total samples
    n_samples = len(df_over)
    samples_per_chain = n_samples // 4
    
    # Build celltype mapping for index→name conversion
    celltype_map = comp_data.get('celltype_map', None)
    idx_to_name, name_to_idx = build_celltype_mappings(celltype_map)

    # Get allowed indices for selected cell
    if selected_cell is not None:
        allowed_idx = get_allowed_indices_for_cell(selected_cell, celltype_map)
        if not allowed_idx:
            st.warning(f"Cell type '{selected_cell}' not found in mapping")
            return None
        cell_cols = [f"celltype_{i}" for i in allowed_idx if f"celltype_{i}" in df_over.columns]
    else:
        # Default: first n_celltypes
        cell_cols = [c for c in df_over.columns if c.startswith('celltype_')][:n_celltypes]

    if not cell_cols:
        st.warning("No cell type columns found in posterior data")
        return None

    # Create subplot titles with actual cell names
    subplot_titles = []
    for col in cell_cols:
        idx_num = int(col.split('_')[1])
        if idx_num in idx_to_name:
            subplot_titles.append(idx_to_name[idx_num].replace('_', ' ').title())
        else:
            subplot_titles.append(f'Cell Type {idx_num}')

    n_cols = 2
    n_rows = int(np.ceil(len(cell_cols) / n_cols))

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, col in enumerate(cell_cols):
        r = idx // n_cols + 1
        c = idx % n_cols + 1
        
        # Split into chains
        for chain in range(4):
            start = chain * samples_per_chain
            end = (chain + 1) * samples_per_chain
            chain_data = df_over[col].iloc[start:end].values
            
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(chain_data)),
                    y=chain_data,
                    mode='lines',
                    name=f'Chain {chain}',
                    line=dict(color=colors[chain], width=1),
                    opacity=0.7,
                    showlegend=(idx == 0),
                    legendgroup=f'chain_{chain}',
                    hovertemplate=f'Chain {chain}<br>Iteration: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>'
                ),
                row=r, col=c
            )
        
        fig.update_xaxes(title_text='Iteration' if r == n_rows else '', row=r, col=c)
        fig.update_yaxes(title_text='Effect Size' if c == 1 else '', row=r, col=c)
    
    # Title reflects selection
    if selected_cell:
        title = f'Trace Plots - Overweight Effect ({selected_cell.replace("_", " ").title()})'
    else:
        title = f'Trace Plots - Overweight Effect (First {len(cell_cols)} Cell Types)'

    fig.update_layout(
        title=title,
        height=n_rows * 300,
        template=PLOTLY_TEMPLATE,
        hovermode='closest'
    )

    return fig


def plot_rank_diagnostic(comp_data, selected_cell=None, n_celltypes=6):
    """
    Generate rank plots for convergence diagnostic (index-based filtering).

    Args:
        comp_data: Dictionary containing posterior_overweight and celltype_map
        selected_cell: Cell type name to filter (if None, shows first n_celltypes)
        n_celltypes: Number of cell types to show if no selection
    """
    if comp_data['posterior_overweight'] is None:
        st.info("ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¹ÃƒÂ¯Ã‚Â¸Ã‚Â Posterior data not available")
        return None
    
    df_over = comp_data['posterior_overweight']

    n_samples = len(df_over)
    samples_per_chain = n_samples // 4

    # Build celltype mapping for index→name conversion
    celltype_map = comp_data.get('celltype_map', None)
    idx_to_name, name_to_idx = build_celltype_mappings(celltype_map)

    # Get allowed indices for selected cell
    if selected_cell is not None:
        allowed_idx = get_allowed_indices_for_cell(selected_cell, celltype_map)
        if not allowed_idx:
            st.warning(f"Cell type '{selected_cell}' not found in mapping")
            return None
        cell_cols = [f"celltype_{i}" for i in allowed_idx if f"celltype_{i}" in df_over.columns]
    else:
        cell_cols = [c for c in df_over.columns if c.startswith('celltype_')][:n_celltypes]

    if not cell_cols:
        st.warning("No cell type columns found in posterior data")
        return None

    # Create subplot titles with actual cell names
    subplot_titles = []
    for col in cell_cols:
        idx_num = int(col.split('_')[1])
        if idx_num in idx_to_name:
            subplot_titles.append(idx_to_name[idx_num].replace('_', ' ').title())
        else:
            subplot_titles.append(f'Cell Type {idx_num}')

    n_cols = 2
    n_rows = int(np.ceil(len(cell_cols) / n_cols))

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, col in enumerate(cell_cols):
        r = idx // n_cols + 1
        c = idx % n_cols + 1

        # Get all samples and compute ranks
        all_samples = df_over[col].values
        ranks = stats.rankdata(all_samples)
        
        # Split ranks by chain
        for chain in range(4):
            start = chain * samples_per_chain
            end = (chain + 1) * samples_per_chain
            chain_ranks = ranks[start:end]
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=chain_ranks,
                    name=f'Chain {chain}',
                    marker=dict(color=colors[chain]),
                    opacity=0.6,
                    showlegend=(idx == 0),
                    legendgroup=f'chain_{chain}',
                    hovertemplate=f'Chain {chain}<br>Rank: %{{x}}<br>Count: %{{y}}<extra></extra>',
                    nbinsx=20
                ),
                row=r, col=c
            )
        
        fig.update_xaxes(title_text='Rank' if r == n_rows else '', row=r, col=c)
        fig.update_yaxes(title_text='Frequency' if c == 1 else '', row=r, col=c)
    
    # Title reflects selection
    if selected_cell:
        title = f'Rank Plots - Convergence Diagnostic ({selected_cell.replace("_", " ").title()})'
    else:
        title = f'Rank Plots - Convergence Diagnostic (First {len(cell_cols)} Cell Types)'

    fig.update_layout(
        title=title,
        height=n_rows * 300,
        template=PLOTLY_TEMPLATE,
        hovermode='closest',
        barmode='overlay'
    )

    return fig


def plot_autocorrelation(comp_data, selected_cell=None, n_celltypes=6, max_lag=40):
    """
    Generate autocorrelation plots (index-based filtering).

    Args:
        comp_data: Dictionary containing posterior_overweight and celltype_map
        selected_cell: Cell type name to filter (if None, shows first n_celltypes)
        n_celltypes: Number of cell types to show if no selection
        max_lag: Maximum lag for autocorrelation computation
    """
    if comp_data['posterior_overweight'] is None:
        st.info("ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¹ÃƒÂ¯Ã‚Â¸Ã‚Â Posterior data not available")
        return None
    
    df_over = comp_data['posterior_overweight']

    n_samples = len(df_over)
    samples_per_chain = n_samples // 4

    # Build celltype mapping for index→name conversion
    celltype_map = comp_data.get('celltype_map', None)
    idx_to_name, name_to_idx = build_celltype_mappings(celltype_map)

    # Get allowed indices for selected cell
    if selected_cell is not None:
        allowed_idx = get_allowed_indices_for_cell(selected_cell, celltype_map)
        if not allowed_idx:
            st.warning(f"Cell type '{selected_cell}' not found in mapping")
            return None
        cell_cols = [f"celltype_{i}" for i in allowed_idx if f"celltype_{i}" in df_over.columns]
    else:
        cell_cols = [c for c in df_over.columns if c.startswith('celltype_')][:n_celltypes]

    if not cell_cols:
        st.warning("No cell type columns found in posterior data")
        return None

    # Create subplot titles with actual cell names
    subplot_titles = []
    for col in cell_cols:
        idx_num = int(col.split('_')[1])
        if idx_num in idx_to_name:
            subplot_titles.append(idx_to_name[idx_num].replace('_', ' ').title())
        else:
            subplot_titles.append(f'Cell Type {idx_num}')

    n_cols = 2
    n_rows = int(np.ceil(len(cell_cols) / n_cols))

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, col in enumerate(cell_cols):
        r = idx // n_cols + 1
        c = idx % n_cols + 1

        # Compute autocorrelation for each chain
        for chain in range(4):
            start = chain * samples_per_chain
            end = (chain + 1) * samples_per_chain
            chain_data = df_over[col].iloc[start:end].values
            
            # Compute autocorrelation
            acf_values = []
            for lag in range(max_lag + 1):
                if lag == 0:
                    acf_values.append(1.0)
                else:
                    acf = np.corrcoef(chain_data[:-lag], chain_data[lag:])[0, 1]
                    acf_values.append(acf)
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(max_lag + 1)),
                    y=acf_values,
                    mode='lines+markers',
                    name=f'Chain {chain}',
                    line=dict(color=colors[chain], width=2),
                    marker=dict(size=4),
                    showlegend=(idx == 0),
                    legendgroup=f'chain_{chain}',
                    hovertemplate=f'Chain {chain}<br>Lag: %{{x}}<br>ACF: %{{y:.3f}}<extra></extra>'
                ),
                row=r, col=c
            )
        
        # Add significance bands
        sig_level = 1.96 / np.sqrt(samples_per_chain)
        fig.add_hline(y=sig_level, line_dash="dash", line_color="gray", opacity=0.5, row=r, col=c)
        fig.add_hline(y=-sig_level, line_dash="dash", line_color="gray", opacity=0.5, row=r, col=c)
        
        fig.update_xaxes(title_text='Lag' if r == n_rows else '', row=r, col=c)
        fig.update_yaxes(title_text='Autocorrelation' if c == 1 else '', row=r, col=c, range=[-0.2, 1.1])
    
    # Title reflects selection
    if selected_cell:
        title = f'Autocorrelation Plots ({selected_cell.replace("_", " ").title()})'
    else:
        title = f'Autocorrelation Plots (First {len(cell_cols)} Cell Types)'

    fig.update_layout(
        title=title,
        height=n_rows * 300,
        template=PLOTLY_TEMPLATE,
        hovermode='closest'
    )

    return fig
 
def plot_ess_rhat(comp_data):
    """Generate ESS and R-hat diagnostic plots with proper cell type names"""
    if comp_data['diagnostics'] is None:
        st.info("❌ Diagnostic summary not available")
        return None
    
    diag = comp_data['diagnostics']
    
    # Check if first column should be index
    if 'Unnamed: 0' in diag.columns:
        diag = diag.set_index('Unnamed: 0')
    elif diag.columns[0] in ['parameter', 'index', 'name']:
        diag = diag.set_index(diag.columns[0])
    
    if isinstance(diag.index, pd.RangeIndex):
        st.warning("❌ Diagnostic data doesn't have parameter names")
        return None
    
    # Convert index to string
    diag.index = diag.index.astype(str)
    
    # Filter for relevant parameters
    diag_filtered = diag[
        diag.index.str.contains('celltype_effect', na=False, case=False) |
        diag.index.str.contains('bmi_slope', na=False, case=False) |
        diag.index.str.contains('feature_bmi_slope', na=False, case=False) |
        diag.index.str.match(r'^celltype_\d+$', na=False)
    ]
    
    if len(diag_filtered) == 0:
        st.warning("❌ No cell type diagnostics found")
        return None
    
    # Load celltype mapping to get actual names
    celltype_map = comp_data.get('celltype_map', None)
    celltype_names = {}
    
    if celltype_map is not None and len(celltype_map) > 0:
        # Build mapping: index -> name
        if 'celltype_idx' in celltype_map.columns and 'celltype_name' in celltype_map.columns:
            celltype_names = dict(zip(
                celltype_map['celltype_idx'].astype(int),
                celltype_map['celltype_name']
            ))
        elif len(celltype_map.columns) >= 2:
            # Fallback: use first column as index, second as name
            celltype_names = dict(zip(
                celltype_map.iloc[:, 0].astype(int),
                celltype_map.iloc[:, 1]
            ))
    
    # Create labels with actual cell type names
    labels = []
    for idx in diag_filtered.index:
        # Extract cell number from parameter name
        import re
        match = re.search(r'\[(\d+)\]', idx)  # Matches feature_bmi_slope[0]
        if not match:
            match = re.search(r'_(\d+)$', idx)  # Matches celltype_0
        if not match:
            match = re.search(r'\[(\d+),', idx)  # Matches celltype_effect_obese[0,]
        
        if match:
            cell_idx = int(match.group(1))
            # Look up actual cell type name
            if cell_idx in celltype_names:
                cell_name = str(celltype_names[cell_idx]).replace('_', ' ').title()
                labels.append(cell_name)
            else:
                labels.append(f"Cell {cell_idx}")
        else:
            # Fallback to original parameter name
            labels.append(idx.replace('_', ' ').title())
    
    # Create figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Effective Sample Size (ESS)', 'R-hat Convergence Statistic'],
        horizontal_spacing=0.15
    )
    
    # ESS plot
    ess_col = None
    for col in ['ess_bulk', 'ess_mean', 'ess', 'n_eff']:
        if col in diag_filtered.columns:
            ess_col = col
            break
    
    if ess_col is None:
        st.warning("❌ ESS column not found")
        return None
    
    ess_bulk = diag_filtered[ess_col].values
    
    fig.add_trace(
        go.Bar(
            y=labels,
            x=ess_bulk,
            orientation='h',
            marker=dict(
                color=ess_bulk,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="ESS", x=0.45)
            ),
            hovertemplate='<b>%{y}</b><br>ESS: %{x:.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_vline(x=400, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1,
                 annotation_text="Min recommended (400)", annotation_position="top")
    
    # R-hat plot
    rhat_col = None
    for col in ['r_hat', 'rhat', 'Rhat', 'R_hat']:
        if col in diag_filtered.columns:
            rhat_col = col
            break
    
    if rhat_col is None:
        st.warning("⚠️ R-hat column not found - showing ESS only")
        fig.update_xaxes(title_text='Effective Sample Size', row=1, col=1)
        fig.update_layout(
            title='Bayesian Diagnostic Statistics (ESS only)',
            height=max(500, len(diag_filtered) * 25),
            template=PLOTLY_TEMPLATE,
            showlegend=False,
            hovermode='closest'
        )
        return fig
    
    rhat = diag_filtered[rhat_col].values
    colors = ['green' if r < 1.01 else 'orange' if r < 1.05 else 'red' for r in rhat]
    
    fig.add_trace(
        go.Bar(
            y=labels,
            x=rhat,
            orientation='h',
            marker=dict(color=colors),
            hovertemplate='<b>%{y}</b><br>R-hat: %{x:.4f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Add reference lines
    fig.add_vline(x=1.01, line_dash="dash", line_color="green", opacity=0.5, row=1, col=2,
                 annotation_text="Excellent (<1.01)", annotation_position="top")
    fig.add_vline(x=1.05, line_dash="dash", line_color="orange", opacity=0.5, row=1, col=2,
                 annotation_text="Acceptable (<1.05)", annotation_position="bottom")
    
    fig.update_xaxes(title_text='Effective Sample Size', row=1, col=1)
    fig.update_xaxes(title_text='R-hat Value', row=1, col=2, range=[0.99, max(1.1, rhat.max() * 1.05)])
    
    fig.update_layout(
        title='Bayesian Diagnostic Statistics by Cell Type',
        height=max(500, len(diag_filtered) * 25),
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig

# ==================================================================================
# ========== SEPARATE DIAGNOSTIC FUNCTIONS (CATEGORICAL vs CONTINUOUS) ============
# ==================================================================================

def plot_ess_rhat_categorical(comp_data, selected_cell=None):
    """
    ESS & R-hat for CATEGORICAL analysis (multi-comparison parameters).

    Filtering logic (name → index → filter):
    1. Extract cell name from parameter string using regex
    2. Normalize the extracted name
    3. Map to index using celltype_mapping
    4. Filter to only the selected cell's index

    Args:
        comp_data: Dictionary containing diagnostics and celltype_map
        selected_cell: Cell type name selected in sidebar (if None, shows all)
    """
    import re

    if comp_data['diagnostics'] is None:
        st.info("❌ Diagnostic summary not available")
        return None

    diag = comp_data['diagnostics'].copy()

    if 'Unnamed: 0' in diag.columns:
        diag = diag.set_index('Unnamed: 0')
    elif len(diag.columns) > 0 and diag.columns[0] in ['parameter', 'index', 'name']:
        diag = diag.set_index(diag.columns[0])

    if isinstance(diag.index, pd.RangeIndex):
        st.warning("❌ Diagnostic data doesn't have parameter names")
        return None

    diag.index = diag.index.astype(str)

    # Filter for celltype_effect parameters (categorical specific)
    diag_filtered = diag[diag.index.str.contains('celltype_effect', na=False, case=False)].copy()

    if len(diag_filtered) == 0:
        st.warning("❌ No cell type diagnostics found")
        return None

    # Build celltype mapping (single source of truth)
    celltype_map = comp_data.get('celltype_map', None)
    idx_to_name, name_to_idx = build_celltype_mappings(celltype_map)

    # Get allowed indices for selected cell
    allowed_idx = None
    if selected_cell is not None:
        allowed_idx = get_allowed_indices_for_cell(selected_cell, celltype_map)

    # Extract cell name from parameter string using regex [CELL_NAME]
    # Then normalize, map to index, and filter
    diag_filtered['cell_name'] = diag_filtered.index.to_series().str.extract(r'\[(.*?)\]', expand=False)
    diag_filtered['cell_name_norm'] = (
        diag_filtered['cell_name']
        .str.upper()
        .str.replace('_', ' ', regex=False)
        .str.strip()
    )
    diag_filtered['celltype_idx'] = diag_filtered['cell_name_norm'].map(name_to_idx)

    # Track how many rows were excluded due to missing mapping
    rows_before = len(diag_filtered)
    diag_filtered = diag_filtered.dropna(subset=['celltype_idx'])
    rows_excluded = rows_before - len(diag_filtered)

    if rows_excluded > 0:
        st.info(f"ℹ️ {rows_excluded} diagnostic rows were excluded because they are not part of the Bayesian model.")

    if len(diag_filtered) == 0:
        st.warning("❌ No diagnostics remaining after filtering")
        return None

    # Filter to selected cell's indices if specified
    if allowed_idx is not None and len(allowed_idx) > 0:
        diag_filtered = diag_filtered[diag_filtered['celltype_idx'].isin(allowed_idx)]

    if len(diag_filtered) == 0:
        st.warning(f"❌ No diagnostics found for {selected_cell}")
        return None

    # Sort by index for consistent ordering
    diag_filtered = diag_filtered.sort_values('celltype_idx')

    # Create labels from cell names
    labels = diag_filtered['cell_name'].str.replace('_', ' ').str.title().tolist()

    # Create figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Effective Sample Size (ESS)', 'R-hat Convergence'],
        horizontal_spacing=0.15
    )

    # ESS
    ess_col = next((c for c in ['ess_bulk', 'ess_mean', 'ess', 'n_eff'] if c in diag_filtered.columns), None)
    if not ess_col:
        st.warning("❌ ESS column not found")
        return None

    ess_vals = diag_filtered[ess_col].values

    fig.add_trace(go.Bar(
        y=labels, x=ess_vals, orientation='h',
        marker=dict(color=ess_vals, colorscale='Viridis', showscale=True,
                   colorbar=dict(title="ESS", x=0.45)),
        hovertemplate='<b>%{y}</b><br>ESS: %{x:.0f}<extra></extra>'
    ), row=1, col=1)

    fig.add_vline(x=400, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)

    # R-hat
    rhat_col = next((c for c in ['r_hat', 'rhat', 'Rhat', 'R_hat'] if c in diag_filtered.columns), None)
    if rhat_col:
        rhat_vals = diag_filtered[rhat_col].values
        colors = ['green' if r < 1.01 else 'orange' if r < 1.05 else 'red' for r in rhat_vals]

        fig.add_trace(go.Bar(
            y=labels, x=rhat_vals, orientation='h',
            marker=dict(color=colors),
            hovertemplate='<b>%{y}</b><br>R-hat: %{x:.4f}<extra></extra>'
        ), row=1, col=2)

        fig.add_vline(x=1.01, line_dash="dash", line_color="green", opacity=0.5, row=1, col=2)
        fig.add_vline(x=1.05, line_dash="dash", line_color="orange", opacity=0.5, row=1, col=2)
        fig.update_xaxes(title_text='R-hat Value', row=1, col=2)

    fig.update_xaxes(title_text='Effective Sample Size', row=1, col=1)

    # Title reflects selection
    title = f'Bayesian Diagnostic Statistics - {selected_cell.replace("_", " ").title()}' if selected_cell else 'Bayesian Diagnostic Statistics by Cell Type'

    fig.update_layout(
        title=title,
        height=max(500, len(diag_filtered) * 25),
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        hovermode='closest'
    )

    return fig


def get_continuous_celltype_index_map(comp_data):
    """
    Build a mapping from UPPER-CASE cell type name → integer index for continuous analysis.

    The celltype_mapping.csv in the continuous data folder only contains integer-to-integer
    mappings (not actual names). The true name→index mapping is determined by the order
    cell types first appear in continuous_results: the first N unique cell types
    (where N = number of posterior columns in posterior_bmi_slope.csv) map to indices 0..N-1.

    Returns:
        dict: {UPPER_CASE_NAME: int_index}  e.g. {'ICAF': 3, 'ACINAR': 0, ...}
    """
    continuous_results = comp_data.get('continuous_results')
    if continuous_results is None or 'cell_type' not in continuous_results.columns:
        return {}

    # Determine N = number of posterior cell types
    posterior = comp_data.get('posterior_bmi_slope')
    if posterior is not None:
        n_celltypes = len([c for c in posterior.columns if c.startswith('celltype_')])
    else:
        credible = comp_data.get('credible_intervals')
        n_celltypes = len(credible) if credible is not None else None

    # Build first-appearance order from continuous_results
    seen = []
    for ct in continuous_results['cell_type'].dropna():
        ct_str = str(ct)
        if ct_str not in seen:
            seen.append(ct_str)

    # Only the first n_celltypes have posterior estimates
    if n_celltypes is not None:
        seen = seen[:n_celltypes]

    # Return upper-case keyed dict for case-insensitive lookup
    return {name.upper(): idx for idx, name in enumerate(seen)}


def plot_ess_rhat_continuous(comp_data, selected_cell=None):
    """ESS & R-hat for CONTINUOUS analysis - single cell or all cells"""
    if comp_data['diagnostics'] is None:
        st.info("❌ Diagnostic summary not available")
        return None
    
    diag = comp_data['diagnostics']
    
    if 'Unnamed: 0' in diag.columns:
        diag = diag.set_index('Unnamed: 0')
    elif diag.columns[0] in ['parameter', 'index', 'name']:
        diag = diag.set_index(diag.columns[0])
    
    if isinstance(diag.index, pd.RangeIndex):
        st.warning("❌ Diagnostic data doesn't have parameter names")
        return None
    
    diag.index = diag.index.astype(str)
    
    # Filter for CELL-TYPE-LEVEL bmi_slope parameters only.
    # We exclude feature_bmi_slope (signature-level) because that produces
    # one bar per signature per cell — dozens of bars that make the chart
    # unreadably wide.  The cell-type-level parameter is the key convergence
    # diagnostic: one bar per cell type shows whether the overall BMI slope
    # for that cell type has converged.
    diag_filtered = diag[
        diag.index.str.startswith('celltype_bmi_slope', na=False)
        | (
            diag.index.str.contains('bmi_slope', na=False, case=False)
            & ~diag.index.str.startswith('feature_bmi_slope', na=False)
        )
    ]
    
    if len(diag_filtered) == 0:
        st.warning("❌ No BMI slope diagnostics found")
        return None
    
    # Build integer→name mapping as fallback for old-format diagnostics
    name_to_idx = get_continuous_celltype_index_map(comp_data)
    idx_to_name = {v: k for k, v in name_to_idx.items()}  # int → UPPER_CASE name

    # Normalise the selected_cell for comparison (upper, underscores→spaces)
    sel_upper = selected_cell.upper().replace('_', ' ') if selected_cell else None

    # Map parameter names to cell type labels and filter if a specific cell is selected.
    # Supports TWO diagnostics formats:
    #   NEW: celltype_bmi_slope[ACINAR]  or  feature_bmi_slope[TUMOR_EPITHELIAL||sig]
    #   OLD: celltype_bmi_slope[0]  (integer index → looked up via idx_to_name)
    import re
    labels = []
    indices_to_keep = []

    for i, param_name in enumerate(diag_filtered.index):
        match = re.search(r'\[([^\]]+)\]', param_name)   # match anything in [ ]
        if not match:
            if sel_upper is None:
                labels.append(param_name.replace('_', ' ').title())
                indices_to_keep.append(i)
            continue

        content = match.group(1)

        if content.isdigit():
            # ── OLD FORMAT: integer index ─────────────────────────────────
            cell_idx = int(content)
            if cell_idx in idx_to_name:
                cell_name_upper = idx_to_name[cell_idx].replace('_', ' ')
                cell_name_display = cell_name_upper.title()
            else:
                cell_name_upper = None
                cell_name_display = f"Cell {cell_idx}"
        else:
            # ── NEW FORMAT: name (or name||signature) in brackets ─────────
            # e.g. 'ACINAR', 'TUMOR EPITHELIAL', 'TUMOR_CLASSICAL||sig_name'
            cell_part = content.split('||')[0].strip().replace('_', ' ')
            cell_name_upper = cell_part.upper()
            cell_name_display = cell_part.title()

        # Apply cell filter
        if sel_upper is None or (cell_name_upper and cell_name_upper == sel_upper):
            labels.append(cell_name_display)
            indices_to_keep.append(i)
    
    if len(indices_to_keep) == 0:
        st.warning(f"❌ No diagnostics found for {selected_cell if selected_cell else 'any cell'}")
        return None
    
    # Filter to selected indices
    diag_filtered = diag_filtered.iloc[indices_to_keep]
    
    # Create figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Effective Sample Size (ESS)', 'R-hat Convergence'],
        horizontal_spacing=0.15
    )
    
    # ESS
    ess_col = next((c for c in ['ess_bulk', 'ess_mean', 'ess', 'n_eff'] if c in diag_filtered.columns), None)
    if not ess_col:
        st.warning("❌ ESS column not found")
        return None
    
    ess_vals = diag_filtered[ess_col].values
    
    fig.add_trace(go.Bar(
        y=labels, x=ess_vals, orientation='h',
        marker=dict(color=ess_vals, colorscale='Viridis', showscale=True,
                   colorbar=dict(title="ESS", x=0.45)),
        hovertemplate='<b>%{y}</b><br>ESS: %{x:.0f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_vline(x=400, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
    
    # R-hat
    rhat_col = next((c for c in ['r_hat', 'rhat', 'Rhat', 'R_hat'] if c in diag_filtered.columns), None)
    if rhat_col:
        rhat_vals = diag_filtered[rhat_col].values
        colors = ['green' if r < 1.01 else 'orange' if r < 1.05 else 'red' for r in rhat_vals]
        
        fig.add_trace(go.Bar(
            y=labels, x=rhat_vals, orientation='h',
            marker=dict(color=colors),
            hovertemplate='<b>%{y}</b><br>R-hat: %{x:.4f}<extra></extra>'
        ), row=1, col=2)
        
        fig.add_vline(x=1.01, line_dash="dash", line_color="green", opacity=0.5, row=1, col=2)
        fig.add_vline(x=1.05, line_dash="dash", line_color="orange", opacity=0.5, row=1, col=2)
        fig.update_xaxes(title_text='R-hat Value', row=1, col=2)
    
    fig.update_xaxes(title_text='Effective Sample Size', row=1, col=1)
    
    title = f'BMI Slope Diagnostics - {selected_cell}' if selected_cell else 'BMI Slope Diagnostics (All Cells)'
    fig.update_layout(
        title=title,
        height=max(400, len(diag_filtered) * 30),
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig

# ==================================================================================
# ================== SIGNATURE SURVIVAL PLOTTING (PLOTLY) ==========================
# ==================================================================================
 
def plot_survival_bmi_vs_time(patient_data, signature_name):
    """Plot 1: BMI vs Follow-up Time (Interactive Scatter)"""
    if 'BMI' not in patient_data.columns or patient_data['BMI'].isna().all():
        return None
    
    patient_data_bmi = patient_data[patient_data['BMI'].notna()].copy()
    
    if len(patient_data_bmi) < 10:
        return None
    
    deceased = patient_data_bmi[patient_data_bmi['vital_status_binary'] == 1]
    alive = patient_data_bmi[patient_data_bmi['vital_status_binary'] == 0]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=alive['BMI'],
        y=alive['follow_up_months'],
        mode='markers',
        name=f'Alive (n={len(alive)})',
        marker=dict(color='#1E88E5', size=8, opacity=0.6, line=dict(color='white', width=1)),
        hovertemplate='<b>Alive</b><br>BMI: %{x:.1f}<br>Follow-up: %{y:.1f} months<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=deceased['BMI'],
        y=deceased['follow_up_months'],
        mode='markers',
        name=f'Deceased (n={len(deceased)})',
        marker=dict(color='#E53935', size=8, opacity=0.6, line=dict(color='white', width=1)),
        hovertemplate='<b>Deceased</b><br>BMI: %{x:.1f}<br>Follow-up: %{y:.1f} months<extra></extra>'
    ))
    
    try:
        from scipy.stats import binned_statistic
        from scipy.ndimage import gaussian_filter1d
        bmi_bins = np.linspace(patient_data_bmi['BMI'].min(), patient_data_bmi['BMI'].max(), 15)
        means, edges, _ = binned_statistic(patient_data_bmi['BMI'], patient_data_bmi['follow_up_months'],
                                           statistic='mean', bins=bmi_bins)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        valid_idx = ~np.isnan(means)
        
        if valid_idx.sum() >= 3:
            means_smooth = gaussian_filter1d(means[valid_idx], sigma=1)
            fig.add_trace(go.Scatter(
                x=bin_centers[valid_idx],
                y=means_smooth,
                mode='lines',
                name='Mean Trend',
                line=dict(color='black', width=3, dash='dash'),
                hovertemplate='BMI: %{x:.1f}<br>Mean survival: %{y:.1f} months<extra></extra>'
            ))
    except:
        pass
    
    fig.add_vline(x=25, line_dash="dot", line_color="orange", opacity=0.5,
                 annotation_text="Overweight (25)", annotation_position="top left")
    fig.add_vline(x=30, line_dash="dot", line_color="red", opacity=0.5,
                 annotation_text="Obese (30)", annotation_position="top right")
    
    fig.update_layout(
        title=dict(text=f'{signature_name}<br>BMI vs Follow-up Time', font=dict(size=14)),
        xaxis_title='BMI',
        yaxis_title='Follow-up Time (months)',
        template=PLOTLY_TEMPLATE,
        hovermode='closest',
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

 
def plot_survival_bmi_vs_hr(patient_data, signature_name):
    """Plot 2: BMI vs Hazard Ratio (Smoothed Curve)"""
    if not LIFELINES_AVAILABLE or 'BMI' not in patient_data.columns:
        return None
    
    patient_data_bmi = patient_data[patient_data['BMI'].notna()].copy()
    
    if len(patient_data_bmi) < 30:
        return None
    
    bmi_min = patient_data_bmi['BMI'].min()
    bmi_max = patient_data_bmi['BMI'].max()
    bmi_points = np.linspace(bmi_min, bmi_max, 30)
    
    hrs, ci_lowers, ci_uppers, valid_bmis = [], [], [], []
    
    for bmi_mid in bmi_points:
        window_size = (bmi_max - bmi_min) / 5.0
        window_patients = patient_data_bmi[
            (patient_data_bmi['BMI'] >= bmi_mid - window_size/2) &
            (patient_data_bmi['BMI'] < bmi_mid + window_size/2)
        ].copy()
        
        if len(window_patients) < 10 or window_patients['vital_status_binary'].sum() < 3:
            continue
        
        try:
            cox_data = window_patients[['follow_up_months', 'vital_status_binary', 'Z']].dropna()
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_data, duration_col='follow_up_months', event_col='vital_status_binary')
            
            hr = np.exp(cph.params_['Z'])
            ci_lower = np.exp(cph.confidence_intervals_.loc['Z', '95% lower-bound'])
            ci_upper = np.exp(cph.confidence_intervals_.loc['Z', '95% upper-bound'])
            
            hrs.append(np.clip(hr, 0.1, 10))
            ci_lowers.append(np.clip(ci_lower, 0.1, 10))
            ci_uppers.append(np.clip(ci_upper, 0.1, 10))
            valid_bmis.append(bmi_mid)
        except:
            continue
    
    if len(hrs) < 3:
        return None
    
    from scipy.ndimage import gaussian_filter1d
    valid_bmis = np.array(valid_bmis)
    hrs = np.array(hrs)
    ci_lowers = np.array(ci_lowers)
    ci_uppers = np.array(ci_uppers)
    
    hrs_smooth = gaussian_filter1d(hrs, sigma=1.5)
    ci_lowers_smooth = gaussian_filter1d(ci_lowers, sigma=1.5)
    ci_uppers_smooth = gaussian_filter1d(ci_uppers, sigma=1.5)
    
    median_hr = np.median(hrs_smooth)
    color = COLOR_POSITIVE_HR if median_hr > 1 else COLOR_NEGATIVE_HR
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([valid_bmis, valid_bmis[::-1]]),
        y=np.concatenate([ci_uppers_smooth, ci_lowers_smooth[::-1]]),
        fill='toself',
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=valid_bmis,
        y=hrs_smooth,
        mode='lines',
        name='Hazard Ratio',
        line=dict(color=color, width=3),
        hovertemplate='BMI: %{x:.1f}<br>HR: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_hline(y=1, line_dash="dash", line_color="gray",
                 annotation_text="HR=1 (No Effect)", annotation_position="right")
    
    fig.add_vline(x=25, line_dash="dot", line_color="orange", opacity=0.3)
    fig.add_vline(x=30, line_dash="dot", line_color="red", opacity=0.3)
    
    fig.update_layout(
        title=dict(text=f'{signature_name}<br>BMI vs Hazard Ratio', font=dict(size=14)),
        xaxis_title='BMI',
        yaxis_title='Hazard Ratio',
        yaxis_type='log',
        template=PLOTLY_TEMPLATE,
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    return fig

 
def plot_survival_bmi_dual_axis(patient_data, signature_name):
    """Plot 3: BMI vs Time & HR (Dual-Axis)"""
    if not LIFELINES_AVAILABLE or 'BMI' not in patient_data.columns:
        return None
    
    patient_data_bmi = patient_data[patient_data['BMI'].notna()].copy()
    
    if len(patient_data_bmi) < 30:
        return None
    
    bmi_min = patient_data_bmi['BMI'].min()
    bmi_max = patient_data_bmi['BMI'].max()
    bmi_points = np.linspace(bmi_min, bmi_max, 30)
    
    hrs, valid_bmis_hr = [], []
    for bmi_mid in bmi_points:
        window_size = (bmi_max - bmi_min) / 5.0
        window_patients = patient_data_bmi[
            (patient_data_bmi['BMI'] >= bmi_mid - window_size/2) &
            (patient_data_bmi['BMI'] < bmi_mid + window_size/2)
        ].copy()
        
        if len(window_patients) < 10 or window_patients['vital_status_binary'].sum() < 3:
            continue
        
        try:
            cox_data = window_patients[['follow_up_months', 'vital_status_binary', 'Z']].dropna()
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_data, duration_col='follow_up_months', event_col='vital_status_binary')
            hr = np.exp(cph.params_['Z'])
            hrs.append(np.clip(hr, 0.1, 10))
            valid_bmis_hr.append(bmi_mid)
        except:
            continue
    
    if len(hrs) < 3:
        return None
    
    from scipy.ndimage import gaussian_filter1d
    hrs_smooth = gaussian_filter1d(np.array(hrs), sigma=1.5)
    
    time_means, time_bmis = [], []
    for bmi_mid in bmi_points:
        window_size = (bmi_max - bmi_min) / 5.0
        window_patients = patient_data_bmi[
            (patient_data_bmi['BMI'] >= bmi_mid - window_size/2) &
            (patient_data_bmi['BMI'] < bmi_mid + window_size/2)
        ]
        if len(window_patients) >= 5:
            time_means.append(window_patients['follow_up_months'].mean())
            time_bmis.append(bmi_mid)
    
    time_smooth = gaussian_filter1d(np.array(time_means), sigma=1.5)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=time_bmis,
            y=time_smooth,
            mode='lines+markers',
            name='Follow-up Time',
            line=dict(color='#1E88E5', width=3),
            marker=dict(size=5),
            hovertemplate='BMI: %{x:.1f}<br>Mean follow-up: %{y:.1f} months<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=valid_bmis_hr,
            y=hrs_smooth,
            mode='lines+markers',
            name='Hazard Ratio',
            line=dict(color='#E53935', width=3),
            marker=dict(size=5, symbol='square'),
            hovertemplate='BMI: %{x:.1f}<br>HR: %{y:.3f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.7, secondary_y=True)
    
    fig.add_vline(x=25, line_dash="dot", line_color="orange", opacity=0.3)
    fig.add_vline(x=30, line_dash="dot", line_color="red", opacity=0.3)
    
    fig.update_xaxes(title_text="BMI")
    fig.update_yaxes(title_text="Mean Follow-up Time (months)", 
                     title_font=dict(color='#1E88E5'),
                     tickfont=dict(color='#1E88E5'),
                     secondary_y=False)
    fig.update_yaxes(title_text="Hazard Ratio", 
                     title_font=dict(color='#E53935'),
                     tickfont=dict(color='#E53935'),
                     type='log',
                     secondary_y=True)
    
    fig.update_layout(
        title=dict(text=f'{signature_name}<br>BMI vs Follow-up Time & Hazard Ratio', font=dict(size=14)),
        template=PLOTLY_TEMPLATE,
        hovermode='x unified',
        height=500
    )
    
    return fig

 
def plot_survival_forest_bmi(patient_data, signature_name):
    """Plot 4: Forest Plot by BMI Category"""
    if not LIFELINES_AVAILABLE or 'BMI' not in patient_data.columns:
        return None
    
    patient_data = patient_data.copy()
    patient_data['bmi_category'] = patient_data['BMI'].apply(assign_bmi_category)
    patient_data = patient_data[patient_data['bmi_category'].notna()]
    
    if len(patient_data) < 30:
        return None
    
    results = []
    category_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
    
    for cat in category_order:
        cat_data = patient_data[patient_data['bmi_category'] == cat].copy()
        
        if len(cat_data) < 15 or cat_data['vital_status_binary'].sum() < 3:
            continue
        
        try:
            cox_data = cat_data[['follow_up_months', 'vital_status_binary', 'Z']].dropna()
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_data, duration_col='follow_up_months', event_col='vital_status_binary')
            
            hr = np.exp(cph.params_['Z'])
            ci_lower = np.exp(cph.confidence_intervals_.loc['Z', '95% lower-bound'])
            ci_upper = np.exp(cph.confidence_intervals_.loc['Z', '95% upper-bound'])
            p_value = cph.summary.loc['Z', 'p']
            
            results.append({
                'category': cat,
                'hr': hr,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value,
                'n': len(cat_data),
                'events': int(cat_data['vital_status_binary'].sum())
            })
        except:
            continue
    
    if len(results) == 0:
        return None
    
    results_df = pd.DataFrame(results)
    
    fig = go.Figure()
    
    for _, row in results_df.iterrows():
        color = BMI_COLORS_SURVIVAL.get(row['category'], 'gray')
        sig_marker = '*' if row['p_value'] < 0.05 else ''
        
        fig.add_trace(go.Scatter(
            x=[row['hr']],
            y=[row['category']],
            mode='markers',
            name=row['category'],
            marker=dict(color=color, size=15, line=dict(color='black', width=2)),
            error_x=dict(
                type='data',
                symmetric=False,
                array=[row['ci_upper'] - row['hr']],
                arrayminus=[row['hr'] - row['ci_lower']],
                color=color,
                thickness=3,
                width=10
            ),
            hovertemplate=f"<b>{row['category']}</b><br>" +
                         f"HR: {row['hr']:.3f}<br>" +
                         f"95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]<br>" +
                         f"p-value: {row['p_value']:.3e}<br>" +
                         f"n={row['n']}, events={row['events']}<br>" +
                         f"{sig_marker}<extra></extra>",
            showlegend=False
        ))
    
    fig.add_vline(x=1, line_dash="dash", line_color="gray",
                 annotation_text="HR=1 (No Effect)", annotation_position="top")
    
    fig.update_layout(
        title=dict(text=f'{signature_name}<br>Hazard Ratio by BMI Category', font=dict(size=14)),
        xaxis_title='Hazard Ratio (95% CI)',
        xaxis_type='log',
        yaxis_title='',
        template=PLOTLY_TEMPLATE,
        hovermode='closest',
        height=500,
        showlegend=False
    )
    
    return fig

 
def plot_survival_interaction_tertile(patient_data, signature_name):
    """Plot 5: BMI Ã— Signature Interaction (Tertiles)"""
    if 'BMI' not in patient_data.columns or patient_data['BMI'].isna().all():
        return None
    
    patient_data = patient_data.copy()
    patient_data['bmi_category'] = patient_data['BMI'].apply(assign_bmi_category)
    patient_data = patient_data[patient_data['bmi_category'].notna()].copy()
    
    if len(patient_data) < 30:
        return None
    
    patient_data['z_group'] = pd.qcut(patient_data['Z'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    
    category_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
    z_order = ['Low', 'Medium', 'High']
    z_colors = {'Low': '#2196F3', 'Medium': '#FF9800', 'High': '#4CAF50'}
    
    results = []
    for cat in category_order:
        for z_grp in z_order:
            subset = patient_data[
                (patient_data['bmi_category'] == cat) &
                (patient_data['z_group'] == z_grp)
            ].copy()
            
            if len(subset) < 3:
                continue
            
            results.append({
                'bmi_category': cat,
                'z_group': z_grp,
                'median_survival': subset['follow_up_months'].median(),
                'event_rate': subset['vital_status_binary'].mean() * 100,
                'n': len(subset),
                'is_confident': len(subset) >= CONFIDENCE_THRESHOLD
            })
    
    if len(results) == 0:
        return None
    
    results_df = pd.DataFrame(results)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Median Survival by BMI Ã— Signature', 'Event Rate by BMI Ã— Signature'),
        horizontal_spacing=0.12
    )
    
    for z_grp in z_order:
        subset = results_df[results_df['z_group'] == z_grp]
        if len(subset) == 0:
            continue
        
        subset = subset.set_index('bmi_category').reindex(category_order).reset_index()
        subset = subset.dropna(subset=['median_survival'])
        
        if len(subset) == 0:
            continue
        
        x_pos = [category_order.index(cat) for cat in subset['bmi_category']]
        
        line_style = 'solid'
        for i in range(len(subset)):
            if not subset.iloc[i]['is_confident']:
                line_style = 'dash'
                break
        
        fig.add_trace(
            go.Scatter(
                x=x_pos,
                y=subset['median_survival'],
                mode='lines+markers',
                name=f'{z_grp} (n={subset["n"].sum()})',
                line=dict(color=z_colors[z_grp], width=2.5, dash=line_style),
                marker=dict(size=8, line=dict(color='white', width=1)),
                hovertemplate='<b>%{fullData.name}</b><br>Category: %{text}<br>Median survival: %{y:.1f} months<extra></extra>',
                text=subset['bmi_category'],
                showlegend=True,
                legendgroup=z_grp
            ),
            row=1, col=1
        )
    
    for z_grp in z_order:
        subset = results_df[results_df['z_group'] == z_grp]
        if len(subset) == 0:
            continue
        
        subset = subset.set_index('bmi_category').reindex(category_order).reset_index()
        subset = subset.dropna(subset=['event_rate'])
        
        if len(subset) == 0:
            continue
        
        x_pos = [category_order.index(cat) for cat in subset['bmi_category']]
        
        line_style = 'solid'
        for i in range(len(subset)):
            if not subset.iloc[i]['is_confident']:
                line_style = 'dash'
                break
        
        fig.add_trace(
            go.Scatter(
                x=x_pos,
                y=subset['event_rate'],
                mode='lines+markers',
                name=f'{z_grp}',
                line=dict(color=z_colors[z_grp], width=2.5, dash=line_style),
                marker=dict(size=8, symbol='square', line=dict(color='white', width=1)),
                hovertemplate='<b>%{fullData.name}</b><br>Category: %{text}<br>Event rate: %{y:.1f}%<extra></extra>',
                text=subset['bmi_category'],
                showlegend=False,
                legendgroup=z_grp
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(ticktext=category_order, tickvals=list(range(len(category_order))),
                    title_text='BMI Category', row=1, col=1)
    fig.update_xaxes(ticktext=category_order, tickvals=list(range(len(category_order))),
                    title_text='BMI Category', row=1, col=2)
    fig.update_yaxes(title_text='Median Survival (months)', row=1, col=1)
    fig.update_yaxes(title_text='Event Rate (%)', row=1, col=2)
    
    fig.update_layout(
        title_text=f'{signature_name}<br>BMI Ã— Signature Interaction (Tertiles)',
        template=PLOTLY_TEMPLATE,
        hovermode='closest',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

 
def plot_survival_interaction_median(patient_data, signature_name):
    """Plot 6: BMI Ã— Signature Interaction (Median Split)"""
    if 'BMI' not in patient_data.columns or patient_data['BMI'].isna().all():
        return None
    
    patient_data = patient_data.copy()
    patient_data['bmi_category'] = patient_data['BMI'].apply(assign_bmi_category)
    patient_data = patient_data[patient_data['bmi_category'].notna()].copy()
    
    if len(patient_data) < 30:
        return None
    
    patient_data['z_group'] = pd.qcut(patient_data['Z'], q=2, labels=['Low', 'High'], duplicates='drop')
    
    category_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
    z_order = ['Low', 'High']
    z_colors = {'Low': '#2196F3', 'High': '#E53935'}
    
    results = []
    for cat in category_order:
        for z_grp in z_order:
            subset = patient_data[
                (patient_data['bmi_category'] == cat) &
                (patient_data['z_group'] == z_grp)
            ].copy()
            
            if len(subset) < 3:
                continue
            
            results.append({
                'bmi_category': cat,
                'z_group': z_grp,
                'median_survival': subset['follow_up_months'].median(),
                'event_rate': subset['vital_status_binary'].mean() * 100,
                'n': len(subset),
                'is_confident': len(subset) >= CONFIDENCE_THRESHOLD
            })
    
    if len(results) == 0:
        return None
    
    results_df = pd.DataFrame(results)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Median Survival by BMI Ã— Signature', 'Event Rate by BMI Ã— Signature'),
        horizontal_spacing=0.12
    )
    
    for z_grp in z_order:
        subset = results_df[results_df['z_group'] == z_grp]
        if len(subset) == 0:
            continue
        
        subset = subset.set_index('bmi_category').reindex(category_order).reset_index()
        subset = subset.dropna(subset=['median_survival'])
        
        if len(subset) == 0:
            continue
        
        x_pos = [category_order.index(cat) for cat in subset['bmi_category']]
        
        line_style = 'solid'
        for i in range(len(subset)):
            if not subset.iloc[i]['is_confident']:
                line_style = 'dash'
                break
        
        fig.add_trace(
            go.Scatter(
                x=x_pos,
                y=subset['median_survival'],
                mode='lines+markers',
                name=f'{z_grp} Sig (n={subset["n"].sum()})',
                line=dict(color=z_colors[z_grp], width=3, dash=line_style),
                marker=dict(size=10, line=dict(color='white', width=1.5)),
                hovertemplate='<b>%{fullData.name}</b><br>Category: %{text}<br>Median survival: %{y:.1f} months<extra></extra>',
                text=subset['bmi_category'],
                showlegend=True,
                legendgroup=z_grp
            ),
            row=1, col=1
        )
    
    for z_grp in z_order:
        subset = results_df[results_df['z_group'] == z_grp]
        if len(subset) == 0:
            continue
        
        subset = subset.set_index('bmi_category').reindex(category_order).reset_index()
        subset = subset.dropna(subset=['event_rate'])
        
        if len(subset) == 0:
            continue
        
        x_pos = [category_order.index(cat) for cat in subset['bmi_category']]
        
        line_style = 'solid'
        for i in range(len(subset)):
            if not subset.iloc[i]['is_confident']:
                line_style = 'dash'
                break
        
        fig.add_trace(
            go.Scatter(
                x=x_pos,
                y=subset['event_rate'],
                mode='lines+markers',
                name=f'{z_grp} Sig',
                line=dict(color=z_colors[z_grp], width=3, dash=line_style),
                marker=dict(size=10, symbol='square', line=dict(color='white', width=1.5)),
                hovertemplate='<b>%{fullData.name}</b><br>Category: %{text}<br>Event rate: %{y:.1f}%<extra></extra>',
                text=subset['bmi_category'],
                showlegend=False,
                legendgroup=z_grp
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(ticktext=category_order, tickvals=list(range(len(category_order))),
                    title_text='BMI Category', row=1, col=1)
    fig.update_xaxes(ticktext=category_order, tickvals=list(range(len(category_order))),
                    title_text='BMI Category', row=1, col=2)
    fig.update_yaxes(title_text='Median Survival (months)', row=1, col=1)
    fig.update_yaxes(title_text='Event Rate (%)', row=1, col=2)
    
    fig.update_layout(
        title_text=f'{signature_name}<br>BMI Ã— Signature Interaction (Median Split: High vs Low)',
        template=PLOTLY_TEMPLATE,
        hovermode='closest',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

 
def plot_survival_hr_with_distribution(patient_data, signature_name):
    """Plot 7: HR + Patient BMI Distribution (Dual-axis)"""
    if not LIFELINES_AVAILABLE or 'BMI' not in patient_data.columns:
        return None
    
    patient_data_bmi = patient_data[patient_data['BMI'].notna()].copy()
    
    if len(patient_data_bmi) < 30:
        return None
    
    bmi_min = patient_data_bmi['BMI'].min()
    bmi_max = patient_data_bmi['BMI'].max()
    bmi_points = np.linspace(bmi_min, bmi_max, 30)
    
    hrs, ci_lowers, ci_uppers, valid_bmis = [], [], [], []
    
    for bmi_mid in bmi_points:
        window_size = (bmi_max - bmi_min) / 5.0
        window_patients = patient_data_bmi[
            (patient_data_bmi['BMI'] >= bmi_mid - window_size/2) &
            (patient_data_bmi['BMI'] < bmi_mid + window_size/2)
        ].copy()
        
        if len(window_patients) < 10 or window_patients['vital_status_binary'].sum() < 3:
            continue
        
        try:
            cox_data = window_patients[['follow_up_months', 'vital_status_binary', 'Z']].dropna()
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_data, duration_col='follow_up_months', event_col='vital_status_binary')
            
            hr = np.exp(cph.params_['Z'])
            ci_lower = np.exp(cph.confidence_intervals_.loc['Z', '95% lower-bound'])
            ci_upper = np.exp(cph.confidence_intervals_.loc['Z', '95% upper-bound'])
            
            hrs.append(np.clip(hr, 0.1, 10))
            ci_lowers.append(np.clip(ci_lower, 0.1, 10))
            ci_uppers.append(np.clip(ci_upper, 0.1, 10))
            valid_bmis.append(bmi_mid)
        except:
            continue
    
    if len(hrs) < 3:
        return None
    
    from scipy.ndimage import gaussian_filter1d
    valid_bmis = np.array(valid_bmis)
    hrs = np.array(hrs)
    ci_lowers = np.array(ci_lowers)
    ci_uppers = np.array(ci_uppers)
    
    hrs_smooth = gaussian_filter1d(hrs, sigma=1.5)
    ci_lowers_smooth = gaussian_filter1d(ci_lowers, sigma=1.5)
    ci_uppers_smooth = gaussian_filter1d(ci_uppers, sigma=1.5)
    
    median_hr = np.median(hrs_smooth)
    color_hr = COLOR_POSITIVE_HR if median_hr > 1 else COLOR_NEGATIVE_HR
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([valid_bmis, valid_bmis[::-1]]),
            y=np.concatenate([ci_uppers_smooth, ci_lowers_smooth[::-1]]),
            fill='toself',
            fillcolor=f'rgba({int(color_hr[1:3], 16)}, {int(color_hr[3:5], 16)}, {int(color_hr[5:7], 16)}, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI',
            showlegend=True,
            hoverinfo='skip'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=valid_bmis,
            y=hrs_smooth,
            mode='lines+markers',
            name='Hazard Ratio',
            line=dict(color=color_hr, width=3.5),
            marker=dict(size=5),
            hovertemplate='BMI: %{x:.1f}<br>HR: %{y:.3f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Histogram(
            x=patient_data_bmi['BMI'],
            nbinsx=20,
            name='Patient Count',
            marker=dict(color='gray', opacity=0.3, line=dict(color='black', width=0.5)),
            hovertemplate='BMI: %{x:.1f}<br>Count: %{y}<extra></extra>'
        ),
        secondary_y=True
    )
    
    fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.7, secondary_y=False)
    
    fig.add_vline(x=25, line_dash="dot", line_color="orange", opacity=0.3)
    fig.add_vline(x=30, line_dash="dot", line_color="red", opacity=0.3)
    
    fig.update_xaxes(title_text="BMI")
    fig.update_yaxes(title_text="Hazard Ratio", 
                     title_font=dict(color=color_hr),
                     tickfont=dict(color=color_hr),
                     type='log',
                     secondary_y=False)
    fig.update_yaxes(title_text="Number of Patients", 
                     title_font=dict(color='gray'),
                     tickfont=dict(color='gray'),
                     secondary_y=True)
    
    fig.update_layout(
        title=dict(text=f'{signature_name}<br>Hazard Ratio & Patient Distribution by BMI', font=dict(size=14)),
        template=PLOTLY_TEMPLATE,
        hovermode='x unified',
        height=500,
        barmode='overlay'
    )
    
    return fig

# ==================================================================================  
# =================== SURVIVAL PLOTTING FUNCTIONS ==================================
# ==================================================================================
 
def plot_survival_bmi_vs_time(patient_data, signature_name):
    """Plot 1: BMI vs Follow-up Time"""
    if 'BMI' not in patient_data.columns or patient_data['BMI'].isna().all():
        return None
    patient_data_bmi = patient_data[patient_data['BMI'].notna()].copy()
    if len(patient_data_bmi) < 10:
        return None
    
    deceased = patient_data_bmi[patient_data_bmi['vital_status_binary'] == 1]
    alive = patient_data_bmi[patient_data_bmi['vital_status_binary'] == 0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=alive['BMI'], y=alive['follow_up_months'], mode='markers',
                             name=f'Alive (n={len(alive)})', marker=dict(color='#1E88E5', size=8, opacity=0.6)))
    fig.add_trace(go.Scatter(x=deceased['BMI'], y=deceased['follow_up_months'], mode='markers',
                             name=f'Deceased (n={len(deceased)})', marker=dict(color='#E53935', size=8, opacity=0.6)))
    
    try:
        from scipy.stats import binned_statistic
        bmi_bins = np.linspace(patient_data_bmi['BMI'].min(), patient_data_bmi['BMI'].max(), 15)
        means, edges, _ = binned_statistic(patient_data_bmi['BMI'], patient_data_bmi['follow_up_months'], 
                                           statistic='mean', bins=bmi_bins)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        valid_idx = ~np.isnan(means)
        if valid_idx.sum() >= 3:
            means_smooth = gaussian_filter1d(means[valid_idx], sigma=1)
            fig.add_trace(go.Scatter(x=bin_centers[valid_idx], y=means_smooth, mode='lines',
                                    name='Mean Trend', line=dict(color='black', width=3, dash='dash')))
    except:
        pass
    
    fig.add_vline(x=25, line_dash="dot", line_color="orange", opacity=0.5)
    fig.add_vline(x=30, line_dash="dot", line_color="red", opacity=0.5)
    fig.update_layout(title=f'{signature_name}<br>BMI vs Follow-up Time', xaxis_title='BMI',
                     yaxis_title='Follow-up Time (months)', template=PLOTLY_TEMPLATE, height=500)
    return fig
 
def plot_survival_bmi_vs_hr(patient_data, signature_name):
    """Plot 2: BMI vs Hazard Ratio"""
    if not LIFELINES_AVAILABLE or 'BMI' not in patient_data.columns:
        return None
    patient_data_bmi = patient_data[patient_data['BMI'].notna()].copy()
    if len(patient_data_bmi) < 30:
        return None
    
    bmi_min, bmi_max = patient_data_bmi['BMI'].min(), patient_data_bmi['BMI'].max()
    bmi_points = np.linspace(bmi_min, bmi_max, 30)
    hrs, ci_lowers, ci_uppers, valid_bmis = [], [], [], []
    
    for bmi_mid in bmi_points:
        window_size = (bmi_max - bmi_min) / 5.0
        window_patients = patient_data_bmi[
            (patient_data_bmi['BMI'] >= bmi_mid - window_size/2) &
            (patient_data_bmi['BMI'] < bmi_mid + window_size/2)
        ].copy()
        if len(window_patients) < 10 or window_patients['vital_status_binary'].sum() < 3:
            continue
        try:
            cox_data = window_patients[['follow_up_months', 'vital_status_binary', 'Z']].dropna()
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_data, duration_col='follow_up_months', event_col='vital_status_binary')
            hr = np.exp(cph.params_['Z'])
            ci_lower = np.exp(cph.confidence_intervals_.loc['Z', '95% lower-bound'])
            ci_upper = np.exp(cph.confidence_intervals_.loc['Z', '95% upper-bound'])
            hrs.append(np.clip(hr, 0.1, 10))
            ci_lowers.append(np.clip(ci_lower, 0.1, 10))
            ci_uppers.append(np.clip(ci_upper, 0.1, 10))
            valid_bmis.append(bmi_mid)
        except:
            continue
    
    if len(hrs) < 3:
        return None
    
    valid_bmis = np.array(valid_bmis)
    hrs_smooth = gaussian_filter1d(np.array(hrs), sigma=1.5)
    ci_lowers_smooth = gaussian_filter1d(np.array(ci_lowers), sigma=1.5)
    ci_uppers_smooth = gaussian_filter1d(np.array(ci_uppers), sigma=1.5)
    
    color = COLOR_POSITIVE_HR if np.median(hrs_smooth) > 1 else COLOR_NEGATIVE_HR
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([valid_bmis, valid_bmis[::-1]]),
        y=np.concatenate([ci_uppers_smooth, ci_lowers_smooth[::-1]]),
        fill='toself', fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
        line=dict(color='rgba(255,255,255,0)'), name='95% CI', showlegend=True, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(x=valid_bmis, y=hrs_smooth, mode='lines', name='Hazard Ratio',
                            line=dict(color=color, width=3)))
    fig.add_hline(y=1, line_dash="dash", line_color="gray")
    fig.update_layout(title=f'{signature_name}<br>BMI vs Hazard Ratio', xaxis_title='BMI',
                     yaxis_title='Hazard Ratio', yaxis_type='log', template=PLOTLY_TEMPLATE, height=500)
    return fig

 
def plot_survival_bmi_dual_axis(patient_data, signature_name):
    """Plot 3: Dual-axis (Time & HR)"""
    if not LIFELINES_AVAILABLE or 'BMI' not in patient_data.columns:
        return None
    patient_data_bmi = patient_data[patient_data['BMI'].notna()].copy()
    if len(patient_data_bmi) < 30:
        return None
    
    bmi_min, bmi_max = patient_data_bmi['BMI'].min(), patient_data_bmi['BMI'].max()
    bmi_points = np.linspace(bmi_min, bmi_max, 30)
    
    # Calculate HRs
    hrs, valid_bmis_hr = [], []
    for bmi_mid in bmi_points:
        window_size = (bmi_max - bmi_min) / 5.0
        window_patients = patient_data_bmi[
            (patient_data_bmi['BMI'] >= bmi_mid - window_size/2) &
            (patient_data_bmi['BMI'] < bmi_mid + window_size/2)
        ].copy()
        if len(window_patients) < 10 or window_patients['vital_status_binary'].sum() < 3:
            continue
        try:
            cox_data = window_patients[['follow_up_months', 'vital_status_binary', 'Z']].dropna()
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_data, duration_col='follow_up_months', event_col='vital_status_binary')
            hrs.append(np.clip(np.exp(cph.params_['Z']), 0.1, 10))
            valid_bmis_hr.append(bmi_mid)
        except:
            continue
    
    if len(hrs) < 3:
        return None
    
    hrs_smooth = gaussian_filter1d(np.array(hrs), sigma=1.5)
    
    # Calculate follow-up times
    time_means, time_bmis = [], []
    for bmi_mid in bmi_points:
        window_size = (bmi_max - bmi_min) / 5.0
        window_patients = patient_data_bmi[
            (patient_data_bmi['BMI'] >= bmi_mid - window_size/2) &
            (patient_data_bmi['BMI'] < bmi_mid + window_size/2)
        ]
        if len(window_patients) >= 5:
            time_means.append(window_patients['follow_up_months'].mean())
            time_bmis.append(bmi_mid)
    
    time_smooth = gaussian_filter1d(np.array(time_means), sigma=1.5)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=time_bmis, y=time_smooth, mode='lines+markers', name='Follow-up Time',
                            line=dict(color='#1E88E5', width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=valid_bmis_hr, y=hrs_smooth, mode='lines+markers', name='Hazard Ratio',
                            line=dict(color='#E53935', width=3)), secondary_y=True)
    
    fig.update_xaxes(title_text="BMI")
    fig.update_yaxes(title_text="Follow-up Time (months)", secondary_y=False)
    fig.update_yaxes(title_text="Hazard Ratio", type='log', secondary_y=True)
    fig.update_layout(title=f'{signature_name}<br>Dual-Axis: Time & HR', template=PLOTLY_TEMPLATE, height=500)
    return fig
 
def plot_survival_forest_bmi(patient_data, signature_name):
    """Plot 4: Forest plot by BMI category"""
    if not LIFELINES_AVAILABLE or 'BMI' not in patient_data.columns:
        return None
    patient_data = patient_data.copy()
    patient_data['bmi_category'] = patient_data['BMI'].apply(assign_bmi_category)
    patient_data = patient_data[patient_data['bmi_category'].notna()]
    if len(patient_data) < 30:
        return None
    
    results = []
    for cat in ['Underweight', 'Normal', 'Overweight', 'Obese']:
        cat_data = patient_data[patient_data['bmi_category'] == cat].copy()
        if len(cat_data) < 15 or cat_data['vital_status_binary'].sum() < 3:
            continue
        try:
            cox_data = cat_data[['follow_up_months', 'vital_status_binary', 'Z']].dropna()
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_data, duration_col='follow_up_months', event_col='vital_status_binary')
            hr = np.exp(cph.params_['Z'])
            ci_lower = np.exp(cph.confidence_intervals_.loc['Z', '95% lower-bound'])
            ci_upper = np.exp(cph.confidence_intervals_.loc['Z', '95% upper-bound'])
            p_value = cph.summary.loc['Z', 'p']
            results.append({'category': cat, 'hr': hr, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
                          'p_value': p_value, 'n': len(cat_data), 'events': int(cat_data['vital_status_binary'].sum())})
        except:
            continue
    
    if len(results) == 0:
        return None
    
    results_df = pd.DataFrame(results)
    fig = go.Figure()
    for _, row in results_df.iterrows():
        color = BMI_COLORS_SURVIVAL.get(row['category'], 'gray')
        fig.add_trace(go.Scatter(
            x=[row['hr']], y=[row['category']], mode='markers', name=row['category'],
            marker=dict(color=color, size=15),
            error_x=dict(type='data', symmetric=False, array=[row['ci_upper']-row['hr']],
                        arrayminus=[row['hr']-row['ci_lower']], color=color, thickness=3),
            showlegend=False
        ))
    
    fig.add_vline(x=1, line_dash="dash", line_color="gray")
    fig.update_layout(title=f'{signature_name}<br>Forest Plot by BMI', xaxis_title='Hazard Ratio',
                     xaxis_type='log', template=PLOTLY_TEMPLATE, height=500)
    return fig
 
def plot_survival_interaction_tertile(patient_data, signature_name):
    """Plot 5: BMI Ã— Signature (Tertiles)"""
    if 'BMI' not in patient_data.columns:
        return None
    patient_data = patient_data.copy()
    patient_data['bmi_category'] = patient_data['BMI'].apply(assign_bmi_category)
    patient_data = patient_data[patient_data['bmi_category'].notna()].copy()
    if len(patient_data) < 30:
        return None
    
    patient_data['z_group'] = pd.qcut(patient_data['Z'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    
    results = []
    for cat in ['Underweight', 'Normal', 'Overweight', 'Obese']:
        for z_grp in ['Low', 'Medium', 'High']:
            subset = patient_data[(patient_data['bmi_category']==cat) & (patient_data['z_group']==z_grp)].copy()
            if len(subset) >= 3:
                results.append({'bmi_category': cat, 'z_group': z_grp,
                              'median_survival': subset['follow_up_months'].median(),
                              'event_rate': subset['vital_status_binary'].mean() * 100,
                              'n': len(subset)})
    
    if len(results) == 0:
        return None
    
    results_df = pd.DataFrame(results)
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Median Survival', 'Event Rate'))
    
    z_colors = {'Low': '#2196F3', 'Medium': '#FF9800', 'High': '#4CAF50'}
    category_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
    
    for z_grp in ['Low', 'Medium', 'High']:
        subset = results_df[results_df['z_group'] == z_grp]
        if len(subset) == 0:
            continue
        subset = subset.set_index('bmi_category').reindex(category_order).reset_index().dropna(subset=['median_survival'])
        x_pos = [category_order.index(cat) for cat in subset['bmi_category']]
        fig.add_trace(go.Scatter(x=x_pos, y=subset['median_survival'], mode='lines+markers', name=z_grp,
                                line=dict(color=z_colors[z_grp], width=2.5)), row=1, col=1)
        subset2 = results_df[results_df['z_group'] == z_grp].set_index('bmi_category').reindex(category_order).reset_index().dropna(subset=['event_rate'])
        x_pos2 = [category_order.index(cat) for cat in subset2['bmi_category']]
        fig.add_trace(go.Scatter(x=x_pos2, y=subset2['event_rate'], mode='lines+markers', name=z_grp,
                                line=dict(color=z_colors[z_grp], width=2.5), showlegend=False), row=1, col=2)
    
    fig.update_xaxes(ticktext=category_order, tickvals=list(range(4)), row=1, col=1)
    fig.update_xaxes(ticktext=category_order, tickvals=list(range(4)), row=1, col=2)
    fig.update_layout(title=f'{signature_name}<br>Tertile Interaction', template=PLOTLY_TEMPLATE, height=500)
    return fig
 
def plot_survival_interaction_median(patient_data, signature_name):
    """Plot 6: BMI Ã— Signature (Median Split)"""
    if 'BMI' not in patient_data.columns:
        return None
    patient_data = patient_data.copy()
    patient_data['bmi_category'] = patient_data['BMI'].apply(assign_bmi_category)
    patient_data = patient_data[patient_data['bmi_category'].notna()].copy()
    if len(patient_data) < 30:
        return None
    
    patient_data['z_group'] = pd.qcut(patient_data['Z'], q=2, labels=['Low', 'High'], duplicates='drop')
    
    results = []
    for cat in ['Underweight', 'Normal', 'Overweight', 'Obese']:
        for z_grp in ['Low', 'High']:
            subset = patient_data[(patient_data['bmi_category']==cat) & (patient_data['z_group']==z_grp)].copy()
            if len(subset) >= 3:
                results.append({'bmi_category': cat, 'z_group': z_grp,
                              'median_survival': subset['follow_up_months'].median(),
                              'event_rate': subset['vital_status_binary'].mean() * 100,
                              'n': len(subset)})
    
    if len(results) == 0:
        return None
    
    results_df = pd.DataFrame(results)
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Median Survival', 'Event Rate'))
    
    z_colors = {'Low': '#2196F3', 'High': '#E53935'}
    category_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
    
    for z_grp in ['Low', 'High']:
        subset = results_df[results_df['z_group'] == z_grp]
        if len(subset) == 0:
            continue
        subset = subset.set_index('bmi_category').reindex(category_order).reset_index().dropna(subset=['median_survival'])
        x_pos = [category_order.index(cat) for cat in subset['bmi_category']]
        fig.add_trace(go.Scatter(x=x_pos, y=subset['median_survival'], mode='lines+markers', name=f'{z_grp} Sig',
                                line=dict(color=z_colors[z_grp], width=3)), row=1, col=1)
        subset2 = results_df[results_df['z_group'] == z_grp].set_index('bmi_category').reindex(category_order).reset_index().dropna(subset=['event_rate'])
        x_pos2 = [category_order.index(cat) for cat in subset2['bmi_category']]
        fig.add_trace(go.Scatter(x=x_pos2, y=subset2['event_rate'], mode='lines+markers', name=f'{z_grp} Sig',
                                line=dict(color=z_colors[z_grp], width=3), showlegend=False), row=1, col=2)
    
    fig.update_xaxes(ticktext=category_order, tickvals=list(range(4)), row=1, col=1)
    fig.update_xaxes(ticktext=category_order, tickvals=list(range(4)), row=1, col=2)
    fig.update_layout(title=f'{signature_name}<br>Median Split', template=PLOTLY_TEMPLATE, height=500)
    return fig
 
def plot_survival_hr_with_distribution(patient_data, signature_name):
    """Plot 7: HR + Patient Distribution"""
    if not LIFELINES_AVAILABLE or 'BMI' not in patient_data.columns:
        return None
    patient_data_bmi = patient_data[patient_data['BMI'].notna()].copy()
    if len(patient_data_bmi) < 30:
        return None
    
    bmi_min, bmi_max = patient_data_bmi['BMI'].min(), patient_data_bmi['BMI'].max()
    bmi_points = np.linspace(bmi_min, bmi_max, 30)
    hrs, valid_bmis = [], []
    
    for bmi_mid in bmi_points:
        window_size = (bmi_max - bmi_min) / 5.0
        window_patients = patient_data_bmi[
            (patient_data_bmi['BMI'] >= bmi_mid - window_size/2) &
            (patient_data_bmi['BMI'] < bmi_mid + window_size/2)
        ].copy()
        if len(window_patients) < 10 or window_patients['vital_status_binary'].sum() < 3:
            continue
        try:
            cox_data = window_patients[['follow_up_months', 'vital_status_binary', 'Z']].dropna()
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_data, duration_col='follow_up_months', event_col='vital_status_binary')
            hrs.append(np.clip(np.exp(cph.params_['Z']), 0.1, 10))
            valid_bmis.append(bmi_mid)
        except:
            continue
    
    if len(hrs) < 3:
        return None
    
    hrs_smooth = gaussian_filter1d(np.array(hrs), sigma=1.5)
    color = COLOR_POSITIVE_HR if np.median(hrs_smooth) > 1 else COLOR_NEGATIVE_HR
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=valid_bmis, y=hrs_smooth, mode='lines+markers', name='Hazard Ratio',
                            line=dict(color=color, width=3.5)), secondary_y=False)
    fig.add_trace(go.Histogram(x=patient_data_bmi['BMI'], nbinsx=20, name='Patient Count',
                              marker=dict(color='gray', opacity=0.3)), secondary_y=True)
    fig.add_hline(y=1, line_dash="dash", line_color="gray", secondary_y=False)
    fig.update_xaxes(title_text="BMI")
    fig.update_yaxes(title_text="Hazard Ratio", type='log', secondary_y=False)
    fig.update_yaxes(title_text="Patient Count", secondary_y=True)
    fig.update_layout(title=f'{signature_name}<br>HR + Distribution', template=PLOTLY_TEMPLATE, height=500)
    return fig

 
def plot_gene_survival_interactive(genes, clinical, tpm):
    """Generate interactive gene-level survival forest plot"""
    if not LIFELINES_AVAILABLE or tpm is None:
        st.info("ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¹ÃƒÂ¯Ã‚Â¸Ã‚Â Survival analysis not available")
        return None
    
    tpm_t = tpm.T
    tpm_t.index.name = 'sample_id'
    tpm_t = tpm_t.reset_index()
    
    merged = clinical.merge(tpm_t, on='sample_id', how='inner')
    merged = merged[merged['follow_up_months'] > 0].copy()
    
    results = []
    for gene in genes:
        if gene not in merged.columns:
            continue
        
        surv_data = merged[['follow_up_months', 'vital_status_binary', gene]].copy()
        surv_data = surv_data.dropna()
        
        if len(surv_data) < 20 or surv_data['vital_status_binary'].sum() < 3:
            continue
        
        if surv_data[gene].std() < 1e-6:
            continue
        
        surv_data['expression_std'] = (surv_data[gene] - surv_data[gene].mean()) / surv_data[gene].std()
        
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(surv_data[['follow_up_months', 'vital_status_binary', 'expression_std']],
                   duration_col='follow_up_months',
                   event_col='vital_status_binary')
            
            hr = np.exp(cph.params_['expression_std'])
            ci_lower = np.exp(cph.confidence_intervals_.loc['expression_std', '95% lower-bound'])
            ci_upper = np.exp(cph.confidence_intervals_.loc['expression_std', '95% upper-bound'])
            p_value = cph.summary.loc['expression_std', 'p']
            
            if hr < 1e6 and ci_upper < 1e6:
                results.append({
                    'gene': gene,
                    'n_patients': len(surv_data),
                    'n_events': int(surv_data['vital_status_binary'].sum()),
                    'hr': hr,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'p_value': p_value
                })
        except:
            continue
    
    if not results:
        st.info("ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¹ÃƒÂ¯Ã‚Â¸Ã‚Â No genes passed survival criteria")
        return None
    
    results_df = pd.DataFrame(results).sort_values('hr', ascending=False).head(20)
    
    # Create interactive forest plot
    fig = go.Figure()
    
    colors = ['#E74C3C' if p < 0.05 else '#95A5A6' for p in results_df['p_value']]
    
    # Add HR points
    fig.add_trace(go.Scatter(
        x=results_df['hr'],
        y=results_df['gene'],
        mode='markers',
        marker=dict(color=colors, size=12, line=dict(color='black', width=1.5)),
        name='Hazard Ratio',
        hovertemplate='<b>%{y}</b><br>HR: %{x:.3f}<br>CI: %{customdata[0]:.2f} - %{customdata[1]:.2f}<br>p-value: %{customdata[2]:.3e}<br>N: %{customdata[3]}<extra></extra>',
        customdata=np.column_stack((results_df['ci_lower'], results_df['ci_upper'], 
                                   results_df['p_value'], results_df['n_patients']))
    ))
    
    # Add confidence intervals
    for _, row in results_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['ci_lower'], row['ci_upper']],
            y=[row['gene'], row['gene']],
            mode='lines',
            line=dict(color='#E74C3C' if row['p_value'] < 0.05 else '#95A5A6', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add HR=1 reference line
    fig.add_vline(x=1, line_dash="dash", line_color="black", line_width=2, opacity=0.7,
                 annotation_text="No effect (HR=1)", annotation_position="top")
    
    fig.update_layout(
        title='Gene-Level Survival Analysis<br>(Cox Proportional Hazards)',
        xaxis_title='Hazard Ratio (95% CI)',
        yaxis_title='Genes',
        height=max(500, len(results_df) * 30),
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        hovermode='closest'
    )
    
    fig.update_xaxes(type='log' if results_df['hr'].max() > 5 else 'linear')
    
    return fig

# ==================================================================================
# ================= CONTINUOUS BMI ANALYSIS PLOTTING ===============================
# ==================================================================================

def plot_continuous_cell_heatmap(selected_cell, comp_data):
    """Cell-specific HEATMAP showing signatures for selected cell with credibility markers"""
    if comp_data['continuous_results'] is None:
        st.warning("❌ Continuous results not available")
        return None
    
    results = comp_data['continuous_results'].copy()
    
    # Parse features — normalise underscores→spaces in cell type so the parsed
    # value matches the cell_type column (e.g. TUMOR_EPITHELIAL → TUMOR EPITHELIAL)
    def parse_feature(feature):
        if "||" in str(feature):
            cell_type, signature = str(feature).split("||", 1)
            return cell_type.strip().replace('_', ' '), signature.strip()
        return "Unknown", str(feature)
    
    results['cell_type_parsed'], results['signature'] = zip(*results['feature'].apply(parse_feature))
    
    # Filter for selected cell only
    cell_results = results[results['cell_type_parsed'].str.upper() == selected_cell.upper()].copy()
    
    if len(cell_results) == 0:
        st.warning(f"❌ No data for {selected_cell}")
        return None
    
    # Assign priorities
    cell_results['is_hdi'] = cell_results.get('bmi_slope_credible', False).fillna(False)
    cell_results['prob_01'] = cell_results.get('bmi_slope_prob_gt_0.1', 0.0).fillna(0.0)
    cell_results['prob_02'] = cell_results.get('bmi_slope_prob_gt_0.2', 0.0).fillna(0.0)
    
    def assign_priority(row):
        if row['is_hdi'] and row['prob_02'] > 0.95:
            return 1  # ★★
        elif row['is_hdi'] and row['prob_01'] > 0.95:
            return 2  # ★
        elif row['is_hdi']:
            return 3  # ○
        else:
            return 4  # No marker
    
    cell_results['priority'] = cell_results.apply(assign_priority, axis=1)
    
    # Sort by priority and absolute slope
    cell_results['abs_slope'] = cell_results['bmi_slope_standardized_mean'].abs()
    cell_results = cell_results.sort_values('abs_slope', ascending=False)
    
    # Create single-column heatmap data (signatures × 1 column)
    slopes = cell_results['bmi_slope_standardized_mean'].values.reshape(-1, 1)
    signatures = cell_results['signature'].values
    priorities = cell_results['priority'].values
    
    # Create heatmap
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=slopes,
        x=['BMI Slope'],  # Single column
        y=signatures,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-slopes.max() if slopes.max() > 0 else -1,
        zmax=slopes.max() if slopes.max() > 0 else 1,
        colorbar=dict(
            title="BMI Slope",
            thickness=20,
            len=0.7
        ),
        hovertemplate='<b>%{y}</b><br>Slope: %{z:.4f}<extra></extra>',
        showscale=True
    ))
    
    # Add credibility markers as annotations
    annotations = []
    for i, (sig, slope_val, pri) in enumerate(zip(signatures, slopes.flatten(), priorities)):
        if pri >= 4:
            continue  # Skip non-credible
        
        # Determine marker
        if pri == 1:
            text = "★★"
            size = 16
        elif pri == 2:
            text = "★"
            size = 18
        elif pri == 3:
            text = "○"
            size = 16
        else:
            continue
        
        # Determine color based on background
        if abs(slope_val) > 0.1:
            marker_color = 'white'
        else:
            marker_color = 'black'
        
        annotations.append(
            dict(
                x='BMI Slope',
                y=sig,
                text=text,
                showarrow=False,
                font=dict(size=size, color=marker_color, family='Arial'),
                xref='x',
                yref='y'
            )
        )
    
    fig.update_layout(
        annotations=annotations,
        title=dict(
            text=f'{selected_cell} - BMI Slope per Signature<br><sub>★★ HDI+ROPE>0.2 | ★ HDI+ROPE>0.1 | ○ HDI only</sub>',
            font=dict(size=16, color='#2c3e50')
        ),
        xaxis_title='',
        yaxis_title='Signature',
        height=max(600, len(cell_results) * 20),
        width=600,
        template=PLOTLY_TEMPLATE,
        hovermode='closest',
        xaxis=dict(side='top'),
        yaxis=dict(autorange='reversed')  # Top to bottom ordering
    )
    
    # Add grid lines for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig


def plot_trace_continuous(comp_data, selected_cell):
    """Trace plot for selected cell in continuous analysis"""
    if comp_data['posterior_bmi_slope'] is None:
        st.info("❌ Posterior data not available")
        return None

    df_slope = comp_data['posterior_bmi_slope']

    # Find cell index using first-appearance order in continuous_results
    # (celltype_map in the continuous folder only has integer indices, not names)
    name_to_idx = get_continuous_celltype_index_map(comp_data)
    cell_idx = name_to_idx.get(selected_cell.upper())

    if cell_idx is None:
        st.warning(f"❌ Cell index not found for {selected_cell}")
        return None
    
    col_name = f'celltype_{cell_idx}'
    if col_name not in df_slope.columns:
        st.warning(f"❌ Column {col_name} not found")
        return None
    
    samples = df_slope[col_name].values
    n_samples = len(samples)
    samples_per_chain = n_samples // 4
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for chain in range(4):
        start = chain * samples_per_chain
        end = (chain + 1) * samples_per_chain
        chain_data = samples[start:end]
        
        fig.add_trace(go.Scatter(
            x=np.arange(len(chain_data)),
            y=chain_data,
            mode='lines',
            name=f'Chain {chain}',
            line=dict(color=colors[chain], width=1),
            opacity=0.7,
            hovertemplate=f'Chain {chain}<br>Iteration: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'Trace Plot - {selected_cell}',
        xaxis_title='Iteration',
        yaxis_title='BMI Slope',
        height=400,
        template=PLOTLY_TEMPLATE,
        hovermode='closest'
    )
    
    return fig


def plot_rank_continuous(comp_data, selected_cell):
    """Rank plot for selected cell"""
    if comp_data['posterior_bmi_slope'] is None:
        st.info("❌ Posterior data not available")
        return None

    df_slope = comp_data['posterior_bmi_slope']

    # Find cell index using first-appearance order in continuous_results
    name_to_idx = get_continuous_celltype_index_map(comp_data)
    cell_idx = name_to_idx.get(selected_cell.upper())

    if cell_idx is None:
        st.warning(f"❌ Cell index not found for {selected_cell}")
        return None
    
    col_name = f'celltype_{cell_idx}'
    if col_name not in df_slope.columns:
        st.warning(f"❌ Column {col_name} not found")
        return None
    
    samples = df_slope[col_name].values
    ranks = stats.rankdata(samples)
    
    n_samples = len(samples)
    samples_per_chain = n_samples // 4
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for chain in range(4):
        start = chain * samples_per_chain
        end = (chain + 1) * samples_per_chain
        chain_ranks = ranks[start:end]
        
        fig.add_trace(go.Histogram(
            x=chain_ranks,
            name=f'Chain {chain}',
            marker=dict(color=colors[chain]),
            opacity=0.6,
            hovertemplate=f'Chain {chain}<br>Rank: %{{x}}<br>Count: %{{y}}<extra></extra>',
            nbinsx=20
        ))
    
    fig.update_layout(
        title=f'Rank Plot - {selected_cell}',
        xaxis_title='Rank',
        yaxis_title='Frequency',
        height=400,
        template=PLOTLY_TEMPLATE,
        hovermode='closest',
        barmode='overlay'
    )
    
    return fig


def plot_autocorrelation_continuous(comp_data, selected_cell, max_lag=40):
    """Autocorrelation plot for selected cell"""
    if comp_data['posterior_bmi_slope'] is None:
        st.info("❌ Posterior data not available")
        return None

    df_slope = comp_data['posterior_bmi_slope']

    # Find cell index using first-appearance order in continuous_results
    name_to_idx = get_continuous_celltype_index_map(comp_data)
    cell_idx = name_to_idx.get(selected_cell.upper())

    if cell_idx is None:
        st.warning(f"❌ Cell index not found for {selected_cell}")
        return None
    
    col_name = f'celltype_{cell_idx}'
    if col_name not in df_slope.columns:
        st.warning(f"❌ Column {col_name} not found")
        return None
    
    samples = df_slope[col_name].values
    n_samples = len(samples)
    samples_per_chain = n_samples // 4
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for chain in range(4):
        start = chain * samples_per_chain
        end = (chain + 1) * samples_per_chain
        chain_data = samples[start:end]
        
        acf_values = []
        for lag in range(max_lag + 1):
            if lag == 0:
                acf_values.append(1.0)
            else:
                acf = np.corrcoef(chain_data[:-lag], chain_data[lag:])[0, 1]
                acf_values.append(acf)
        
        fig.add_trace(go.Scatter(
            x=list(range(max_lag + 1)),
            y=acf_values,
            mode='lines+markers',
            name=f'Chain {chain}',
            line=dict(color=colors[chain], width=2),
            marker=dict(size=4),
            hovertemplate=f'Chain {chain}<br>Lag: %{{x}}<br>ACF: %{{y:.3f}}<extra></extra>'
        ))
    
    # Significance bands
    sig_level = 1.96 / np.sqrt(samples_per_chain)
    fig.add_hline(y=sig_level, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=-sig_level, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f'Autocorrelation - {selected_cell}',
        xaxis_title='Lag',
        yaxis_title='Autocorrelation',
        yaxis_range=[-0.2, 1.1],
        height=400,
        template=PLOTLY_TEMPLATE,
        hovermode='closest'
    )
    
    return fig


def plot_continuous_ridge_plot(selected_cell, comp_data):
    """Plot ridge plot for SINGLE CELL TYPE in continuous analysis"""
    if comp_data['posterior_bmi_slope'] is None:
        st.info("❌ Posterior BMI slope data not available")
        return None
    
    try:
        df_slope = comp_data['posterior_bmi_slope']

        # Find the cell index using first-appearance order in continuous_results
        # (celltype_map in the continuous folder only has integer indices, not names)
        name_to_idx = get_continuous_celltype_index_map(comp_data)
        cell_idx = name_to_idx.get(selected_cell.upper())

        if cell_idx is None:
            st.warning(f"❌ Could not find index for {selected_cell}")
            return None
        
        col_name = f'celltype_{cell_idx}'
        
        if col_name not in df_slope.columns:
            st.warning(f"❌ Column {col_name} not found in posterior data")
            return None
        
        # Extract samples for this cell
        samples = df_slope[col_name].values
        
        # Create single ridge plot
        from scipy.stats import gaussian_kde
        
        fig = go.Figure()
        
        KDE_POINTS = 200
        x_min, x_max = np.percentile(samples, [0.5, 99.5])
        x_span = max(1e-6, x_max - x_min)
        xgrid = np.linspace(x_min - 0.03*x_span, x_max + 0.03*x_span, KDE_POINTS)
        
        try:
            kde = gaussian_kde(samples)
            density = kde(xgrid)
        except:
            density = np.zeros_like(xgrid)
        
        mean_slope = samples.mean()
        color = '#E53935' if mean_slope > 0 else '#1E88E5'
        fill_color = 'rgba(229, 57, 53, 0.3)' if mean_slope > 0 else 'rgba(30, 136, 229, 0.3)'
        
        fig.add_trace(go.Scatter(
            x=xgrid,
            y=density,
            fill='tozeroy',
            fillcolor=fill_color,
            line=dict(color=color, width=2),
            hovertemplate=f'<b>{selected_cell}</b><br>BMI Slope: %{{x:.3f}}<extra></extra>',
            name=selected_cell
        ))
        
        # Add mean marker
        fig.add_vline(x=mean_slope, line_dash="dash", line_color="black", line_width=2,
                     annotation_text=f"Mean: {mean_slope:.3f}", annotation_position="top")
        
        # Add zero reference
        fig.add_vline(x=0, line_dash="dash", line_color="darkred", line_width=2, opacity=0.7)
        
        fig.update_layout(
            title=f'Posterior BMI Slope Distribution - {selected_cell}',
            xaxis_title='BMI Slope (Standardized)',
            yaxis_title='Density',
            height=400,
            template=PLOTLY_TEMPLATE,
            hovermode='closest',
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"❌ Error creating ridge plot: {e}")
        return None

def plot_continuous_slope_heatmap(compartment, comp_data):
    """
    Compartment-level heatmap: All signatures × All cell types
    Marks credible features: ★★ (ROPE 0.2), ★ (ROPE 0.1), ○ (HDI only)
    """
    if comp_data['continuous_results'] is None:
        st.warning("❌ Continuous results not available")
        return None
    
    results = comp_data['continuous_results'].copy()
    
    # Parse feature to get cell type and signature — normalise underscores→spaces
    def parse_feature(feature):
        if "||" in str(feature):
            cell_type, signature = str(feature).split("||", 1)
            return cell_type.strip().replace('_', ' '), signature.strip()
        return "Unknown", str(feature)
    
    results['cell_type_parsed'], results['signature'] = zip(*results['feature'].apply(parse_feature))
    
    # Determine credibility priority
    results['is_hdi'] = results.get('bmi_slope_credible', False).fillna(False)
    results['prob_01'] = results.get('bmi_slope_prob_gt_0.1', 0.0).fillna(0.0)
    results['prob_02'] = results.get('bmi_slope_prob_gt_0.2', 0.0).fillna(0.0)
    
    # Priority: 1 = ★★, 2 = ★, 3 = ○, 4 = no marker
    def assign_priority(row):
        if row['is_hdi'] and row['prob_02'] > 0.95:
            return 1  # ★★
        elif row['is_hdi'] and row['prob_01'] > 0.95:
            return 2  # ★
        elif row['is_hdi']:
            return 3  # ○
        else:
            return 4  # No marker
    
    results['priority'] = results.apply(assign_priority, axis=1)
    
    # Filter to top credible features (limit display size)
    credible = results[results['priority'] <= 3].copy()
    
    if len(credible) == 0:
        st.warning("❌ No credible features found")
        return None
    
    # Sort and limit
    credible = credible.sort_values(['priority', 'prob_02'], ascending=[True, False])
    max_features = 50  # Limit for readability
    credible = credible.head(max_features)
    
    # Get all data for these signatures (including non-credible cell types for context)
    selected_sigs = credible['signature'].unique()
    display_data = results[results['signature'].isin(selected_sigs)].copy()
    
    # Create pivot tables
    pivot_slope = display_data.pivot_table(
        index='signature',
        columns='cell_type_parsed',
        values='bmi_slope_standardized_mean',
        aggfunc='first'
    )
    
    pivot_priority = display_data.pivot_table(
        index='signature',
        columns='cell_type_parsed',
        values='priority',
        aggfunc='first'
    ).fillna(4)
    
    if pivot_slope.empty:
        st.warning("❌ No data to display")
        return None
    
    # Sort rows and columns
    row_order = credible.drop_duplicates('signature').sort_values('priority')['signature'].tolist()
    pivot_slope = pivot_slope.reindex(row_order)
    pivot_priority = pivot_priority.reindex(row_order)
    
    col_order = sorted(pivot_slope.columns)
    pivot_slope = pivot_slope[col_order]
    pivot_priority = pivot_priority[col_order]
    
    # Create figure
    n_rows, n_cols = pivot_slope.shape
    
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=pivot_slope.values,
        x=pivot_slope.columns,
        y=pivot_slope.index,
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title="BMI Slope<br>(Std)"),
        hovertemplate='<b>%{y}</b><br>Cell: %{x}<br>Slope: %{z:.4f}<extra></extra>'
    ))
    
    # Add markers for credible features
    annotations = []
    for i, sig in enumerate(pivot_slope.index):
        for j, cell in enumerate(pivot_slope.columns):
            priority = pivot_priority.iloc[i, j]
            slope_val = pivot_slope.iloc[i, j]
            
            if pd.isna(slope_val) or priority >= 4:
                continue
            
            # Determine marker
            if priority == 1:
                text = "★★"
                size = 16
            elif priority == 2:
                text = "★"
                size = 18
            elif priority == 3:
                text = "○"
                size = 16
            else:
                continue
            
            # Determine color (contrast with background)
            if abs(slope_val) > 0.3:
                color = 'white'
            else:
                color = 'black'
            
            annotations.append(
                dict(
                    x=cell,
                    y=sig,
                    text=text,
                    showarrow=False,
                    font=dict(size=size, color=color, family='Arial'),
                    xref='x',
                    yref='y'
                )
            )
    
    fig.update_layout(
        annotations=annotations,
        title=dict(
            text=f'{compartment} - Continuous BMI Slopes<br><sub>★★ ROPE>0.2 | ★ ROPE>0.1 | ○ HDI only</sub>',
            font=dict(size=16, color='#2c3e50')
        ),
        xaxis_title='Cell Type',
        yaxis_title='Signature',
        height=max(600, n_rows * 25),
        width=max(1000, n_cols * 80),
        template=PLOTLY_TEMPLATE,
        hovermode='closest'
    )
    
    fig.update_xaxes(side='top', tickangle=-45)
    
    return fig


def plot_continuous_ridge_plot(cell_type, comp_data):
    """Plot ridge plot for continuous BMI slopes."""
    if comp_data['posterior_bmi_slope'] is None:
        st.info("❌ Posterior BMI slope data not available")
        return None
    
    try:
        df_slope = comp_data['posterior_bmi_slope']
        post_slope = df_slope.iloc[:, 1:].values
        
        ct_map = comp_data['celltype_map']
        n_cells = post_slope.shape[1]

        cell_names = []
        if ct_map is not None and len(ct_map) > 0 and 'celltype_name' in ct_map.columns and 'celltype_idx' in ct_map.columns:
            for _, row in ct_map.iterrows():
                name = str(row['celltype_name'])
                cell_names.append(name.replace('_', ' ').title())
        else:
            cell_names = [f"Cell {i}" for i in range(n_cells)]
        
        sorted_pairs = sorted(enumerate(cell_names), key=lambda x: x[1].lower())
        indices = [p[0] for p in sorted_pairs]
        names = [p[1] for p in sorted_pairs]
        
        if len(indices) > 14:
            means = post_slope.mean(axis=0)
            abs_order = np.argsort(np.abs(means))[::-1][:14]
            indices = [idx for idx in indices if idx in abs_order]
            names = [name for idx, name in zip(indices, names) if idx in abs_order]
        
        indices = indices[::-1]
        names = names[::-1]
        
        fig = go.Figure()
        
        KDE_POINTS = 200
        RIDGE_HEIGHT = 1.0
        SPACING = 1.5
        
        all_samples = post_slope[:, indices].flatten()
        x_min, x_max = np.percentile(all_samples, [0.5, 99.5])
        x_span = max(1e-6, x_max - x_min)
        xgrid = np.linspace(x_min - 0.03*x_span, x_max + 0.03*x_span, KDE_POINTS)
        
        means = post_slope.mean(axis=0)
        
        y_base = 0
        for i, (ct_idx, ct_name) in enumerate(zip(indices, names)):
            samples = post_slope[:, ct_idx]
            
            try:
                kde = gaussian_kde(samples)
                density = kde(xgrid)
                density = (density / density.max()) * RIDGE_HEIGHT
            except:
                density = np.zeros_like(xgrid)
            
            y_offset = y_base + i * SPACING
            
            mean_slope = means[ct_idx]
            color = '#E53935' if mean_slope > 0 else '#1E88E5'
            fill_color = f'rgba(229, 57, 53, 0.5)' if mean_slope > 0 else f'rgba(30, 136, 229, 0.5)'
            
            fig.add_trace(go.Scatter(
                x=[x_min - 0.02 * x_span],
                y=[y_offset + RIDGE_HEIGHT * 0.5],
                mode='text',
                text=[ct_name],
                textposition='middle right',
                textfont=dict(size=12, color='#2c3e50'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=xgrid,
                y=density + y_offset,
                fill='tonexty' if i > 0 else 'tozeroy',
                fillcolor=fill_color,
                line=dict(color=color, width=2),
                hovertemplate=f'<b>{ct_name}</b><br>BMI Slope: %{{x:.3f}}<extra></extra>',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[mean_slope],
                y=[y_offset + RIDGE_HEIGHT * 0.5],
                mode='markers',
                marker=dict(color='black', size=8, symbol='line-ns-open'),
                hovertemplate=f'<b>{ct_name}</b><br>Mean: {mean_slope:.3f}<extra></extra>',
                showlegend=False
            ))
        
        fig.add_vline(x=0, line_dash="dash", line_color="darkred", line_width=2, opacity=0.7)
        
        fig.update_layout(
            title='Posterior BMI Slope Distributions by Cell Type',
            xaxis_title='BMI Slope (Standardized)',
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            height=max(600, len(indices) * 80),
            width=850,
            template=PLOTLY_TEMPLATE,
            hovermode='closest',
            margin=dict(l=180, r=60, t=80, b=60)
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"❌ Error creating ridge plot: {e}")
        return None


def plot_continuous_tier_summary(comp_data):
    """Plot tier-based evidence summary."""
    if comp_data['continuous_results'] is None:
        st.warning("❌ Continuous results not available")
        return None
    
    results = comp_data['continuous_results'].copy()
    
    tier_counts = {
        'Tier 1: Large': results['tier1_large_credible'].sum(),
        'Tier 2: Medium': results['tier2_medium_credible'].sum(),
        'Tier 3: Any': results['tier3_any_credible'].sum(),
        'Not Credible': len(results) - results['tier3_any_credible'].sum()
    }
    
    credible = results[results['tier3_any_credible'] == True]
    if len(credible) > 0:
        direction_counts = credible['bmi_slope_direction'].value_counts().to_dict()
    else:
        direction_counts = {}
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Evidence Tiers', 'Direction of Effects'),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )
    
    colors = ['#4CAF50', '#FF9800', '#2196F3', '#E0E0E0']
    fig.add_trace(
        go.Bar(
            x=list(tier_counts.keys()),
            y=list(tier_counts.values()),
            marker=dict(color=colors, line=dict(color='black', width=1)),
            text=list(tier_counts.values()),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    if direction_counts:
        fig.add_trace(
            go.Pie(
                labels=list(direction_counts.keys()),
                values=list(direction_counts.values()),
                marker=dict(colors=['#E53935', '#1E88E5', '#9E9E9E']),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title='Evidence Summary',
        height=500,
        template=PLOTLY_TEMPLATE,
        showlegend=False
    )
    
    return fig

# ==================================================================================
# ============================= SIGNATURE EXPLORER =================================
# ==================================================================================
def render_signature_explorer():
    """Render the signature database explorer interface"""
    st.markdown('<div class="sub-header">🔍 Signature Database Explorer</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>📚 Browse the Complete Signature Database</b><br>
    Explore all metabolic and functional signatures across different cell types and compartments.
    </div>
    """, unsafe_allow_html=True)
    
    # Load signatures
    signatures = load_signatures()
    
    if not signatures:
        st.error("❌ Failed to load signature database")
        return
    
    # --- MOVED FROM SIDEBAR TO MAIN PAGE ---
    st.markdown("### 🛠️ Search Criteria")
    
    # Create two columns for the dropdowns
    sel_col1, sel_col2 = st.columns(2)
    
    # Step 1: Compartment selection (Main Page)
    with sel_col1:
        compartment = st.selectbox(
            "1. Choose compartment:",
            options=['Immune Fine', 'Immune Coarse', 'Non-Immune'],
            index=0,
            key='explorer_compartment'
        )
    
    # Get available cells based on compartment
    available_cells = get_available_cells(compartment)
    
    if not available_cells:
        st.warning(f"⚠️ No cell types found for {compartment}")
        # Show what is available in the JSON just in case z-scores are missing
        all_cell_types = sorted(list(set([s['cell_type'] for s in signatures])))
        with st.expander("See all cell types available in database", expanded=False):
            st.write(all_cell_types)
        return
    
    # Format cell names for display
    cell_display = {cell.replace('_', ' ').title(): cell for cell in available_cells}
    
    # Step 2: Cell Type Selection (Main Page)
    with sel_col2:
        selected_cell_display = st.selectbox(
            f"2. Choose cell type ({len(available_cells)} available):",
            options=list(cell_display.keys()),
            index=0,
            key='explorer_cell'
        )
    
    selected_cell = cell_display[selected_cell_display]
    
    # Get signatures for this cell type
    cell_signatures = get_cell_signatures(selected_cell)
    
    # --- END OF SELECTION SECTION ---

    # Display summary metrics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📂 Compartment", compartment)
    with col2:
        st.metric("🔬 Cell Type", selected_cell_display)
    with col3:
        st.metric("📝 Signatures Found", len(cell_signatures))
    
    st.markdown("---")
    
    if not cell_signatures:
        st.warning(f"⚠️ No signatures found for {selected_cell}")
        return
    
    # Create tabs for different views
    sig_tabs = st.tabs(["📋 Summary Table", "🔬 Detailed View", "📊 Statistics"])
    
    # Tab 1: Summary Table
    with sig_tabs[0]:
        st.markdown("#### Quick Overview")
        
        summary_data = []
        for sig in cell_signatures:
            summary_data.append({
                'Signature Name': format_signature_name(sig['signature'], max_length=50),
                'Full Name': sig['signature'],
                'Number of Genes': len(sig['genes']),
                'First 5 Genes': ', '.join(sig['genes'][:5]) + ('...' if len(sig['genes']) > 5 else '')
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        st.dataframe(
            summary_df[['Signature Name', 'Number of Genes', 'First 5 Genes']],
            width='stretch',
            height=min(600, len(summary_df) * 35 + 38)
        )
        
    
    # Tab 2: Detailed View (Updated for new JSON structure)
    with sig_tabs[1]:
        st.markdown("#### Detailed Signature Information")
        
        # Signature selector
        sig_names = [format_signature_name(s['signature'], max_length=60) for s in cell_signatures]
        selected_sig_idx = st.selectbox(
            "Select a signature to view details:",
            options=range(len(sig_names)),
            format_func=lambda x: sig_names[x],
            key='detailed_sig_select'
        )
        
        selected_sig = cell_signatures[selected_sig_idx]
        
        # Display detailed info
        st.markdown(f"#### 🧬 {selected_sig['signature']}")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Metadata:**")
            st.info(f"""
            **Cell Type:** {selected_cell_display}  
            **Compartment:** {compartment}  
            **Gene Count:** {len(selected_sig['genes'])}  
            **Signature ID:** {selected_sig['signature']}
            """)
        
        with col2:
            st.markdown("**Gene List:**")
            
            # Create list of gene badges (Clean style, no positive/negative logic)
            gene_badges = []
            for gene in sorted(selected_sig['genes']):
                badge = f'<span style="display: inline-block; background: #e3f2fd; color: #1565c0; padding: 2px 8px; border-radius: 12px; margin: 3px; font-size: 0.85rem; font-weight: 500; border: 1px solid #bbdefb;">{gene}</span>'
                gene_badges.append(badge)
            
            genes_html = "".join(gene_badges)
            
            st.markdown(
                f"""
                <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                            border: 1px solid #e0e0e0; max-height: 300px; overflow-y: auto; 
                            box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);">
                {genes_html}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("""
            <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #666;">
            ℹ️ <b>Note:</b> Genes are listed alphabetically.
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 3: Statistics
    with sig_tabs[2]:
        st.markdown("#### 📊 Database Statistics")
        
        # Signature size distribution for current cell type
        st.markdown(f"##### Signature Sizes for {selected_cell_display}")
        
        if cell_signatures:
            sig_sizes = [len(s['genes']) for s in cell_signatures]
            sig_names_short = [format_signature_name(s['signature'], max_length=40) for s in cell_signatures]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=sig_names_short,
                y=sig_sizes,
                marker=dict(
                    color=sig_sizes,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Gene Count")
                ),
                text=sig_sizes,
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Genes: %{y}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'Number of Genes per Signature - {selected_cell_display}',
                xaxis_title='Signature',
                yaxis_title='Number of Genes',
                height=500,
                template=PLOTLY_TEMPLATE,
                xaxis=dict(tickangle=-45),
                hovermode='closest'
            )
            
            st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        
        # Overall Stats
        total_signatures = len(signatures)
        total_cell_types = len(set([s['cell_type'] for s in signatures]))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Signatures in DB", total_signatures)
        with col2:
            st.metric("Total Cell Types in DB", total_cell_types)


# ==================================================================================
# ============================= MAIN APP ===========================================
# ==================================================================================
# ==================================================================================
# ===================== MODE 3: SIGNATURE SURVIVAL =================================
# ==================================================================================
PLOT_EXPLANATIONS = {
    "BMI vs Time": """
Shows patient BMI vs follow-up time. Each dot is a patient.
Helps see raw survival patterns across BMI before modeling.
""",

    "BMI vs HR": """
Shows how the hazard ratio of the signature changes across BMI.
HR > 1 = higher risk, HR < 1 = protective.
""",

    "Dual-Axis": """
Shows both mean follow-up time and hazard ratio vs BMI.
Helps interpret risk in context of follow-up depth.
""",

    "Forest Plot": """
Shows hazard ratios of the signature within BMI categories.
Identifies where the signature is prognostic.
""",

    "Tertile": """
BMI × signature interaction using Low/Medium/High signature groups.
Tests whether BMI modifies signature effect.
""",

    "Median Split": """
BMI × signature interaction using Low vs High (50/50 split).
Higher power and simpler interpretation.
""",

    "HR + Distribution": """
Shows HR trend across BMI plus patient count histogram.
Reveals where estimates are reliable or sparse.
"""
}

COMPARISONS = {
    "Overweight vs Normal": "overweight_vs_normal",
    "Obese vs Normal": "obese_vs_normal",
    "Obese vs Overweight": "obese_vs_overweight"
}
 
def _try_send_contact_email(sender_name, subject, message):
    """Send contact email via Gmail SMTP. Returns (sent: bool, status: str)."""
    try:
        smtp_user = st.secrets["CONTACT_SMTP_USER"]
        smtp_pass = st.secrets["CONTACT_SMTP_PASSWORD"]
    except Exception:
        return False, "queued"
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    recipients = ["arunv@rgcb.res.in", "arunviswanathan91@gmail.com"]
    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = f"[Cell Analysis Viewer] {subject}"
    msg.attach(MIMEText(f"From: {sender_name}\n\n{message}", "plain"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipients, msg.as_string())
        return True, "sent"
    except Exception:
        return False, "queued"


def render_signature_survival():
    """Mode 3: Dedicated Signature Survival Analysis"""
    st.markdown('<div class="sub-header">🎯 Signature-Level Survival Analysis</div>', unsafe_allow_html=True)

    st.warning(
        "**Note:** Signature Survival Analysis is not part of the published paper. "
        "This section is provided as an exploratory tool. The paper citation will be updated here upon publication.",
        icon="📌"
    )

    st.markdown("""
    <div class="info-box">
    <b>📋 BMI-Stratified Survival Analysis</b><br>
    Explore how signature expression affects patient outcomes across BMI categories using Cox proportional hazards modeling.
    </div>
    """, unsafe_allow_html=True)

    # ================= LOAD DATA =================
    clinical = load_clinical_data()
    sig_features = load_significant_features()
    zscore_data = load_zscore_data_survival()

    if sig_features is None or len(sig_features) == 0:
        st.error("❌ No survival features available")
        return

    # ── Main-page selection controls ──────────────────────────────────────────
    st.markdown("### 🛠️ Data Selection")

    sel_col1, sel_col2 = st.columns(2)

    # Step 1: Comparison
    with sel_col1:
        comparison_display = st.selectbox(
            "1. Choose comparison:",
            options=list(COMPARISONS.keys()),
            index=0,
            key='surv_comparison',
        )
    comparison_key = COMPARISONS[comparison_display]

    # Step 2: Compartment (from CSV)
    available_compartments = sorted(
        sig_features
        .loc[sig_features['comparison'] == comparison_key, 'compartment']
        .dropna()
        .unique()
        .tolist()
    )

    if not available_compartments:
        st.warning("⚠️ No compartments available for this comparison")
        return

    with sel_col2:
        compartment = st.selectbox(
            "2. Choose compartment:",
            options=available_compartments,
            index=0,
            key='surv_compartment',
        )

    # ---- Filter by comparison + compartment ----
    filtered_sigs = sig_features[
        (sig_features['comparison'] == comparison_key) &
        (sig_features['compartment'] == compartment)
    ].copy()

    if filtered_sigs.empty:
        st.warning("⚠️ No signatures for this selection")
        return

    # ================= CELL TYPE =================
    filtered_sigs['cell_type'] = filtered_sigs['feature'].apply(
        lambda x: x.split('||')[0] if '||' in str(x) else None
    )

    available_cells = sorted(filtered_sigs['cell_type'].dropna().unique())
    if not available_cells:
        st.warning("⚠️ No cell types found")
        return

    sel_col3, sel_col4 = st.columns(2)

    cell_display = {c.replace('_', ' ').title(): c for c in available_cells}
    with sel_col3:
        selected_cell_display = st.selectbox(
            "3. Choose cell type:",
            options=list(cell_display.keys()),
            key='surv_cell',
        )
    selected_cell = cell_display[selected_cell_display]

    cell_filtered = filtered_sigs[filtered_sigs['cell_type'] == selected_cell].copy()
    if cell_filtered.empty:
        st.warning("⚠️ No signatures for selected cell type")
        return

    # ================= SIGNATURE =================
    sig_display_map = {}
    for _, row in cell_filtered.iterrows():
        feat = row['feature']
        sig_name = feat.split('||')[1] if '||' in str(feat) else feat
        sig_display_map[clean_label_text(sig_name)] = feat

    with sel_col4:
        selected_sig_display = st.selectbox(
            "4. Choose signature:",
            options=list(sig_display_map.keys()),
            key='surv_signature',
        )
    selected_feature = sig_display_map[selected_sig_display]

    sig_row = cell_filtered[cell_filtered['feature'] == selected_feature].iloc[0]

    # Sidebar summary
    st.sidebar.markdown("### Current Selection")
    st.sidebar.info(f"""
    **Comparison:** {comparison_display}
    **Compartment:** {compartment}
    **Cell Type:** {selected_cell_display}
    **Signature:** {selected_sig_display}
    **HR:** {sig_row['hr']:.3f}
    **p-value:** {sig_row['hr_p']:.3e}
    """)

    st.divider()

    # ================= MAIN ANALYSIS =================
    feature_data = zscore_data[zscore_data['feature'] == selected_feature].copy()
    if feature_data.empty:
        st.error("❌ No z-score data found")
        return

    patient_data = clinical.merge(
        feature_data[['base_sample_id', 'Z']],
        left_on='sample_id',
        right_on='base_sample_id',
        how='inner'
    )

    patient_data = patient_data[
        (patient_data['follow_up_months'] > 0) &
        (patient_data['vital_status_binary'].notna())
    ]

    if len(patient_data) < 30:
        st.warning("⚠️ Insufficient data for survival analysis")
        return

    st.markdown("### 📊 Interactive Survival Plots")

    plot_configs = [
        ("BMI vs Time", plot_survival_bmi_vs_time),
        ("BMI vs HR", plot_survival_bmi_vs_hr),
        ("Dual-Axis", plot_survival_bmi_dual_axis),
        ("Forest Plot", plot_survival_forest_bmi),
        ("Tertile", plot_survival_interaction_tertile),
        ("Median Split", plot_survival_interaction_median),
        ("HR + Distribution", plot_survival_hr_with_distribution)
    ]

    for i in range(0, len(plot_configs), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(plot_configs):
                name, fn = plot_configs[i + j]
                with cols[j]:
                    fig = fn(patient_data, selected_sig_display)
                    if fig:
                        st.plotly_chart(fig, width='stretch')
                        with st.expander("What does this plot mean?", expanded=False):
                            st.markdown(PLOT_EXPLANATIONS[name])

def render_continuous_analysis():
    """Mode: Continuous BMI Analysis - Cell-Level"""
    
    st.markdown('<div class="header-spacer"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">📈 Continuous BMI Association Analysis</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>📊 Cell-Level BMI Slope Analysis</h3>
        <p>Explore how signatures change continuously with BMI for a specific cell type. 
        Credible features marked: ★★ (ROPE > 0.2), ★ (ROPE > 0.1), ○ (HDI only).</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("**About the Analysis**", expanded=False):
        st.markdown("""
        ### 🔬 Continuous BMI Modeling
        
        Models BMI as continuous variable (dose-response).
        
        **BMI Slope:** Change per 1 SD BMI increase (~5 units)
        
        **Credibility Markers:**
        - **★★:** HDI + ROPE > 0.2 (large effect)
        - **★:** HDI + ROPE > 0.1 (medium effect)
        - **○:** HDI credible only
        """)
    
    # ── Main-page selection controls ──────────────────────────────────────────
    st.markdown("### 🛠️ Data Selection")

    sel_col1, sel_col2 = st.columns(2)

    # Step 1: Compartment
    with sel_col1:
        compartment = st.selectbox(
            "1. Choose compartment:",
            options=['Immune Fine', 'Immune Coarse', 'Non-Immune'],
            index=0,
            key='continuous_compartment',
        )

    with st.status("Loading analysis data...", expanded=False) as _load_status:
        st.write("Loading continuous model data...")
        comp_data_cont = load_compartment_data_continuous(compartment)
        st.write("Loading categorical data...")
        comp_data_cat = load_compartment_data(compartment)
        _load_status.update(label="Data ready", state="complete", expanded=False)

    # Step 2: Cell Type Selection
    available_cells = get_available_cells_continuous(compartment)

    if not available_cells:
        st.error("❌ No cell types with sufficient data for continuous modeling found")
        return

    cell_display = {cell.replace('_', ' ').title(): cell for cell in available_cells}

    with sel_col2:
        selected_cell_display = st.selectbox(
            f"2. Choose cell type ({len(available_cells)} available):",
            options=list(cell_display.keys()),
            index=0,
            key='continuous_cell',
        )
    selected_cell = cell_display[selected_cell_display]

    # Metrics
    n_total = 0
    n_credible = 0
    if comp_data_cont['continuous_results'] is not None:
        results = comp_data_cont['continuous_results']

        # Filter for selected cell
        results['cell_type_parsed'] = results['feature'].apply(
            lambda x: str(x).split('||')[0].strip().replace('_', ' ') if '||' in str(x) else "Unknown"
        )

        cell_results = results[results['cell_type_parsed'].str.upper() == selected_cell.upper()]
        n_total = len(cell_results)
        n_credible = cell_results.get('bmi_slope_credible', pd.Series([False]*len(cell_results))).sum()

    # Sidebar summary
    st.sidebar.markdown("### Current Selection")
    st.sidebar.info(f"""
    **Compartment:** {compartment}
    **Cell Type:** {selected_cell_display}
    **Total Signatures:** {n_total}
    **Credible:** {n_credible}
    """)

    st.divider()
    
    st.markdown(f'<div class="sub-header">📈 {selected_cell_display} - Continuous Analysis</div>', 
               unsafe_allow_html=True)
        
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Compartment", compartment)
    with col2:
        st.metric("Cell Type", selected_cell_display)
    with col3:
        st.metric("Credible Signatures", n_credible)
    
    # Create tabs
    tabs = st.tabs(["📊 Heatmap", "🌊 Ridge Plot", "🔍 Diagnostics"])
    
    # Tab 1: Cell-specific Heatmap
    with tabs[0]:
        st.markdown(f"### 📊 BMI Slope Heatmap - {selected_cell_display}")
        st.markdown("""
        <div class="method-box">
        <b>💡 Reading the Heatmap</b><br>
        Shows signatures (rows) for this cell type only.<br>
        • <b>Color:</b> Red = positive slope, Blue = negative<br>
        • <b>★★:</b> HDI + ROPE > 0.2 | <b>★:</b> ROPE > 0.1 | <b>○:</b> HDI only
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Loading chart..."):
            fig = plot_continuous_cell_heatmap(selected_cell, comp_data_cont)
            if fig:
                st.plotly_chart(fig, width='stretch')
    
    # Tab 2: Ridge Plot
    with tabs[1]:
        st.markdown(f"### 🌊 Posterior Distribution - {selected_cell_display}")
        
        with st.expander("📖 Understanding Ridge Plots", expanded=False):
            st.markdown("""
            Shows the full posterior distribution of BMI slopes.
            - **Width:** Uncertainty in slope estimate
            - **Peak:** Most likely slope value
            - **Color:** Red = positive, Blue = negative
            - **Black line:** Mean slope
            """)
        
        with st.spinner("Loading chart..."):
            fig = plot_continuous_ridge_plot(selected_cell, comp_data_cont)
            if fig:
                st.plotly_chart(fig, width='stretch')
    
    # Tab 3: Diagnostics - 2-Column Layout
    with tabs[2]:
        st.markdown("### MCMC Diagnostics")

        with st.expander("Understanding MCMC Diagnostics", expanded=False):
            st.markdown("""
            **ESS & R-hat:** Check convergence (ESS > 400, R-hat < 1.01).
            **Energy:** Monitor sampling quality (smooth = good).
            **Trace:** "Hairy caterpillar" = good mixing.
            **Rank:** Uniform distributions = converged.
            **Autocorrelation:** Rapid decay = independent samples.
            """)

        # Check whether the selected cell has MCMC posterior data.
        # Primary check: look for celltype_bmi_slope[CELL_NAME] in diagnostics index
        # (works for new name-based format). Fallback: use integer index mapping.
        _diag = comp_data_cont.get('diagnostics')
        _cell_upper = selected_cell.upper().replace('_', ' ')
        if _diag is not None:
            _diag_idx = _diag.index.astype(str)
            # Match both "CELL NAME" (spaces) and "CELL_NAME" (underscores)
            _cell_in_model = (
                _diag_idx.str.contains(
                    r'celltype_bmi_slope\[' + _cell_upper.replace(' ', '[_ ]') + r'\]',
                    regex=True, case=False, na=False
                ).any()
                or _diag_idx.str.contains(
                    r'celltype_bmi_slope\[' + _cell_upper.replace(' ', '_') + r'\]',
                    regex=True, case=False, na=False
                ).any()
            )
        else:
            _name_to_idx = get_continuous_celltype_index_map(comp_data_cont)
            _cell_in_model = _cell_upper in _name_to_idx

        if not _cell_in_model:
            # Cells with full MCMC posteriors for this compartment
            _modeled_cells = sorted(
                k.replace('_', ' ').title() for k in _name_to_idx.keys()
            )
            st.info(
                f"ℹ️ **Full MCMC diagnostics are not available for {selected_cell_display}.**\n\n"
                f"The continuous Bayesian model ran full posterior sampling for "
                f"{len(_modeled_cells)} of the {len(available_cells)} cell types in this "
                f"compartment. **{selected_cell_display}** has slope estimates (visible in the "
                f"Heatmap tab) derived from a separate model pass, but its posterior chains "
                f"were not stored, so ESS, R-hat, trace, rank, and autocorrelation plots "
                f"cannot be generated.\n\n"
                f"**Cells with full diagnostics available:** "
                f"{', '.join(_modeled_cells)}"
            )
        if _cell_in_model:
            # Row 1: ESS/R-hat and Energy (side by side)
            cont_col1, cont_col2 = st.columns(2)

            with cont_col1:
                st.markdown("#### ESS & R-hat")
                with st.expander("What does this show?", expanded=False):
                    st.markdown("**ESS:** Independent samples (target > 400). **R-hat:** Chain agreement (< 1.01).")
                with st.spinner("Loading chart..."):
                    fig = plot_ess_rhat_continuous(comp_data_cont, selected_cell=selected_cell_display)
                    if fig:
                        st.plotly_chart(fig, width='stretch')

            with cont_col2:
                st.markdown("#### Energy")
                with st.expander("What does this show?", expanded=False):
                    st.markdown("**Energy:** HMC sampling quality. Smooth transitions = good mixing.")
                with st.spinner("Loading chart..."):
                    fig = plot_energy_diagnostic(comp_data_cont)
                    if fig:
                        st.plotly_chart(fig, width='stretch')

            # Row 2: Trace and Rank (side by side)
            cont_col3, cont_col4 = st.columns(2)

            with cont_col3:
                st.markdown("#### Trace Plot")
                with st.expander("What does this show?", expanded=False):
                    st.markdown("**Good:** 'Hairy caterpillar' with overlapping chains. **Bad:** Trends or stuck chains.")
                with st.spinner("Loading chart..."):
                    fig = plot_trace_continuous(comp_data_cont, selected_cell_display)
                    if fig:
                        st.plotly_chart(fig, width='stretch')

            with cont_col4:
                st.markdown("#### Rank Plot")
                with st.expander("What does this show?", expanded=False):
                    st.markdown("**Good:** Uniform distributions. **Bad:** Non-uniform = exploring different regions.")
                with st.spinner("Loading chart..."):
                    fig = plot_rank_continuous(comp_data_cont, selected_cell_display)
                    if fig:
                        st.plotly_chart(fig, width='stretch')

            # Row 3: Autocorrelation (full width)
            st.markdown("#### Autocorrelation")
            with st.expander("What does this show?", expanded=False):
                st.markdown("**Good:** Rapid decay to zero. **Bad:** Slow decay = high autocorrelation.")
            with st.spinner("Loading chart..."):
                fig = plot_autocorrelation_continuous(comp_data_cont, selected_cell_display)
                if fig:
                    st.plotly_chart(fig, width='stretch')


    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
    <b>Continuous BMI Analysis</b><br>Cell-level dose-response modeling
    </div>
    """, unsafe_allow_html=True)

# ==================================================================================
# ====================== INTERACTOME CONFIGURATION =================================
# ==================================================================================

# Configuration: Maximum number of interactions to display for "All Interactions"
# Adjust this value based on browser performance
MAX_ALL_INTERACTIONS = 200  # Cap at 200 interactions to prevent browser crashes

INDIVIDUAL_INTERACTION_SIGNATURES = {
    "Bindea": {
        "normal": "data/interactome/cell/cellinteraction_normalweight.csv",
        "overweight": "data/interactome/cell/cellinteraction_overweight.csv"
    },
    "Newman": {
        "normal": "data/interactome/cell/cellinteraction_NW_NMan.csv",
        "overweight": "data/interactome/cell/cellinteraction_OW_NMan.csv"
    },
    "Zheng": {
        "normal": "data/interactome/cell/cellinteraction_NW_Zheng.csv",
        "overweight": "data/interactome/cell/cellinteraction_OW_Zheng.csv"
    }
}

# ==================================================================================
# ====================== INTERACTOME HELPER FUNCTIONS ==============================
# ==================================================================================

def load_interactome_data():
    """Load all 6 cell interaction CSV files from data/interactome/cell/"""
    base_path = "data/interactome/cell"

    datasets = {
        'Bindea_Normal': 'cellinteraction_normalweight.csv',
        'Bindea_Overweight': 'cellinteraction_overweight.csv',
        'Zheng_Normal': 'cellinteraction_NW_Zheng.csv',
        'Zheng_Overweight': 'cellinteraction_OW_Zheng.csv',
        'Newman_Normal': 'cellinteraction_NW_NMan.csv',
        'Newman_Overweight': 'cellinteraction_OW_NMan.csv'
    }

    data = {}
    for name, filename in datasets.items():
        filepath = os.path.join(base_path, filename)
        try:
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                parts = name.split('_')
                df['Dataset'] = parts[0]
                df['Condition'] = parts[1]
                data[name] = df
            else:
                st.warning(f"⚠️ File not found: {filepath}")
        except Exception as e:
            st.error(f"❌ Error loading {filename}: {str(e)}")

    return data


def get_all_cell_types_interactome(interactome_data):
    """Extract all unique cell types from interactome data"""
    all_cells = set()
    for df in interactome_data.values():
        all_cells.update(df['Favorable.Cell.Type'].unique())
        all_cells.update(df['Unfavorable.Cell.Type'].unique())
    return all_cells


def get_cell_group_interactome(cell_name):
    """Biological cell type grouping for interactome"""
    if 'Tumor' in cell_name:
        return 'Tumor'

    elif cell_name in ['CD4 Tn', 'CD4+ADSL+ Tn', 'CD4+Tn']:
        return 'CD4_Naive'
    elif cell_name in ['CD4+CAPG+CREM- Tm', 'CD4+CCL5+ Tm', 'CD4+CREM+ Tm',
                       'CD4+TIMP1+ Tm', 'aCD4 Tm', 'rCD4 Tm', 'Tcm']:
        return 'CD4_Memory'
    elif cell_name in ['CD4+GZMK+ Tem', 'Tem']:
        return 'CD4_Effector_Memory'
    elif cell_name in ['Th', 'Th1', 'Th2', 'CD4+ISG+ Th']:
        return 'CD4_Th1_Th2'
    elif cell_name in ['CD4+CCR6+ Th17', 'CD4+IL26+ Th17']:
        return 'CD4_Th17'
    elif cell_name in ['CD4+CXCR5+ pre-Tfh', 'CD4+IFNG+ Tfh/Th1', 'Tfh']:
        return 'CD4_Tfh'
    elif cell_name in ['CD4+ISG+ Treg', 'CD4+S1PR1+ Treg',
                       'CD4+TNFRSF9+ Treg', 'Treg']:
        return 'CD4_Treg'
    elif cell_name in ['CD4+NME1+CCR4+ T', 'CD4+NME1+CCR4- T', 'CD4+TNF+ T']:
        return 'CD4_Other'

    elif cell_name in ['CD8+Tn']:
        return 'CD8_Naive'
    elif cell_name in ['CD8 T', 'CD8+Tc17', 'Cytotoxic']:
        return 'CD8_Effector'
    elif cell_name in ['CD8+GZMK+ Tem', 'CD8+GZMK+ early Tem']:
        return 'CD8_Effector_Memory'
    elif cell_name in ['CD8+GZMK+ Tex', 'CD8+OXPHOS- Tex',
                       'CD8+TCF7+ Tex', 'CD8+terminal Tex']:
        return 'CD8_Exhausted'
    elif cell_name in ['CD8+KIR+EOMES+ NK-like', 'CD8+KIR+TXK+ NK-like']:
        return 'CD8_NK_like'
    elif cell_name in ['CD8+ISG+ CD8+ T', 'CD8+NME1+ T']:
        return 'CD8_Other'

    elif cell_name == 'Tgd':
        return 'T_Gamma_Delta'
    elif cell_name == 'T':
        return 'T_General'

    elif cell_name in ['B', 'Bn']:
        return 'B_Naive'
    elif cell_name == 'Bm':
        return 'B_General'
    elif cell_name == 'Plasma':
        return 'Plasma'

    elif cell_name in ['NK', 'CD56dim NK', 'aNK', 'rNK']:
        return 'NK_cells'

    elif cell_name == 'Monocyte':
        return 'Monocyte'
    elif cell_name in ['Macrophage', 'M0', 'M1', 'M2']:
        return 'Macrophage'
    elif cell_name in ['DC', 'aDC', 'iDC', 'rDC']:
        return 'Dendritic'
    elif cell_name == 'Neutrophil':
        return 'Neutrophil'
    elif cell_name == 'Eosinophil':
        return 'Eosinophil'
    elif cell_name in ['Mast', 'aMast', 'rMast']:
        return 'Mast'

    else:
        return 'Other'


def get_cell_color_interactome(cell_name):
    """Color scheme for biological cell groups"""
    group = get_cell_group_interactome(cell_name)

    color_map = {
        'Tumor': '#8B0000',
        'CD4_Naive': '#AED6F1',
        'CD4_Memory': '#85C1E9',
        'CD4_Effector_Memory': '#5DADE2',
        'CD4_Th1_Th2': '#3498DB',
        'CD4_Th17': '#2E86C1',
        'CD4_Tfh': '#2874A6',
        'CD4_Treg': '#21618C',
        'CD4_Other': '#1B4F72',
        'CD8_Naive': '#D7BDE2',
        'CD8_Effector': '#BB8FCE',
        'CD8_Effector_Memory': '#A569BD',
        'CD8_Exhausted': '#884EA0',
        'CD8_NK_like': '#76448A',
        'CD8_Other': '#633974',
        'T_Gamma_Delta': '#512E5F',
        'T_General': '#4A235A',
        'B_Naive': '#C39BD3',
        'B_General': '#AF7AC5',
        'Plasma': '#9B59B6',
        'NK_cells': '#E74C3C',
        'Monocyte': '#F8C471',
        'Macrophage': '#1ABC9C',
        'Dendritic': '#F39C12',
        'Neutrophil': '#E67E22',
        'Eosinophil': '#95A5A6',
        'Mast': '#34495E',
        'Other': '#7F8C8D'
    }

    return color_map.get(group, '#7F8C8D')


def cap_interactions(df, max_interactions, sort_by='Enrichment.Ratio'):
    """
    Cap the number of interactions to prevent browser crashes.
    Returns the top N interactions sorted by the specified column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The interactions dataframe
    max_interactions : int
        Maximum number of interactions to keep
    sort_by : str
        Column name to sort by (default: 'Enrichment.Ratio')
    
    Returns:
    --------
    tuple : (capped_df, was_capped, original_count)
    """
    original_count = len(df)
    
    if original_count <= max_interactions:
        return df, False, original_count
    
    # Sort by the specified column and take top N
    capped_df = df.nlargest(max_interactions, sort_by)
    
    return capped_df, True, original_count


def filter_interactome_data(interactome_data, dataset_choice, sig_filter, selected_cells, condition_filter):
    """Filter interactions based on user selections"""

    # First, select the dataset
    if dataset_choice == "All Combined":
        combined_df = pd.concat(interactome_data.values(), ignore_index=True)
    else:
        relevant_dfs = [df for name, df in interactome_data.items()
                        if name.startswith(dataset_choice)]
        if not relevant_dfs:
            return pd.DataFrame(), False, 0
        combined_df = pd.concat(relevant_dfs, ignore_index=True)

    # Apply condition filter
    if condition_filter != "Both":
        if condition_filter == "Normal Weight":
            combined_df = combined_df[combined_df['Condition'] == 'Normal']
        elif condition_filter == "Overweight":
            combined_df = combined_df[combined_df['Condition'] == 'Overweight']

    # Track if we capped the data
    was_capped = False
    original_count = 0

    # Apply significance filter
    if sig_filter == "All Interactions (Per Signature)":
        # Return all interactions for the selected signature, but cap it
        filtered_df, was_capped, original_count = cap_interactions(
            combined_df, 
            MAX_ALL_INTERACTIONS, 
            sort_by='Enrichment.Ratio'
        )

    elif sig_filter == "Top 30 All":
        filtered_df = combined_df.nlargest(30, 'Enrichment.Ratio')

    elif sig_filter == "Significant Across All":
        # This requires multiple datasets/conditions to compare
        sig_df = combined_df[combined_df['Permutation.FDR'] < 0.05]
        sig_df = sig_df.copy()
        sig_df['Interaction_Key'] = sig_df['Cell.Interaction']
        interaction_counts = sig_df.groupby('Interaction_Key').size()
        multi_dataset = interaction_counts[interaction_counts >= 2].index
        filtered_df = sig_df[sig_df['Interaction_Key'].isin(multi_dataset)]
        if len(filtered_df) > 50:
            filtered_df = filtered_df.nlargest(50, 'Enrichment.Ratio')

    elif sig_filter == "Significant Per Dataset":
        filtered_dfs = []
        for (_, _), group in combined_df.groupby(['Dataset', 'Condition']):
            sig_group = group[group['Permutation.FDR'] < 0.05]
            filtered_dfs.append(sig_group.nsmallest(10, 'Permutation.FDR'))
        filtered_df = pd.concat(filtered_dfs, ignore_index=True) if filtered_dfs else pd.DataFrame()

    else:
        filtered_df = combined_df

    # Apply cell type filter
    if selected_cells:
        mask = (
            filtered_df['Favorable.Cell.Type'].isin(selected_cells) |
            filtered_df['Unfavorable.Cell.Type'].isin(selected_cells)
        )
        filtered_df = filtered_df[mask]

    return filtered_df, was_capped, original_count


# ==================================================================================
# ====================== INTERACTOME PLOTTING FUNCTIONS ============================
# ==================================================================================
 
def plot_interactome_sankey(filtered_data, dataset_choice, condition_filter):
    """
    Create a COMPACT Sankey diagram designed to fit in a single screen view.
    """
    import plotly.graph_objects as go

    if filtered_data.empty:
        st.warning("No data to plot")
        return None

    try:
        # --- 1. Helper to Truncate Long Names ---
        # This keeps the plot from getting too wide
        def shorten_name(name, max_len=15):
            name = str(name)
            if len(name) > max_len:
                return name[:max_len] + "..."
            return name

        # Get raw names for mapping
        raw_sources = filtered_data['Favorable.Cell.Type'].unique().tolist()
        raw_targets = filtered_data['Unfavorable.Cell.Type'].unique().tolist()
        raw_nodes = list(set(raw_sources + raw_targets))
        
        # Create SHORT names for display
        short_nodes = [shorten_name(n) for n in raw_nodes]
        
        # Mapping dict using RAW names to find indices
        node_dict = {node: idx for idx, node in enumerate(raw_nodes)}

        # Build links
        source_indices = [node_dict[cell] for cell in filtered_data['Favorable.Cell.Type']]
        target_indices = [node_dict[cell] for cell in filtered_data['Unfavorable.Cell.Type']]
        values = filtered_data['Enrichment.Ratio'].tolist()
        
        # Colors
        node_colors = [get_cell_color_interactome(node) for node in raw_nodes]

        # Link Colors
        link_colors = []
        for _, row in filtered_data.iterrows():
            if condition_filter == "Both":
                link_colors.append('rgba(231, 76, 60, 0.4)' if row['Condition'] == 'Overweight' else 'rgba(46, 204, 113, 0.4)')
            elif condition_filter == "Overweight":
                link_colors.append('rgba(231, 76, 60, 0.6)')
            elif condition_filter == "Normal Weight":
                link_colors.append('rgba(46, 204, 113, 0.6)')
            else:
                link_colors.append('rgba(149, 165, 166, 0.4)')

        # Custom Hover (shows full names)
        link_labels = [f"<b>{row['Favorable.Cell.Type']} ➞ {row['Unfavorable.Cell.Type']}</b><br>Ratio: {row['Enrichment.Ratio']:.2f}" for _, row in filtered_data.iterrows()]

        # --- 2. Create Plot with "Fit to Screen" Settings ---
        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            
            # Global Font Settings
            textfont=dict(size=10, color="black", family="Arial"),
            
            node=dict(
                pad=10,        # Tight padding to fit more vertical bars
                thickness=15,  # Thinner bars
                line=dict(color="black", width=0.5),
                label=short_nodes, # Use the SHORT names here
                color=node_colors,
                # Show FULL name on hover
                customdata=raw_nodes,
                hovertemplate='<b>%{customdata}</b><br>Flow: %{value:.2f}<extra></extra>'
            ),
            
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                customdata=link_labels,
                hovertemplate='%{customdata}<extra></extra>',
                color=link_colors
            )
        )])

        # --- 3. Layout: Fixed Height & Tight Margins ---
        fig.update_layout(
            title=dict(
                text=f"<b>{dataset_choice} Interactions</b>",
                font=dict(size=16),
                x=0.5
            ),
            # Add top annotations
            annotations=[
                dict(x=0, y=1.06, xref="paper", yref="paper", text="<b>Source</b>", showarrow=False),
                dict(x=1, y=1.06, xref="paper", yref="paper", text="<b>Target</b>", showarrow=False)
            ],
            
            # HEIGHT: 600px fits comfortably on almost all laptop screens
            height=600,
            
            # MARGINS: Reduced l/r because names are now short
            margin=dict(l=80, r=80, t=60, b=20),
            
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        return fig

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def show_interaction_stats(filtered_data, dataset_choice, sig_filter, condition_filter):
    """
    Display summary statistics for the filtered interactions
    """
    if filtered_data.empty:
        st.warning("No data to display statistics")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Interactions", len(filtered_data))
        
    with col2:
        unique_cells = len(set(filtered_data['Favorable.Cell.Type'].unique().tolist() + 
                        filtered_data['Unfavorable.Cell.Type'].unique().tolist()))
        st.metric("Unique Cell Types", unique_cells)
    
    with col3:
        sig_interactions = len(filtered_data[filtered_data['Permutation.FDR'] < 0.05])
        st.metric("Significant (FDR < 0.05)", sig_interactions)
        
    with col4:
        st.metric("Avg Enrichment Ratio", f"{filtered_data['Enrichment.Ratio'].mean():.2f}")
    
    # Condition breakdown
    if 'Condition' in filtered_data.columns and condition_filter == "Both":
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            normal_count = len(filtered_data[filtered_data['Condition'] == 'Normal'])
            st.metric("🟢 Normal Weight Interactions", normal_count)
        
        with col2:
            overweight_count = len(filtered_data[filtered_data['Condition'] == 'Overweight'])
            st.metric("🔴 Overweight Interactions", overweight_count)
    
    # Dataset breakdown
    if 'Dataset' in filtered_data.columns:
        st.markdown("---")
        st.markdown("#### 📊 Dataset Distribution")
        dataset_counts = filtered_data['Dataset'].value_counts()
        col1, col2, col3 = st.columns(3)
        
        for idx, (dataset, count) in enumerate(dataset_counts.items()):
            with [col1, col2, col3][idx % 3]:
                st.metric(f"{dataset}", count)
    
    # Top interactions table
    st.markdown("---")
    st.markdown("#### 🔝 Top 10 Interactions by Enrichment Ratio")
    
    display_cols = ['Favorable.Cell.Type', 'Unfavorable.Cell.Type', 
                   'Enrichment.Ratio', 'Permutation.FDR']
    
    if 'Dataset' in filtered_data.columns:
        display_cols.append('Dataset')
    if 'Condition' in filtered_data.columns:
        display_cols.append('Condition')
    
    top_interactions = filtered_data.nlargest(10, 'Enrichment.Ratio')[display_cols].copy()
    top_interactions['Enrichment.Ratio'] = top_interactions['Enrichment.Ratio'].round(2)
    top_interactions['Permutation.FDR'] = top_interactions['Permutation.FDR'].round(4)
    
    st.dataframe(top_interactions, width='stretch', hide_index=True)
    
    # Distribution plots
    st.markdown("---")
    st.markdown("#### 📊 Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enrichment ratio distribution
        fig = px.histogram(
            filtered_data, 
            x='Enrichment.Ratio',
            nbins=30,
            title='<b>Enrichment Ratio Distribution</b>',
            labels={'Enrichment.Ratio': 'Enrichment Ratio', 'count': 'Frequency'},
            color_discrete_sequence=['#3498db']
        )
        fig.update_layout(
            height=300, 
            showlegend=False,
            title_font=dict(size=16, color='#2c3e50', family='Arial Black'),
            font=dict(color='#2c3e50')
        )
        
        st.plotly_chart(
            fig,
            width='stretch',
            config={
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "svg",
                    "filename": "interactome_sankey",
                    "height": 800,
                    "width": 1200,
                    "scale": 1
                }
            }
        )

    with col2:
        # FDR distribution
        fig = px.histogram(
            filtered_data, 
            x='Permutation.FDR',
            nbins=30,
            title='<b>FDR Distribution</b>',
            labels={'Permutation.FDR': 'Permutation FDR', 'count': 'Frequency'},
            color_discrete_sequence=['#e74c3c']
        )
        fig.update_layout(
            height=300, 
            showlegend=False,
            title_font=dict(size=16, color='#2c3e50', family='Arial Black'),
            font=dict(color='#2c3e50')
        )
        st.plotly_chart(fig, width='stretch')


# ==================================================================================
# ====================== INTERACTOME ANALYSIS MODE =================================
# ==================================================================================
 
def render_interactome_analysis():
    """Render the Interactome Analysis mode"""
    
    # Add spacer
# Spacer div
    st.markdown('<div class="header-spacer"></div>', unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔗 Cell-Cell Interactome Network Analysis</h1>', unsafe_allow_html=True)

    # Info box
    st.markdown("""
    <div class="info-box">
        <h3>📊 Interactive Cell-Cell Interaction Network</h3>
        <p>Explore cell-cell interactions across different immune profiling signatures (Bindea, Zheng, Newman) 
        comparing normal weight vs overweight conditions.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        interactome_data = load_interactome_data()

    if not interactome_data:
        st.error("❌ Could not load interactome data.")
        return

    # First row: Dataset and Condition selection
    col1, col2 = st.columns([1, 1])

    with col1:
        dataset_choice = st.radio("📊 **Dataset:**",
                                  ["All Combined", "Bindea", "Zheng", "Newman"],
                                  index=0, horizontal=True)

    with col2:
        condition_filter = st.radio("🎨 **Condition Display:**",
                                   ["Both", "Normal Weight", "Overweight"],
                                   index=0, horizontal=True,
                                   help="🟢 Green = Normal Weight | 🔴 Red = Overweight")

    # Second row: Significance filter
    st.markdown("---")
    
    # Determine available options based on dataset choice
    if dataset_choice in ["Bindea", "Zheng", "Newman"]:
        # For specific signatures: allow "All Interactions (Per Signature)"
        sig_options = ["All Interactions (Per Signature)", "Top 30 All", "Significant Per Dataset"]
        default_index = 0
        disabled_notes = [
            f"✅ Shows top {MAX_ALL_INTERACTIONS} interactions (sorted by Enrichment Ratio) for the selected signature",
            "⚠️ *'Significant Across All' requires 'All Combined' dataset*"
        ]
    else:
        # For "All Combined": don't allow "All Interactions"
        sig_options = ["Top 30 All", "Significant Across All", "Significant Per Dataset"]
        default_index = 0
        disabled_notes = [
            "⚠️ *'All Interactions' is only available when selecting a specific signature (Bindea, Zheng, or Newman)*",
            "✅ This option compares interactions across multiple datasets"
        ]

    sig_filter = st.radio("🎯 **Show Interactions:**",
                          sig_options,
                          index=default_index, 
                          horizontal=True)
    
    # Show appropriate notes
    for note in disabled_notes:
        st.caption(note)

    # Third row: Cell type filter
    st.markdown("---")
    cell_filter_mode = st.radio("🔬 **Cell Type Filter:**",
                                ["All Cell Types", "Select Specific"],
                                index=0, horizontal=True)

    selected_cells = None
    if cell_filter_mode == "Select Specific":
        
        if dataset_choice == "All Combined":
            available_cell_types = get_all_cell_types_interactome(interactome_data)
            helper_text = "Showing cell types from all datasets"
        else:
            dataset_specific_data = {
                name: df for name, df in interactome_data.items() 
                if name.startswith(dataset_choice)
            }
            available_cell_types = get_all_cell_types_interactome(dataset_specific_data)
            helper_text = f"Showing cell types from {dataset_choice} dataset only"
        
        st.caption(f"ℹ️ {helper_text}")
        
        selected_cells = st.multiselect(
            "Choose one or more cell types:",
            sorted(available_cell_types),
            default=None,
            placeholder="Select cell types..."
        )

    # Filter data
    filtered_data, was_capped, original_count = filter_interactome_data(
        interactome_data, dataset_choice, sig_filter, selected_cells, condition_filter
    )

    if filtered_data.empty:
        st.warning("⚠️ No interactions found with current filters.")
        return

    # Show capping warning if data was capped
    if was_capped:
        st.warning(f"""
        ⚠️ **Data Capped for Performance**  
        Original interactions: **{original_count}**  
        Displaying top: **{MAX_ALL_INTERACTIONS}** (sorted by Enrichment Ratio)  
        """)
    # Display Sankey diagram
    st.markdown("---")
    st.markdown("### 🌐 Interaction Network")
    
    # Add legend
    if condition_filter == "Both":
        st.markdown("""
        <div style='padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 20px;'>
        <b>Legend:</b> 
        <span style='color: #2ecc71; font-weight: bold;'>🟢 Green connections</span> = Normal Weight | 
        <span style='color: #e74c3c; font-weight: bold;'>🔴 Red connections</span> = Overweight
        </div>
        """, unsafe_allow_html=True)
    
    fig = plot_interactome_sankey(filtered_data, dataset_choice, condition_filter)
    if fig:
        st.plotly_chart(fig, width='stretch')

    # Statistics section
    st.markdown("---")
    with st.expander("Interaction Statistics", expanded=False):
        show_interaction_stats(filtered_data, dataset_choice, sig_filter, condition_filter)


# ==================================================================================
# ====================== INDIVIDUAL INTERACTION EXPLORER ===========================
# ==================================================================================

def parse_imgp_to_dataframe(imgp_string):
    """Parse Shared.IMGP string into a DataFrame of gene pairs.
    Format: 'GENE1_GENE2/GENE3_GENE4/...'
    Splits by '/' to get pairs, then by '_' (first occurrence) to get source/target genes.
    """
    if not imgp_string or pd.isna(imgp_string):
        return pd.DataFrame(columns=['source_gene', 'target_gene', 'value'])
    rows = []
    for pair in str(imgp_string).split('/'):
        pair = pair.strip()
        if '_' in pair:
            src, tgt = pair.split('_', 1)
            rows.append({'source_gene': src, 'target_gene': tgt, 'value': 1})
    return pd.DataFrame(rows)


def _sample_chord_ribbon(a1, a2, r, hw1, hw2, n_bezier=20, n_arc=10):
    """Return (xs, ys) tracing the closed ribbon boundary for a go.Scatter trace."""
    ctrl = np.array([0.0, 0.0])
    t = np.linspace(0, 1, n_bezier)

    def _bezier(p0, p2):
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * ctrl[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * ctrl[1] + t ** 2 * p2[1]
        return x.tolist(), y.tolist()

    def _arc(radius, a_start, a_end):
        angles = np.linspace(a_start, a_end, n_arc)
        return (radius * np.cos(angles)).tolist(), (radius * np.sin(angles)).tolist()

    p0 = (r * np.cos(a1 - hw1), r * np.sin(a1 - hw1))
    p1 = (r * np.cos(a2 + hw2), r * np.sin(a2 + hw2))
    p2 = (r * np.cos(a2 - hw2), r * np.sin(a2 - hw2))
    p3 = (r * np.cos(a1 + hw1), r * np.sin(a1 + hw1))

    bx1, by1 = _bezier(p0, p1)
    ax2, ay2 = _arc(r, a2 + hw2, a2 - hw2)
    bx2, by2 = _bezier(p2, p3)
    ax1, ay1 = _arc(r, a1 + hw1, a1 - hw1)

    xs = bx1 + ax2[1:] + bx2 + ax1[1:] + [bx1[0]]
    ys = by1 + ay2[1:] + by2 + ay1[1:] + [by1[0]]
    return xs, ys


def _sample_wedge(a_center, half_w, r_out, r_in, n=15):
    """Return (xs, ys) for a closed annular wedge polygon for go.Scatter."""
    outer = np.linspace(a_center - half_w, a_center + half_w, n)
    inner = np.linspace(a_center + half_w, a_center - half_w, n)
    xs = (r_out * np.cos(outer)).tolist() + (r_in * np.cos(inner)).tolist() + [float(r_out * np.cos(a_center - half_w))]
    ys = (r_out * np.sin(outer)).tolist() + (r_in * np.sin(inner)).tolist() + [float(r_out * np.sin(a_center - half_w))]
    return xs, ys


def _rgba_to_plotly(rgba_tuple, alpha=None):
    """Convert a matplotlib RGBA float tuple to a Plotly CSS rgba() string."""
    r, g, b, a = rgba_tuple
    if alpha is not None:
        a = alpha
    return f'rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{a:.3f})'


def build_chord_figure_plotly(gene_df, title):
    """
    Build an interactive Plotly chord/circos diagram.
    Source genes (Favorable) → LEFT arc, pink/RdPu palette.
    Target genes (Unfavorable) → RIGHT arc, blue palette.
    Hover on ribbons shows the gene pair; hover on arcs shows the gene + connection count.
    """
    if gene_df.empty:
        return None

    from collections import Counter

    sources = gene_df['source_gene'].tolist()
    targets = gene_df['target_gene'].tolist()

    source_set = sorted(set(sources))
    target_set = sorted(set(targets))

    MAX_PER_SIDE = 40
    if len(source_set) > MAX_PER_SIDE:
        source_set = sorted(g for g, _ in Counter(sources).most_common(MAX_PER_SIDE))
    if len(target_set) > MAX_PER_SIDE:
        target_set = sorted(g for g, _ in Counter(targets).most_common(MAX_PER_SIDE))

    mask = gene_df['source_gene'].isin(source_set) & gene_df['target_gene'].isin(target_set)
    gene_df = gene_df[mask].reset_index(drop=True)

    if len(gene_df) == 0:
        return None

    n_src = len(source_set)
    n_tgt = len(target_set)

    # ── Angle layout (identical to the matplotlib version) ────────────────────
    GAP     = 0.14
    ARC_PAD = 0.008
    R_OUT   = 1.0
    R_IN    = 0.86
    LABEL_R = R_OUT + 0.11

    src_start = np.pi / 2 + GAP
    src_end   = 3 * np.pi / 2 - GAP
    src_span  = src_end - src_start

    tgt_start = -np.pi / 2 + GAP
    tgt_end   =  np.pi / 2 - GAP
    tgt_span  = tgt_end - tgt_start

    src_arc_w = max((src_span - (n_src - 1) * ARC_PAD) / n_src, 0.005)
    tgt_arc_w = max((tgt_span - (n_tgt - 1) * ARC_PAD) / n_tgt, 0.005)

    src_angles = {g: src_start + i * (src_arc_w + ARC_PAD) + src_arc_w / 2
                  for i, g in enumerate(source_set)}
    tgt_angles = {g: tgt_start + i * (tgt_arc_w + ARC_PAD) + tgt_arc_w / 2
                  for i, g in enumerate(target_set)}

    # ── Colour palettes ───────────────────────────────────────────────────────
    src_cmap = plt.cm.RdPu
    tgt_cmap = plt.cm.Blues
    src_colors = {g: src_cmap(0.30 + 0.60 * i / max(n_src - 1, 1))
                  for i, g in enumerate(source_set)}
    tgt_colors = {g: tgt_cmap(0.35 + 0.55 * i / max(n_tgt - 1, 1))
                  for i, g in enumerate(target_set)}

    # ── Connection counts for hover labels ────────────────────────────────────
    src_conn = Counter(gene_df['source_gene'])
    tgt_conn = Counter(gene_df['target_gene'])

    n_pairs      = len(gene_df)
    ribbon_alpha = float(np.clip(1.5 / np.sqrt(max(n_pairs, 1)), 0.06, 0.55))
    hw_src = src_arc_w / 2
    hw_tgt = tgt_arc_w / 2

    fig = go.Figure()

    # ── Ribbons (drawn first, behind arcs) ────────────────────────────────────
    for _, row in gene_df.iterrows():
        sg = row['source_gene']
        tg = row['target_gene']
        if sg not in src_angles or tg not in tgt_angles:
            continue
        xs, ys = _sample_chord_ribbon(
            src_angles[sg], tgt_angles[tg], R_IN, hw_src, hw_tgt
        )
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='lines',
            fill='toself',
            fillcolor=_rgba_to_plotly(src_colors[sg], alpha=ribbon_alpha),
            line=dict(width=0),
            hovertemplate=(
                '<b>Source:</b> %{customdata[0]}<br>'
                '<b>Target:</b> %{customdata[1]}'
                '<extra></extra>'
            ),
            customdata=[[sg, tg]] * len(xs),
            showlegend=False,
            name='',
        ))

    # ── Source gene arc segments ──────────────────────────────────────────────
    for gene in source_set:
        xs, ys = _sample_wedge(src_angles[gene], hw_src, R_OUT, R_IN)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='lines',
            fill='toself',
            fillcolor=_rgba_to_plotly(src_colors[gene]),
            line=dict(color='white', width=0.6),
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'Favorable (source) gene<br>'
                'Connections: %{customdata[1]}'
                '<extra></extra>'
            ),
            customdata=[[gene, src_conn[gene]]] * len(xs),
            showlegend=False,
            name='',
        ))

    # ── Target gene arc segments ──────────────────────────────────────────────
    for gene in target_set:
        xs, ys = _sample_wedge(tgt_angles[gene], hw_tgt, R_OUT, R_IN)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='lines',
            fill='toself',
            fillcolor=_rgba_to_plotly(tgt_colors[gene]),
            line=dict(color='white', width=0.6),
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'Unfavorable (target) gene<br>'
                'Connections: %{customdata[1]}'
                '<extra></extra>'
            ),
            customdata=[[gene, tgt_conn[gene]]] * len(xs),
            showlegend=False,
            name='',
        ))

    # ── Rotated gene labels via annotations ───────────────────────────────────
    fontsize = 9 if (n_src + n_tgt) < 50 else 7
    all_gene_angles = {**src_angles, **tgt_angles}
    for gene, a in all_gene_angles.items():
        rot_mpl = np.degrees(a)
        xanchor = 'left'
        if np.cos(a) < 0:
            rot_mpl += 180
            xanchor = 'right'
        fig.add_annotation(
            x=LABEL_R * np.cos(a),
            y=LABEL_R * np.sin(a),
            text=gene,
            showarrow=False,
            textangle=-rot_mpl,   # Plotly is CW-positive; matplotlib is CCW-positive
            xanchor=xanchor,
            yanchor='middle',
            font=dict(size=fontsize, color='#1a1a2e'),
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color='#1a1a2e'), x=0.5),
        xaxis=dict(range=[-1.75, 1.75], visible=False, scaleanchor='y'),
        yaxis=dict(range=[-1.75, 1.75], visible=False),
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode='closest',
        showlegend=False,
        dragmode=False,
    )
    return fig, len(gene_df)


def chord_html_with_hover(fig, n_ribbons, height=670):
    """
    Wrap a chord go.Figure as a self-contained HTML page with mouseover
    ribbon-highlighting behaviour injected as a post-render JS script.

    Hover over an arc  → fade all ribbons not connected to that gene (opacity 0.06).
    Hover over a ribbon → fade all other ribbons (opacity 0.06).
    Mouse-out           → restore all ribbons to full opacity.
    """
    js = f"""
    (function() {{
        var gd = document.getElementById('{{plot_id}}');
        var nR = {n_ribbons};

        gd.on('plotly_hover', function(ev) {{
            var pt = ev.points[0];
            var cd = pt.customdata;
            if (!cd) return;

            var isRibbon   = (pt.curveNumber < nR);
            var geneA      = cd[0];
            var geneBOrNull = isRibbon ? cd[1] : null;

            var opacities = [];
            for (var i = 0; i < nR; i++) {{
                var d    = gd.data[i];
                var rSrc = d.customdata[0][0];
                var rTgt = d.customdata[0][1];
                var show = isRibbon
                    ? (rSrc === geneA && rTgt === geneBOrNull)
                    : (rSrc === geneA || rTgt === geneA);
                opacities.push(show ? 1.0 : 0.06);
            }}

            var idxs = Array.from({{length: nR}}, function(_, k) {{ return k; }});
            Plotly.restyle(gd, {{opacity: opacities}}, idxs);
        }});

        gd.on('plotly_unhover', function() {{
            var idxs = Array.from({{length: nR}}, function(_, k) {{ return k; }});
            Plotly.restyle(gd, {{opacity: 1.0}}, idxs);
        }});
    }})();
    """
    return fig.to_html(
        include_plotlyjs='cdn',
        full_html=True,
        post_script=js,
        config={'displaylogo': False},
    )


def _fmt_stat(v):
    """Format a numeric stat value for display."""
    try:
        f = float(v)
        if f == 0.0:
            return "0"
        if abs(f) < 0.001:
            return f"{f:.2e}"
        return f"{f:.4f}"
    except (TypeError, ValueError):
        return str(v)


def _show_interaction_stats_row(row_df, label):
    """Display per-interaction statistics in a compact grid."""
    if row_df is None or (hasattr(row_df, 'empty') and row_df.empty):
        st.info(f"No {label} statistics available for this pair.")
        return

    row = row_df.iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("P-Value", _fmt_stat(row.get('P.Value', 'N/A')))
    with c2:
        st.metric("Adj. P-Value", _fmt_stat(row.get('Adjust.P.Value', 'N/A')))
    with c3:
        st.metric("Shared IMGP", str(int(row['No.Shared.IMGP']))
                  if pd.notna(row.get('No.Shared.IMGP')) else 'N/A')
    with c4:
        st.metric("Total IMGP", str(int(row['No.Total.IMGP']))
                  if pd.notna(row.get('No.Total.IMGP')) else 'N/A')
    with c5:
        st.metric("Enrichment Ratio", _fmt_stat(row.get('Enrichment.Ratio', 'N/A')))


@st.cache_data
def _load_interaction_map(sig_key):
    """
    Load both NW and OW CSVs for a signature (cached).
    Returns nw_df, ow_df, fav_cells, unfav_cells, fav_to_unfav, unfav_to_fav.
    """
    paths = INDIVIDUAL_INTERACTION_SIGNATURES[sig_key]
    nw_df = pd.read_csv(paths["normal"])
    try:
        ow_df = pd.read_csv(paths["overweight"])
    except Exception:
        ow_df = pd.DataFrame(columns=nw_df.columns)

    combined = pd.concat([nw_df, ow_df], ignore_index=True)

    fav_cells   = sorted(combined['Favorable.Cell.Type'].dropna().unique())
    unfav_cells = sorted(combined['Unfavorable.Cell.Type'].dropna().unique())

    fav_to_unfav = {
        f: sorted(combined[combined['Favorable.Cell.Type'] == f]
                  ['Unfavorable.Cell.Type'].dropna().unique())
        for f in fav_cells
    }
    unfav_to_fav = {
        u: sorted(combined[combined['Unfavorable.Cell.Type'] == u]
                  ['Favorable.Cell.Type'].dropna().unique())
        for u in unfav_cells
    }
    return nw_df, ow_df, fav_cells, unfav_cells, fav_to_unfav, unfav_to_fav


def render_individual_interaction():
    """Render the Explore Individual Interaction mode with beautiful chord diagrams."""
    st.markdown('<div class="header-spacer"></div>', unsafe_allow_html=True)
    st.markdown(
        '<h1 class="main-header">🎵 Gene-Level Interaction Explorer</h1>',
        unsafe_allow_html=True
    )

    # ── Colour key legend ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex; align-items:center; gap:24px; padding:10px 16px;
                background:#f8f9fa; border-radius:8px; margin-bottom:16px;
                border-left:4px solid #aaa; font-size:0.9rem;">
      <strong>Chord Diagram Key:</strong>
      <span style="display:inline-flex; align-items:center; gap:6px;">
        <span style="width:14px; height:14px; border-radius:3px;
                     background:#c8488c; display:inline-block;"></span>
        <span><b style="color:#1a7a3a;">Favorable (source)</b> gene arcs — left half, pink palette</span>
      </span>
      <span style="display:inline-flex; align-items:center; gap:6px;">
        <span style="width:14px; height:14px; border-radius:3px;
                     background:#3182bd; display:inline-block;"></span>
        <span><b style="color:#b22222;">Unfavorable (target)</b> gene arcs — right half, blue palette</span>
      </span>
      <span style="display:inline-flex; align-items:center; gap:6px;">
        <span style="width:14px; height:14px; border-radius:3px;
                     background:linear-gradient(90deg,#f7a8c8,#c8488c); display:inline-block;"></span>
        <span>Ribbons = Shared IMGP gene pairs</span>
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h3>🧬 Gene Interaction Chord Diagrams</h3>
        <p>Use the controls below to select a signature and cell type pair. Both Normal Weight (left) and Overweight (right) chord diagrams are
        shown side by side, plotting the <b>Shared IMGP</b> gene pairs for each group.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Main-page controls ────────────────────────────────────────────────────
    st.markdown("---")

    # Row 1 — Signature
    signature = st.selectbox(
        "📊 Signature dataset",
        options=list(INDIVIDUAL_INTERACTION_SIGNATURES.keys()),
        key="indiv_signature",
    )

    # Load interaction map (cached)
    try:
        nw_df, ow_df, fav_cells, unfav_cells, fav_to_unfav, unfav_to_fav = \
            _load_interaction_map(signature)
    except Exception as e:
        st.error(f"Could not load data for {signature}: {e}")
        return

    # Row 2 — Selection direction
    start_mode = st.radio(
        "🔀 Start selection from:",
        options=["Favorable cell", "Unfavorable cell"],
        horizontal=True,
        key="indiv_start_mode",
        help="Choose which cell type to pick first — the second dropdown will show only cells that interact with your first choice.",
    )
    st.caption("① pick first  →  ② list is filtered to matching interactions only")

    col_a, col_b = st.columns(2)

    if start_mode == "Favorable cell":
        with col_a:
            source_cell = st.selectbox(
                "① Favorable Cell Type",
                options=fav_cells,
                key="indiv_source_cell",
            )
        related_unfav = fav_to_unfav.get(source_cell, unfav_cells)
        if st.session_state.get("indiv_target_cell") not in related_unfav:
            st.session_state["indiv_target_cell"] = related_unfav[0]
        with col_b:
            target_cell = st.selectbox(
                "② Unfavorable Cell Type (filtered)",
                options=related_unfav,
                key="indiv_target_cell",
            )
    else:  # Unfavorable cell first
        with col_a:
            target_cell = st.selectbox(
                "① Unfavorable Cell Type",
                options=unfav_cells,
                key="indiv_target_cell",
            )
        related_fav = unfav_to_fav.get(target_cell, fav_cells)
        if st.session_state.get("indiv_source_cell") not in related_fav:
            st.session_state["indiv_source_cell"] = related_fav[0]
        with col_b:
            source_cell = st.selectbox(
                "② Favorable Cell Type (filtered)",
                options=related_fav,
                key="indiv_source_cell",
            )

    st.markdown("---")

    # ── Fetch matching rows ───────────────────────────────────────────────────
    nw_row = nw_df[
        (nw_df['Favorable.Cell.Type'] == source_cell) &
        (nw_df['Unfavorable.Cell.Type'] == target_cell)
    ]
    ow_row = ow_df[
        (ow_df['Favorable.Cell.Type'] == source_cell) &
        (ow_df['Unfavorable.Cell.Type'] == target_cell)
    ]

    # ── Parse Shared IMGP gene pairs ─────────────────────────────────────────
    nw_imgp = nw_row['Shared.IMGP'].values[0] if not nw_row.empty else None
    ow_imgp = ow_row['Shared.IMGP'].values[0] if not ow_row.empty else None

    nw_genes = parse_imgp_to_dataframe(nw_imgp)
    ow_genes = parse_imgp_to_dataframe(ow_imgp)

    # ── Coloured interaction title ─────────────────────────────────────────────
    st.markdown(
        f"### Gene Interactions: "
        f"<span style='color:#1a7a3a; font-weight:bold;'>{source_cell}</span>"
        f" <span style='color:#555;'>→</span> "
        f"<span style='color:#b22222; font-weight:bold;'>{target_cell}</span>",
        unsafe_allow_html=True
    )
    st.caption(
        "🟢 Green = Favorable cell type (source genes on LEFT arc) | "
        "🔴 Red = Unfavorable cell type (target genes on RIGHT arc)"
    )
    st.markdown("---")

    # ── Side-by-side chord diagrams ───────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<h4 style='color:#1a7a3a;'>Normal Weight</h4>",
            unsafe_allow_html=True
        )
        # Stats first so they're visible without scrolling
        st.markdown("**Interaction Statistics — Normal Weight**")
        _show_interaction_stats_row(nw_row if not nw_row.empty else None, "Normal Weight")
        if nw_genes.empty:
            st.info("No Shared IMGP gene pairs found for this interaction in the Normal Weight dataset.")
        else:
            try:
                result_nw = build_chord_figure_plotly(
                    nw_genes,
                    f"Normal Weight — {source_cell} → {target_cell}"
                )
                if result_nw is not None:
                    fig_nw, nr_nw = result_nw
                    components.html(
                        chord_html_with_hover(fig_nw, nr_nw),
                        height=690, scrolling=False
                    )
            except Exception as e:
                st.error(f"Could not render Normal Weight chord diagram: {e}")

    with col2:
        st.markdown(
            "<h4 style='color:#b22222;'>Overweight</h4>",
            unsafe_allow_html=True
        )
        # Stats first so they're visible without scrolling
        st.markdown("**Interaction Statistics — Overweight**")
        _show_interaction_stats_row(ow_row if not ow_row.empty else None, "Overweight")
        if ow_genes.empty:
            st.info("No Shared IMGP gene pairs found for this interaction in the Overweight dataset.")
        else:
            try:
                result_ow = build_chord_figure_plotly(
                    ow_genes,
                    f"Overweight — {source_cell} → {target_cell}"
                )
                if result_ow is not None:
                    fig_ow, nr_ow = result_ow
                    components.html(
                        chord_html_with_hover(fig_ow, nr_ow),
                        height=690, scrolling=False
                    )
            except Exception as e:
                st.error(f"Could not render Overweight chord diagram: {e}")

    # ── Summary counts ────────────────────────────────────────────────────────
    st.markdown("---")
    s1, s2 = st.columns(2)
    with s1:
        st.metric("Normal Weight — Shared IMGP gene pairs", len(nw_genes))
    with s2:
        st.metric("Overweight — Shared IMGP gene pairs", len(ow_genes))


def main():
    # Analysis type selector
    st.sidebar.title("Analysis Type")

    analysis_mode = st.sidebar.radio(
        "Select Analysis:",
        options=[
            "Signature Explorer",
            "Categorical Analysis",
            "Continuous Analysis",
            "Signature Survival",
            "Interactome Analysis",
            "Explore Individual Interaction",
            "📖 Study Methodology",
            "🧮 Bayesian Model Explained",
        ],
        index=1,
        key="analysis_mode_selector"
    )

    if analysis_mode == "Signature Explorer":
        st.sidebar.info("Browse the signature database")
    elif analysis_mode == "Categorical Analysis":
        st.sidebar.success("Compare BMI groups (Normal, Overweight, Obese)")
    elif analysis_mode == "Continuous Analysis":
        st.sidebar.success("Model BMI as continuous (dose-response)")
    elif analysis_mode == "Signature Survival":
        st.sidebar.warning("Survival analysis stratified by BMI")
    elif analysis_mode == "Interactome Analysis":
        st.sidebar.info("Explore cell-cell interaction networks")
    elif analysis_mode == "📖 Study Methodology":
        st.sidebar.info("Full walkthrough of every analytical step")
    elif analysis_mode == "🧮 Bayesian Model Explained":
        st.sidebar.info("Deep-dive into the Bayesian hierarchical model")
    else:
        st.sidebar.info("Drill into gene-level interactions for a specific cell pair")

    st.sidebar.markdown("---")

    with st.sidebar.expander("Contact the Author", expanded=False):
        components.html("""
        <style>
          * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
          label { display: block; margin-bottom: 3px; color: #555; font-size: 12px; }
          input, textarea {
            width: 100%; padding: 6px 8px; margin-bottom: 10px;
            border: 1px solid #d0d0d0; border-radius: 4px; font-size: 13px;
            background: #fafafa; color: #111;
          }
          textarea { height: 72px; resize: vertical; }
          button {
            background: #1f6feb; color: white; padding: 7px 0;
            border: none; border-radius: 4px; cursor: pointer;
            font-size: 13px; width: 100%;
          }
          button:disabled { background: #aaa; cursor: not-allowed; }
          #msg { margin-top: 8px; font-size: 12px; text-align: center; }
        </style>
        <form id="cf">
          <label>Email</label>
          <input type="email" name="email" required placeholder="your@email.com" />
          <label>Message</label>
          <textarea name="message" required placeholder="Your message"></textarea>
          <button type="submit" id="btn">Send</button>
          <div id="msg"></div>
        </form>
        <script>
          document.getElementById('cf').addEventListener('submit', async function(e) {
            e.preventDefault();
            const btn = document.getElementById('btn');
            const msg = document.getElementById('msg');
            btn.disabled = true; btn.textContent = 'Sending...';
            try {
              const res = await fetch('https://formspree.io/f/xreopraq', {
                method: 'POST', body: new FormData(this),
                headers: { 'Accept': 'application/json' }
              });
              if (res.ok) {
                msg.style.color = '#2d7d2d';
                msg.textContent = 'Message sent. Thank you.';
                this.reset();
              } else { throw new Error(); }
            } catch(_) {
              msg.style.color = '#c0392b';
              msg.textContent = 'Failed to send. Please try again.';
            }
            btn.disabled = false; btn.textContent = 'Send';
          });
        </script>
        """, height=290)

    # ── Full-page HTML doc views: exit BEFORE any Streamlit header is rendered ──
    if analysis_mode == "📖 Study Methodology":
        render_study_methodology()
        return
    elif analysis_mode == "🧮 Bayesian Model Explained":
        render_bayesian_explained()
        return

    # Spacer div (expands when header becomes fixed)
    st.markdown('<div class="header-spacer"></div>', unsafe_allow_html=True)

    # Main header
    st.markdown('''
    <h1 class="main-header">
        Obesity-Driven Pancreatic Ductal Adenocarcinoma: An ML driven bayesian model
    </h1>
    ''', unsafe_allow_html=True)

    # Info box
    st.markdown("""
    <div class="info-box">
        <p>
        This platform accompanies a study characterising obesity-driven remodeling of the tumor microenvironment
        in pancreatic ductal adenocarcinoma (PDAC). Results are derived from a Bayesian hierarchical model
        applied to the CPTAC-PAAD cohort (140 samples). Select an analysis module from the sidebar to explore
        cell-type-resolved BMI effects, dose-response models, cell-cell interaction networks, and survival associations.
        </p>
        <p style="margin-top: 10px;">
            <a href="https://github.com/arunviswanathan91/cell-analysis-viewer" target="_blank"
               style="color: #1f6feb; text-decoration: none; margin-right: 20px;">
               Viewer repository
            </a>
            <a href="https://github.com/arunviswanathan91/obese-model" target="_blank"
               style="color: #1f6feb; text-decoration: none;">
               Analysis code repository
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)


    # Route to appropriate analysis mode
    if analysis_mode == "Signature Explorer":
        render_signature_explorer()
        return

    elif analysis_mode == "Continuous Analysis":
        render_continuous_analysis()
        return

    elif analysis_mode == "Signature Survival":
        render_signature_survival()
        return

    elif analysis_mode == "Interactome Analysis":
        render_interactome_analysis()
        return

    elif analysis_mode == "Explore Individual Interaction":
        render_individual_interaction()
        return

    # Continue with Statistical Analysis mode
    
    # Methodology Section (Collapsible)
    with st.expander("**About the Analysis Methods**", expanded=False):
        st.markdown("""
        ###  Data & Methods Overview
        
        This analysis integrates multiple computational approaches to understand how obesity affects the tumor microenvironment in pancreatic cancer:
        
        ---
        
        #### **BayesPrism** - Cell Type Deconvolution
        A fully Bayesian method that infers tumor microenvironment composition from bulk RNA-seq data. BayesPrism estimates the proportion of different cell types in each tumor sample, providing cell-type-specific gene expression profiles.
        
        **Reference:** [Danko-Lab/BayesPrism](https://github.com/Danko-Lab/BayesPrism)
        
        ---
        
        #### **Stabl** - Feature Selection
        Stability-driven feature selection that identifies the most robust biomarkers associated with BMI status. Stabl uses bootstrapping to find features that consistently show effects across multiple random samplings, reducing false positives.
        
        **Reference:** [gregbellan/Stabl](https://github.com/gregbellan/Stabl)
        
        ---
        
        #### **Bayesian Hierarchical Model** - Effect Size Estimation
        A three-group model comparing: 
        - **Normal BMI** (< 25) vs **Overweight** (25-30) vs **Obese** (>30)
        
        The model estimates cell-type-specific effects of obesity on metabolic signatures while accounting for between-sample variability. Uses **Markov Chain Monte Carlo (MCMC)** for posterior sampling.
        
        **References:**
        - [Bayesian Hierarchical Modeling - Wikipedia](https://en.wikipedia.org/wiki/Bayesian_hierarchical_modeling)
        - [Markov Chain Monte Carlo - Wikipedia](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)
        
        ---
        
        #### **Diagnostic Metrics**
        - **R-hat:** Measures convergence (should be < 1.01 for good convergence)
        - **ESS (Effective Sample Size):** Number of independent samples (higher is better, > 400 recommended)
        - **Energy:** Hamiltonian Monte Carlo diagnostic (identifies sampling problems)
        - **Credible Intervals:** Bayesian equivalent of confidence intervals (95% HDI)
        
        ---
        
        #### **Dataset**
        - **Source:** CPTAC Pancreatic Adenocarcinoma (PAAD) cohort
        - **Samples:** 140 tumor samples with clinical annotations
        - **Cell Types:** Deconvolved into immune and non-immune cell populations
        - **Signatures:** 30+ metabolic and functional gene signatures per cell type
        
        ---
        
        #### **Analysis Workflow**
        1. **Deconvolution:** BayesPrism ➜ Cell type proportions/Cell-specific expression matrix
        2. **Expression:** TPM values from CPTAC-3 ➜ Gene expression matrix of PDAC patients
        3. **Signatures:** Custom signature databse ➜ Signature scores (Z-scores)
        4. **Selection:** Sabl ML based  ➜ Robust BMI-associated features
        5. **Modeling:** Bayesian hierarchical with MCMC ➜ Effect sizes with uncertainty (Feature level/ Cell level)
        6. **Validation:** MCMC diagnostics ➜ Convergence checks
        7. **Survival:** Cox regression ➜ Clinical relevance of creble signature/features and cell type
        """)
    
    # ── Main-page selection controls ──────────────────────────────────────────
    st.markdown("### 🛠️ Data Selection")

    sel_col1, sel_col2 = st.columns(2)

    # Step 1: Compartment
    with sel_col1:
        compartment = st.selectbox(
            "1. Choose compartment:",
            options=['Immune Fine', 'Immune Coarse', 'Non-Immune'],
            index=0,
            key='cat_compartment',
        )

    # Load data
    with st.status("Loading analysis data...", expanded=False) as _load_status:
        st.write("Loading compartment data...")
        comp_data = load_compartment_data(compartment)
        st.write("Loading clinical data...")
        clinical = load_clinical_data()
        st.write("Loading TPM data...")
        tpm = load_tpm_data()
        _load_status.update(label="Data ready", state="complete", expanded=False)

    # Step 2: Cell Type
    available_cells = get_available_cells(compartment)

    if not available_cells:
        st.error("❌ No cell types found")
        return

    cell_display = {cell.replace('_', ' ').title(): cell for cell in available_cells}

    with sel_col2:
        selected_cell_display = st.selectbox(
            f"2. Choose cell type ({len(available_cells)} available):",
            options=list(cell_display.keys()),
            index=0,
            key='cat_cell',
        )
    selected_cell = cell_display[selected_cell_display]

    # Step 3: Signature
    signatures = get_cell_signatures(selected_cell)

    if not signatures:
        st.warning(f"❌ No signatures found for {selected_cell}")
        return

    sig_options = {}
    for s in signatures:
        formatted_name = format_signature_name(s['signature'], max_length=35)
        display_text = f"{formatted_name} ({len(s['genes'])} genes)"
        sig_options[display_text] = s

    selected_sig_display = st.selectbox(
        f"3. Choose signature ({len(signatures)} available):",
        options=list(sig_options.keys()),
        index=0,
        help="Signature names are truncated for readability. Full name shown in results.",
        key='cat_signature',
    )
    selected_sig_info = sig_options[selected_sig_display]
    sig_name = selected_sig_info['signature']
    genes = selected_sig_info['genes']

    # Sidebar summary
    st.sidebar.markdown("### Current Selection")
    st.sidebar.info(f"""
    **Compartment:** {compartment}
    **Cell Type:** {selected_cell_display}
    **Signature:** {format_signature_name(sig_name, max_length=50)}
    **Genes:** {len(genes)}
    """)
    if len(sig_name) > 50:
        st.sidebar.caption(f"Full name: {sig_name.replace('_', ' ')}")

    st.divider()

    # Main content
    st.markdown(f'<div class="sub-header"> Interactive Analysis Results</div>', 
               unsafe_allow_html=True)
        
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Compartment", compartment)
    with col2:
        st.metric("Cell Type", selected_cell_display)
    with col3:
        st.metric("Signature", sig_name.replace('_', ' '))
    with col4:
        st.metric("Genes", len(genes))
    
    # Tabs (Ask Model is the global mode — no per-tab Ask AI)
    tabs = st.tabs([
        "STABL & Bayesian",
        "Ridge Plot",
        "Diagnostics",
        "Gene BMI",
        "Gene Survival",
        ])
    
    # Tab 1: STABL & Bayesian
    with tabs[0]:
        st.markdown("### STABL Feature Selection")
        
        st.markdown("""
        <div class="method-box">
        <b>❓ What is STABL?</b><br>
        STABL (STABility-driven feature seLection) identifies robust biomarkers by:
        <ol>
        <li>Running feature selection on multiple bootstrap samples</li>
        <li>Counting how often each feature is selected</li>
        <li>Keeping only features selected consistently (stable features)</li>
        </ol>
        <b>⭐ Stars mark STABL-selected features</b> - these show the most robust associations with BMI status.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Z-score Heatmap")
        st.caption("Z-scores represent standardized signature expression across BMI categories")
        with st.spinner("Loading chart..."):
            fig = plot_stabl_heatmap_interactive(selected_cell, sig_name, comp_data, clinical)
            if fig:
                st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        
        st.markdown("### 📝 Bayesian Effect Size Estimation")
        
        st.markdown("""
        <div class="method-box">
        <b>📛 Bayesian Hierarchical Model</b><br>
        Estimates how much each cell type's signature changes with increasing BMI:
        <ul>
        <li><b>Blue bars:</b> Overweight vs Normal effect</li>
        <li><b>Red bars:</b> Obese vs Normal effect</li>
        <li><b>Green bars:</b> Obese vs Overweight effect</li>
        <li><b>Error bars:</b> 95% Credible Intervals (uncertainty)</li>
        </ul>
        <b>Interpretation:</b> Positive = signature increased with higher BMI, Negative = signature decreased
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📝 Effect Sizes with Credible Intervals")
        st.caption("Hover for exact effect sizes | Click legend to toggle comparisons")
        with st.spinner("Loading chart..."):
            fig = plot_bayesian_heatmap_interactive(selected_cell, sig_name, comp_data)
            if fig:
                st.plotly_chart(fig, width='stretch')
    
    
    # Tab 2: Ridge Plot
    with tabs[1]:
        st.markdown("### 🧾 Posterior Distribution Visualization")
    
        # 🔽 Dropdown explanation
        with st.expander("📖 How to interpret the ridge plot", expanded=False):
            st.markdown("""
            <div class="method-box">
            <b>Ridge Plots Explained</b><br>
            Each "ridge" shows the full distribution of MCMC samples for one cell type:
            <ul>
            <li><b>Width:</b> Uncertainty in effect size estimate</li>
            <li><b>Peak location:</b> Most likely effect size</li>
            <li><b>Overlap with zero:</b> Effect may not be significant</li>
            <li><b>Vertical lines:</b> Mean effect sizes for each BMI comparison</li>
            </ul>
            <b>Colors:</b> Blue = Overweight, Red = Obese, Green = Obese vs Overweight
            </div>
            """, unsafe_allow_html=True)
    
        # ✅ Now actually draw the ridge plot
        st.markdown("#### 📊 Overlapped Posterior Distributions")
        st.caption("Interactive ridge plot | Hover for details | Scroll to zoom | Double-click to reset")
    
        with st.spinner("Loading chart..."):
            fig = plot_overlapped_ridges_interactive(selected_cell, comp_data)
            if fig:
                st.plotly_chart(fig, width='content')
            else:
                st.info("ℹ️ Ridge plot not available for this selection.")



    
# Tab 3: Bayesian Diagnostics - 2-Column Layout for Visual Comparison
    with tabs[2]:
        st.markdown("### MCMC Diagnostics")
        
        # Collapsible guide at the top
        with st.expander("Understanding MCMC Diagnostics", expanded=False):
            st.markdown("""
            **MCMC** (Markov Chain Monte Carlo) explores parameter space to estimate posterior distributions.
            Diagnostics verify the sampler converged and explored properly.

            **Good Signs:** R-hat < 1.01, ESS > 400, smooth energy, "hairy caterpillar" traces, uniform ranks.

            **Warning Signs:** R-hat > 1.05, ESS < 100, divergent transitions, trending traces.
            """)

        # Row 1: ESS/R-hat and Energy (side by side)
        diag_col1, diag_col2 = st.columns(2)

        with diag_col1:
            st.markdown("#### ESS & R-hat")
            with st.expander("What does this show?", expanded=False):
                st.markdown("""
                **ESS:** Number of independent samples (target > 400).
                **R-hat:** Chain agreement (excellent < 1.01, acceptable < 1.05).
                """)
            with st.spinner("Loading chart..."):
                fig = plot_ess_rhat_categorical(comp_data, selected_cell=selected_cell)
                if fig:
                    st.plotly_chart(fig, width='stretch')

        with diag_col2:
            st.markdown("#### Energy")
            with st.expander("What does this show?", expanded=False):
                st.markdown("""
                **Energy:** HMC sampling quality. Smooth transitions = good mixing.
                Divergent transitions indicate sampling difficulties.
                """)
            with st.spinner("Loading chart..."):
                fig = plot_energy_diagnostic(comp_data)
                if fig:
                    st.plotly_chart(fig, width='stretch')

        # Row 2: Trace Plot (full width — needs space for many chains)
        st.markdown("#### Trace Plot")
        with st.expander("What does this show?", expanded=False):
            st.markdown("""
            **Good:** "Hairy caterpillar" pattern with overlapping chains.
            **Bad:** Trends, stuck chains, or chains not overlapping.
            """)
        with st.spinner("Loading chart..."):
            fig = plot_trace_diagnostic(comp_data, selected_cell=selected_cell)
            if fig:
                st.plotly_chart(fig, width='stretch')

        # Row 3: Rank Plot (full width)
        st.markdown("#### Rank Plot")
        with st.expander("What does this show?", expanded=False):
            st.markdown("""
            **Good:** Uniform distributions across all chains.
            **Bad:** Non-uniform = chains exploring different regions.
            """)
        with st.spinner("Loading chart..."):
            fig = plot_rank_diagnostic(comp_data, selected_cell=selected_cell)
            if fig:
                st.plotly_chart(fig, width='stretch')

        # Row 3: Autocorrelation (full width)
        st.markdown("#### Autocorrelation")
        with st.expander("What does this show?", expanded=False):
            st.markdown("""
            **Good:** Rapid decay to zero (independent samples).
            **Bad:** Slow decay = high autocorrelation, low effective samples.
            """)
        with st.spinner("Loading chart..."):
            fig = plot_autocorrelation(comp_data, selected_cell=selected_cell, max_lag=40)
            if fig:
                st.plotly_chart(fig, width='stretch')

                
    # Tab 4: Gene BMI
    with tabs[3]:
        st.markdown("### 📈  Gene-Level BMI Associations")
        st.info(" Hover for statistics| Click-drag to zoom | Double-click to reset")
        with st.spinner("Loading chart..."):
            fig1, fig2 = plot_gene_bmi_interactive(genes, clinical, tpm)
            if fig1:
                st.plotly_chart(fig1, width='stretch')
            if fig2:
                st.plotly_chart(fig2, width='stretch')
    
    # Tab 5: Gene Survival
    with tabs[4]:
        st.markdown("### 📈  Gene-Level Survival Analysis")
        st.info("Forest plot with confidence intervals| Hover for full statistics")
        with st.spinner("Loading chart..."):
            fig = plot_gene_survival_interactive(genes, clinical, tpm)
            if fig:
                st.plotly_chart(fig, width='stretch')

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
    <b>Interactive Cell Analysis Viewer</b><br>
    Real-time interactive visualizations with Plotly<br>
    <i>Zoom | Pan | Hover | Explore</i>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
