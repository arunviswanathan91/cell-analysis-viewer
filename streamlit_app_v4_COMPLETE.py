"""
Obesity-Driven Pancreatic Cancer: Cell-Signature Analysis Platform
Version 4.0 - Professional, Publication-Ready

Complete implementation with:
- Professional styling (no emojis)
- Dual heatmap system (all signatures + STABL-selected)
- Fixed cell type mapping
- Selected cell diagnostics only
- ArviZ energy plots
- Comprehensive methodology descriptions
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import arviz as az
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Obesity-Driven Pancreatic Cancer Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PROFESSIONAL STYLING CONFIGURATION
# ============================================================================

# Color Palette (Professional, Muted)
COLOR_PRIMARY_TEXT = "#1a1a1a"
COLOR_SECONDARY_TEXT = "#4a4a4a"
COLOR_BACKGROUND = "#ffffff"
COLOR_ACCENT = "#2c5aa0"
COLOR_OVERWEIGHT = "#5b8fa3"
COLOR_OBESE = "#a44a4a"
COLOR_OB_OW = "#6b8e6b"

BMI_COLORS = {
    'overweight': COLOR_OVERWEIGHT,
    'obese': COLOR_OBESE,
    'obese_vs_overweight': COLOR_OB_OW
}

BMI_LABELS = {
    'overweight': 'Overweight vs Normal',
    'obese': 'Obese vs Normal',
    'obese_vs_overweight': 'Obese vs Overweight'
}

# Professional CSS (No Emojis)
PROFESSIONAL_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    * {{
        font-family: 'Inter', -apple-system, sans-serif !important;
    }}
    
    .main-title {{
        font-size: 28px;
        font-weight: 600;
        color: {COLOR_PRIMARY_TEXT};
        margin-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }}
    
    .section-title {{
        font-size: 22px;
        font-weight: 600;
        color: {COLOR_PRIMARY_TEXT};
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }}
    
    .info-box {{
        background-color: #f8f8f8;
        padding: 1rem;
        border-left: 3px solid {COLOR_ACCENT};
        border-radius: 2px;
        margin: 1rem 0;
        color: {COLOR_SECONDARY_TEXT};
    }}
    
    .method-box {{
        background-color: #f0f8ff;
        padding: 1.5rem;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        margin: 1.5rem 0;
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    .stButton>button {{
        background-color: {COLOR_ACCENT};
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        border-radius: 2px;
    }}
    
    .stSelectbox label {{
        font-weight: 500 !important;
        color: {COLOR_SECONDARY_TEXT} !important;
    }}
    
    .caption {{
        font-size: 12px;
        color: #6a6a6a;
        font-style: italic;
    }}
</style>
"""

st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_clinical_data():
    """Load clinical data with BMI information"""
    try:
        clinical = pd.read_csv('data/clinical/cptac_complete_clinical.csv')
        return clinical
    except Exception as e:
        st.error(f"Error loading clinical data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_tpm_data():
    """Load TPM expression data"""
    try:
        tpm = pd.read_csv('data/tpm_expression/bulk_combined_with_symbols_cleaned.csv', index_col=0)
        return tpm
    except Exception as e:
        st.error(f"Error loading TPM data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_signatures():
    """Load signature definitions"""
    try:
        with open('data/signatures/ALL_CELL_SIGNATURES_FLAT.json', 'r') as f:
            data = json.load(f)
        return data.get('entries', [])
    except Exception as e:
        st.error(f"Error loading signatures: {e}")
        return []

def load_compartment_data(compartment):
    """Load all data for a compartment"""
    comp_map = {
        'Immune Fine': {
            'dir': 'data/bayesian_csvs/immune_fine',
            'zscore': 'data/zscores/immune_fine_zscores.csv',
            'stabl': 'data/stabl/immune_fine_selected.csv',
            'key': 'immune_fine'
        },
        'Immune Coarse': {
            'dir': 'data/bayesian_csvs/immune_coarse',
            'zscore': 'data/zscores/immune_coarse_zscores.csv',
            'stabl': 'data/stabl/immune_coarse_selected.csv',
            'key': 'immune_coarse'
        },
        'Non-Immune': {
            'dir': 'data/bayesian_csvs/non_immune',
            'zscore': 'data/zscores/non_immune_zscores.csv',
            'stabl': 'data/stabl/non_immune_selected.csv',
            'key': 'non_immune'
        }
    }
    
    comp_data = comp_map[compartment]
    
    # Load celltype mapping
    try:
        ct_path = f"{comp_data['dir']}/celltype_mapping.csv"
        comp_data['celltype_mapping'] = pd.read_csv(ct_path)
    except Exception as e:
        st.error(f"Error loading celltype mapping: {e}")
        comp_data['celltype_mapping'] = pd.DataFrame()
    
    # Load trace if exists
    try:
        trace_path = f"{comp_data['dir']}/posterior_trace.nc"
        if os.path.exists(trace_path):
            comp_data['trace'] = az.from_netcdf(trace_path)
        else:
            comp_data['trace'] = None
    except:
        comp_data['trace'] = None
    
    return comp_data

# ============================================================================
# CELL TYPE NAME CANONICALIZATION
# ============================================================================

CELL_ALIASES = {
    "acinar": "Acinar", "adipocytes": "Adipocytes", "b cells": "B cells",
    "b cells naive": "B cells naive", "basophils": "Basophils",
    "cd4+ t cells": "CD4+ T cells", "cd4 t th1": "CD4 T Th1",
    "cd4 t naive": "CD4 T naive", "cd4 t regulatory": "CD4 T regulatory",
    "cd4 tfh": "CD4 Tfh", "cd8 t effector": "CD8 T effector",
    "cd8 t exhausted": "CD8 T exhausted", "cd8 tmemory": "CD8 Tmemory",
    "cd8 t general": "CD8 T general", "dendritic cells": "Dendritic cells",
    "pdc": "pDC", "mdc": "mDC", "endothelial": "Endothelial",
    "fibroblasts": "Fibroblasts", "icaf": "iCAF", "apcaf": "apCAF",
    "islet endocrine": "Islet endocrine", "monocytes": "Monocytes",
    "monocyte classical": "Monocyte classical",
    "monocyte non classical": "Monocyte non-classical",
    "nk cells": "NK cells", "neural": "Neural", "normal ductal": "Normal ductal",
    "pericytes smc": "Pericytes SMC", "schwann": "Schwann",
    "stellate": "Stellate", "t cells": "T cells",
    "t gamma delta": "T gamma delta", "tam": "TAM",
    "tumor classical": "Tumor classical", "tumor epithelial": "Tumor epithelial"
}

def canon_cell(cell_name):
    """Canonicalize cell type names"""
    if pd.isna(cell_name):
        return cell_name
    
    s = str(cell_name).strip().lower()
    s = s.replace('_', ' ').replace('-', ' ')
    s = ' '.join(s.split())
    
    if s in CELL_ALIASES:
        return CELL_ALIASES[s]
    
    # Fallback formatting
    return ' '.join(word.capitalize() for word in s.split())

def get_available_cells(compartment):
    """Get list of available cell types for compartment"""
    comp_data = load_compartment_data(compartment)
    ct_map = comp_data.get('celltype_mapping')
    
    if ct_map is None or ct_map.empty:
        return []
    
    # Try to get cell names
    if 'celltype_name' in ct_map.columns:
        cells = ct_map['celltype_name'].unique()
    elif 'name' in ct_map.columns:
        cells = ct_map['name'].unique()
    else:
        return []
    
    # Canonicalize
    cells = [canon_cell(c) for c in cells if pd.notna(c)]
    return sorted([c for c in cells if c])

def get_cell_index(cell_name, celltype_mapping):
    """Get numeric index for a cell type"""
    try:
        if 'celltype_name' in celltype_mapping.columns:
            match = celltype_mapping[
                celltype_mapping['celltype_name'].str.upper().str.replace('_', ' ').str.replace('-', ' ') ==
                cell_name.upper().replace('_', ' ').replace('-', ' ')
            ]
            if len(match) > 0:
                return int(match.iloc[0]['celltype_idx'])
        
        # Fallback
        for idx, row in celltype_mapping.iterrows():
            row_name = canon_cell(row.get('celltype_name', row.get('name', '')))
            if row_name.upper() == cell_name.upper():
                return int(row.get('celltype_idx', idx))
        
        return None
    except Exception as e:
        st.error(f"Error getting cell index: {e}")
        return None

# ============================================================================
# SIGNATURE FUNCTIONS
# ============================================================================

def get_cell_signatures(cell_type):
    """Get signatures for a cell type"""
    entries = load_signatures()
    cell_sigs = []
    
    for e in entries:
        entry_cell = canon_cell(e.get('cell_type', ''))
        if entry_cell.upper() == cell_type.upper():
            cell_sigs.append(e)
    
    return cell_sigs

def format_signature_name(sig_name, max_length=40):
    """Format signature name for display"""
    display_name = sig_name.replace('_Signature', '').replace('_signature', '')
    display_name = display_name.replace('_', ' ')
    
    if len(display_name) > max_length:
        display_name = display_name[:max_length-3] + '...'
    
    return display_name

# ============================================================================
# DUAL HEATMAP SYSTEM
# ============================================================================

def load_stabl_selected_features(compartment):
    """Load STABL-selected features"""
    stabl_map = {
        'Immune Fine': 'data/stabl/immune_fine_selected.csv',
        'Immune Coarse': 'data/stabl/immune_coarse_selected.csv',
        'Non-Immune': 'data/stabl/non_immune_selected.csv'
    }
    
    try:
        df = pd.read_csv(stabl_map[compartment])
        return set(df['feature'].values)
    except Exception as e:
        st.error(f"Could not load STABL features: {e}")
        return set()

def compute_tpm_zscores_by_bmi(tpm_data, clinical_data, signature_genes):
    """Compute Z-scores from TPM data across BMI groups"""
    # Define BMI groups
    clinical = clinical_data.copy()
    clinical['BMI_Group'] = pd.cut(
        clinical['BMI'],
        bins=[0, 25, 30, 100],
        labels=['Normal', 'Overweight', 'Obese']
    )
    
    results = []
    
    for gene in signature_genes:
        if gene not in tpm_data.index:
            continue
        
        gene_expr = tpm_data.loc[gene]
        
        # Collect values by BMI group
        groups = {'Normal': [], 'Overweight': [], 'Obese': []}
        
        for _, row in clinical.iterrows():
            sample_id = row['sampleId']
            bmi_group = row['BMI_Group']
            
            if sample_id in gene_expr.index and pd.notna(bmi_group):
                val = gene_expr[sample_id]
                if pd.notna(val):
                    groups[bmi_group].append(val)
        
        # Calculate means
        means = {k: np.mean(v) if v else 0 for k, v in groups.items()}
        
        # Calculate Z-scores
        all_vals = sum(groups.values(), [])
        if len(all_vals) > 1:
            overall_mean = np.mean(all_vals)
            overall_std = np.std(all_vals)
            
            if overall_std > 0:
                zscores = {k: (means[k] - overall_mean) / overall_std for k in means}
            else:
                zscores = {k: 0 for k in means}
        else:
            zscores = {k: 0 for k in means}
        
        results.append({
            'Gene': gene,
            'Normal': zscores['Normal'],
            'Overweight': zscores['Overweight'],
            'Obese': zscores['Obese'],
            'Mean_Z': np.mean([abs(z) for z in zscores.values()])
        })
    
    return pd.DataFrame(results)

def create_heatmap_all_signatures(cell_type, sig_name, signature_genes, 
                                  tpm_data, clinical_data, stabl_features):
    """Create heatmap showing ALL signatures with stars for STABL-selected"""
    
    zscore_df = compute_tpm_zscores_by_bmi(tpm_data, clinical_data, signature_genes)
    
    if zscore_df.empty:
        return None
    
    # Sort by mean Z-score
    zscore_df = zscore_df.sort_values('Mean_Z', ascending=False).head(50)
    
    # Check which are STABL-selected
    feature_string = f"{cell_type}||{sig_name}"
    is_stabl = feature_string in stabl_features
    
    # Create labels
    labels = []
    for gene in zscore_df['Gene']:
        if is_stabl:
            labels.append(f"{gene} ★")
        else:
            labels.append(gene)
    
    # Create heatmap
    z_matrix = zscore_df[['Normal', 'Overweight', 'Obese']].values
    
    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=['Normal', 'Overweight', 'Obese'],
        y=labels,
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title="Z-score"),
        hovertemplate='<b>%{y}</b><br>BMI: %{x}<br>Z-score: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"{cell_type} - {format_signature_name(sig_name)}<br>" +
              "<sub>All Signatures (★ = STABL-selected)</sub>",
        xaxis_title="BMI Category",
        yaxis_title="Gene",
        height=max(400, len(labels) * 15),
        template='plotly_white'
    )
    
    return fig

def create_heatmap_stabl_only(cell_type, sig_name, compartment, clinical_data):
    """Create heatmap showing ONLY STABL-selected features"""
    
    zscore_file = {
        'Immune Fine': 'data/zscores/immune_fine_zscores.csv',
        'Immune Coarse': 'data/zscores/immune_coarse_zscores.csv',
        'Non-Immune': 'data/zscores/non_immune_zscores.csv'
    }
    
    try:
        zscores = pd.read_csv(zscore_file[compartment])
        
        # Filter for this cell and signature
        filtered = zscores[
            (zscores['CellType'].str.upper().str.replace('_', ' ').str.replace('-', ' ') == 
             cell_type.upper().replace('_', ' ').replace('-', ' '))
        ]
        
        if filtered.empty:
            return None
        
        # Get top signatures by Z-score variability
        sig_var = filtered.groupby('Signature')['Z'].std().sort_values(ascending=False).head(30)
        filtered = filtered[filtered['Signature'].isin(sig_var.index)]
        
        # Pivot
        pivot = filtered.pivot_table(
            index='Signature',
            columns='Sample',
            values='Z',
            aggfunc='mean'
        )
        
        # Merge with clinical
        clinical_subset = clinical_data[['sampleId', 'BMI']].copy()
        clinical_subset['BMI_Group'] = pd.cut(
            clinical_subset['BMI'],
            bins=[0, 25, 30, 100],
            labels=['Normal', 'Overweight', 'Obese']
        )
        clinical_subset.set_index('sampleId', inplace=True)
        
        # Get samples that exist in both
        common_samples = pivot.columns.intersection(clinical_subset.index)
        pivot = pivot[common_samples]
        
        # Group by BMI
        bmi_groups = clinical_subset.loc[common_samples, 'BMI_Group']
        grouped = pd.DataFrame({
            'Normal': pivot.loc[:, bmi_groups == 'Normal'].mean(axis=1),
            'Overweight': pivot.loc[:, bmi_groups == 'Overweight'].mean(axis=1),
            'Obese': pivot.loc[:, bmi_groups == 'Obese'].mean(axis=1)
        })
        
        # Format signature names
        formatted_sigs = [format_signature_name(s, max_length=50) for s in grouped.index]
        
        fig = go.Figure(data=go.Heatmap(
            z=grouped.values,
            x=grouped.columns,
            y=formatted_sigs,
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title="Mean Z-score"),
            hovertemplate='<b>%{y}</b><br>BMI: %{x}<br>Mean Z: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{cell_type} - STABL-Selected Features Only",
            xaxis_title="BMI Category",
            yaxis_title="Signature",
            height=max(400, len(formatted_sigs) * 20),
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating STABL heatmap: {e}")
        return None

# ============================================================================
# DIAGNOSTIC PLOTS (SELECTED CELL ONLY)
# ============================================================================

def plot_ess_rhat_single_cell(selected_cell, comp_data):
    """Plot ESS and R-hat for selected cell"""
    
    cell_idx = get_cell_index(selected_cell, comp_data['celltype_mapping'])
    if cell_idx is None:
        return None
    
    try:
        diag_file = f"{comp_data['dir']}/diagnostics_summary.csv"
        diag = pd.read_csv(diag_file)
        
        cell_diag = diag[diag['celltype_idx'] == cell_idx]
        if cell_diag.empty:
            return None
        
        ess_bulk = float(cell_diag.iloc[0].get('ess_bulk', 0))
        r_hat = float(cell_diag.iloc[0].get('r_hat', 1.0))
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Effective Sample Size', 'R-hat Convergence'),
            horizontal_spacing=0.15
        )
        
        # ESS
        fig.add_trace(
            go.Bar(
                x=['ESS'],
                y=[ess_bulk],
                marker_color='steelblue',
                text=[f'{int(ess_bulk)}'],
                textposition='outside',
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_hline(y=400, line_dash="dash", line_color="green",
                     annotation_text="Recommended (400)", row=1, col=1)
        
        # R-hat
        color = 'green' if r_hat < 1.01 else ('orange' if r_hat < 1.05 else 'red')
        status = 'Excellent' if r_hat < 1.01 else ('Acceptable' if r_hat < 1.05 else 'Poor')
        
        fig.add_trace(
            go.Bar(
                x=['R-hat'],
                y=[r_hat],
                marker_color=color,
                text=[f'{r_hat:.4f}'],
                textposition='outside',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_hline(y=1.01, line_dash="dash", line_color="green",
                     annotation_text="Excellent (<1.01)", row=1, col=2)
        fig.add_hline(y=1.05, line_dash="dash", line_color="orange",
                     annotation_text="Acceptable (<1.05)", row=1, col=2)
        
        fig.update_layout(
            title=f"MCMC Diagnostics: {selected_cell}<br>" +
                  f"<sub>Convergence Status: {status}</sub>",
            height=400,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="R-hat", range=[0.99, max(1.1, r_hat + 0.02)], row=1, col=2)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating ESS/R-hat plot: {e}")
        return None

def plot_energy_arviz(comp_data, selected_cell):
    """Create ArviZ-style energy plot"""
    
    try:
        if comp_data['trace'] is None:
            return None
        
        trace = comp_data['trace']
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # ArviZ energy plot
        az.plot_energy(
            trace,
            ax=ax,
            fill_color=("steelblue", "coral"),
            fill_alpha=[0.6, 0.4]
        )
        
        ax.set_title(f"Energy Diagnostic: {selected_cell}", fontsize=16, weight='bold')
        ax.set_xlabel("Energy", fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Convert to image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return buf
        
    except Exception as e:
        st.error(f"Error creating energy plot: {e}")
        return None

def plot_trace_single_cell(selected_cell, comp_data, comparison='overweight'):
    """Plot trace for selected cell"""
    
    try:
        if comp_data['trace'] is None:
            return None
        
        trace = comp_data['trace']
        cell_idx = get_cell_index(selected_cell, comp_data['celltype_mapping'])
        
        if cell_idx is None:
            return None
        
        # Extract posterior
        if comparison == 'overweight':
            posterior = trace.posterior['celltype_effect_overweight'].values[..., cell_idx]
        elif comparison == 'obese':
            posterior = trace.posterior['celltype_effect_obese'].values[..., cell_idx]
        else:
            ow = trace.posterior['celltype_effect_overweight'].values[..., cell_idx]
            ob = trace.posterior['celltype_effect_obese'].values[..., cell_idx]
            posterior = ob - ow
        
        n_chains = posterior.shape[0]
        
        # Create plot
        fig = go.Figure()
        
        colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
        
        for chain_idx in range(n_chains):
            chain_data = posterior[chain_idx, :]
            
            fig.add_trace(go.Scatter(
                y=chain_data,
                mode='lines',
                name=f'Chain {chain_idx}',
                line=dict(width=0.5, color=colors[chain_idx % len(colors)]),
                opacity=0.7
            ))
        
        mean_val = np.mean(posterior)
        fig.add_hline(y=mean_val, line_dash="dash", line_color="black",
                     annotation_text=f"Mean: {mean_val:.3f}")
        
        fig.update_layout(
            title=f"Posterior Trace: {selected_cell}<br>" +
                  f"<sub>{BMI_LABELS.get(comparison, comparison)}</sub>",
            xaxis_title="Iteration",
            yaxis_title="Effect Size",
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating trace plot: {e}")
        return None

# ============================================================================
# BAYESIAN HEATMAP
# ============================================================================

def plot_bayesian_heatmap(selected_cell, comp_data):
    """Plot Bayesian effect sizes for selected cell"""
    
    try:
        results_file = f"{comp_data['dir']}/bayesian_results.csv"
        if not os.path.exists(results_file):
            # Try alternative location
            results_file = f"data/bayesian/{comp_data['key']}_results.csv"
        
        results = pd.read_csv(results_file)
        
        # Filter for selected cell
        cell_results = results[
            results['cell_type'].str.upper().str.replace('_', ' ').str.replace('-', ' ') ==
            selected_cell.upper().replace('_', ' ').replace('-', ' ')
        ]
        
        if cell_results.empty:
            return None
        
        # Get top features by absolute effect size
        cell_results['max_effect'] = cell_results[[
            'overweight_vs_normal_mean',
            'obese_vs_normal_mean',
            'obese_vs_overweight_mean'
        ]].abs().max(axis=1)
        
        cell_results = cell_results.nlargest(20, 'max_effect')
        
        # Format signature names
        cell_results['sig_formatted'] = cell_results['signature'].apply(
            lambda x: format_signature_name(x, max_length=40)
        )
        
        # Create data for plotting
        comparisons = ['overweight', 'obese', 'obese_vs_overweight']
        data = []
        
        for comp in comparisons:
            mean_col = f'{comp}_vs_normal_mean' if comp != 'obese_vs_overweight' else f'{comp}_mean'
            low_col = f'{comp}_vs_normal_hdi_low' if comp != 'obese_vs_overweight' else f'{comp}_hdi_low'
            high_col = f'{comp}_vs_normal_hdi_high' if comp != 'obese_vs_overweight' else f'{comp}_hdi_high'
            
            if mean_col in cell_results.columns:
                data.append({
                    'comparison': BMI_LABELS[comp],
                    'means': cell_results[mean_col].values,
                    'lows': cell_results[low_col].values if low_col in cell_results.columns else None,
                    'highs': cell_results[high_col].values if high_col in cell_results.columns else None
                })
        
        # Create grouped bar plot
        fig = go.Figure()
        
        for i, comp_data_item in enumerate(data):
            comp_name = comp_data_item['comparison']
            color = BMI_COLORS[comparisons[i]]
            
            fig.add_trace(go.Bar(
                name=comp_name,
                x=cell_results['sig_formatted'],
                y=comp_data_item['means'],
                marker_color=color,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=comp_data_item['highs'] - comp_data_item['means'] if comp_data_item['highs'] is not None else None,
                    arrayminus=comp_data_item['means'] - comp_data_item['lows'] if comp_data_item['lows'] is not None else None
                ) if comp_data_item['highs'] is not None else None
            ))
        
        fig.update_layout(
            title=f"Bayesian Effect Sizes: {selected_cell}<br>" +
                  "<sub>Error bars show 95% credible intervals</sub>",
            xaxis_title="Signature",
            yaxis_title="Effect Size",
            barmode='group',
            height=500,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating Bayesian heatmap: {e}")
        return None

# ============================================================================
# RIDGE PLOT
# ============================================================================

def plot_ridge_for_cell(selected_cell, comp_data):
    """Create ridge plot for selected cell"""
    
    try:
        if comp_data['trace'] is None:
            return None
        
        trace = comp_data['trace']
        cell_idx = get_cell_index(selected_cell, comp_data['celltype_mapping'])
        
        if cell_idx is None:
            return None
        
        # Extract posteriors
        ow = trace.posterior['celltype_effect_overweight'].values[..., cell_idx].flatten()
        ob = trace.posterior['celltype_effect_obese'].values[..., cell_idx].flatten()
        ob_ow = ob - ow
        
        posteriors = {
            'Overweight vs Normal': ow,
            'Obese vs Normal': ob,
            'Obese vs Overweight': ob_ow
        }
        
        colors_list = [COLOR_OVERWEIGHT, COLOR_OBESE, COLOR_OB_OW]
        
        fig = go.Figure()
        
        for i, (label, posterior) in enumerate(posteriors.items()):
            kde = gaussian_kde(posterior)
            x = np.linspace(posterior.min(), posterior.max(), 200)
            y = kde(x)
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y + i*0.5,
                name=f"{selected_cell}: {label}",
                fill='tozeroy',
                fillcolor=colors_list[i],
                line=dict(color=colors_list[i], width=2),
                opacity=0.6
            ))
            
            # Add mean line
            mean_val = np.mean(posterior)
            fig.add_vline(x=mean_val, line_dash="dash", 
                         line_color=colors_list[i], opacity=0.8,
                         annotation_text=f"{mean_val:.3f}")
        
        fig.update_layout(
            title=f"Posterior Distributions: {selected_cell}",
            xaxis_title="Effect Size",
            yaxis_title="Density (offset for visibility)",
            height=500,
            template='plotly_white',
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1)
        )
        
        fig.add_vline(x=0, line_dash="solid", line_color="black", opacity=0.3)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating ridge plot: {e}")
        return None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-title">Obesity-Driven Pancreatic Cancer: Cell-Signature Analysis</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>Interactive Analysis Platform</b><br>
    Exploring relationships between BMI, tumor microenvironment cell types, and metabolic 
    signatures in pancreatic adenocarcinoma.
    </div>
    """, unsafe_allow_html=True)
    
    # Methodology expander
    with st.expander("About the Analysis Methods", expanded=False):
        st.markdown("""
        ### Data & Methods Overview
        
        This platform integrates multiple computational approaches:
        
        **BayesPrism** - Cell type deconvolution from bulk RNA-seq  
        **STABL** - Stability-driven feature selection for robust biomarkers  
        **Bayesian Hierarchical Model** - Effect size estimation across BMI groups  
        **MCMC Diagnostics** - Convergence assessment (R-hat, ESS, energy)  
        
        **Dataset:** CPTAC Pancreatic Adenocarcinoma (140 samples, 3 BMI groups)  
        **Cell Types:** Deconvolved into immune and non-immune populations  
        **Signatures:** 30+ metabolic and functional gene signatures per cell type  
        
        **Analysis Workflow:**  
        1. Deconvolution → Cell type proportions  
        2. Expression → TPM matrix  
        3. Signatures → Z-scores  
        4. STABL selection → Robust features  
        5. Bayesian modeling → Effect sizes  
        6. Diagnostics → Validation  
        """)
    
    # Sidebar
    st.sidebar.title("Data Selection")
    
    st.sidebar.markdown("### 1. Select Compartment")
    compartment = st.sidebar.selectbox(
        "Compartment:",
        options=['Immune Fine', 'Immune Coarse', 'Non-Immune'],
        index=0
    )
    
    # Load data
    with st.spinner(f"Loading {compartment} data..."):
        comp_data = load_compartment_data(compartment)
        clinical = load_clinical_data()
        tpm = load_tpm_data()
    
    # Select cell type
    st.sidebar.markdown("### 2. Select Cell Type")
    available_cells = get_available_cells(compartment)
    
    if not available_cells:
        st.error("No cell types available for this compartment")
        return
    
    selected_cell = st.sidebar.selectbox(
        f"Cell type ({len(available_cells)} available):",
        options=available_cells,
        index=0
    )
    
    # Select signature
    st.sidebar.markdown("### 3. Select Signature")
    signatures = get_cell_signatures(selected_cell)
    
    if not signatures:
        st.warning(f"No signatures found for {selected_cell}")
        return
    
    sig_options = {}
    for s in signatures:
        formatted_name = format_signature_name(s['signature'], max_length=35)
        display_text = f"{formatted_name} ({len(s['genes'])} genes)"
        sig_options[display_text] = s
    
    selected_sig_display = st.sidebar.selectbox(
        f"Signature ({len(signatures)} available):",
        options=list(sig_options.keys()),
        index=0,
        help="Names truncated for readability"
    )
    
    selected_sig_info = sig_options[selected_sig_display]
    sig_name = selected_sig_info['signature']
    genes = selected_sig_info['genes']
    
    # Generate button
    st.sidebar.markdown("---")
    generate = st.sidebar.button("Generate Analysis", type="primary")
    
    # Current selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current Selection")
    st.sidebar.info(f"""
    **Compartment:** {compartment}  
    **Cell Type:** {selected_cell}  
    **Signature:** {format_signature_name(sig_name, max_length=40)}  
    **Genes:** {len(genes)}
    """)
    
    if len(sig_name) > 50:
        st.sidebar.caption(f"Full name: {sig_name.replace('_', ' ')}")
    
    # Main content
    if generate:
        st.markdown("---")
        
        # Create tabs
        tabs = st.tabs([
            "STABL & Bayesian",
            "Ridge Plot",
            "Diagnostics"
        ])
        
        # Tab 1: STABL & Bayesian
        with tabs[0]:
            st.markdown("### Complete Signature Analysis")
            
            # Load STABL features
            stabl_features = load_stabl_selected_features(compartment)
            
            # Heatmap 1: All signatures
            st.markdown("#### All Signatures - Z-scores by BMI Group")
            st.caption("★ indicates feature selected by STABL for downstream analysis")
            
            with st.spinner("Computing Z-scores from TPM data..."):
                fig_all = create_heatmap_all_signatures(
                    selected_cell, sig_name, genes,
                    tpm, clinical, stabl_features
                )
            
            if fig_all:
                st.plotly_chart(fig_all, use_container_width=True)
            else:
                st.warning("No data available for all signatures")
            
            st.markdown("---")
            
            # Heatmap 2: STABL-selected only
            st.markdown("#### STABL-Selected Features Only")
            st.caption("Subset showing only features that passed STABL selection")
            
            with st.spinner("Loading STABL-selected features..."):
                fig_stabl = create_heatmap_stabl_only(
                    selected_cell, sig_name, compartment, clinical
                )
            
            if fig_stabl:
                st.plotly_chart(fig_stabl, use_container_width=True)
            else:
                st.info("This signature was not selected by STABL for this cell type")
            
            st.markdown("---")
            
            # Bayesian heatmap
            st.markdown("#### Bayesian Effect Size Estimates")
            st.caption("Effect sizes with 95% credible intervals")
            
            with st.spinner("Loading Bayesian results..."):
                fig_bayes = plot_bayesian_heatmap(selected_cell, comp_data)
            
            if fig_bayes:
                st.plotly_chart(fig_bayes, use_container_width=True)
            else:
                st.info("Bayesian results not available")
        
        # Tab 2: Ridge Plot
        with tabs[1]:
            st.markdown("### Posterior Distribution Visualization")
            
            st.markdown("""
            <div class="method-box">
            <b>Ridge Plots Explained</b><br>
            Each distribution shows MCMC samples for this cell type's effect size.
            Width indicates uncertainty, peak shows most likely value.
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Generating ridge plot..."):
                fig_ridge = plot_ridge_for_cell(selected_cell, comp_data)
            
            if fig_ridge:
                st.plotly_chart(fig_ridge, use_container_width=True)
            else:
                st.warning("Ridge plot not available")
        
        # Tab 3: Diagnostics
        with tabs[2]:
            st.markdown(f"### MCMC Diagnostics: {selected_cell}")
            
            st.markdown("""
            <div class="method-box">
            <b>Understanding MCMC Diagnostics</b><br><br>
            <b>ESS (Effective Sample Size):</b> Number of independent samples. Target > 400.<br>
            <b>R-hat:</b> Convergence metric. < 1.01 = excellent, < 1.05 = acceptable.<br>
            <b>Energy:</b> HMC diagnostic. Smooth overlap = good sampling.
            </div>
            """, unsafe_allow_html=True)
            
            # ESS & R-hat
            st.markdown("#### Convergence Metrics")
            
            with st.spinner("Loading diagnostics..."):
                fig_ess = plot_ess_rhat_single_cell(selected_cell, comp_data)
            
            if fig_ess:
                st.plotly_chart(fig_ess, use_container_width=True)
            else:
                st.warning("Convergence metrics not available")
            
            st.markdown("---")
            
            # Energy
            st.markdown("#### Energy Diagnostic")
            st.caption("ArviZ energy plot showing marginal energy and transitions")
            
            with st.spinner("Generating energy plot..."):
                energy_img = plot_energy_arviz(comp_data, selected_cell)
            
            if energy_img:
                st.image(energy_img, use_column_width=True)
            else:
                st.warning("Energy diagnostic not available")
            
            st.markdown("---")
            
            # Traces
            st.markdown("#### Posterior Traces")
            st.caption("Should show good mixing across chains ('hairy caterpillars')")
            
            comparison = st.selectbox(
                "Select comparison:",
                options=['overweight', 'obese', 'obese_vs_overweight'],
                format_func=lambda x: BMI_LABELS[x],
                key='trace_comparison'
            )
            
            with st.spinner("Generating trace plot..."):
                fig_trace = plot_trace_single_cell(selected_cell, comp_data, comparison)
            
            if fig_trace:
                st.plotly_chart(fig_trace, use_container_width=True)
            else:
                st.warning("Trace plot not available")

if __name__ == "__main__":
    main()
