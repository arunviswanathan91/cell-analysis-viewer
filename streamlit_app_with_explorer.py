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

# Try optional imports
try:
    from lifelines import CoxPHFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

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
        text-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .sub-header {
        font-size: 1.875rem;
        font-weight: 700;
        color: var(--md-grey-900);
        margin-top: 3.5rem;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        position: relative;
        letter-spacing: -0.02em;
    }
    
    .sub-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 4px;
        background: var(--gradient-primary);
        border-radius: 2px;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
    }
    
    /* ========== GLASSMORPHISM CARDS - Premium Look ========== */
    
    .info-box {
        background: var(--glass-bg);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        border: 1px solid var(--glass-border);
        box-shadow: var(--shadow-lg), var(--glass-shadow);
        position: relative;
        overflow: hidden;
        transition: all 0.4s var(--ease-smooth);
    }
    
    .info-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: var(--gradient-info);
    }
    
    .info-box::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(33, 150, 243, 0.08) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.4s var(--ease-smooth);
    }
    
    .info-box:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-xl), var(--shadow-glow);
        border-color: rgba(33, 150, 243, 0.3);
    }
    
    .info-box:hover::after {
        opacity: 1;
    }
    
    .method-box {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2.5rem 0;
        border: 1px solid rgba(76, 175, 80, 0.2);
        box-shadow: var(--shadow-lg), 0 8px 32px rgba(76, 175, 80, 0.15);
        position: relative;
        overflow: hidden;
        transition: all 0.4s var(--ease-smooth);
    }
    
    .method-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: var(--gradient-success);
    }
    
    .method-box:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow: var(--shadow-2xl), 0 12px 40px rgba(76, 175, 80, 0.25);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 248, 225, 0.95) 0%, rgba(255, 248, 225, 0.85) 100%);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border-left: 4px solid var(--md-warning-500);
        box-shadow: var(--shadow-md);
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
    
    /* ========== ELEVATED METRICS - Card with Depth ========== */
    
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.75rem;
        border-radius: 16px;
        box-shadow: var(--shadow-md);
        transition: all 0.3s var(--ease-smooth);
        border: 1px solid var(--md-grey-200);
        position: relative;
        overflow: hidden;
    }
    
    .stMetric::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: var(--gradient-primary);
    }
    
    .stMetric:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: var(--shadow-xl);
        border-color: var(--md-primary-200);
    }
    
    .stMetric label {
        font-weight: 700 !important;
        color: var(--md-grey-600) !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.25rem !important;
        font-weight: 800 !important;
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
    
    /* ========== PLOTLY CHARTS - Premium Card ========== */
    
    .js-plotly-plot {
        border-radius: 16px;
        overflow: hidden;
        background: white;
        box-shadow: var(--shadow-lg);
        border: 1px solid var(--md-grey-200);
        transition: all 0.3s var(--ease-smooth);
    }
    
    .js-plotly-plot:hover {
        box-shadow: var(--shadow-xl);
        transform: translateY(-4px);
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
    
    .stSpinner > div {
        border-color: var(--md-primary-500) transparent transparent transparent !important;
        animation: spinner 1s cubic-bezier(0.5, 0, 0.5, 1) infinite;
    }
    
    @keyframes spinner {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* ========== CHECKBOX & RADIO - Modern Toggle ========== */
    
    .stCheckbox, .stRadio {
        padding: 0.75rem;
        border-radius: 8px;
        transition: background 0.3s var(--ease-smooth);
    }
    
    .stCheckbox:hover, .stRadio:hover {
        background: rgba(33, 150, 243, 0.05);
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

# ==================================================================================
# ============================= CONFIGURATION ======================================
# ==================================================================================

DATA_DIR = "data"
LOG_TRANSFORM = True
BMI_COLORS = {'Normal': '#2ECC71', 'Overweight': '#F39C12', 'Obese': '#E74C3C'}
COLOR_OVERWEIGHT = "#1f78b4"
COLOR_OBESE = "#e31a1c"
COLOR_OBO = "#33a02c"

# Plotly template
PLOTLY_TEMPLATE = "plotly_white"

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

@st.cache_data
def load_signatures():
    """Load signatures from JSON"""
    try:
        sig_file = os.path.join(DATA_DIR, "signatures", "ALL_CELL_SIGNATURES_FLAT.json")
        with open(sig_file, 'r') as f:
            data = json.load(f)
        return data.get('entries', [])
    except Exception as e:
        st.error(f"Error loading signatures: {e}")
        return []

@st.cache_data
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

@st.cache_data
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

@st.cache_data
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

@st.cache_data
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

@st.cache_data
def load_zscore_data_survival():
    """Load z-score data for survival analysis"""
    all_data = []
    
    comp_map = {
        'Immune Fine': 'immune_fine',
        'Immune Coarse': 'immune_coarse',
        'Non-Immune': 'non_immune'
    }
    
    for compartment_name, comp_key in comp_map.items():
        zfile = os.path.join(DATA_DIR, "zscores", f"{comp_key}_zscores.csv")
        
        if not os.path.exists(zfile):
            continue
        
        try:
            df = pd.read_csv(zfile, low_memory=False)
            
            # Get sample column (first column)
            sample_col = df.columns[0]
            
            # Get feature columns (contain "||")
            feature_cols = [c for c in df.columns if "||" in str(c)]
            
            if not feature_cols:
                continue
            
            # Melt to long format
            df_long = df.melt(id_vars=[sample_col], value_vars=feature_cols,
                             var_name="feature", value_name="Z")
            
            # Extract base sample ID
            df_long['base_sample_id'] = df_long[sample_col].apply(extract_base_sample_id)
            
            # Add compartment info
            df_long['compartment'] = compartment_name
            
            all_data.append(df_long)
        except Exception as e:
            continue
    
    if not all_data:
        return None
    
    return pd.concat(all_data, ignore_index=True)

def assign_bmi_category(bmi):
    """Assign BMI category using WHO standards"""
    if pd.isna(bmi):
        return None
    for cat, (low, high) in BMI_CATEGORIES.items():
        if low <= bmi < high:
            return cat
    return 'Obese'

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
            st.sidebar.warning(f"Ã¢Å¡Â Ã¯Â¸Â No cell types found in z-score data. Available columns: {list(comp_data['zscores'].columns)}")
        
        return cells
    else:
        st.sidebar.error(f"Ã¢ÂÅ’ Z-score data not loaded for {compartment}. Check if file exists: data/zscores/{compartment.lower().replace(' ', '_').replace('-', '_')}_zscores.csv")
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
    """Generate interactive STABL Z-score heatmap"""
    if comp_data['zscores'] is None or comp_data['stabl'] is None:
        st.warning("Ã¢Å¡Â Ã¯Â¸Â STABL data not available")
        return None
    
    zscores = comp_data['zscores']
    zscores = zscores[zscores['CellType'].str.upper() == cell_type.upper()].copy()
    
    if len(zscores) == 0:
        st.warning(f"Ã¢Å¡Â Ã¯Â¸Â No Z-scores found for {cell_type}")
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
    
    # Add STABL marker to signature names
    signatures = []
    for sig in heatmap_data.index:
        feature_name = f"{cell_type}||{sig}"
        if feature_name in stabl_features:
            signatures.append(f"{sig} Ã¢Â­Â")
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
            text=f'{cell_type} - {sig_name}<br>STABL Z-scores by BMI Group',
            font=dict(size=16, color='#2c3e50')
        ),
        xaxis_title='BMI Category',
        yaxis_title='Signatures (Ã¢Â­Â = STABL-selected)',
        height=max(600, len(heatmap_data) * 25),
        template=PLOTLY_TEMPLATE,
        hovermode='closest'
    )
    
    return fig

def plot_bayesian_heatmap_interactive(cell_type, sig_name, comp_data):
    """Generate interactive Bayesian effect size heatmap"""
    if comp_data['bayesian'] is None:
        st.warning("Ã¢Å¡Â Ã¯Â¸Â Bayesian data not available")
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
        st.warning(f"Ã¢Å¡Â Ã¯Â¸Â No Bayesian results for {cell_type}")
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
        st.warning("Ã¢Å¡Â Ã¯Â¸Â No effect size columns found")
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
        st.info("Ã¢â€žÂ¹Ã¯Â¸Â Posterior data not available - ridge plot skipped")
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
        for i in range(n_cells):
            if ct_map is not None and len(ct_map) > 0:
                # Try to get from celltype_name column (more robust)
                if 'celltype_name' in ct_map.columns and 'celltype_idx' in ct_map.columns:
                    ct_row = ct_map[ct_map['celltype_idx'] == i]
                    if len(ct_row) > 0:
                        name = str(ct_row['celltype_name'].iloc[0])
                    else:
                        name = f"Cell_{i}"
                # Fallback: use column 1 (celltype_name) instead of column 0 (celltype_idx)
                elif len(ct_map.columns) >= 2 and i < len(ct_map):
                    name = str(ct_map.iloc[i, 1])  # FIX: Changed from 0 to 1
                else:
                    name = f"Cell_{i}"
            else:
                name = f"Cell_{i}"
            cell_names.append(name.replace('_', ' ').title())
        
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
            
            # Add traces for each comparison
            fig.add_trace(go.Scatter(
                x=xgrid, y=d_ob + y_offset,
                fill='tonexty' if i > 0 else 'tozeroy',
                fillcolor=f'rgba(227, 26, 28, 0.5)',
                line=dict(color='rgba(227, 26, 28, 0.8)', width=1.5),
                name=f'{ct_name} - Obese',
                hovertemplate=f'<b>{ct_name}</b><br>Obese vs Normal<br>Effect: %{{x:.3f}}<extra></extra>',
                showlegend=(i == 0),
                legendgroup='obese'
            ))
            
            fig.add_trace(go.Scatter(
                x=xgrid, y=d_over + y_offset,
                fill='tonexty',
                fillcolor=f'rgba(31, 120, 180, 0.5)',
                line=dict(color='rgba(31, 120, 180, 0.8)', width=1.5),
                name=f'{ct_name} - Overweight',
                hovertemplate=f'<b>{ct_name}</b><br>Overweight vs Normal<br>Effect: %{{x:.3f}}<extra></extra>',
                showlegend=(i == 0),
                legendgroup='overweight'
            ))
            
            fig.add_trace(go.Scatter(
                x=xgrid, y=d_obo + y_offset,
                fill='tonexty',
                fillcolor=f'rgba(51, 160, 44, 0.5)',
                line=dict(color='rgba(51, 160, 44, 0.8)', width=1.5),
                name=f'{ct_name} - Obese vs Overweight',
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
            template=PLOTLY_TEMPLATE,
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Ã¢Å¡Â Ã¯Â¸Â Error creating ridge plot: {e}")
        return None

def plot_gene_bmi_interactive(genes, clinical, tpm):
    """Generate interactive gene-level BMI analysis plots"""
    if tpm is None:
        st.warning("Ã¢Å¡Â Ã¯Â¸Â TPM data not available")
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
        st.warning("Ã¢Å¡Â Ã¯Â¸Â No genes analyzed")
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
        hovertemplate='<b>%{y}</b><br>Slope: %{x:.4f}<br>RÃ‚Â²: %{customdata[0]:.3f}<br>p-value: %{customdata[1]:.3e}<extra></extra>',
        customdata=np.column_stack((plot_df['r_squared'], plot_df['p_value']))
    ))
    
    fig1.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
    
    fig1.update_layout(
        title='Gene-Level BMI Association<br>ÃŽâ€ Expression per ÃŽâ€ BMI',
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
        subplot_titles=[f"{row['gene']} (slope={row['slope']:.4f}, RÃ‚Â²={row['r_squared']:.3f})" 
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
        st.info("Ã¢â€žÂ¹Ã¯Â¸Â Energy data not available")
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

def plot_trace_diagnostic(comp_data, n_celltypes=6):
    """Generate trace plots for first N cell types"""
    if comp_data['posterior_overweight'] is None:
        st.info("Ã¢â€žÂ¹Ã¯Â¸Â Posterior data not available")
        return None
    
    # Get posterior data
    df_over = comp_data['posterior_overweight']
    
    # Assume 4 chains based on total samples
    n_samples = len(df_over)
    samples_per_chain = n_samples // 4
    
    # Select first n_celltypes
    cell_cols = [c for c in df_over.columns if c.startswith('celltype_')][:n_celltypes]
    
    n_cols = 2
    n_rows = int(np.ceil(len(cell_cols) / n_cols))
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'Cell Type {col.split("_")[1]}' for col in cell_cols],
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
    
    fig.update_layout(
        title='Trace Plots - Overweight Effect (First 6 Cell Types)',
        height=n_rows * 300,
        template=PLOTLY_TEMPLATE,
        hovermode='closest'
    )
    
    return fig

def plot_rank_diagnostic(comp_data, n_celltypes=6):
    """Generate rank plots for convergence diagnostic"""
    if comp_data['posterior_overweight'] is None:
        st.info("Ã¢â€žÂ¹Ã¯Â¸Â Posterior data not available")
        return None
    
    df_over = comp_data['posterior_overweight']
    
    n_samples = len(df_over)
    samples_per_chain = n_samples // 4
    
    cell_cols = [c for c in df_over.columns if c.startswith('celltype_')][:n_celltypes]
    
    n_cols = 2
    n_rows = int(np.ceil(len(cell_cols) / n_cols))
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'Cell Type {col.split("_")[1]}' for col in cell_cols],
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
    
    fig.update_layout(
        title='Rank Plots - Convergence Diagnostic (First 6 Cell Types)',
        height=n_rows * 300,
        template=PLOTLY_TEMPLATE,
        hovermode='closest',
        barmode='overlay'
    )
    
    return fig

def plot_autocorrelation(comp_data, n_celltypes=6, max_lag=40):
    """Generate autocorrelation plots"""
    if comp_data['posterior_overweight'] is None:
        st.info("Ã¢â€žÂ¹Ã¯Â¸Â Posterior data not available")
        return None
    
    df_over = comp_data['posterior_overweight']
    
    n_samples = len(df_over)
    samples_per_chain = n_samples // 4
    
    cell_cols = [c for c in df_over.columns if c.startswith('celltype_')][:n_celltypes]
    
    n_cols = 2
    n_rows = int(np.ceil(len(cell_cols) / n_cols))
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'Cell Type {col.split("_")[1]}' for col in cell_cols],
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
    
    fig.update_layout(
        title='Autocorrelation Plots (First 6 Cell Types)',
        height=n_rows * 300,
        template=PLOTLY_TEMPLATE,
        hovermode='closest'
    )
    
    return fig

def plot_ess_rhat(comp_data):
    """Generate ESS and R-hat diagnostic plots"""
    if comp_data['diagnostics'] is None:
        st.info("Ã¢â€žÂ¹Ã¯Â¸Â Diagnostic summary not available")
        return None
    
    diag = comp_data['diagnostics']
    
    # Check if first column should be index
    if 'Unnamed: 0' in diag.columns:
        diag = diag.set_index('Unnamed: 0')
    elif diag.columns[0] in ['parameter', 'index', 'name']:
        diag = diag.set_index(diag.columns[0])
    
    # Filter for cell type effects
    if isinstance(diag.index, pd.RangeIndex):
        st.warning("Ã¢Å¡Â Ã¯Â¸Â Diagnostic data doesn't have parameter names")
        return None
    
    # Convert index to string if needed
    diag.index = diag.index.astype(str)
    
    # Filter for celltype_effect parameters
    diag = diag[diag.index.str.contains('celltype_effect', na=False, case=False)]
    
    if len(diag) == 0:
        st.warning("Ã¢Å¡Â Ã¯Â¸Â No cell type diagnostics found in data")
        return None
    
    # Create figure with two subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Effective Sample Size (ESS)', 'R-hat Convergence Statistic'],
        horizontal_spacing=0.15
    )
    
    # ESS plot - try different column names
    ess_col = None
    for col in ['ess_bulk', 'ess_mean', 'ess', 'n_eff']:
        if col in diag.columns:
            ess_col = col
            break
    
    if ess_col is None:
        st.warning("Ã¢Å¡Â Ã¯Â¸Â ESS column not found in diagnostics")
        return None
    
    ess_bulk = diag[ess_col].values
    
    fig.add_trace(
        go.Bar(
            y=[f"Cell {i}" for i in range(len(ess_bulk))],
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
    
    # Add reference line for good ESS (>400)
    fig.add_vline(x=400, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1,
                 annotation_text="Min recommended (400)", annotation_position="top")
    
    # R-hat plot
    rhat_col = None
    for col in ['r_hat', 'rhat', 'Rhat', 'R_hat']:
        if col in diag.columns:
            rhat_col = col
            break
    
    if rhat_col is None:
        st.warning("Ã¢Å¡Â Ã¯Â¸Â R-hat column not found in diagnostics")
        # Continue with ESS plot only
        fig.update_xaxes(title_text='Effective Sample Size', row=1, col=1)
        fig.update_layout(
            title='Bayesian Diagnostic Statistics (ESS only)',
            height=max(500, len(diag) * 25),
            template=PLOTLY_TEMPLATE,
            showlegend=False,
            hovermode='closest'
        )
        return fig
    
    rhat = diag[rhat_col].values
    
    colors = ['green' if r < 1.01 else 'orange' if r < 1.05 else 'red' for r in rhat]
    
    fig.add_trace(
        go.Bar(
            y=[f"Cell {i}" for i in range(len(rhat))],
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
        title='Bayesian Diagnostic Statistics',
        height=max(500, len(diag) * 25),
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
    """Plot 5: BMI × Signature Interaction (Tertiles)"""
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
        subplot_titles=('Median Survival by BMI × Signature', 'Event Rate by BMI × Signature'),
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
        title_text=f'{signature_name}<br>BMI × Signature Interaction (Tertiles)',
        template=PLOTLY_TEMPLATE,
        hovermode='closest',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_survival_interaction_median(patient_data, signature_name):
    """Plot 6: BMI × Signature Interaction (Median Split)"""
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
        subplot_titles=('Median Survival by BMI × Signature', 'Event Rate by BMI × Signature'),
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
        title_text=f'{signature_name}<br>BMI × Signature Interaction (Median Split: High vs Low)',
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


def plot_gene_survival_interactive(genes, clinical, tpm):
    """Generate interactive gene-level survival forest plot"""
    if not LIFELINES_AVAILABLE or tpm is None:
        st.info("Ã¢â€žÂ¹Ã¯Â¸Â Survival analysis not available")
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
        st.info("Ã¢â€žÂ¹Ã¯Â¸Â No genes passed survival criteria")
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
# ============================= SIGNATURE EXPLORER =================================
# ==================================================================================

def render_signature_explorer():
    """Render the signature database explorer interface"""
    st.markdown('<div class="sub-header"> Signature Database Explorer</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b> Browse the Complete Signature Database</b><br>
    Explore all metabolic and functional signatures across different cell types and compartments.
    Select a compartment and cell type to view available signatures and their gene lists.
    </div>
    """, unsafe_allow_html=True)
    
    # Load signatures
    signatures = load_signatures()
    
    if not signatures:
        st.error("âŒ Failed to load signature database")
        return
    
    # Create sidebar for selection
    st.sidebar.markdown("###  Signature Selection")
    st.sidebar.markdown("Select compartment and cell type to explore signatures:")
    st.sidebar.markdown("")
    
    # Step 1: Compartment selection
    compartment = st.sidebar.selectbox(
        "Choose compartment:",
        options=['Immune Fine', 'Immune Coarse', 'Non-Immune'],
        index=0,
        key='explorer_compartment'
    )
    
    # Step 2: Get available cells for this compartment
    available_cells = get_available_cells(compartment)
    
    if not available_cells:
        st.warning(f"âš ï¸ No cell types found for {compartment}")
        st.info("This may indicate missing z-score data files. The signature database may still contain entries for this compartment.")
        
        # Show all unique cell types from signature database
        all_cell_types = sorted(list(set([s['cell_type'] for s in signatures])))
        st.sidebar.markdown("### Available in Signature Database:")
        for ct in all_cell_types:
            st.sidebar.caption(f"* {ct.replace('_', ' ').title()}")
        return
    
    # Format cell names for display
    cell_display = {cell.replace('_', ' ').title(): cell for cell in available_cells}
    
    selected_cell_display = st.sidebar.selectbox(
        f"Choose cell type ({len(available_cells)} available):",
        options=list(cell_display.keys()),
        index=0,
        key='explorer_cell'
    )
    selected_cell = cell_display[selected_cell_display]
    
    # Get signatures for this cell type
    cell_signatures = get_cell_signatures(selected_cell)
    
    # Display summary
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(" Compartment", compartment)
    with col2:
        st.metric(" Cell Type", selected_cell_display)
    with col3:
        st.metric(" Signatures Found", len(cell_signatures))
    
    st.markdown("---")
    
    if not cell_signatures:
        st.warning(f"âš ï¸ No signatures found for {selected_cell}")
        st.info("This cell type exists in the z-score data but may not have associated signatures in the database.")
        return
    
    # Display signatures in an organized way
    st.markdown(f"###  Signatures for {selected_cell_display}")
    
    # Create tabs for different views
    sig_tabs = st.tabs([" Summary Table", " Detailed View", " Statistics"])
    
    # Tab 1: Summary Table
    with sig_tabs[0]:
        st.markdown("#### Quick Overview")
        
        # Create summary dataframe
        summary_data = []
        for sig in cell_signatures:
            summary_data.append({
                'Signature Name': format_signature_name(sig['signature'], max_length=50),
                'Full Name': sig['signature'],
                'Number of Genes': len(sig['genes']),
                'First 5 Genes': ', '.join(sig['genes'][:5]) + ('...' if len(sig['genes']) > 5 else '')
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Display with formatting
        st.dataframe(
            summary_df[['Signature Name', 'Number of Genes', 'First 5 Genes']],
            use_container_width=True,
            height=min(600, len(summary_df) * 35 + 38)
        )
        
        # Download button
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label=" Download Summary as CSV",
            data=csv,
            file_name=f"{selected_cell}_signatures_summary.csv",
            mime="text/csv"
        )
    
    # Tab 2: Detailed View
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
        st.markdown(f"####  {selected_sig['signature']}")
        
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
            
            # Display genes in a nice format
            genes_text = ', '.join(selected_sig['genes'])
            st.text_area(
                "Genes in this signature:",
                genes_text,
                height=200,
                key='gene_list_display'
            )
            
            # Download genes
            genes_csv = '\n'.join(selected_sig['genes'])
            st.download_button(
                label=" Download Gene List",
                data=genes_csv,
                file_name=f"{selected_sig['signature']}_genes.txt",
                mime="text/plain"
            )
        
        # Show gene list as table
        st.markdown("####  Gene List (Table View)")
        genes_df = pd.DataFrame({'Gene Symbol': selected_sig['genes']})
        genes_df['Index'] = range(1, len(genes_df) + 1)
        genes_df = genes_df[['Index', 'Gene Symbol']]
        
        st.dataframe(
            genes_df,
            use_container_width=True,
            height=min(400, len(genes_df) * 35 + 38)
        )
    
    # Tab 3: Statistics
    with sig_tabs[2]:
        st.markdown("####  Database Statistics")
        
        # Overall statistics
        st.markdown("#####  Overall Database Stats")
        
        total_signatures = len(signatures)
        total_cell_types = len(set([s['cell_type'] for s in signatures]))
        avg_genes = np.mean([len(s['genes']) for s in signatures])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Signatures", total_signatures)
        with col2:
            st.metric("Cell Types", total_cell_types)
        with col3:
            st.metric("Avg Genes/Signature", f"{avg_genes:.1f}")
        with col4:
            st.metric("Current Cell Sigs", len(cell_signatures))
        
        st.markdown("---")
        
        # Signature size distribution for current cell type
        st.markdown(f"#####  Signature Sizes for {selected_cell_display}")
        
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
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # All cell types signature counts
        st.markdown("#####  Signatures per Cell Type (All Compartments)")
        
        cell_sig_counts = {}
        for sig in signatures:
            ct = sig['cell_type']
            cell_sig_counts[ct] = cell_sig_counts.get(ct, 0) + 1
        
        ct_df = pd.DataFrame([
            {'Cell Type': ct.replace('_', ' ').title(), 'Signature Count': count}
            for ct, count in sorted(cell_sig_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            y=ct_df['Cell Type'],
            x=ct_df['Signature Count'],
            orientation='h',
            marker=dict(
                color=ct_df['Signature Count'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Count")
            ),
            text=ct_df['Signature Count'],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Signatures: %{x}<extra></extra>'
        ))
        
        fig2.update_layout(
            title='Signature Count by Cell Type',
            xaxis_title='Number of Signatures',
            yaxis_title='Cell Type',
            height=max(500, len(ct_df) * 25),
            template=PLOTLY_TEMPLATE,
            hovermode='closest'
        )
        
        st.plotly_chart(fig2, use_container_width=True)

# ==================================================================================
# ============================= MAIN APP ===========================================
# ==================================================================================

def main():
    # Sidebar Mode Selection with Toggle Button
    st.sidebar.title("âš™ï¸ Application Mode")
    
    analysis_mode = st.sidebar.toggle(
        "Statistical Analysis Mode",
        value=False,
        help="Toggle ON for Statistical Analysis, OFF for Signature Explorer"
    )
    
    # Show current mode with colored indicator
    if analysis_mode:
        st.sidebar.success(" **Current Mode:** Statistical Analysis")
    else:
        st.sidebar.info(" **Current Mode:** Signature Explorer")
    
    st.sidebar.markdown("---")
    
    # Main Header
    st.markdown('<div class="main-header"> Obesity-Driven Pancreatic Cancer: Cell-Signature Analysis</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b> Interactive Analysis Platform</b><br>
    Exploring the relationship between BMI, tumor microenvironment cell types, and metabolic signatures in pancreatic adenocarcinoma (PAAD).
    All visualizations powered by Plotly: Hover for details * Zoom with box select * Pan with click-drag * Reset with double-click
    </div>
    """, unsafe_allow_html=True)
    
    # Route to appropriate section
    if not analysis_mode:  # Signature Explorer mode (toggle OFF)
        render_signature_explorer()
        return
    
    # Continue with Statistical Analysis mode (toggle ON)
    
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
        
        #### **STABL** - Feature Selection
        Stability-driven feature selection that identifies the most robust biomarkers associated with BMI status. STABL uses bootstrapping to find features that consistently show effects across multiple random samplings, reducing false positives.
        
        **Reference:** [gregbellan/Stabl](https://github.com/gregbellan/Stabl)
        
        ---
        
        #### **Bayesian Hierarchical Model** - Effect Size Estimation
        A three-group hierarchical model comparing:
        - **Normal BMI** (< 25) vs **Overweight** (25-30) vs **Obese** (Ã¢â€°Â¥ 30)
        
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
        1. **Deconvolution:** BayesPrism Ã¢â€™ Cell type proportions/Cell-specific expression matrix
        2. **Expression:** TPM values Ã¢â€™ Gene expression matrix
        3. **Signatures:** Aggregate genes Ã¢â€™ Signature scores (Z-scores)
        4. **Selection:** STABL Ã¢â€™ Robust BMI-associated features
        5. **Modeling:** Bayesian hierarchical Ã¢â€™ Effect sizes with uncertainty
        6. **Validation:** MCMC diagnostics Ã¢â€™ Convergence checks
        7. **Survival:** Cox regression Ã¢â€™ Clinical relevance
        """)
    
    # Sidebar
    st.sidebar.title("Ã°Å¸â€œÅ  Data Selection")
    
    # Step 1: Compartment
    st.sidebar.markdown("### Step 1: Select Compartment")
    compartment = st.sidebar.selectbox(
        "Choose compartment:",
        options=['Immune Fine', 'Immune Coarse', 'Non-Immune'],
        index=0
    )
    
    # Load data
    with st.spinner(f"Loading {compartment} data..."):
        comp_data = load_compartment_data(compartment)
        clinical = load_clinical_data()
        tpm = load_tpm_data()
    
    # Step 2: Cell Type
    st.sidebar.markdown("### Step 2: Select Cell Type")
    available_cells = get_available_cells(compartment)
    
    if not available_cells:
        st.error("Ã¢ÂÅ’ No cell types found")
        return
    
    cell_display = {cell.replace('_', ' ').title(): cell for cell in available_cells}
    selected_cell_display = st.sidebar.selectbox(
        f"Choose cell type ({len(available_cells)} available):",
        options=list(cell_display.keys()),
        index=0
    )
    selected_cell = cell_display[selected_cell_display]
    
    # Step 3: Signature
    st.sidebar.markdown("### Step 3: Select Signature")
    signatures = get_cell_signatures(selected_cell)
    
    if not signatures:
        st.warning(f"Ã¢Å¡Â Ã¯Â¸Â No signatures found for {selected_cell}")
        return
    
    # Create formatted options for display
    sig_options = {}
    for s in signatures:
        formatted_name = format_signature_name(s['signature'], max_length=35)
        display_text = f"{formatted_name} ({len(s['genes'])} genes)"
        sig_options[display_text] = s
    
    selected_sig_display = st.sidebar.selectbox(
        f"Choose signature ({len(signatures)} available):",
        options=list(sig_options.keys()),
        index=0,
        help="Signature names are truncated for readability. Full name shown in results."
    )
    selected_sig_info = sig_options[selected_sig_display]
    sig_name = selected_sig_info['signature']
    genes = selected_sig_info['genes']
    
    # Generate button
    st.sidebar.markdown("---")
    generate = st.sidebar.button("Ã°Å¸Å¡â‚¬ Generate Analysis", type="primary")
    
    # Current selection with full signature name
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current Selection")
    st.sidebar.info(f"""
    **Compartment:** {compartment}  
    **Cell Type:** {selected_cell_display}  
    **Signature:** {format_signature_name(sig_name, max_length=50)}  
    **Genes:** {len(genes)}
    """)
    
    # Show full signature name in a smaller font if truncated
    if len(sig_name) > 50:
        st.sidebar.caption(f"Full name: {sig_name.replace('_', ' ')}")
    
    # Main content
    if generate:
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
        
        # Tabs
        tabs = st.tabs([
            "STABL & Bayesian",
            "Ridge Plot",
            "Diagnostics",
            "Gene BMI",
            "Gene Survival",
            "Signature Survival"
        ])
        
        # Tab 1: STABL & Bayesian
        with tabs[0]:
            st.markdown("### STABL Feature Selection")
            
            st.markdown("""
            <div class="method-box">
            <b>Ã°Å¸â€Â¬ What is STABL?</b><br>
            STABL (STABility-driven feature seLection) identifies robust biomarkers by:
            <ol>
            <li>Running feature selection on multiple bootstrap samples</li>
            <li>Counting how often each feature is selected</li>
            <li>Keeping only features selected consistently (stable features)</li>
            </ol>
            <b>Ã¢Â­Â Stars mark STABL-selected features</b> - these show the most robust associations with BMI status.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Z-score Heatmap")
            st.caption("Z-scores represent standardized signature expression across BMI categories")
            with st.spinner("Generating interactive STABL heatmap..."):
                fig = plot_stabl_heatmap_interactive(selected_cell, sig_name, comp_data, clinical)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            st.markdown("### Ã°Å¸â€œÅ  Bayesian Effect Size Estimation")
            
            st.markdown("""
            <div class="method-box">
            <b>Ã°Å¸Â§Â® Bayesian Hierarchical Model</b><br>
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
            
            st.markdown("#### Ã°Å¸â€œÅ  Effect Sizes with Credible Intervals")
            st.caption("Hover for exact effect sizes | Click legend to toggle comparisons")
            with st.spinner("Generating interactive Bayesian heatmap..."):
                fig = plot_bayesian_heatmap_interactive(selected_cell, sig_name, comp_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Ridge Plot
        with tabs[1]:
            st.markdown("### Ã°Å¸Å’Å  Posterior Distribution Visualization")
            
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
            
            st.markdown("#### Ã°Å¸Å’Å  Overlapped Posterior Distributions")
            st.caption("Interactive ridge plot | Hover for details | Scroll to zoom | Double-click to reset")
            with st.spinner("Generating interactive ridge plot..."):
                fig = plot_overlapped_ridges_interactive(selected_cell, comp_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
# Tab 3: Bayesian Diagnostics
        with tabs[2]:
            st.markdown("### ðŸ”¬ Bayesian MCMC Diagnostics")
            
            # Collapsible guide at the top
            with st.expander("ðŸ“– Understanding MCMC Diagnostics â€” Click to Learn More", expanded=False):
                st.markdown("""
                <div class="method-box">
                <b>ðŸ” What is MCMC?</b><br><br>
            
                Our analysis uses <b>Bayesian modeling</b> to estimate how biological signatures change with BMI.
                Rather than giving a single fixed value, the model learns a <b>range of plausible values</b>,
                explicitly capturing uncertainty in the data.
            
                To achieve this, we use a method called <b>Markov Chain Monte Carlo (MCMC)</b>.
            
                ðŸ—ºï¸ <b>Analogy:</b><br>
                Imagine exploring a foggy mountain range to find the highest peaks.
                You can only take small steps, guided by the terrain around you.
                Over time, you spend more time near the highest regions.
            
                MCMC works in the same way:
                <ul>
                    <li>It takes many small steps through possible parameter values,</li>
                    <li>Visits more likely values more often,</li>
                    <li>And builds a map of what values best explain the data.</li>
                </ul>
            
                This map is called the <b>posterior distribution</b> â€” it represents our updated belief
                about the true effects after seeing the data.
            
                <hr>
            
                ðŸ“Š <b>Why do we show diagnostics?</b><br><br>
            
                Because MCMC is a stochastic (random) process, we must verify that it explored the space properly.
                The diagnostic plots help answer:
            
                <ul>
                    <li>Did the sampler <b>converge</b> to a stable solution?</li>
                    <li>Do different chains agree with each other?</li>
                    <li>Do we have enough effective samples?</li>
                    <li>Does the model reproduce patterns seen in the data?</li>
                </ul>
            
                If these checks look good, we can trust both the estimated effects <i>and</i> their uncertainty.
            
                âœ”ï¸ <b>In short:</b><br>
                <b>MCMC explores uncertainty, and diagnostics ensure the exploration is reliable.</b>
                </div>
                """, unsafe_allow_html=True)
            
                col1, col2 = st.columns(2)
            
                with col1:
                    st.markdown("""
                    **âœ… Good Convergence Indicators**
                    - **R-hat < 1.01:** Chains agree very well  
                    - **ESS > 400:** Enough independent samples  
                    - **Smooth energy transitions:** Good mixing  
                    - **â€œHairy caterpillarâ€ traces:** Stable, well-mixed chains  
                    """)
            
                with col2:
                    st.markdown("""
                    **âš ï¸ Warning Signs**
                    - **R-hat > 1.05:** Chains disagree (no convergence)  
                    - **ESS < 100:** Strong autocorrelation  
                    - **Divergent transitions:** Geometry or tuning issues  
                    - **Trending traces:** Sampler not at equilibrium  
                    """)
                    
            st.markdown("---")

            
            # ESS and R-hat
            st.markdown("#### ðŸ“Š ESS & R-hat Statistics")
            
            with st.expander("â„¹ï¸ What do ESS and R-hat mean?", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **Effective Sample Size (ESS)**
                    - Measures number of *independent* samples
                    - Accounts for autocorrelation
                    - **Target:** ESS > 400 per parameter
                    - **Good:** Green bars (high ESS)
                    - **Poor:** Red bars (low ESS, need longer chains)
                    """)
                with col2:
                    st.markdown("""
                    **R-hat (Gelman-Rubin)**
                    - Compares within-chain vs between-chain variance
                    - Tests if multiple chains converged to same distribution
                    - **Excellent:** R-hat < 1.01 (chains agree perfectly)
                    - **Acceptable:** R-hat < 1.05
                    - **Problem:** R-hat > 1.05 (chains disagree, not converged)
                    """)
            
            with st.spinner("Generating ESS/R-hat plot..."):
                fig = plot_ess_rhat(comp_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Energy plot
            st.markdown("#### âš¡ Energy Diagnostic")
            
            with st.expander("â„¹ï¸ What is the Energy diagnostic?", expanded=False):
                st.markdown("""
                **Hamiltonian Monte Carlo Energy**
                - Monitors the "energy" of the sampling process (from physics analogy)
                - **Good:** Energy transitions are smooth and explore well
                - **Problem:** Divergent transitions indicate sampling difficulties
                - **Interpretation:** Chains should transition smoothly between energy states
                """)
            
            with st.spinner("Generating energy plot..."):
                fig = plot_energy_diagnostic(comp_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Trace plots
            st.markdown("#### ðŸ“ˆ Trace Plots (First 6 Cell Types)")
            
            with st.expander("â„¹ï¸ How to read trace plots?", expanded=False):
                st.markdown("""
                **What to Look For:**
                - **"Hairy caterpillar":** âœ… Good mixing (chains bouncing around randomly)
                - **Flat mixing:** âœ… All chains overlap (converged to same distribution)
                - **âš ï¸ Trends:** Bad (chain drifting, not converged)
                - **âš ï¸ Stuck chains:** Bad (chain not exploring)
                """)
            
            with st.spinner("Generating trace plots..."):
                fig = plot_trace_diagnostic(comp_data, n_celltypes=6)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Rank plots
            st.markdown("#### ðŸ“Š Rank Plots (First 6 Cell Types)")
            
            with st.expander("â„¹ï¸ What are rank plots?", expanded=False):
                st.markdown("""
                **Rank histograms:** 
                - All chains should have uniform distributions (good mixing)
                - Non-uniform = chains exploring different regions (bad convergence)
                """)
            
            with st.spinner("Generating rank plots..."):
                fig = plot_rank_diagnostic(comp_data, n_celltypes=6)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Autocorrelation
            st.markdown("#### ðŸ“‰ Autocorrelation Plots (First 6 Cell Types)")
            
            with st.expander("â„¹ï¸ What is autocorrelation?", expanded=False):
                st.markdown("""
                **Autocorrelation:** 
                - Measures how correlated successive samples are
                - Should decay quickly to zero (independent samples)
                - High autocorrelation = low ESS (need to thin or run longer)
                - Dashed lines show significance threshold
                """)
            
            with st.spinner("Generating autocorrelation plots..."):
                fig = plot_autocorrelation(comp_data, n_celltypes=6, max_lag=40)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
        # Tab 4: Gene BMI
        with tabs[3]:
            st.markdown("### Ã°Å¸â€œË† Gene-Level BMI Associations")
            st.info("Ã°Å¸â€™Â¡ Hover for statistics Ã¢â‚¬Â¢ Click-drag to zoom Ã¢â‚¬Â¢ Double-click to reset")
            with st.spinner("Running BMI regression analysis..."):
                fig1, fig2 = plot_gene_bmi_interactive(genes, clinical, tpm)
                if fig1:
                    st.plotly_chart(fig1, use_container_width=True)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
        
        # Tab 5: Gene Survival
        with tabs[4]:
            st.markdown("### Ã°Å¸â€™Å  Gene-Level Survival Analysis")
            st.info("Ã°Å¸â€™Â¡ Forest plot with confidence intervals Ã¢â‚¬Â¢ Hover for full statistics")
            with st.spinner("Running survival analysis..."):
                fig = plot_gene_survival_interactive(genes, clinical, tpm)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
        
        # Tab 6: Signature Survival
        with tabs[5]:
            st.markdown("### 🎯 Signature-Level Survival Analysis")
            
            st.markdown("""
            <div class="method-box">
            <b>📊 BMI-Stratified Survival Analysis</b><br>
            7 interactive Plotly plots examining signature-survival relationships across BMI categories.
            </div>
            """, unsafe_allow_html=True)
            
            # Load survival data
            sig_features = load_significant_features()
            zscore_data = load_zscore_data_survival()
            
            if sig_features is None or zscore_data is None:
                st.warning("⚠️ Survival data not available")
            else:
                filtered_sigs = sig_features[sig_features['compartment'] == compartment].copy()
                
                if 'feature' in filtered_sigs.columns:
                    filtered_sigs['cell_from_feature'] = filtered_sigs['feature'].apply(
                        lambda x: x.split('||')[0] if '||' in str(x) else None
                    )
                    
                    def normalize_cell(name):
                        return str(name).upper().replace('_', ' ').strip()
                    
                    filtered_sigs = filtered_sigs[
                        filtered_sigs['cell_from_feature'].apply(normalize_cell) == normalize_cell(selected_cell)
                    ].copy()
                
                if len(filtered_sigs) == 0:
                    st.info(f"ℹ️ No significant survival data for {compartment} / {selected_cell_display}")
                else:
                    st.markdown("#### 🔍 Select Significant Signature")
                    
                    current_feature = f"{selected_cell}||{sig_name}"
                    sidebar_sig_is_significant = current_feature in filtered_sigs['feature'].values
                    default_idx = filtered_sigs['feature'].tolist().index(current_feature) if sidebar_sig_is_significant else 0
                    
                    selected_survival_feature = st.selectbox(
                        f"Significant signatures:",
                        options=filtered_sigs['feature'].tolist(),
                        format_func=lambda x: clean_label_text(x.split('||')[1] if '||' in x else x),
                        index=default_idx,
                        key='survival_sig_dropdown'
                    )
                    
                    feature_data = zscore_data[zscore_data['feature'] == selected_survival_feature].copy()
                    
                    if len(feature_data) > 0:
                        patient_data = clinical.merge(
                            feature_data[['base_sample_id', 'Z']],
                            left_on='sample_id',
                            right_on='base_sample_id',
                            how='inner'
                        )
                        
                        patient_data = patient_data[
                            (patient_data['follow_up_months'] > 0) &
                            (patient_data['follow_up_months'].notna()) &
                            (patient_data['vital_status_binary'].notna())
                        ].copy()
                        
                        if len(patient_data) >= 30:
                            n_total = len(patient_data)
                            n_events = int(patient_data['vital_status_binary'].sum())
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Patients", n_total)
                            with col2:
                                st.metric("Events", n_events)
                            with col3:
                                st.metric("Event Rate", f"{100*n_events/n_total:.1f}%")
                            
                            st.markdown("---")
                            st.markdown("#### 📊 Interactive Survival Plots")
                            
                            sig_display_name = clean_label_text(
                                selected_survival_feature.split('||')[1] if '||' in selected_survival_feature 
                                else selected_survival_feature
                            )
                            
                            plot_functions = [
                                ("BMI vs Time", plot_survival_bmi_vs_time),
                                ("BMI vs HR", plot_survival_bmi_vs_hr),
                                ("Dual-Axis", plot_survival_bmi_dual_axis),
                                ("Forest Plot", plot_survival_forest_bmi),
                                ("Tertile Interaction", plot_survival_interaction_tertile),
                                ("Median Split", plot_survival_interaction_median),
                                ("HR + Distribution", plot_survival_hr_with_distribution)
                            ]
                            
                            plot_count = 0
                            for row_idx in range(3):
                                if row_idx < 2:
                                    cols = st.columns(3)
                                    for col_idx in range(3):
                                        if plot_count < len(plot_functions):
                                            with cols[col_idx]:
                                                plot_name, plot_func = plot_functions[plot_count]
                                                fig = plot_func(patient_data, sig_display_name)
                                                if fig:
                                                    st.plotly_chart(fig, use_container_width=True)
                                                plot_count += 1
                                else:
                                    if plot_count < len(plot_functions):
                                        col1, col2, col3 = st.columns([1, 2, 1])
                                        with col2:
                                            plot_name, plot_func = plot_functions[plot_count]
                                            fig = plot_func(patient_data, sig_display_name)
                            if fig:
                                                    st.plotly_chart(fig, use_container_width=True)
                                            plot_count += 1
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
