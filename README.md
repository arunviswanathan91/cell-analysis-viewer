# 🔬 Obesity-Driven Pancreatic Cancer: Cell-Signature Analysis

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://obese-pdac-model.streamlit.app/)

## 📊 Interactive Analysis Platform

This repository accompanies a manuscript investigating **obesity-driven remodeling of the tumor microenvironment in pancreatic ductal adenocarcinoma (PAAD)**. It provides an **interactive Streamlit-based viewer** that allows readers to explore cell-type–resolved molecular signatures, BMI-associated effects, and survival-relevant patterns.

The application is fully web-based and requires no local installation.

---

## 📖 About the Analysis

The interactive viewer enables exploration of relationships between **body mass index (BMI)**, **tumor microenvironment cell types**, and **metabolic and functional gene signatures** in pancreatic cancer.

All visualizations are rendered using **Plotly**, supporting:

* Hover for detailed values
* Zoom using box selection
* Pan via click-drag
* Reset views with double-click

---

## 🧬 Data & Methods Overview

This analysis integrates multiple computational frameworks to characterize obesity-associated tumor microenvironment alterations.

### 🔹 BayesPrism — Cell-Type Deconvolution

BayesPrism is a fully Bayesian framework used to infer tumor microenvironment composition from bulk RNA-seq data. It estimates both **cell-type proportions** and **cell-type–specific gene expression profiles** for each tumor sample.

**Reference:** Danko-Lab/BayesPrism

---

### 🔹 STABL — Stability-Driven Feature Selection

STABL identifies **robust BMI-associated molecular features** using repeated subsampling and bootstrapping. This approach prioritizes features with consistent effects across resampled datasets, thereby reducing false-positive discoveries.

**Reference:** gregbellan/Stabl

---

### 🔹 Bayesian Hierarchical Modeling — Effect Size Estimation

A three-group Bayesian hierarchical model is used to compare:

* Normal BMI (< 25)
* Overweight (25–30)
* Obese (≥ 30)

The model estimates **cell-type–specific obesity effects** on molecular signatures while accounting for between-sample variability. Posterior distributions are inferred using **Markov Chain Monte Carlo (MCMC)** sampling.

**References:**

* Bayesian hierarchical modeling
* Markov Chain Monte Carlo

---

### 🔹 Diagnostic Metrics

Model validity and convergence are assessed using:

* **R-hat** (target < 1.01 for good convergence)
* **Effective Sample Size (ESS)** (> 400 recommended)
* **Energy diagnostic** (Hamiltonian Monte Carlo sampling health)
* **95% Highest Density Intervals (HDI)** as Bayesian credible intervals

---

## 📊 Dataset Summary

* **Source:** CPTAC Pancreatic Adenocarcinoma (PAAD) cohort
* **Samples:** 140 tumor samples with clinical annotation
* **Cell Types:** Immune and non-immune populations inferred via deconvolution
* **Signatures:** 30+ metabolic and functional gene signatures per cell type

---

## 🎯 Analysis Workflow

1. **Deconvolution:** BayesPrism → cell-type proportions
2. **Expression processing:** TPM normalization → gene expression matrices
3. **Signature scoring:** Gene aggregation → Z-score–based signature scores
4. **Feature selection:** STABL → robust BMI-associated features
5. **Modeling:** Bayesian hierarchical modeling → effect sizes with uncertainty
6. **Validation:** MCMC diagnostics → convergence assessment
7. **Survival analysis:** Cox proportional hazards regression → clinical relevance

---

## 🧭 Interactive Cell Analysis Viewer

The Streamlit application supports:

* Real-time interactive heatmaps and summary plots
* Cell-type–resolved inspection of BMI effects
* Exploration of effect sizes, uncertainty estimates, and survival associations

🔗 **Launch the interactive viewer:**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://obese-pdac-model.streamlit.app/)

---

## 📌 Citation

Citation details will be provided upon publication of the associated manuscript.
