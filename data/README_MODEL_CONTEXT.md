# Model Context and Limitations

## About This Analysis

This application presents results from a Bayesian hierarchical analysis of obesity effects on the pancreatic ductal adenocarcinoma (PDAC) tumor microenvironment. The analysis uses cell-type deconvolution from bulk RNA-seq data combined with gene signature scoring.

## Data Source

- **Cohort:** CPTAC Pancreatic Adenocarcinoma cohort
- **Sample Size:** ~140 tumor samples
- **BMI Categories:** Normal (BMI < 25), Overweight (25-30), Obese (> 30)
- **Compartments:** Immune Fine, Immune Coarse, Non-Immune cell populations

## Statistical Modeling Limitations

### Bayesian Model Constraints

1. **Effect Sizes Are Relative:** All effect sizes are standardized (Z-scores). They indicate relative changes between BMI groups, not absolute biological magnitudes.

2. **Credible Intervals ≠ Confidence Intervals:** The 95% HDI (Highest Density Interval) represents a Bayesian credible interval, which has a different interpretation than frequentist confidence intervals.

3. **Convergence Requirements:** Results are only reliable when:
   - R-hat < 1.01 (chains converged)
   - ESS > 400 (sufficient effective samples)
   - No divergent transitions

4. **Hierarchical Shrinkage:** Cell-type estimates are partially pooled. Small sample sizes for some cell types lead to stronger shrinkage toward the population mean.

### What This Analysis CANNOT Claim

1. **Causality:** Observational data cannot establish causal relationships. Associations between BMI and signatures do not prove BMI causes these changes.

2. **Individual Predictions:** This is a population-level analysis. Results should not be used to predict outcomes for individual patients.

3. **Mechanism:** Statistical associations do not explain biological mechanisms. Further experimental validation is required.

4. **Generalizability:** Results are specific to the CPTAC PDAC cohort and may not generalize to other cancer types or populations.

5. **Clinical Utility:** These results are exploratory and not validated for clinical decision-making.

### Cell Type Deconvolution Caveats

1. **Signature Overlap:** Cell type signatures may overlap, leading to correlation between cell type estimates.

2. **Rare Cell Types:** Estimates for rare cell types have higher uncertainty and should be interpreted cautiously.

3. **Bulk Deconvolution:** BayesPrism deconvolution from bulk RNA-seq has inherent limitations compared to single-cell analysis.

### Survival Analysis Caveats

1. **Confounding:** Survival associations may be confounded by unmeasured factors (treatment, stage, comorbidities).

2. **Multiple Testing:** Multiple signatures are tested, increasing false discovery risk despite FDR correction.

3. **Sample Size:** Some BMI-stratified analyses have small sample sizes, limiting statistical power.

## Interpretation Guidelines

### When Discussing Results

- **DO:** Report effect sizes with uncertainty (credible intervals)
- **DO:** Acknowledge the observational nature of the study
- **DO:** Note convergence diagnostics for the specific analysis
- **DO:** Consider biological plausibility

- **DO NOT:** Claim causal relationships
- **DO NOT:** Make clinical recommendations
- **DO NOT:** Overstate confidence in exploratory findings
- **DO NOT:** Ignore non-converged or low-ESS estimates

### Credibility Markers

For continuous analysis:
- **Two Stars (★★):** HDI excludes zero AND ROPE probability > 0.2 (large effect)
- **One Star (★):** HDI excludes zero AND ROPE probability > 0.1 (medium effect)
- **Circle (○):** HDI excludes zero only (small or uncertain effect)
- **No marker:** Effect not credibly different from zero

## Technical Specifications

- **MCMC Sampler:** Hamiltonian Monte Carlo (HMC/NUTS)
- **Chains:** 4 parallel chains
- **Warmup:** Standard warmup phase
- **Posterior Samples:** Stored for all parameters

## Questions the Model Cannot Answer

1. "Should patients change their diet based on these results?" - No clinical recommendations.
2. "Does losing weight improve prognosis?" - Cannot establish causality.
3. "Which cell type is most important?" - Rankings are uncertain and context-dependent.
4. "Are these findings definitive?" - All findings are exploratory and require validation.

## Contact

For questions about the methodology, contact the study authors.
