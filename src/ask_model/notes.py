"""Curated facts about the study design, taken from the viewer's own methodology text.
The assistant may state these as background; everything quantitative must come from SQL."""

STUDY_NOTES = {
    "Study design": (
        "The study investigates obesity-driven remodeling of the tumor microenvironment in pancreatic ductal "
        "adenocarcinoma (PDAC). Data: CPTAC-PAAD cohort, 140 tumor samples with clinical annotation, bulk RNA-seq (TPM). "
        "BMI groups: Normal (BMI < 25), Overweight (25 to 30), Obese (>= 30). Compartments: immune_fine, immune_coarse, "
        "non_immune."),
    "Workflow": (
        "1. Deconvolution: BayesPrism infers cell-type proportions and cell-type-specific expression from bulk RNA-seq. "
        "2. Expression: TPM values from CPTAC-3 give the gene expression matrix. "
        "3. Signatures: a custom signature database (30+ metabolic and functional signatures per cell type) yields "
        "signature scores (z-scores) per patient. "
        "4. Selection: STABL (stability selection with machine learning) picks robust BMI-associated features. "
        "5. Modeling: Bayesian hierarchical models fitted by MCMC give effect sizes with uncertainty at feature (signature) "
        "level and cell level. 6. Validation: MCMC diagnostics (r_hat, ESS) check convergence. "
        "7. Survival: Cox regression tests clinical relevance of credible signatures and cell types."),
    "Credibility definitions": (
        "Credible: the 95% highest density interval (HDI) of the effect excludes zero (both bounds share a sign). "
        "Not credible: the HDI crosses zero, even when the posterior mean is nonzero. ROPE (region of practical "
        "equivalence) probabilities, prob_gt_0.1 / 0.2 / 0.3 / 0.5, give the posterior probability that the effect "
        "magnitude exceeds that threshold. Continuous-analysis markers: two stars = HDI excludes zero and ROPE probability "
        "> 0.2 (large effect); one star = HDI excludes zero and ROPE probability > 0.1 (medium effect); circle = HDI "
        "excludes zero only; no marker = not credibly different from zero."),
    "Categorical versus continuous": (
        "Categorical analysis compares discrete BMI groups (overweight vs normal, obese vs normal, obese vs overweight). "
        "Continuous analysis models BMI as a continuous variable and estimates a slope per 1 SD of BMI. A slope near zero "
        "does not mean there is no group difference. Use the categorical tables for group questions and the continuous "
        "tables for dose-response questions."),
    "STABL": (
        "STABL (stability-driven feature selection) identifies robust biomarkers by running feature selection on many "
        "bootstrap or subsampled datasets and keeping features that are selected consistently. A feature in the stabl table "
        "was selected as robustly BMI-associated; a feature missing from it was not selected."),
    "Interactome": (
        "Cell-cell interactions are scored per signature dataset (Bindea, Newman, Zheng) and per BMI category "
        "(normal weight vs overweight). Each interaction has a favorable cell and an unfavorable cell; shared IMGP are the "
        "shared interacting gene pairs, and the enrichment ratio with its p-value, adjusted p-value and FDR measures how "
        "enriched the interaction is. In the data the Bindea set is labeled 'Bindeya'."),
    "Survival": (
        "Survival analysis uses Cox proportional hazards on the signature z-score within a BMI comparison. HR > 1 means "
        "higher signature score is associated with worse survival, HR < 1 with better survival. This part is exploratory "
        "and not part of the published paper."),
    "Limitations - what this analysis cannot claim": (
        "It cannot establish causality (observational data), predict outcomes for individual patients, explain biological "
        "mechanism, generalize beyond the CPTAC PDAC cohort, or support clinical decisions. Cell-type estimates come from "
        "bulk deconvolution and are partially pooled, so rare cell types are uncertain."),
}
