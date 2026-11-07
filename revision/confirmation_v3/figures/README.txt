EXPORTABLE FIGURES

study_design.pdf/png: current manuscript Figure1, showing matched verified-history
budgets, frozen models and untouched-history confirmation. Recreate with
python -m revision.confirmation_v3.plot_design.

safety.pdf/png: supplementary scatter plot of the prespecified budget64,target.99
safety comparison. Each point averages24 block/model-seed evaluations. The x-axis
is new damage per1000 initially correct events; y is the fraction of initially
incorrect events fully recovered. The source panels use different vertical ranges.
All three generated-plus-real loan policies coincide; overlapping markers express
identical predictions, not missing conditions. Intervals and identity-recovery
costs are in paper_table_contrasts.tex, paper_table_safety.tex and RESULTS.txt.
Recreate with python -m revision.confirmation_v3.present.

Plots use audited analysis.json. No human verification-time or natural-error
performance is inferred from either figure. The manuscript uses study_design;
the complete supplementary safety comparison is also retained numerically.
