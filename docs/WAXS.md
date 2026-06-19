# WAXS data processing

BayesianBinding can infer a binding free energy `delta_g` from a wide-angle X-ray solution
scattering (WAXS) titration series, using the multivariate curve resolution (MCR) approach of
Minh & Makowski (2013). This note describes the data model and the two-stage processing pipeline.

## Data model

The scattering data are modeled as `D = C(delta_g)·R + ε` (multivariate curve resolution): each
measured curve is a concentration-weighted mixture of a small set of reference species patterns
`R`, where the concentrations `C(delta_g)` come from the binding model. The reference patterns `R`
are profiled out by weighted least squares, so the only free binding parameter in the likelihood is
`delta_g`. WAXS is enthalpy-blind (it carries no `delta_h`).

## What the repository ships

To keep the repository small and reproducible, only the **preprocessed** dataset needed by the
regression test is included, as a compressed NumPy archive:

```python
from bayesian_binding import WAXSDataset, waxs_delta_g_posterior

waxs = WAXSDataset.load("data/lysozyme_waxs_nag2/waxs_nag2.npz")
result = waxs_delta_g_posterior(waxs)
print(result.delta_g, result.ci68)   # reproduces Minh & Makowski (2013), Table 1
```

The raw `.chi` curves, per-run analysis tables, and per-angle integration weights used to *produce*
that archive are part of the original experimental pipeline and are not distributed here. The
functions that consume them (Stage 1, below) are included for completeness and for users who bring
their own raw data.

## Stage 1 — calibration (raw `.chi` → normalized intensities)

`load_waxs_calibrated(...)` is a faithful port of the original MATLAB `loadTable.m`. Given raw
`.chi` curves, an analysis table, and the per-angle integration weights for a run, it:

1. drops exposures flagged in the notes column (outliers, buffer/water);
2. performs a per-condition ion-chamber calibration — fits the summed scattering
   `Isum = intensities · counts` against the ion-chamber reading and divides each curve by the fit,
   removing beam-intensity drift between exposures;
3. truncates to the beam-stop angular range and then to the small-angle window
   `1/d <= one_over_d_max` (default `0.2 Å⁻¹`, where `1/d = 2 sin(θ)/λ`).

The wide-angle region (`1/d > 0.2 Å⁻¹`) is excluded because the solvent/ligand nonlinearity there
is not a linear concentration effect. The result is an analysis-ready `WAXSDataset` — the same
object `WAXSDataset.load` returns from a shipped `.npz`.

## Stage 2 — inference (the `delta_g` posterior)

`waxs_delta_g_posterior(dataset, ...)` is a faithful port of the paper's `analyze1site.m`. It:

1. forms the condition means of the series and does an SVD to get an angle-space basis;
2. projects each exposure onto that basis normalized by the singular value (the SVD *coordinates*)
   and fits the four-species model to the top `K = 4` coordinates by weighted least squares,
   weighting by the inverse shrinkage variance (the scatter of each coordinate about its condition
   mean, pooled over the protein / protein-free groups, divided by the replicate count);
3. forms the 1-D posterior over `delta_g` (point estimate = median; 68% CI = 16th/84th percentiles).

Fitting the SVD *coordinates* (rather than raw angles) is what surfaces the subtle binding signal
that the dominant free-ligand scattering otherwise swamps; with more conditions than species, the
four-coordinate fit is over-determined.

## Other binding models, and joint ITC + WAXS

The species/`C` model is a pluggable callable `(binding_params, protein, ligand, T) -> C`, so a model
other than 1:1 can be used without touching the likelihood. `binding_model_concentrations(model)`
adapts any `models.py` binding model (two-component, cooperative two-site, dimerization) into such a
C-model, so WAXS resolves the *same* equilibrium an ITC fit uses.

`build_joint_numpyro_model(itc_experiments, waxs_reduced, ...)` samples a single shared `delta_g` and
adds both the ITC heat likelihood (with ITC-only `delta_h`/offset/noise) and the WAXS observation, so
the two modalities are fit together under one binding free energy. `build_waxs_numpyro_model` is the
WAXS-only variant; both run under the generic `run_nuts`.

## Reference

Minh & Makowski (2013), *Wide-angle X-ray solution scattering for protein–ligand binding:
multivariate curve resolution with Bayesian confidence intervals*, Biophysical Journal 104(4):
873–883. <https://doi.org/10.1016/j.bpj.2012.12.019>
