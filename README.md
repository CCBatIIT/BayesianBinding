# BayesianBinding

**Bayesian uncertainty quantification for protein–ligand binding from calorimetry and X-ray scattering.**

BayesianBinding fits binding models to experimental data and returns *full posterior distributions* —
credible intervals on binding free energies, enthalpies, and cooperativity — rather than single
point estimates. It is built on a modern [NumPyro](https://num.pyro.ai)/[JAX](https://jax.dev) stack
(NUTS, autodiff, JIT) and can **fit multiple modalities at once**, sharing a single binding free energy 
across them. Presently isothermal titration calorimetry (ITC) and wide-angle X-ray solution scattering 
(WAXS) are implemented as the starting point, not a fixed scope — the modality-independent core is designed so
that further experimental methods can be added later.

The codebase reproduces results from several published analyses (Nguyen et al. 2018/2022, Bonin et
al. 2022, Minh & Makowski 2013) as runnable, tested examples.

## Highlights

- **Multiple modalities, one free energy.** Fit ITC heats and WAXS scattering jointly, 
  sharing `delta_g` — or fit either alone. WAXS uses the multivariate curve resolution of Minh & Makowski (2013).
- **Full posteriors, not point estimates.** NUTS sampling with exact gradients; MAP initialization;
  ArviZ summaries and diagnostics.
- **A library of binding models** — simple 1:1, two-site cooperative, monomer–dimer, and competing
  enantiomeric mixtures — each with closed-form or implicitly-differentiated equilibria for stable
  gradients.
- **Principled model selection.** Bayes factors by per-model bridge sampling — the recommended
  estimator, reliable across the full range of factor magnitudes — with closed-form asymptotic errors
  and a Bennett-overlap reliability flag; a nested warp-BAR alternative is also provided. See
  [`docs/BAYES_FACTORS.md`](docs/BAYES_FACTORS.md).
- **Global fits.** Share thermodynamic parameters across many experiments while keeping each one's
  own nuisances (baseline offset, noise, concentration scale).
- **Reproducible.** Every example ships a fit script and an analysis notebook; published numbers are
  pinned by regression tests.

## Install

Requires Python ≥ 3.10.

```bash
python -m pip install -e ".[dev]"
pytest          # fast suite (smoke tests fit every model end to end), ~1-2 minutes
```

The slower tests that reproduce published posteriors are excluded by default; run them with
`pytest -m regression`.

## Quick start

**Fit an ITC titration** and get a posterior:

```python
from bayesian_binding import load_dat, fit_map, run_mcmc

experiment = load_dat("data/nguyen_2018_mg_edta/Mg1EDTAp1a.DAT")
start = fit_map(experiment, model_name="two_component")        # MAP point estimate
mcmc = run_mcmc(experiment, model_name="two_component",        # full posterior (NUTS)
                init_params=start.params)
mcmc.print_summary()                                           # delta_g, delta_h + credible intervals
```

**Analyze a WAXS titration** (single binding free energy from the scattering series):

```python
from bayesian_binding import WAXSDataset, waxs_delta_g_posterior

waxs = WAXSDataset.load("data/lysozyme_waxs_nag2/waxs_nag2.npz")
result = waxs_delta_g_posterior(waxs)
print(result.delta_g, result.ci68)   # -5.74 kcal/mol, (-5.80, -5.67) -- reproduces Minh & Makowski Table 1
```

**Fit ITC and WAXS jointly**, sharing `delta_g`, with `build_joint_numpyro_model(...)` + `run_nuts(...)`
— see [`docs/WAXS.md`](docs/WAXS.md) for the WAXS data model and processing pipeline.

## Binding models

| Model | Description |
| --- | --- |
| `TwoComponentBindingModel` | one ligand binding one receptor (1:1). |
| `CooperativeBindingModel` | two-site sequential/cooperative binding, parameterized by `delta_g` and `delta_delta_g` (set `equivalent_sites=True` for a no-cooperativity null). |
| `DimerizationCooperativeBindingModel` | monomer–dimer receptor equilibrium plus two ligand-binding steps on the dimer. |
| `EnantiomericMixtureBindingModel` | two enantiomers competing for one receptor (Nguyen et al. 2022); `racemic` flag selects fixed `rho = 0.5` vs. a free composition. |

Free energies and enthalpies are in kcal/mol; concentrations in molar, volumes in liters, heats in
microcalories. Equilibria use closed-form solutions where available (e.g. Wang 1995 for competitive
binding) and an implicit-function gradient (one Newton step) for the bisection-solved cooperative
model, giving exact gradients for NUTS and gradient-based MAP.

## Examples

Each example under [`examples/`](examples/) reproduces a published analysis as a fit script
(`fit.py`, with `--quick`/`--full` modes) plus a companion notebook (`analysis.ipynb`). See
[`examples/README.md`](examples/README.md) for runtimes.

| Example | What it reproduces |
| --- | --- |
| [`nguyen_2018_mg_edta/`](examples/nguyen_2018_mg_edta/) | Mg–EDTA two-component regression (Nguyen et al. 2018). |
| [`nguyen_2022_enantiomeric_mixture/`](examples/nguyen_2022_enantiomeric_mixture/) | racemic- and enantiomeric-mixture fits with 95% credible intervals (Nguyen et al. 2022). |
| [`nguyen_2022_model_selection/`](examples/nguyen_2022_model_selection/) | per-model bridge Bayes factors among 2C/RM/EM models, with the nested warp-BAR alternative (Nguyen et al. 2022). |
| [`bonin_2022_delta25_dump/`](examples/bonin_2022_delta25_dump/) | global cooperative fit of three **public** Δ25-hTS/dUMP isotherms — the N-terminal truncation abolishes cooperativity (ΔΔG ≈ 0; Bonin et al. 2022, eLife). |

## How it works

The regression machinery is split into a modality-independent **binding model** (free energies →
equilibrium species concentrations) and a modality-specific **observation model** (concentrations →
predicted signal). Today ITC (`expected_heats`) and WAXS (`D = C(delta_g)·R + ε`, reference patterns
`R` profiled out by weighted least squares) are realized; WAXS data processing and the joint
ITC + WAXS fit are documented in [`docs/WAXS.md`](docs/WAXS.md).

- **Model selection**: per-model bridge sampling (`bayesian_binding.evidence.bayes_factor_bridge` /
  `bridge_sampling` + `log_bayes_factor`) estimates each model's marginal likelihood and takes the
  ratio — the recommended estimator. A nested between-model alternative
  (`bayesian_binding.bayes_factor.nested_bayes_factor`, warp-BAR by default) is also provided; both
  report a closed-form asymptotic SE and a Bennett-overlap reliability flag. See
  [`docs/BAYES_FACTORS.md`](docs/BAYES_FACTORS.md).
- **Global / joint fits**: `build_global_numpyro_model` (many ITC experiments, shared thermodynamics)
  and `build_joint_numpyro_model` (ITC + WAXS, shared `delta_g`) keep per-dataset nuisances separate.
- **WAXS from raw data**: `load_waxs_calibrated` reproduces the original MATLAB `loadTable.m`
  normalization (per-condition ion-chamber calibration) so `waxs_delta_g_posterior` can run straight
  from `.chi` curves; `waxs_delta_g_posterior` is a faithful port of the paper's `analyze1site.m`.

## Status

Research code, version 0.1. The API is still evolving. The modality roadmap is deliberately demand-driven: 
the project begins with ITC and WAXS and will integrate additional experimental methods later, as they are
required by ongoing projects or requested by collaborators, rather than speculatively. Contributions
and issues are welcome.

## References

- Nguyen et al. 2018. *Bayesian analysis of isothermal titration calorimetry for binding
  thermodynamics.* PLOS ONE 13(9): e0203224.
  [doi.org/10.1371/journal.pone.0203224](https://doi.org/10.1371/journal.pone.0203224)
- Nguyen et al. 2022. *Bayesian regression and model selection for isothermal titration calorimetry
  with enantiomeric mixtures.* PLOS ONE 17(9): e0273656.
  [doi.org/10.1371/journal.pone.0273656](https://doi.org/10.1371/journal.pone.0273656)
- Bonin et al. 2019. *Positive cooperativity in substrate binding by human thymidylate synthase.*
  Biophysical Journal 117(6): 1074–1084.
  [doi.org/10.1016/j.bpj.2019.08.015](https://doi.org/10.1016/j.bpj.2019.08.015)
- Bonin, Sapienza & Lee 2022. *Dynamic allostery in substrate binding by human thymidylate synthase.*
  eLife 11: e79915. [doi.org/10.7554/eLife.79915](https://doi.org/10.7554/eLife.79915)
  (source data also at Dryad [doi.org/10.5061/dryad.j9kd51cfx](https://doi.org/10.5061/dryad.j9kd51cfx))
- Minh & Makowski 2013. *Wide-angle X-ray solution scattering for protein–ligand binding: multivariate
  curve resolution with Bayesian confidence intervals.* Biophysical Journal 104(4): 873–883.
  [doi.org/10.1016/j.bpj.2012.12.019](https://doi.org/10.1016/j.bpj.2012.12.019)
