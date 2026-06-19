# Examples

Each example reproduces a published binding analysis and is split into two halves:

- a **fitting script** (`fit.py`) that runs a MAP fit and MCMC and saves the posterior
  samples, summary, and run metadata under `results/`; and
- a companion **analysis notebook** (`analysis.ipynb`) that loads those saved samples and
  reproduces the tables and figures (no sampling happens in the notebook).

Run the script first, then open the notebook.

| Example | Model | Data | Reference |
| --- | --- | --- | --- |
| [`nguyen_2018_mg_edta/`](nguyen_2018_mg_edta/) | two-component (single isotherm) | Mg-EDTA | Nguyen et al. 2018 |
| [`nguyen_2022_enantiomeric_mixture/`](nguyen_2022_enantiomeric_mixture/) | racemic- & enantiomeric-mixture | Fokkens_1d, Baum_59 | Nguyen et al. 2022 |
| [`nguyen_2022_model_selection/`](nguyen_2022_model_selection/) | Bayes factors among 2C / RM / EM | Fokkens_1d | Nguyen et al. 2022 |
| [`bonin_2022_delta25_dump/`](bonin_2022_delta25_dump/) | cooperative (global, 3 isotherms) | Δ25-hTS/dUMP (public) | Bonin et al. 2022 (eLife) |

The [`nguyen_2022_model_selection/`](nguyen_2022_model_selection/) example has an extra step: after
`fit.py` saves the 2C/RM/EM posterior samples, `bayes_factors.py` estimates the pairwise Bayes
factors by per-model bridge sampling (the recommended estimator), printing the nested warp-BAR
alternative alongside, which the notebook then plots.

The [`bonin_2022_delta25_dump/`](bonin_2022_delta25_dump/) example fits a cooperative (sequential
two-site) model globally to three **public** Δ25-hTS/dUMP isotherms (eLife 2022 source data; Dryad
doi:10.5061/dryad.j9kd51cfx). The N-terminal Δ25 truncation abolishes the positive cooperativity of
full-length hTS/dUMP (Bonin et al. 2019), so the fit returns delta_delta_g ~ 0 (cooperativity ratio
~ 1) -- a worked example of a *non-cooperative* result from the same machinery.

## Running

Each `fit.py` takes a run mode and writes to its own `results/` directory (git-ignored):

```bash
# Short run -- checks the code works (seconds to ~1 minute):
python examples/nguyen_2018_mg_edta/fit.py --quick

# Full run -- reproduces the published result and stores the MCMC samples:
python examples/nguyen_2018_mg_edta/fit.py --full
```

`--quick` is the default. Useful flags: `--full`, `--seed N`, `--out DIR`, `--no-progress`
(disable the NUTS progress bar). The mixture example also takes `--dataset {Fokkens_1d,Baum_59,both}`.

Approximate runtimes on a laptop (the full runs use four chains; `conftest.py`-style CPU
device exposure is set inside each script):

| Example | `--quick` | `--full` |
| --- | --- | --- |
| `nguyen_2018_mg_edta` | ~30 s | ~1-2 min |
| `nguyen_2022_enantiomeric_mixture` | ~1 min (both datasets) | several minutes (4 chains x 8k-16k samples) |
| `bonin_2022_delta25_dump` | ~30 s | ~1-3 min |

Only the full runs are intended to reproduce the published credible intervals; the quick
runs are short and only confirm the pipeline works.

## Outputs

`fit.py` writes three files per result directory:

- `posterior_samples.npz` -- the raw posterior draws (loadable with `numpy.load`);
- `posterior_summary.csv` -- the ArviZ parameter summary (excluding the deterministic
  predicted-heat vectors);
- `run_metadata.json` -- model, dataset, run mode, MCMC settings, timing, and environment.

The analysis notebooks read these via `_common.load_results(...)`.

## Other utilities

- [`model_comparison.py`](model_comparison.py) -- bridge-sampling marginal likelihoods and
  Bayes factors between models for a single dataset.
- [`_common.py`](_common.py) -- shared helpers (dataset paths, the `--quick`/`--full`
  scaffolding, and saving/loading the results bundle).
