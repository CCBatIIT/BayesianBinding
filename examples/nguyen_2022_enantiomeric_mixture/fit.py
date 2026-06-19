#!/usr/bin/env python
"""Fit the Nguyen et al. 2022 enantiomeric-mixture ITC datasets.

Two competitive-binding datasets, each with the appropriate model (Nguyen et al. 2022,
PLOS ONE 17(9):e0273656):

- ``Fokkens_1d`` -- a racemic mixture, fit with the racemic-mixture (RM) model (rho = 0.5).
- ``Baum_59`` -- a non-racemic mixture, fit with the enantiomeric-mixture (EM) model
  (rho is a free parameter, peaked near 0.15).

Per-concentration priors (lognormal for a stated concentration, uniform for an unavailable
one) are read from the dataset ``metadata.json``. The companion notebook ``analysis.ipynb``
loads the saved samples and reproduces the S3-Table 95% credible intervals and the
composition posterior. Each dataset's samples are saved under its own subdirectory.

    # Short run to check the code works (~1 min):
    python examples/nguyen_2022_enantiomeric_mixture/fit.py --quick

    # Full run: reproduces the S3-Table credible intervals and stores the samples
    # (4 chains; several minutes). Use --dataset to fit just one:
    python examples/nguyen_2022_enantiomeric_mixture/fit.py --full
    python examples/nguyen_2022_enantiomeric_mixture/fit.py --full --dataset Baum_59
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
# Expose several CPU devices to JAX before it is imported so the four --full chains can
# sample in parallel. Must precede any jax/numpyro (i.e. bayesian_binding) import.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import argparse
import json
import sys
import time
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR.parent))  # so `import _common` works when run as a script

import _common  # noqa: E402

from bayesian_binding.data import load_dat  # noqa: E402
from bayesian_binding.inference import run_mcmc, to_inference_data  # noqa: E402
from bayesian_binding.optimization import fit_map, initial_params  # noqa: E402

DATA_SUBDIR = _common.DATA_DIR / "nguyen_2022_racemic_mixture"

# Per-dataset model, NUTS initialization, output subdirectory, and the number of posterior
# samples per chain in the full run (matching the regression test: the EM posterior for
# Baum_59 is harder to sample, so it uses fewer but is still well determined for rho).
DATASETS = {
    "Fokkens_1d": {"model": "racemic_mixture", "init": "map", "full_samples": 16000, "out": "fokkens_1d_rm"},
    "Baum_59": {"model": "enantiomeric_mixture", "init": "physics", "full_samples": 8000, "out": "baum_59_em"},
}

QUICK = _common.RunSettings(num_warmup=400, num_samples=1000, num_chains=1)


def fit_one(dataset_name: str, *, mode: str, seed: int, out_root: Path, progress: bool) -> Path:
    spec = DATASETS[dataset_name]
    model_name = spec["model"]
    meta = json.loads((DATA_SUBDIR / "metadata.json").read_text())["datasets"][dataset_name]
    experiment = load_dat(
        DATA_SUBDIR / meta["file"],
        cell_concentration_mM=meta["cell_concentration_mM"],
        syringe_concentration_mM=meta["syringe_concentration_mM"],
        cell_volume_mL=meta["cell_volume_mL"],
        temperature_k=meta["temperature_K"],
        name=dataset_name,
    )
    # A stated concentration (S1 Table) uses a lognormal prior; an unavailable one uses an
    # uninformative uniform prior on the linear concentration.
    uniform_kwargs = dict(
        uniform_cell_concentration=meta["cell_concentration_prior"] == "uniform",
        uniform_syringe_concentration=meta["syringe_concentration_prior"] == "uniform",
    )

    if mode == "full":
        settings = _common.RunSettings(num_warmup=2000, num_samples=spec["full_samples"], num_chains=4)
    else:
        settings = QUICK

    # Fokkens_1d (RM) has a minor secondary mode; initialize from the MAP estimate so the
    # chains stay in the dominant best-fit mode. Baum_59 (EM) starts from a physics-based
    # point, as in the regression test.
    if spec["init"] == "map":
        init_params = fit_map(experiment, model_name=model_name, **uniform_kwargs).params
    else:
        init_params = initial_params(experiment, model_name=model_name, **uniform_kwargs)

    print(f"\n=== {dataset_name}: {model_name} ({mode}) ===")
    start = time.time()
    mcmc = run_mcmc(
        experiment,
        model_name=model_name,
        init_params=init_params,
        rng_seed=seed,
        num_warmup=settings.num_warmup,
        num_samples=settings.num_samples,
        num_chains=settings.num_chains,
        chain_method="parallel",
        target_accept_prob=0.9,
        progress_bar=progress,
        **uniform_kwargs,
    )
    elapsed = time.time() - start

    idata = to_inference_data(mcmc)
    print(_common.posterior_parameter_summary(idata))

    metadata = {
        "example": "nguyen_2022_enantiomeric_mixture",
        "dataset": dataset_name,
        "model": model_name,
        "mode": mode,
        "seed": seed,
        "mcmc": _common.mcmc_metadata(settings),
        "elapsed_seconds": round(elapsed, 1),
        "n_injections": experiment.n_injections,
        "uniform_cell_concentration": uniform_kwargs["uniform_cell_concentration"],
        "uniform_syringe_concentration": uniform_kwargs["uniform_syringe_concentration"],
        **_common.run_environment(),
    }
    out = _common.save_results(out_root / spec["out"], samples=mcmc.get_samples(), idata=idata, metadata=metadata)
    print(f"Saved {dataset_name} results to {out} ({elapsed:.1f} s of sampling)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        choices=["both", *DATASETS],
        default="both",
        help="Which dataset to fit (default: both).",
    )
    _common.add_run_arguments(parser, default_out=EXAMPLE_DIR / "results")
    args = parser.parse_args()

    names = list(DATASETS) if args.dataset == "both" else [args.dataset]
    for dataset_name in names:
        fit_one(dataset_name, mode=args.mode, seed=args.seed, out_root=args.out, progress=not args.no_progress)


if __name__ == "__main__":
    main()
