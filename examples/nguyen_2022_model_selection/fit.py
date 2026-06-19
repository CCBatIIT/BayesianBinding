#!/usr/bin/env python
"""Fit the two-component, racemic-mixture, and enantiomeric-mixture models to one dataset.

This is the first half of the Nguyen et al. 2022 model-selection example: all three models are fit
to the *same* dataset (Fokkens_1d) so their posterior samples can be compared with Bayes factors by
``bayes_factors.py``. Each model's samples are saved under ``results/<model>/``.

    # Short run to check the code works (~1-2 min):
    python examples/nguyen_2022_model_selection/fit.py --quick

    # Full run for reliable Bayes factors (4 chains x many samples; several minutes -- Bayes
    # factors need far more samples than posterior summaries, see bayes_factors.py):
    python examples/nguyen_2022_model_selection/fit.py --full
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
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
DATASET = "Fokkens_1d"
MODELS = ("two_component", "racemic_mixture", "enantiomeric_mixture")

# Bayes factors need many more samples than posterior summaries to converge (Nguyen et al. 2022).
RUN_SETTINGS = {
    "quick": _common.RunSettings(num_warmup=1000, num_samples=2000, num_chains=1),
    "full": _common.RunSettings(num_warmup=2000, num_samples=15000, num_chains=4),
}


def load_experiment():
    meta = json.loads((DATA_SUBDIR / "metadata.json").read_text())["datasets"][DATASET]
    experiment = load_dat(
        DATA_SUBDIR / meta["file"],
        cell_concentration_mM=meta["cell_concentration_mM"],
        syringe_concentration_mM=meta["syringe_concentration_mM"],
        cell_volume_mL=meta["cell_volume_mL"],
        temperature_k=meta["temperature_K"],
        name=DATASET,
    )
    # Per-concentration priors from the dataset metadata (Fokkens_1d: both lognormal). The three
    # models must share the same concentration prior so their nuisance sites line up for the
    # Bayes-factor reparameterization.
    uniform_kwargs = dict(
        uniform_cell_concentration=meta["cell_concentration_prior"] == "uniform",
        uniform_syringe_concentration=meta["syringe_concentration_prior"] == "uniform",
    )
    return experiment, uniform_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _common.add_run_arguments(parser, default_out=EXAMPLE_DIR / "results")
    args = parser.parse_args()
    settings = RUN_SETTINGS[args.mode]

    experiment, uniform_kwargs = load_experiment()
    print(f"Loaded {experiment.name}: {experiment.n_injections} injections; fitting {MODELS}")

    for model_name in MODELS:
        # The competitive-binding (mixture) posteriors have a spurious 'no binding' basin; initialize
        # the mixtures from a physics-based point (as in the nguyen_2022_enantiomeric_mixture example)
        # and the two-component model from its MAP.
        if model_name == "two_component":
            init_params = fit_map(experiment, model_name=model_name, **uniform_kwargs).params
        else:
            init_params = initial_params(experiment, model_name=model_name, **uniform_kwargs)

        print(f"\n=== {model_name} ({args.mode}) ===")
        start = time.time()
        mcmc = run_mcmc(
            experiment,
            model_name=model_name,
            init_params=init_params,
            rng_seed=args.seed,
            num_warmup=settings.num_warmup,
            num_samples=settings.num_samples,
            num_chains=settings.num_chains,
            chain_method="parallel",
            target_accept_prob=0.9,
            progress_bar=not args.no_progress,
            **uniform_kwargs,
        )
        elapsed = time.time() - start
        idata = to_inference_data(mcmc)
        print(_common.posterior_parameter_summary(idata))

        metadata = {
            "example": "nguyen_2022_model_selection",
            "dataset": DATASET,
            "model": model_name,
            "mode": args.mode,
            "seed": args.seed,
            "mcmc": _common.mcmc_metadata(settings),
            "elapsed_seconds": round(elapsed, 1),
            "n_injections": experiment.n_injections,
            "uniform_cell_concentration": uniform_kwargs["uniform_cell_concentration"],
            "uniform_syringe_concentration": uniform_kwargs["uniform_syringe_concentration"],
            **_common.run_environment(),
        }
        out = _common.save_results(args.out / model_name, samples=mcmc.get_samples(), idata=idata, metadata=metadata)
        print(f"Saved {model_name} to {out} ({elapsed:.1f} s)")

    print("\nNow compute the Bayes factors: python examples/nguyen_2022_model_selection/bayes_factors.py")


if __name__ == "__main__":
    main()
