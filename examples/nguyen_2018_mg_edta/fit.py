#!/usr/bin/env python
"""Fit the Nguyen et al. 2018 Mg-EDTA isotherm with the two-component binding model.

This is the single-experiment two-component example. The companion notebook
``analysis.ipynb`` loads the samples this script saves and reproduces the posterior
summary, the regression comparison, and the posterior-predictive plot.

    # Short run to check the code works (~30 s):
    python examples/nguyen_2018_mg_edta/fit.py --quick

    # Full run: reproduces the Nguyen et al. 2018 two-component posterior and stores
    # the MCMC samples (~1-2 min on a laptop):
    python examples/nguyen_2018_mg_edta/fit.py --full
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
# Expose several CPU devices to JAX before it is imported so the --full run's chains can
# sample in parallel. Must precede any jax/numpyro (i.e. bayesian_binding) import.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import argparse
import sys
import time
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR.parent))  # so `import _common` works when run as a script

import _common  # noqa: E402

from bayesian_binding.data import load_dat  # noqa: E402
from bayesian_binding.inference import run_mcmc, to_inference_data  # noqa: E402
from bayesian_binding.optimization import fit_map  # noqa: E402

MODEL_NAME = "two_component"
RUN_SETTINGS = {
    "quick": _common.RunSettings(num_warmup=200, num_samples=400, num_chains=1),
    "full": _common.RunSettings(num_warmup=1000, num_samples=2000, num_chains=4),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _common.add_run_arguments(parser, default_out=EXAMPLE_DIR / "results")
    args = parser.parse_args()
    settings = RUN_SETTINGS[args.mode]

    experiment = load_dat(_common.DATA_DIR / "nguyen_2018_mg_edta" / "Mg1EDTAp1a.DAT")
    print(f"Loaded {experiment.name}: {experiment.n_injections} injections")

    # MAP first, then initialize NUTS from it (avoids poor far-from-mode starts).
    map_result = fit_map(experiment, model_name=MODEL_NAME)
    print(f"MAP log-posterior: {map_result.log_posterior:.3f}")

    start = time.time()
    mcmc = run_mcmc(
        experiment,
        model_name=MODEL_NAME,
        init_params=map_result.params,
        rng_seed=args.seed,
        num_warmup=settings.num_warmup,
        num_samples=settings.num_samples,
        num_chains=settings.num_chains,
        target_accept_prob=0.9,
        progress_bar=not args.no_progress,
    )
    elapsed = time.time() - start

    idata = to_inference_data(mcmc)
    print(_common.posterior_parameter_summary(idata))

    metadata = {
        "example": "nguyen_2018_mg_edta",
        "dataset": experiment.name,
        "model": MODEL_NAME,
        "mode": args.mode,
        "seed": args.seed,
        "mcmc": _common.mcmc_metadata(settings),
        "elapsed_seconds": round(elapsed, 1),
        "n_injections": experiment.n_injections,
        "map_log_posterior": float(map_result.log_posterior),
        **_common.run_environment(),
    }
    out = _common.save_results(args.out, samples=mcmc.get_samples(), idata=idata, metadata=metadata)
    print(f"\nSaved results to {out} ({elapsed:.1f} s of sampling)")


if __name__ == "__main__":
    main()
