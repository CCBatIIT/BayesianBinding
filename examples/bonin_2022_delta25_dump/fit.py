#!/usr/bin/env python
"""Global cooperative fit of the public Δ25-hTS/dUMP isotherms (a non-cooperative control).

This multi-dataset example is built entirely on **public** data: three dUMP-into-Δ25
(N-terminally truncated) human thymidylate synthase isotherms from Bonin, Sapienza & Lee
(2022, eLife 11:e79915; Appendix 1 Figure 3 source data; also Dryad doi:10.5061/dryad.j9kd51cfx).
The sequential two-site (cooperative-capable) thermodynamics and a single cell-concentration
scaling factor are shared across the three isotherms, while each isotherm keeps its own heat offset
and noise scale. Unlike full-length hTS/dUMP (which is ~9-fold positively cooperative; Bonin et al.
2019), the Δ25 truncation abolishes cooperativity: the fit returns delta_delta_g ~ 0 (cooperativity
ratio ~ 1, DG1 ~ DG2). The companion notebook ``analysis.ipynb`` loads the saved samples and shows
this.

    # Short run to check the code works (~30 s):
    python examples/bonin_2022_delta25_dump/fit.py --quick

    # Full run (4 chains; ~1-3 min on a laptop):
    python examples/bonin_2022_delta25_dump/fit.py --full
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
# Expose several CPU devices to JAX before it is imported so the --full chains can sample in
# parallel. Must precede any jax/numpyro (i.e. bayesian_binding) import.
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
from bayesian_binding.inference import run_mcmc_global, to_inference_data  # noqa: E402
from bayesian_binding.optimization import fit_map_global  # noqa: E402

MODEL_NAME = "cooperative"
DATA_SUBDIR = _common.DATA_DIR / "bonin_2022_delta25_dump"
RUN_SETTINGS = {
    "quick": _common.RunSettings(num_warmup=200, num_samples=400, num_chains=1),
    "full": _common.RunSettings(num_warmup=1000, num_samples=2000, num_chains=4),
}


def load_experiments():
    metadata = json.loads((DATA_SUBDIR / "metadata.json").read_text())
    return [
        load_dat(
            DATA_SUBDIR / spec["file"],
            cell_concentration_mM=spec["cell_concentration_mM"],
            syringe_concentration_mM=spec["syringe_concentration_mM"],
            cell_volume_mL=metadata["cell_volume_mL"],
            temperature_k=metadata["temperature_K"],
            name=name,
        )
        for name, spec in metadata["datasets"].items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _common.add_run_arguments(parser, default_out=EXAMPLE_DIR / "results")
    args = parser.parse_args()
    settings = RUN_SETTINGS[args.mode]

    experiments = load_experiments()
    print(f"Loaded {len(experiments)} isotherms: {[e.name for e in experiments]}")

    # Global MAP fit (shared thermodynamics + concentration scale; per-isotherm nuisances
    # concentrated out), then initialize NUTS from it.
    map_result = fit_map_global(experiments, model_name=MODEL_NAME)
    print(f"MAP log-posterior: {map_result.log_posterior:.3f}")
    print(f"MAP cell_concentration_scale: {float(map_result.params['cell_concentration_scale']):.3f}")
    print(
        f"MAP delta_delta_g: {float(map_result.params['delta_delta_g']):+.3f} kcal/mol "
        "(0 = no cooperativity)"
    )

    start = time.time()
    mcmc = run_mcmc_global(
        experiments,
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
        "example": "bonin_2022_delta25_dump",
        "datasets": [e.name for e in experiments],
        "model": MODEL_NAME,
        "mode": args.mode,
        "seed": args.seed,
        "mcmc": _common.mcmc_metadata(settings),
        "elapsed_seconds": round(elapsed, 1),
        "n_injections": [e.n_injections for e in experiments],
        "map_log_posterior": float(map_result.log_posterior),
        **_common.run_environment(),
    }
    out = _common.save_results(args.out, samples=mcmc.get_samples(), idata=idata, metadata=metadata)
    print(f"\nSaved results to {out} ({elapsed:.1f} s of sampling)")


if __name__ == "__main__":
    main()
