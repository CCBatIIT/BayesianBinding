#!/usr/bin/env python
"""Pairwise Bayes factors among the 2C / RM / EM models for one dataset (Nguyen et al. 2022).

The **recommended** estimator -- per-model bridge sampling (each model's log marginal likelihood,
then the ratio) -- is computed first for all three pairs; it is reliable across the full range of
Bayes-factor magnitudes and chain-consistent by construction. The **nested warp-BAR** estimator (the
alternative) applies only to a true sub-model nesting (racemic -> enantiomeric, which simply adds the
composition rho), and is printed alongside with its closed-form asymptotic SE and the Bennett overlap:
when the two posteriors overlap too little it is flagged unreliable and defers to the per-model bridge.
See ``docs/BAYES_FACTORS.md``.

    python examples/nguyen_2022_model_selection/fit.py --full   # first, to produce the samples
    python examples/nguyen_2022_model_selection/bayes_factors.py
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import argparse
import json
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR))  # for `import fit`
sys.path.insert(0, str(EXAMPLE_DIR.parent))  # for `import _common`

import numpy as np  # noqa: E402

import _common  # noqa: E402
from fit import DATASET, MODELS, load_experiment  # noqa: E402

from bayesian_binding.bayes_factor import DirectNesting, nested_bayes_factor  # noqa: E402
from bayesian_binding.evidence import bridge_sampling  # noqa: E402
from bayesian_binding.inference import build_numpyro_model  # noqa: E402

_LN10 = np.log(10.0)

# All (simple, complex) pairs for the per-model bridge -- it needs no nesting, just each model's logZ.
_PAIRS = [
    ("two_component", "racemic_mixture"),
    ("two_component", "enantiomeric_mixture"),
    ("racemic_mixture", "enantiomeric_mixture"),
]
# Nested warp-BAR only applies to a true sub-model: racemic -> enantiomeric simply adds the composition
# rho (DirectNesting). The 2C-vs-mixture pairs are not a drop-one-parameter nesting, so they are
# compared by per-model bridge only.
_NESTINGS = {
    ("racemic_mixture", "enantiomeric_mixture"): DirectNesting(gamma_sites=("rho",)),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, default=EXAMPLE_DIR / "results")
    parser.add_argument("--out", type=Path, default=EXAMPLE_DIR / "results" / "bayes_factors")
    args = parser.parse_args()

    experiment, uniform_kwargs = load_experiment()
    models = {m: build_numpyro_model(experiment, model_name=m, **uniform_kwargs) for m in MODELS}
    samples = {m: _common.load_results(args.results / m)[0] for m in MODELS}

    # --- Recommended: per-model bridge marginal likelihoods, then ratios. ---
    print("Per-model bridge log marginal likelihoods (recommended):")
    evidence = {}
    for model_name in MODELS:
        result = bridge_sampling(samples[model_name], models[model_name])
        evidence[model_name] = result
        print(
            f"  logZ[{model_name:21s}] = {result.log_marginal_likelihood:+10.2f} "
            f"± {result.log_marginal_likelihood_se:.2f}  overlap={result.overlap:.2f} "
            f"reliable={result.reliable}"
        )

    bridge_bf: list[dict] = []
    print("\nBayes factors (log10, complex over simple) -- per-model bridge [recommended]:")
    for simple, complex_ in _PAIRS:
        ln_bf = evidence[complex_].log_marginal_likelihood - evidence[simple].log_marginal_likelihood
        se = float(
            np.hypot(
                evidence[complex_].log_marginal_likelihood_se,
                evidence[simple].log_marginal_likelihood_se,
            )
        )
        row = {
            "pair": f"{complex_}_over_{simple}",
            "log10_bf": ln_bf / _LN10,
            "log10_bf_se": se / _LN10,
            "reliable": bool(evidence[complex_].reliable and evidence[simple].reliable),
        }
        bridge_bf.append(row)
        print(f"  {row['pair']:42s} {row['log10_bf']:+9.1f} ± {row['log10_bf_se']:.1f}")

    # --- Alternative (diagnostic): nested warp-BAR, with overlap-gated reliability flag. ---
    nested_bf: list[dict] = []
    print("\nBayes factors (log10) -- nested warp-BAR [alternative; flagged when overlap is low]:")
    for (simple, complex_), nesting in _NESTINGS.items():
        result = nested_bayes_factor(samples[simple], samples[complex_], models[simple], models[complex_], nesting)
        row = {
            "pair": f"{complex_}_over_{simple}",
            "log10_bf": result.ln_bayes_factor / _LN10,
            "log10_bf_se": result.ln_bayes_factor_se / _LN10,
            "overlap": result.overlap,
            "reliable": result.reliable,
            "warning": result.warning,
        }
        nested_bf.append(row)
        flag = "" if result.reliable else "   [LOW OVERLAP -> trust the per-model bridge above]"
        se_str = f"{row['log10_bf_se']:.1f}" if np.isfinite(row["log10_bf_se"]) else "inf"
        print(f"  {row['pair']:42s} {row['log10_bf']:+9.1f} ± {se_str}  overlap={row['overlap']:.2f}{flag}")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": DATASET,
        "per_model_log_evidence": {
            m: {
                "log_marginal_likelihood": evidence[m].log_marginal_likelihood,
                "log_marginal_likelihood_se": evidence[m].log_marginal_likelihood_se,
                "overlap": evidence[m].overlap,
                "reliable": evidence[m].reliable,
            }
            for m in MODELS
        },
        "bridge_bayes_factors": bridge_bf,
        "nested_warp_bar_bayes_factors": nested_bf,
    }
    (args.out / "bayes_factor_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved Bayes factors to {args.out}")


if __name__ == "__main__":
    main()
