#!/usr/bin/env python
"""Compare ITC models with bridge-sampling marginal likelihood estimates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

from bayesian_binding.data import load_dat, load_itc
from bayesian_binding.evidence import bridge_sampling
from bayesian_binding.inference import build_numpyro_model, run_mcmc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat", type=Path)
    parser.add_argument("--itc", type=Path)
    parser.add_argument("--integrated-heats", type=Path)
    parser.add_argument("--models", nargs="+", default=["two_component", "cooperative"])
    parser.add_argument("--cell-concentration-mM", type=float)
    parser.add_argument("--syringe-concentration-mM", type=float)
    parser.add_argument("--cell-volume-mL", type=float, default=1.434)
    parser.add_argument("--temperature-K", type=float, default=298.15)
    parser.add_argument("--num-warmup", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--init-map", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/model_comparison.json"))
    args = parser.parse_args()

    common = {
        "cell_concentration_mM": args.cell_concentration_mM,
        "syringe_concentration_mM": args.syringe_concentration_mM,
        "cell_volume_mL": args.cell_volume_mL,
        "temperature_k": args.temperature_K,
    }
    if args.dat:
        experiment = load_dat(args.dat, **common)
    elif args.itc:
        experiment = load_itc(args.itc, integrated_heats=args.integrated_heats, **common)
    else:
        raise SystemExit("Provide --dat or --itc.")

    results = {}
    for model_name in args.models:
        mcmc = run_mcmc(
            experiment,
            model_name=model_name,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            initialize_with_map=args.init_map,
        )
        model = build_numpyro_model(experiment, model_name=model_name)
        evidence = bridge_sampling(mcmc.get_samples(), model)
        results[model_name] = {
            "log_marginal_likelihood": evidence.log_marginal_likelihood,
            "log_marginal_likelihood_se": evidence.log_marginal_likelihood_se,
            "overlap": evidence.overlap,
            "reliable": evidence.reliable,
            "bridge_converged": evidence.converged,
            "bridge_iterations": evidence.n_iterations,
        }

    names = list(results)
    bayes_factors = {}
    for left in names:
        for right in names:
            if left == right:
                continue
            bayes_factors[f"log_BF_{left}_vs_{right}"] = (
                results[left]["log_marginal_likelihood"] - results[right]["log_marginal_likelihood"]
            )
    output = {"models": results, "bayes_factors": bayes_factors}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
