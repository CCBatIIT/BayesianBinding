from pathlib import Path

import pytest

from bayesian_binding.data import load_dat
from bayesian_binding.inference import run_mcmc
from bayesian_binding.optimization import fit_map
from bayesian_binding.regression import (
    assert_nguyen_2018_mg_edta_regression,
    nguyen_2018_mg_edta_regression_rows,
    summarize_mg_edta_posterior,
)


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.regression
def test_nguyen_2018_mg_edta_posterior_regression():
    experiment = load_dat(REPO / "data" / "nguyen_2018_mg_edta" / "Mg1EDTAp1a.DAT")
    map_result = fit_map(experiment, model_name="two_component")
    # Four MAP-initialized chains (parallel across the CPU devices set in conftest.py). The
    # delta_h 95% CI lower bound sits right at the regression window edge, so a single
    # 1000-sample chain is too Monte-Carlo-noisy to be a stable check -- its exact value
    # depends on negligible (1e-5) shifts in the MAP starting point. Four chains give a
    # stable, starting-point-independent posterior estimate.
    mcmc = run_mcmc(
        experiment,
        model_name="two_component",
        init_params=map_result.params,
        rng_seed=20260610,
        num_warmup=1000,
        num_samples=1000,
        num_chains=4,
        chain_method="parallel",
        progress_bar=False,
    )
    summary = summarize_mg_edta_posterior(mcmc.get_samples())
    assert summary["n_samples"] == 4000
    assert_nguyen_2018_mg_edta_regression(summary)
    assert all(row["passes"] for row in nguyen_2018_mg_edta_regression_rows(summary))
