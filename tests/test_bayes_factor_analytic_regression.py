"""Analytic ground-truth regression for the recommended single-model MVN per-model bridge.

The per-model bridge (:func:`bayesian_binding.evidence.bayes_factor_bridge` / ``bridge_sampling``) is
checked end to end -- real NUTS sampling, then bridging -- against a *closed-form* Bayes factor, in the
regime where it is **unbiased**: a conjugate linear-Gaussian model has an exactly Gaussian posterior, so
the multivariate-normal reference is exact (Bennett overlap ~ 1) and the marginal-likelihood estimate is
not biased.

    model A (complex): y = X_A @ beta_A,  beta_A in R^3,  beta ~ N(0, tau)
    model B (simple):  y = X_B @ beta_B,  beta_B in R^2  (X_B = the first two columns of X_A)

Closed form: ``Z_M = N(y; 0, sigma^2 I + tau^2 X_M X_M^T)`` and ``BF = Z_A / Z_B``.

This complements ``test_bayes_factor_bridge_regression.py`` (a real-ITC reproduction, value pinned to a
band) with an *exact* ground-truth check, and ``test_bayes_factor_consistency.py`` (the same recovery from
exact Gaussian draws, no MCMC) by exercising the full sampling-to-bridge pipeline.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from bayesian_binding.evidence import bayes_factor_bridge

_LN10 = np.log(10.0)
_N_DATA = 30
_SIGMA = 0.5
_TAU = 4.0
_NUM_CHAINS = 4
_NUM_SAMPLES = 4000


def _build_conjugate_pair():
    rng = np.random.default_rng(3)
    design = rng.standard_normal((_N_DATA, 3))  # correlated (but Gaussian) posterior
    y = design @ np.array([1.0, -0.5, 0.4]) + _SIGMA * rng.standard_normal(_N_DATA)
    design_simple = design[:, :2]

    def analytic_logz(matrix):
        cov = _SIGMA**2 * np.eye(_N_DATA) + _TAU**2 * (matrix @ matrix.T)
        return float(multivariate_normal.logpdf(y, mean=np.zeros(_N_DATA), cov=cov))

    def make_model(matrix, names):
        matrix_j, y_j = jnp.asarray(matrix), jnp.asarray(y)

        def model():
            coeffs = jnp.stack([numpyro.sample(name, dist.Normal(0.0, _TAU)) for name in names])
            numpyro.sample("y", dist.Normal(matrix_j @ coeffs, _SIGMA), obs=y_j)

        return model

    return {
        "analytic_logz_complex": analytic_logz(design),
        "analytic_logz_simple": analytic_logz(design_simple),
        "complex_model": make_model(design, ["beta0", "beta1", "beta2"]),
        "simple_model": make_model(design_simple, ["beta0", "beta1"]),
    }


def _nuts(model, seed):
    mcmc = MCMC(
        NUTS(model, target_accept_prob=0.9),
        num_warmup=1000,
        num_samples=_NUM_SAMPLES,
        num_chains=_NUM_CHAINS,
        chain_method="parallel",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc.get_samples()


@pytest.mark.regression
def test_per_model_bridge_recovers_analytic_conjugate_bayes_factor():
    """NUTS + per-model bridge recovers each analytic marginal likelihood and the analytic Bayes factor."""
    problem = _build_conjugate_pair()
    analytic_log10_bf = (problem["analytic_logz_complex"] - problem["analytic_logz_simple"]) / _LN10

    complex_samples = _nuts(problem["complex_model"], 0)
    simple_samples = _nuts(problem["simple_model"], 1)
    assert complex_samples["beta0"].size == _NUM_CHAINS * _NUM_SAMPLES

    out = bayes_factor_bridge(
        complex_samples, problem["complex_model"], simple_samples, problem["simple_model"], rng_seed=7
    )

    # Gaussian posterior -> the MVN reference is exact -> unbiased, near-perfect overlap.
    assert out["reliable"]
    assert out["overlap_a"] > 0.9 and out["overlap_b"] > 0.9
    # Each marginal likelihood and the Bayes factor match the closed form (realized errors ~1e-3).
    assert abs(out["logZ_a"] - problem["analytic_logz_complex"]) < 0.2
    assert abs(out["logZ_b"] - problem["analytic_logz_simple"]) < 0.2
    assert abs(out["log10_bf"] - analytic_log10_bf) < 0.1
