"""Fast tests for per-model bridge sampling (the recommended Bayes-factor estimator).

A conjugate normal-normal model has an analytic log marginal likelihood, so the bridge estimate, its
closed-form asymptotic SE, the Bennett overlap, and the per-model-bridge Bayes factor can all be
checked against ground truth without MCMC (the posterior is sampled in closed form).
"""

from __future__ import annotations

import numpy as np
import numpyro
import numpyro.distributions as dist

from bayesian_binding.evidence import bayes_factor_bridge, bridge_sampling, log_bayes_factor


def _conjugate_normal(m0: float, s0: float, sy: float, y: float, *, n: int = 6000, seed: int = 0):
    """A single-observation conjugate normal model with a known marginal likelihood.

    ``mu ~ Normal(m0, s0)``; ``y ~ Normal(mu, sy)`` observed. Returns (posterior samples of ``mu``,
    no-arg NumPyro model, analytic ``log Z``). ``Z = Normal(y; m0, sqrt(s0^2 + sy^2))`` and the
    posterior of ``mu`` is Normal, sampled here directly.
    """
    post_var = 1.0 / (1.0 / s0**2 + 1.0 / sy**2)
    post_mean = post_var * (m0 / s0**2 + y / sy**2)
    rng = np.random.default_rng(seed)
    samples = {"mu": rng.normal(post_mean, np.sqrt(post_var), size=n)}

    def model():
        mu = numpyro.sample("mu", dist.Normal(m0, s0))
        numpyro.sample("obs", dist.Normal(mu, sy), obs=y)

    marginal_var = s0**2 + sy**2
    log_z = float(-0.5 * np.log(2 * np.pi * marginal_var) - 0.5 * (y - m0) ** 2 / marginal_var)
    return samples, model, log_z


def test_bridge_sampling_recovers_analytic_normal_logz():
    samples, model, log_z = _conjugate_normal(0.0, 5.0, 1.0, 2.0)
    result = bridge_sampling(samples, model)
    assert abs(result.log_marginal_likelihood - log_z) < 0.05
    assert result.converged and result.reliable
    assert result.overlap > 0.9
    assert 0.0 < result.log_marginal_likelihood_se < 0.1


def test_bayes_factor_bridge_matches_analytic():
    samples_a, model_a, log_z_a = _conjugate_normal(0.0, 5.0, 1.0, 2.0, seed=1)
    samples_b, model_b, log_z_b = _conjugate_normal(0.0, 0.5, 1.0, 2.0, seed=2)  # tighter prior
    out = bayes_factor_bridge(samples_a, model_a, samples_b, model_b)
    analytic_log10 = (log_z_a - log_z_b) / np.log(10.0)
    assert abs(out["log10_bf"] - analytic_log10) < 0.05
    assert out["converged"] and out["reliable"]
    assert np.isfinite(out["se"]) and out["se"] >= 0.0


def test_log_bayes_factor_chain_consistent():
    """Per-model factors are differences of independent log evidences -> chain-consistent exactly."""
    sa, ma, _ = _conjugate_normal(0.0, 5.0, 1.0, 1.0, seed=3)
    sb, mb, _ = _conjugate_normal(0.0, 3.0, 1.0, 1.0, seed=4)
    sc, mc, _ = _conjugate_normal(0.0, 1.5, 1.0, 1.0, seed=5)
    ra, rb, rc = bridge_sampling(sa, ma), bridge_sampling(sb, mb), bridge_sampling(sc, mc)
    log_bf_ac = log_bayes_factor(ra, rc)
    log_bf_ab = log_bayes_factor(ra, rb)
    log_bf_bc = log_bayes_factor(rb, rc)
    assert abs(log_bf_ac - (log_bf_ab + log_bf_bc)) < 1e-10
