"""Marginal likelihood and Bayes-factor utilities -- per-model bridge sampling (recommended).

This module implements the **recommended** Bayes-factor estimator: per-model bridge sampling
(Meng & Wong 1996; Gronau et al. 2017). Each model's log marginal likelihood ``log Z`` is estimated
by bridging its posterior to a multivariate-normal reference fit to that same posterior, then
``log BF = log Z_a - log Z_b``. Unlike the nested between-model bridge in
:mod:`bayesian_binding.bayes_factor`, the only overlap required is between a posterior and its own
Gaussian fit, so the estimator stays reliable as the Bayes factor grows.

Each estimate carries a closed-form **asymptotic** standard error (the optimal-bridge / BAR
asymptotic variance; Bennett 1976; Shirts et al. 2003; Frühwirth-Schnatter 2004) and a **Bennett
overlap integral** ``O in [0, 1]``. A low overlap flags the estimate as unreliable -- see
:func:`bridge_diagnostics`, shared with the nested estimator. See ``docs/BAYES_FACTORS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from bayesian_binding import _jax_config as _jax_config

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import log_density
from scipy.special import expit, logsumexp
from scipy.stats import multivariate_normal

# Default Bennett overlap below which a bridge estimate is flagged unreliable.
DEFAULT_OVERLAP_THRESHOLD = 0.10


@dataclass(frozen=True)
class BridgeSamplingResult:
    log_marginal_likelihood: float
    log_marginal_likelihood_se: float  # closed-form asymptotic SE (Bennett 1976 / Shirts 2003)
    overlap: float  # Bennett harmonic-mean overlap integral in [0, 1]
    reliable: bool  # overlap >= threshold and the asymptotic SE is finite
    n_posterior_samples: int
    n_proposal_samples: int
    parameter_names: tuple[str, ...]
    n_iterations: int
    converged: bool
    warning: str | None = None


def _flatten_samples(
    samples: dict[str, np.ndarray], parameter_names: tuple[str, ...]
) -> tuple[np.ndarray, list[tuple[str, tuple[int, ...], int]]]:
    pieces = []
    shapes = []
    for name in parameter_names:
        arr = np.asarray(samples[name])
        flat = arr.reshape(arr.shape[0], -1)
        pieces.append(flat)
        shapes.append((name, arr.shape[1:], flat.shape[1]))
    return np.column_stack(pieces), shapes


def _unflatten_row(row, shapes: list[tuple[str, tuple[int, ...], int]]) -> dict[str, jnp.ndarray]:
    params = {}
    offset = 0
    for name, shape, size in shapes:
        params[name] = jnp.asarray(row[offset : offset + size].reshape(shape))
        offset += size
    return params


def _logmeanexp(values: np.ndarray) -> float:
    return float(logsumexp(values) - np.log(values.size))


def _regularized_covariance(matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric positive definite covariance estimate."""
    n_samples, n_dim = matrix.shape
    if n_samples < 2:
        return np.eye(n_dim) * 1.0e-4
    covariance = np.atleast_2d(np.cov(matrix, rowvar=False))
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(covariance).copy()
    positive = diagonal[np.isfinite(diagonal) & (diagonal > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    diagonal_target = np.diag(np.where(np.isfinite(diagonal) & (diagonal > 0.0), diagonal, scale))
    covariance = 0.95 * covariance + 0.05 * diagonal_target
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=scale, neginf=0.0)
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
    jitter = max(1.0e-10 * scale, -min_eigenvalue + 1.0e-10 * scale, 1.0e-12)
    return covariance + np.eye(n_dim) * jitter


def _batched_log_density(model, shapes: list[tuple[str, tuple[int, ...], int]]):
    """Jitted+vmapped unnormalized log-density over a flat (n, n_dim) parameter matrix."""

    def single(row):
        return log_density(model, (), {}, _unflatten_row(row, shapes))[0]

    batched = jax.jit(jax.vmap(single))

    def run(matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(batched(jnp.asarray(matrix, dtype=float)), dtype=float)
        return np.where(np.isfinite(values), values, -np.inf)

    return run


def _bridge_recursion(
    logp_post: np.ndarray,
    logq_post: np.ndarray,
    logp_prop: np.ndarray,
    logq_prop: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
) -> tuple[float, int, bool]:
    """Meng-Wong optimal-bridge fixed point for ``log Z`` (log space, ``logsumexp``-stable)."""
    n_post = logp_post.size
    n_prop = logp_prop.size
    log_s1 = np.log(n_post / (n_post + n_prop))
    log_s2 = np.log(n_prop / (n_post + n_prop))
    log_z = _logmeanexp(logp_prop - logq_prop)
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        log_den_prop = np.logaddexp(log_s1 + logp_prop, log_s2 + log_z + logq_prop)
        log_den_post = np.logaddexp(log_s1 + logp_post, log_s2 + log_z + logq_post)
        updated = _logmeanexp(logp_prop - log_den_prop) - _logmeanexp(logq_post - log_den_post)
        if np.isfinite(updated) and abs(updated - log_z) < tolerance:
            log_z = updated
            converged = True
            break
        log_z = updated
    return float(log_z), int(iteration), bool(converged)


def bridge_diagnostics(
    w_forward: np.ndarray, w_reverse: np.ndarray, delta_f: float
) -> tuple[float, float]:
    """Bennett overlap ``O in [0, 1]`` and the closed-form asymptotic SE of ``delta_f``.

    This is the **unified** diagnostic for any bridge between two states 0 and 1, whether state 0 is a
    fitted multivariate-normal reference (per-model bridge) or another model's posterior (nested
    bridge). ``w_forward`` are forward works ``u1 - u0`` for state-0 samples; ``w_reverse`` are reverse
    works ``u0 - u1`` for state-1 samples; ``delta_f = f1 - f0 = -ln(Z1/Z0)`` is the BAR/optimal-bridge
    estimate. Returns ``(overlap, se)`` where ``se`` is the standard error of ``delta_f`` (equivalently
    of ``log Z`` or ``ln BF``, up to sign).

    The Fermi summands at the solution, ``a = sigmoid(delta_f - M - w_forward)`` and
    ``b = sigmoid(M - delta_f - w_reverse)`` with ``M = ln(n_0 / n_1)``, give both the harmonic-mean
    overlap integral ``O = <a> + <b>`` (Bennett 1976; ``O -> 1`` identical, ``-> 0`` disjoint) and the
    BAR asymptotic variance (Shirts et al. 2003).
    """
    w_f = np.asarray(w_forward, dtype=float)
    w_r = np.asarray(w_reverse, dtype=float)
    w_f = w_f[np.isfinite(w_f)]
    w_r = w_r[np.isfinite(w_r)]
    n_f = w_f.size
    n_r = w_r.size
    if n_f == 0 or n_r == 0:
        return 0.0, float("inf")
    log_ratio = np.log(n_f / n_r)
    a = expit(delta_f - log_ratio - w_f)  # state-0 (forward) Fermi summand
    b = expit(log_ratio - delta_f - w_r)  # state-1 (reverse) Fermi summand
    a_mean = float(a.mean())
    b_mean = float(b.mean())
    overlap = float(np.clip(a_mean + b_mean, 0.0, 1.0))
    if a_mean <= 0.0 or b_mean <= 0.0:
        return overlap, float("inf")
    variance = (
        (float((a**2).mean()) / a_mean**2) / n_f
        + (float((b**2).mean()) / b_mean**2) / n_r
        - (n_f + n_r) / (n_f * n_r)
    )
    if variance > 0.0:
        se = float(np.sqrt(variance))
    elif variance == 0.0:
        se = 0.0  # exact overlap (degenerate): zero asymptotic variance
    else:
        se = float("inf")  # negative -> formula has broken down (no usable overlap)
    return overlap, se


def bridge_sampling(
    posterior_samples: dict[str, np.ndarray],
    model,
    *,
    parameter_names: tuple[str, ...] | None = None,
    n_proposal_samples: int | None = None,
    rng_seed: int = 20260610,
    max_iterations: int = 1000,
    tolerance: float = 1.0e-6,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> BridgeSamplingResult:
    """Estimate ``log Z`` (log marginal likelihood) by normal-proposal bridge sampling.

    ``model`` is a no-argument NumPyro model closure (e.g. from
    :func:`bayesian_binding.inference.build_numpyro_model`) built for the dataset that produced
    ``posterior_samples``; its unnormalized log posterior is evaluated batched (jitted ``vmap``). The
    result carries a closed-form asymptotic SE and the Bennett overlap (see
    :func:`bridge_diagnostics`); ``reliable`` is False when the overlap is below ``overlap_threshold``.
    """
    if parameter_names is None:
        parameter_names = tuple(
            name
            for name, value in posterior_samples.items()
            if name != "q_model" and np.asarray(value).ndim >= 1
        )
    matrix, shapes = _flatten_samples(posterior_samples, parameter_names)
    n_posterior, _ = matrix.shape
    n_proposal = int(n_proposal_samples or n_posterior)

    mean = matrix.mean(axis=0)
    covariance = _regularized_covariance(matrix)
    rng = np.random.default_rng(rng_seed)
    proposal_matrix = rng.multivariate_normal(mean, covariance, size=n_proposal)
    proposal = multivariate_normal(mean=mean, cov=covariance, allow_singular=False)

    density = _batched_log_density(model, shapes)
    logp_posterior = density(matrix)
    logq_posterior = proposal.logpdf(matrix)
    logp_proposal = density(proposal_matrix)
    logq_proposal = proposal.logpdf(proposal_matrix)

    finite = np.isfinite(logp_posterior)
    if not np.all(finite):
        matrix = matrix[finite]
        logp_posterior = logp_posterior[finite]
        logq_posterior = logq_posterior[finite]
        n_posterior = matrix.shape[0]
    if n_posterior == 0:
        raise ValueError("No finite posterior samples available for bridge sampling.")

    log_z, n_iterations, converged = _bridge_recursion(
        logp_posterior, logq_posterior, logp_proposal, logq_proposal,
        max_iterations=max_iterations, tolerance=tolerance,
    )

    # Unified diagnostics: treat state 0 = proposal, state 1 = posterior (Z0 = 1, Z1 = Z).
    w_forward = logq_proposal - logp_proposal  # u1 - u0 at proposal samples
    w_reverse = logp_posterior - logq_posterior  # u0 - u1 at posterior samples
    overlap, se = bridge_diagnostics(w_forward, w_reverse, delta_f=-log_z)
    reliable = bool(overlap >= overlap_threshold and np.isfinite(se))
    warning = None
    if not reliable:
        warning = (
            f"bridge overlap {overlap:.3f} < {overlap_threshold:.2f}: the Gaussian reference fits the "
            "posterior poorly. Bridge in the unconstrained space, or use a heavier-tailed/warped "
            "reference (see docs/BAYES_FACTORS.md)."
        )

    return BridgeSamplingResult(
        log_marginal_likelihood=float(log_z),
        log_marginal_likelihood_se=float(se),
        overlap=float(overlap),
        reliable=reliable,
        n_posterior_samples=int(n_posterior),
        n_proposal_samples=int(n_proposal),
        parameter_names=parameter_names,
        n_iterations=int(n_iterations),
        converged=bool(converged),
        warning=warning,
    )


def make_numpyro_log_density(model) -> Callable[[dict[str, jnp.ndarray]], float]:
    """Create a per-sample unnormalized log-density function for a no-argument NumPyro model.

    Provided for ad-hoc/manual use; :func:`bridge_sampling` evaluates the model batched internally.
    """

    def evaluate(params: dict[str, jnp.ndarray]) -> float:
        value, _ = log_density(model, (), {}, params)
        return float(value)

    return evaluate


def log_bayes_factor(result_a: BridgeSamplingResult, result_b: BridgeSamplingResult) -> float:
    """Return ``log BF`` (natural log) in favor of model A over model B."""
    return result_a.log_marginal_likelihood - result_b.log_marginal_likelihood


def bayes_factor_bridge(
    samples_a: dict[str, np.ndarray],
    model_a,
    samples_b: dict[str, np.ndarray],
    model_b,
    *,
    rng_seed: int = 20260610,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    **bridge_kwargs: Any,
) -> dict[str, Any]:
    """Per-model bridge Bayes factor for a model pair, with asymptotic SE and overlap flags.

    Returns a dict with ``log10_bf`` (A over B) and its asymptotic ``se`` (logZ SEs added in
    quadrature), the per-model ``logZ_a``/``logZ_b``, ``converged``, ``reliable`` (both overlaps pass),
    and any ``warnings``. This is the recommended estimator (see ``docs/BAYES_FACTORS.md``).
    """
    result_a = bridge_sampling(
        samples_a, model_a, rng_seed=rng_seed, overlap_threshold=overlap_threshold, **bridge_kwargs
    )
    result_b = bridge_sampling(
        samples_b, model_b, rng_seed=rng_seed + 1, overlap_threshold=overlap_threshold, **bridge_kwargs
    )
    ln10 = np.log(10.0)
    ln_bf = result_a.log_marginal_likelihood - result_b.log_marginal_likelihood
    se = float(
        np.sqrt(result_a.log_marginal_likelihood_se**2 + result_b.log_marginal_likelihood_se**2)
    )
    warnings = [w for w in (result_a.warning, result_b.warning) if w]
    return {
        "log10_bf": float(ln_bf / ln10),
        "se": float(se / ln10),
        "ln_bayes_factor": float(ln_bf),
        "logZ_a": float(result_a.log_marginal_likelihood),
        "logZ_b": float(result_b.log_marginal_likelihood),
        "overlap_a": float(result_a.overlap),
        "overlap_b": float(result_b.overlap),
        "converged": bool(result_a.converged and result_b.converged),
        "reliable": bool(result_a.reliable and result_b.reliable),
        "warnings": warnings,
    }
