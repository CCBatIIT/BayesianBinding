"""MAP optimization helpers for initializing MCMC."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from numpyro.infer.util import log_density as numpyro_log_density

from bayesian_binding.inference import (
    DEFAULT_BOUNDS,
    GLOBAL_THERMODYNAMIC_PARAMETERS,
    PriorBounds,
    _heat_offset_bounds,
    _log_sigma_bounds,
    _MIXTURE_MODEL_NAMES,
    _resolve_bounds,
    _thermo_to_heat_kwargs,
    build_numpyro_model,
)
from bayesian_binding.data import ITCExperiment
from bayesian_binding.models import MODEL_REGISTRY


@dataclass(frozen=True)
class MAPResult:
    params: dict[str, jnp.ndarray]
    log_posterior: float
    success: bool
    message: str
    n_iterations: int


def sample_site_names(
    model_name: str,
    fixed: Mapping[str, float] | None = None,
    *,
    uniform_cell_concentration: bool = False,
    uniform_syringe_concentration: bool = False,
) -> tuple[str, ...]:
    """Return latent sample-site names for a model after applying fixed values."""
    fixed = dict(fixed or {})
    cell_name = "cell_concentration_molar" if uniform_cell_concentration else "log_cell_concentration"
    syringe_name = (
        "syringe_concentration_molar" if uniform_syringe_concentration else "log_syringe_concentration"
    )
    names = [cell_name, syringe_name, "heat_offset", "log_sigma"]
    if model_name in ("two_component", "cooperative_equivalent_sites"):
        names.extend(["delta_g", "delta_h"])
    elif model_name == "cooperative":
        names.extend(["delta_g", "delta_delta_g", "delta_h_first", "delta_h_second"])
    elif model_name == "cooperative_equal_affinity":
        names.extend(["delta_g", "delta_h_first", "delta_h_second"])
    elif model_name == "dimerization_cooperative":
        names.extend(
            [
                "delta_g_dimer",
                "delta_g_binding",
                "delta_delta_g_binding",
                "delta_h_dimer",
                "delta_h_first",
                "delta_h_second",
            ]
        )
    elif model_name == "dimerization_monomer_cooperative":
        names.extend(
            [
                "delta_g_dimer",
                "delta_g_binding",
                "delta_delta_g_binding",
                "delta_delta_g_monomer",
                "delta_h_dimer",
                "delta_h_first",
                "delta_h_second",
                "delta_h_monomer",
            ]
        )
    elif model_name in ("racemic_mixture", "enantiomeric_mixture"):
        names.extend(["delta_g1", "delta_delta_g", "delta_h1", "delta_h2"])
        if model_name == "enantiomeric_mixture":
            names.append("rho")
    else:
        raise KeyError(f"Unknown model {model_name!r}")
    return tuple(name for name in names if name not in fixed)


def initial_params(
    experiment: ITCExperiment,
    *,
    model_name: str = "two_component",
    fixed: Mapping[str, float] | None = None,
    uniform_cell_concentration: bool = False,
    uniform_syringe_concentration: bool = False,
) -> dict[str, float]:
    """Return a conservative initial parameter dictionary."""
    fixed = dict(fixed or {})
    offset_low, offset_high = _heat_offset_bounds(np.asarray(experiment.heats_microcalorie, dtype=float))
    log_sigma_low, log_sigma_high = _log_sigma_bounds(np.asarray(experiment.heats_microcalorie, dtype=float))
    if uniform_cell_concentration:
        params = {"cell_concentration_molar": float(experiment.cell_concentration_molar)}
    else:
        params = {"log_cell_concentration": float(np.log(experiment.cell_concentration_molar))}
    if uniform_syringe_concentration:
        params["syringe_concentration_molar"] = float(experiment.syringe_concentration_molar)
    else:
        params["log_syringe_concentration"] = float(np.log(experiment.syringe_concentration_molar))
    params["heat_offset"] = float(np.clip(np.median(experiment.heats_microcalorie[-4:]), offset_low, offset_high))
    params["log_sigma"] = float(
        np.clip(np.log(max(np.std(experiment.heats_microcalorie[-4:]), 1.0e-3)), log_sigma_low, log_sigma_high)
    )
    if model_name in ("two_component", "cooperative_equivalent_sites"):
        params.update({"delta_g": -8.0, "delta_h": -2.5})
    elif model_name in ("racemic_mixture", "enantiomeric_mixture"):
        params.update({"delta_g1": -10.0, "delta_delta_g": 2.0, "delta_h1": -7.0, "delta_h2": -2.5})
        if model_name == "enantiomeric_mixture":
            params["rho"] = 0.5
    elif model_name == "cooperative":
        params.update(
            {
                "delta_g": -8.0,
                "delta_delta_g": 0.0,
                "delta_h_first": -2.5,
                "delta_h_second": -2.5,
            }
        )
    elif model_name == "cooperative_equal_affinity":
        params.update({"delta_g": -8.0, "delta_h_first": -2.5, "delta_h_second": -2.5})
    elif model_name == "dimerization_cooperative":
        params.update(
            {
                "delta_g_dimer": -7.0,
                "delta_g_binding": -7.0,
                "delta_delta_g_binding": 0.0,
                "delta_h_dimer": -2.0,
                "delta_h_first": -5.0,
                "delta_h_second": -5.0,
            }
        )
    elif model_name == "dimerization_monomer_cooperative":
        params.update(
            {
                "delta_g_dimer": -8.0,
                "delta_g_binding": -7.0,
                "delta_delta_g_binding": 0.0,
                "delta_delta_g_monomer": 2.0,
                "delta_h_dimer": -2.0,
                "delta_h_first": -5.0,
                "delta_h_second": -5.0,
                "delta_h_monomer": -5.0,
            }
        )
    else:
        raise KeyError(f"Unknown model {model_name!r}")
    for name in fixed:
        params.pop(name, None)
    return params


def parameter_bounds(
    experiment: ITCExperiment,
    *,
    model_name: str,
    fixed: Mapping[str, float] | None = None,
    concentration_relative_uncertainty: float = 0.10,
    bounds: Mapping[str, PriorBounds] | None = None,
    uniform_cell_concentration: bool = False,
    uniform_syringe_concentration: bool = False,
    concentration_range_factor: float = 10.0,
) -> dict[str, tuple[float | None, float | None]]:
    """Return optimizer bounds for active sample sites."""
    fixed = dict(fixed or {})
    all_bounds = dict(DEFAULT_BOUNDS)
    all_bounds.update(bounds or {})
    log_conc_sigma = float(np.sqrt(np.log1p(concentration_relative_uncertainty**2)))
    offset_low, offset_high = _heat_offset_bounds(np.asarray(experiment.heats_microcalorie, dtype=float))
    log_sigma_low, log_sigma_high = _log_sigma_bounds(np.asarray(experiment.heats_microcalorie, dtype=float))
    if uniform_cell_concentration:
        cell_bounds = {
            "cell_concentration_molar": (
                experiment.cell_concentration_molar / concentration_range_factor,
                experiment.cell_concentration_molar * concentration_range_factor,
            )
        }
    else:
        cell_bounds = {
            "log_cell_concentration": (
                np.log(experiment.cell_concentration_molar) - 8.0 * log_conc_sigma,
                np.log(experiment.cell_concentration_molar) + 8.0 * log_conc_sigma,
            )
        }
    if uniform_syringe_concentration:
        syringe_bounds = {
            "syringe_concentration_molar": (
                experiment.syringe_concentration_molar / concentration_range_factor,
                experiment.syringe_concentration_molar * concentration_range_factor,
            )
        }
    else:
        syringe_bounds = {
            "log_syringe_concentration": (
                np.log(experiment.syringe_concentration_molar) - 8.0 * log_conc_sigma,
                np.log(experiment.syringe_concentration_molar) + 8.0 * log_conc_sigma,
            )
        }
    result: dict[str, tuple[float | None, float | None]] = {
        **cell_bounds,
        **syringe_bounds,
        "heat_offset": (offset_low, offset_high),
        "log_sigma": (log_sigma_low, log_sigma_high),
    }
    if model_name in ("two_component", "cooperative_equivalent_sites"):
        for name in ["delta_g", "delta_h"]:
            b = all_bounds[name]
            result[name] = (b.low, b.high)
    elif model_name in ("racemic_mixture", "enantiomeric_mixture"):
        result["delta_g1"] = (-40.0, 40.0)
        result["delta_delta_g"] = (0.0, 40.0)
        result["delta_h1"] = (-100.0, 100.0)
        result["delta_h2"] = (-100.0, 100.0)
        if model_name == "enantiomeric_mixture":
            result["rho"] = (0.0, 1.0)
    elif model_name == "cooperative":
        for name in ["delta_g", "delta_delta_g", "delta_h_first", "delta_h_second"]:
            b = all_bounds[name]
            result[name] = (b.low, b.high)
    elif model_name == "cooperative_equal_affinity":
        for name in ["delta_g", "delta_h_first", "delta_h_second"]:
            b = all_bounds[name]
            result[name] = (b.low, b.high)
    elif model_name in ("dimerization_cooperative", "dimerization_monomer_cooperative"):
        for name in GLOBAL_THERMODYNAMIC_PARAMETERS[model_name]:
            b = all_bounds[name]
            result[name] = (b.low, b.high)
    else:
        raise KeyError(f"Unknown model {model_name!r}")
    for name in fixed:
        result.pop(name, None)
    return result


def fit_map(
    experiment: ITCExperiment,
    *,
    model_name: str = "two_component",
    fixed: Mapping[str, float] | None = None,
    concentration_relative_uncertainty: float = 0.10,
    bounds: Mapping[str, PriorBounds] | None = None,
    uniform_cell_concentration: bool = False,
    uniform_syringe_concentration: bool = False,
    concentration_range_factor: float = 10.0,
    maxiter: int = 2000,
) -> MAPResult:
    """Optimize the NumPyro log posterior and return active initial values."""
    fixed = dict(fixed or {})
    concentration_kwargs = dict(
        uniform_cell_concentration=uniform_cell_concentration,
        uniform_syringe_concentration=uniform_syringe_concentration,
    )
    model = build_numpyro_model(
        experiment,
        model_name=model_name,
        fixed=fixed,
        concentration_relative_uncertainty=concentration_relative_uncertainty,
        bounds=bounds,
        concentration_range_factor=concentration_range_factor,
        **concentration_kwargs,
    )
    names = sample_site_names(model_name, fixed, **concentration_kwargs)
    starts = [initial_params(experiment, model_name=model_name, fixed=fixed, **concentration_kwargs)]
    if model_name in {"two_component", "cooperative"}:
        for delta_g in [-6.0, -8.0, -10.0]:
            start = initial_params(experiment, model_name=model_name, fixed=fixed, **concentration_kwargs)
            if "delta_g" in start:
                start["delta_g"] = delta_g
            if "delta_h" in start:
                start["delta_h"] = -2.5
            starts.append(start)
    elif model_name in {"racemic_mixture", "enantiomeric_mixture"}:
        # Competitive-binding posteriors have a spurious "no binding" basin
        # (delta_g1 -> 0, sigma large); a small grid over the binding free
        # energies finds the real mode reliably.
        for delta_g1 in [-13.0, -11.0, -9.0, -7.0]:
            for delta_delta_g in [0.5, 2.0, 4.0]:
                start = initial_params(experiment, model_name=model_name, fixed=fixed, **concentration_kwargs)
                if "delta_g1" in start:
                    start["delta_g1"] = delta_g1
                if "delta_delta_g" in start:
                    start["delta_delta_g"] = delta_delta_g
                starts.append(start)
    bounds_by_name = parameter_bounds(
        experiment,
        model_name=model_name,
        fixed=fixed,
        concentration_relative_uncertainty=concentration_relative_uncertainty,
        bounds=bounds,
        concentration_range_factor=concentration_range_factor,
        **concentration_kwargs,
    )
    opt_bounds = [bounds_by_name[name] for name in names]

    def pack(params: Mapping[str, float]) -> np.ndarray:
        return np.asarray([float(params[name]) for name in names], dtype=float)

    def unpack(theta: np.ndarray) -> dict[str, jnp.ndarray]:
        return {name: jnp.asarray(value) for name, value in zip(names, theta)}

    # Exact JAX gradients (jitted) make MAP optimization fast; finite differences over the
    # un-jitted NumPyro density were the dominant cost. The constrained sample values are
    # optimized directly, with L-BFGS-B keeping them inside the prior support via opt_bounds.
    def negative_log_density(theta):
        params = {name: theta[index] for index, name in enumerate(names)}
        value, _ = numpyro_log_density(model, (), {}, params)
        return -value

    value_and_grad = jax.jit(jax.value_and_grad(negative_log_density))

    def objective(theta: np.ndarray):
        value, grad = value_and_grad(jnp.asarray(theta))
        value = float(value)
        if not np.isfinite(value):
            return 1.0e100, np.zeros_like(theta)
        grad = np.asarray(grad, dtype=float)
        return value, np.where(np.isfinite(grad), grad, 0.0)

    best = None
    for start in starts:
        theta0 = pack(start)
        result = minimize(
            objective,
            theta0,
            jac=True,
            method="L-BFGS-B",
            bounds=opt_bounds,
            options={"maxiter": maxiter, "ftol": 1.0e-9, "gtol": 1.0e-6},
        )
        if best is None or result.fun < best.fun:
            best = result
    assert best is not None
    params = unpack(np.asarray(best.x, dtype=float))
    return MAPResult(
        params=params,
        log_posterior=float(-best.fun),
        success=bool(best.success),
        message=str(best.message),
        n_iterations=int(best.nit),
    )


def initial_global_params(
    experiments: Sequence[ITCExperiment],
    *,
    model_name: str = "cooperative",
    fixed: Mapping[str, float] | None = None,
    fit_concentration_scale: bool = True,
    concentration_scale_prior_mean: float = 1.0,
) -> dict[str, float]:
    """Return a conservative initial parameter dictionary for a global fit."""
    fixed = dict(fixed or {})
    if model_name in ("two_component", "cooperative_equivalent_sites"):
        params: dict[str, float] = {"delta_g": -8.0, "delta_h": -5.0}
    elif model_name == "cooperative":
        params = {
            "delta_g": -7.0,
            "delta_delta_g": -1.0,
            "delta_h_first": -9.0,
            "delta_h_second": -5.0,
        }
    elif model_name == "cooperative_equal_affinity":
        params = {"delta_g": -7.0, "delta_h_first": -9.0, "delta_h_second": -5.0}
    elif model_name == "dimerization_cooperative":
        params = {
            "delta_g_dimer": -7.0,
            "delta_g_binding": -7.0,
            "delta_delta_g_binding": 0.0,
            "delta_h_dimer": -2.0,
            "delta_h_first": -5.0,
            "delta_h_second": -5.0,
        }
    elif model_name == "dimerization_monomer_cooperative":
        params = {
            "delta_g_dimer": -8.0,
            "delta_g_binding": -7.0,
            "delta_delta_g_binding": 0.0,
            "delta_delta_g_monomer": 2.0,
            "delta_h_dimer": -2.0,
            "delta_h_first": -5.0,
            "delta_h_second": -5.0,
            "delta_h_monomer": -5.0,
        }
    elif model_name in _MIXTURE_MODEL_NAMES:
        params = {"delta_g1": -10.0, "delta_delta_g": 2.0, "delta_h1": -7.0, "delta_h2": -2.5}
        if model_name == "enantiomeric_mixture":
            params["rho"] = 0.45
    else:
        raise KeyError(f"Global fitting is not supported for model {model_name!r}")
    scale = float(fixed.get("cell_concentration_scale", concentration_scale_prior_mean))
    if fit_concentration_scale and "cell_concentration_scale" not in fixed:
        params["cell_concentration_scale"] = scale
    # Mixture models need the shared composition rho when evaluating expected_heats below.
    extra = {}
    if model_name in _MIXTURE_MODEL_NAMES:
        extra["rho"] = float(fixed.get("rho", params.get("rho", 0.5)))
    # Seed each experiment's offset and noise scale from the residuals at the
    # initial thermodynamic guess. Seeding log_sigma from the tail standard
    # deviation alone collapses it toward zero for saturated curves, which makes
    # the residual/sigma term explode and drives the optimizer into a spurious
    # noise-dominated mode; the residual RMS keeps every scale commensurate with
    # the heats.
    model = MODEL_REGISTRY[model_name]
    thermodynamics = {name: fixed.get(name, params[name]) for name in GLOBAL_THERMODYNAMIC_PARAMETERS[model_name]}
    for index, experiment in enumerate(experiments):
        heats = np.asarray(experiment.heats_microcalorie, dtype=float)
        offset_low, offset_high = _heat_offset_bounds(heats)
        log_sigma_low, log_sigma_high = _log_sigma_bounds(heats)
        predicted = np.asarray(
            model.expected_heats(
                jnp.asarray(experiment.injection_volumes_liter),
                cell_volume_liter=float(experiment.cell_volume_liter),
                cell_concentration_molar=float(experiment.cell_concentration_molar) * scale,
                syringe_concentration_molar=float(experiment.syringe_concentration_molar),
                temperature_k=float(experiment.temperature_k),
                heat_offset=0.0,
                **_thermo_to_heat_kwargs(model_name, thermodynamics),
                **extra,
            ),
            dtype=float,
        )
        offset = float(np.clip(np.median(heats - predicted), offset_low, offset_high))
        residual_rms = float(np.sqrt(np.mean((heats - predicted - offset) ** 2)))
        params[f"heat_offset_{index}"] = offset
        params[f"log_sigma_{index}"] = float(
            np.clip(np.log(max(residual_rms, 0.05)), log_sigma_low, log_sigma_high)
        )
    for name in fixed:
        params.pop(name, None)
    return params


_GLOBAL_SIGMA_FLOOR_MICROCALORIE = 0.05


def fit_map_global(
    experiments: Sequence[ITCExperiment],
    *,
    model_name: str = "cooperative",
    fixed: Mapping[str, float] | None = None,
    bounds: Mapping[str, PriorBounds] | None = None,
    fit_concentration_scale: bool = True,
    concentration_scale_prior_mean: float = 1.0,
    concentration_scale_relative_sd: float = 0.25,
    maxiter: int = 3000,
) -> MAPResult:
    """Optimize the global multi-experiment log posterior with multistart L-BFGS-B.

    Each experiment's heat offset and noise scale are concentrated out analytically
    (offset = mean residual, sigma = root-mean-square residual), so the optimizer
    only navigates the shared thermodynamics and the concentration scaling factor.
    This is both the standard global-ITC treatment of per-curve nuisance scales and
    far better conditioned than the full joint optimum: leaving the per-experiment
    sigmas free lets the likelihood gradient flatten near poor starting points and
    strands gradient-based optimizers at spurious local minima. The returned params
    include the profiled offsets and ``log_sigma`` values so MCMC can initialize from
    them. Exact JAX gradients drive the optimization.
    """
    if model_name not in GLOBAL_THERMODYNAMIC_PARAMETERS:
        raise KeyError(f"Global fitting is not supported for model {model_name!r}")
    fixed = dict(fixed or {})
    model = MODEL_REGISTRY[model_name]
    all_bounds = _resolve_bounds(model_name, bounds)
    thermodynamic_names = GLOBAL_THERMODYNAMIC_PARAMETERS[model_name]

    injection_volumes = [jnp.asarray(e.injection_volumes_liter) for e in experiments]
    observed = [jnp.asarray(e.heats_microcalorie) for e in experiments]
    cell_concentrations = [float(e.cell_concentration_molar) for e in experiments]
    syringe_concentrations = [float(e.syringe_concentration_molar) for e in experiments]
    cell_volumes = [float(e.cell_volume_liter) for e in experiments]
    temperatures = [float(e.temperature_k) for e in experiments]
    offset_bounds = [_heat_offset_bounds(np.asarray(e.heats_microcalorie, dtype=float)) for e in experiments]
    log_sigma_bounds = [_log_sigma_bounds(np.asarray(e.heats_microcalorie, dtype=float)) for e in experiments]

    scale_is_free = fit_concentration_scale and "cell_concentration_scale" not in fixed
    fixed_scale = float(fixed.get("cell_concentration_scale", concentration_scale_prior_mean))
    free_thermodynamic_names = [name for name in thermodynamic_names if name not in fixed]
    # The enantiomeric mixture adds a free, shared composition rho (the racemic model fixes
    # it at 0.5). Optimizer variables, in order: free thermodynamics, then rho if free, then
    # log(scale) if free.
    rho_is_free = model_name == "enantiomeric_mixture" and "rho" not in fixed
    fixed_rho = float(fixed.get("rho", 0.5))
    rho_index = len(free_thermodynamic_names)
    log_scale_index = rho_index + (1 if rho_is_free else 0)
    log_scale_prior_mean = float(np.log(concentration_scale_prior_mean))

    def assemble(theta):
        thermodynamics = {}
        index = 0
        for name in thermodynamic_names:
            if name in fixed:
                thermodynamics[name] = jnp.asarray(fixed[name])
            else:
                thermodynamics[name] = theta[index]
                index += 1
        extra = {}
        if model_name in _MIXTURE_MODEL_NAMES:
            extra["rho"] = theta[rho_index] if rho_is_free else jnp.asarray(fixed_rho)
        if scale_is_free:
            scale = jnp.exp(theta[log_scale_index])
        else:
            scale = jnp.asarray(fixed_scale)
        return thermodynamics, extra, scale

    def predicted_unoffset(thermodynamics, extra, scale, j):
        return model.expected_heats(
            injection_volumes[j],
            cell_volume_liter=cell_volumes[j],
            cell_concentration_molar=cell_concentrations[j] * scale,
            syringe_concentration_molar=syringe_concentrations[j],
            temperature_k=temperatures[j],
            heat_offset=0.0,
            **_thermo_to_heat_kwargs(model_name, thermodynamics),
            **extra,
        )

    def profiled_negative_log_posterior(theta):
        thermodynamics, extra, scale = assemble(theta)
        total = 0.0
        for j in range(len(experiments)):
            residual = observed[j] - predicted_unoffset(thermodynamics, extra, scale, j)
            count = residual.shape[0]
            if f"heat_offset_{j}" in fixed:
                residual = residual - fixed[f"heat_offset_{j}"]
            else:
                residual = residual - jnp.mean(residual)
            if f"log_sigma_{j}" in fixed:
                variance = jnp.exp(2.0 * fixed[f"log_sigma_{j}"])
                total = total + jnp.sum(0.5 * residual**2 / variance) + 0.5 * count * jnp.log(
                    2.0 * jnp.pi * variance
                )
            else:
                variance = jnp.maximum(jnp.mean(residual**2), _GLOBAL_SIGMA_FLOOR_MICROCALORIE**2)
                total = total + 0.5 * count * (jnp.log(2.0 * jnp.pi * variance) + 1.0)
        if scale_is_free:
            log_scale = theta[log_scale_index]
            # Negative log of the LogNormal scale prior (the +log_scale term).
            total = total + 0.5 * ((log_scale - log_scale_prior_mean) / concentration_scale_relative_sd) ** 2
            total = total + log_scale
        return total

    value_and_grad = jax.jit(jax.value_and_grad(profiled_negative_log_posterior))

    def objective(theta: np.ndarray):
        value, grad = value_and_grad(jnp.asarray(theta))
        value = float(value)
        grad = np.asarray(grad, dtype=float)
        if not np.isfinite(value):
            return 1.0e100, np.zeros_like(theta)
        return value, np.where(np.isfinite(grad), grad, 0.0)

    optimizer_bounds = []
    for name in free_thermodynamic_names:
        b = all_bounds[name]
        optimizer_bounds.append((b.low, b.high))
    if rho_is_free:
        optimizer_bounds.append((0.0, 1.0))
    if scale_is_free:
        optimizer_bounds.append(
            (
                log_scale_prior_mean - 8.0 * concentration_scale_relative_sd,
                log_scale_prior_mean + 8.0 * concentration_scale_relative_sd,
            )
        )

    # Defaults and multistart grid over the (well-conditioned) shared parameters.
    # The first binding free energy correlates with the first enthalpy in these data
    # (Bonin et al.), so the grid spans both, plus two concentration-scale starts.
    defaults = initial_global_params(
        experiments,
        model_name=model_name,
        fixed=fixed,
        fit_concentration_scale=False,
        concentration_scale_prior_mean=concentration_scale_prior_mean,
    )
    scale_starts = (concentration_scale_prior_mean * 0.75, concentration_scale_prior_mean)
    starts: list[dict[str, float]] = [dict(defaults)]
    if model_name == "cooperative":
        for delta_g in (-5.5, -6.3, -7.0, -8.0):
            for delta_delta_g in (-2.5, -1.3, 0.0):
                for delta_h_first in (-7.0, -9.0, -12.0):
                    start = dict(defaults)
                    start["delta_g"] = delta_g
                    start["delta_delta_g"] = delta_delta_g
                    start["delta_h_first"] = delta_h_first
                    starts.append(start)
    elif model_name == "cooperative_equal_affinity":
        for delta_g in (-5.5, -6.3, -7.0, -8.0):
            for delta_h_first in (-7.0, -9.0, -12.0):
                start = dict(defaults)
                start["delta_g"] = delta_g
                start["delta_h_first"] = delta_h_first
                starts.append(start)
    elif model_name in ("two_component", "cooperative_equivalent_sites"):
        for delta_g in (-6.0, -8.0, -10.0):
            for delta_h in (-2.5, -8.0):
                start = dict(defaults)
                start["delta_g"] = delta_g
                start["delta_h"] = delta_h
                starts.append(start)
    elif model_name in _MIXTURE_MODEL_NAMES:
        # Competitive-binding posteriors have a spurious "no binding" basin; a small grid
        # over the binding free energies (mirroring the single-experiment fit_map) finds
        # the real mode reliably.
        for delta_g1 in (-13.0, -11.0, -9.0, -7.0):
            for delta_delta_g in (0.5, 2.0, 4.0):
                start = dict(defaults)
                start["delta_g1"] = delta_g1
                start["delta_delta_g"] = delta_delta_g
                starts.append(start)

    best = None
    for start in starts:
        base_theta = [float(start[name]) for name in free_thermodynamic_names]
        scale_inits = scale_starts if scale_is_free else (fixed_scale,)
        for scale_init in scale_inits:
            theta0 = list(base_theta)
            if rho_is_free:
                theta0.append(float(start.get("rho", 0.45)))
            if scale_is_free:
                theta0.append(float(np.log(scale_init)))
            result = minimize(
                objective,
                np.asarray(theta0, dtype=float),
                jac=True,
                method="L-BFGS-B",
                bounds=optimizer_bounds,
                options={"maxiter": maxiter, "ftol": 1.0e-12, "gtol": 1.0e-9, "maxls": 50},
            )
            if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
                best = result
    if best is None:
        raise RuntimeError("Global MAP optimization failed from every starting point.")

    # Reconstruct the full parameter dictionary, including the profiled nuisances.
    theta_best = jnp.asarray(best.x, dtype=float)
    thermodynamics, extra, scale = assemble(theta_best)
    params: dict[str, jnp.ndarray] = {}
    for name in free_thermodynamic_names:
        params[name] = jnp.asarray(thermodynamics[name])
    if rho_is_free:
        params["rho"] = jnp.asarray(extra["rho"])
    if scale_is_free:
        params["cell_concentration_scale"] = jnp.asarray(scale)
    for j in range(len(experiments)):
        residual = np.asarray(observed[j]) - np.asarray(predicted_unoffset(thermodynamics, extra, scale, j))
        if f"heat_offset_{j}" not in fixed:
            offset_low, offset_high = offset_bounds[j]
            params[f"heat_offset_{j}"] = jnp.asarray(float(np.clip(np.mean(residual), offset_low, offset_high)))
        if f"log_sigma_{j}" not in fixed:
            log_sigma_low, log_sigma_high = log_sigma_bounds[j]
            centered = residual - np.mean(residual)
            sigma = max(float(np.sqrt(np.mean(centered**2))), _GLOBAL_SIGMA_FLOOR_MICROCALORIE)
            params[f"log_sigma_{j}"] = jnp.asarray(float(np.clip(np.log(sigma), log_sigma_low, log_sigma_high)))
    return MAPResult(
        params=params,
        log_posterior=float(-best.fun),
        success=bool(best.success),
        message=str(best.message),
        n_iterations=int(best.nit),
    )

