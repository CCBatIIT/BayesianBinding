"""Thermodynamic ITC heat models implemented with JAX."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

from bayesian_binding import _jax_config as _jax_config

import jax
import jax.numpy as jnp

from bayesian_binding.constants import MICROCALORIES_PER_KCAL, beta_mol_per_kcal


def association_constant_from_delta_g(delta_g: float, temperature_k: float) -> jnp.ndarray:
    """Return association constant in reciprocal molar from standard free energy."""
    return jnp.exp(-beta_mol_per_kcal(temperature_k) * delta_g)


def dissociation_constant_from_delta_g(delta_g: float, temperature_k: float) -> jnp.ndarray:
    """Return dissociation constant in molar from standard free energy."""
    return jnp.exp(beta_mol_per_kcal(temperature_k) * delta_g)


@dataclass(frozen=True)
class TwoComponentBindingModel:
    """One ligand binding one receptor/macromolecule."""

    name: str = "two_component"

    @staticmethod
    def expected_heats(
        injection_volumes_liter,
        *,
        cell_volume_liter,
        cell_concentration_molar,
        syringe_concentration_molar,
        temperature_k,
        delta_g,
        delta_h,
        heat_offset,
    ):
        kd = dissociation_constant_from_delta_g(delta_g, temperature_k)
        volumes = jnp.asarray(injection_volumes_liter)

        def step(carry, injection_volume):
            previous_complex, dilution_cumulative = carry
            dilution = 1.0 - injection_volume / cell_volume_liter
            dilution_cumulative = dilution_cumulative * dilution
            total_receptor_mol = cell_volume_liter * cell_concentration_molar * dilution_cumulative
            total_ligand_mol = cell_volume_liter * syringe_concentration_molar * (1.0 - dilution_cumulative)
            total_r = total_receptor_mol / cell_volume_liter
            total_l = total_ligand_mol / cell_volume_liter
            root = jnp.sqrt(jnp.maximum((total_r + total_l + kd) ** 2 - 4.0 * total_r * total_l, 0.0))
            complex_conc = 0.5 * ((total_r + total_l + kd) - root)
            delta_complex_mol = cell_volume_liter * (complex_conc - dilution * previous_complex)
            heat = delta_h * delta_complex_mol * MICROCALORIES_PER_KCAL + heat_offset
            return (complex_conc, dilution_cumulative), heat

        (_complex, _dilution_cumulative), heats = jax.lax.scan(step, (0.0, 1.0), volumes)
        return heats

    @staticmethod
    def equilibrium_species(protein_molar, ligand_molar, temperature_k, *, delta_g):
        """Equilibrium concentrations of ``{free_ligand, free_protein, complex}`` for 1:1 binding.

        Vectorized over the ``(protein, ligand)`` totals. Exposes the binding model's species to
        other modalities (e.g. ``scattering.binding_model_concentrations`` builds a WAXS
        concentration model from it). Same convention as :func:`scattering.binding_1to1_concentrations`.
        """
        kd = dissociation_constant_from_delta_g(delta_g, temperature_k)
        protein = jnp.asarray(protein_molar)
        ligand = jnp.asarray(ligand_molar)
        total = protein + ligand + kd
        complex_conc = 0.5 * (total - jnp.sqrt(jnp.maximum(total**2 - 4.0 * protein * ligand, 0.0)))
        return {
            "free_ligand": ligand - complex_conc,
            "free_protein": protein - complex_conc,
            "complex": complex_conc,
        }


@dataclass(frozen=True)
class AdKADPSteadyStateModel:
    """AdK mixed with ADP, allowing ADP dismutation plus cooperative ADP binding.

    This is a concentration model for AdK WAXS/other enthalpy-blind modalities, not an ITC heat
    model. The standalone AdK+AMP and AdK+ATP series remain ordinary one-site/two-component systems;
    this model reuses their binding free energies as ``delta_g_amp`` and ``delta_g_atp`` for the ADP
    series. The ADP series adds its own cooperative two-ADP binding branch:

    ``2 ADP <-> AMP + ATP`` with ``Keq = [AMP_free] [ATP_free] / [ADP_free]^2``.

    ``AdK + AMP <-> AdK:AMP``, ``AdK + ATP <-> AdK:ATP``, and the mixed
    ``AdK + AMP + ATP <-> AdK:AMP:ATP`` state.

    ``AdK + ADP <-> AdK:ADP`` and ``AdK:ADP + ADP <-> AdK:ADP2``.

    The bound catalytic ratio ``[AdK:AMP:ATP] / [AdK:ADP2]`` is not an independent
    parameter. It follows by detailed balance from the free nucleotide equilibrium and the
    binding constants:
    ``K_bound = K_sol * B_AMP_ATP / B_ADP2``.

    Starting from ADP alone imposes equal total AMP-like and ATP-like products, including bound AMP/ATP
    nucleotide: ``[AMP_free] + [AdK:AMP] = [ATP_free] + [AdK:ATP]``.
    """

    name: str = "adk_adp_steady_state"

    @staticmethod
    def _species_from_free(
        total_protein,
        free_amp,
        free_adp,
        free_atp,
        ka_amp,
        ka_atp,
        ka_amp_atp,
        ka_adp_first,
        ka_adp_second,
    ):
        z = (
            1.0
            + ka_amp * free_amp
            + ka_atp * free_atp
            + ka_amp_atp * free_amp * free_atp
            + 2.0 * ka_adp_first * free_adp
            + ka_adp_first * ka_adp_second * free_adp * free_adp
        )
        free_protein = total_protein / z
        protein_amp = free_protein * ka_amp * free_amp
        protein_atp = free_protein * ka_atp * free_atp
        protein_amp_atp = free_protein * ka_amp_atp * free_amp * free_atp
        protein_adp = free_protein * 2.0 * ka_adp_first * free_adp
        protein_adp2 = free_protein * ka_adp_first * ka_adp_second * free_adp * free_adp
        return free_protein, protein_amp, protein_atp, protein_amp_atp, protein_adp, protein_adp2

    @staticmethod
    def _species(total_protein, total_adp, ka_amp, ka_atp, ka_amp_atp, ka_adp_first, ka_adp_second, keq):
        total_adp = jnp.maximum(total_adp, 0.0)
        solve_total_adp = jnp.maximum(total_adp, 1.0e-30)
        keq = jnp.maximum(keq, 1.0e-30)
        log_keq = jnp.log(keq)
        scale = jnp.maximum(solve_total_adp, 1.0e-12)
        product_ratio = jnp.sqrt(keq)
        free_amp0 = product_ratio * solve_total_adp / (1.0 + 2.0 * product_ratio)
        free_adp0 = solve_total_adp / (1.0 + 2.0 * product_ratio)
        free_atp0 = free_amp0
        y0 = jnp.log(jnp.asarray([free_amp0, free_adp0, free_atp0]) + 1.0e-30)

        def residual(log_free):
            free_amp, free_adp, free_atp = jnp.exp(log_free)
            free_protein, protein_amp, protein_atp, protein_amp_atp, protein_adp, protein_adp2 = (
                AdKADPSteadyStateModel._species_from_free(
                    total_protein,
                    free_amp,
                    free_adp,
                    free_atp,
                    ka_amp,
                    ka_atp,
                    ka_amp_atp,
                    ka_adp_first,
                    ka_adp_second,
                )
            )
            amp_total = free_amp + protein_amp + protein_amp_atp
            atp_total = free_atp + protein_atp + protein_amp_atp
            nucleotide_total = (
                amp_total + atp_total + free_adp + protein_adp + 2.0 * protein_adp2
            )
            return jnp.asarray(
                [
                    (amp_total - atp_total) / scale,
                    (nucleotide_total - solve_total_adp) / scale,
                    log_free[0] + log_free[2] - 2.0 * log_free[1] - log_keq,
                ]
            )

        def newton_step(_i, log_free):
            r = residual(log_free)
            jacobian = jax.jacfwd(residual)(log_free)
            step = jnp.linalg.solve(jacobian + 1.0e-10 * jnp.eye(3), r)
            step = jnp.clip(step, -2.0, 2.0)
            return log_free - step

        log_free = jax.lax.fori_loop(0, 40, newton_step, y0)
        free_amp, free_adp, free_atp = jnp.exp(log_free)
        free_protein, protein_amp, protein_atp, protein_amp_atp, protein_adp, protein_adp2 = (
            AdKADPSteadyStateModel._species_from_free(
                total_protein,
                free_amp,
                free_adp,
                free_atp,
                ka_amp,
                ka_atp,
                ka_amp_atp,
                ka_adp_first,
                ka_adp_second,
            )
        )
        active = total_adp > 0.0
        return (
            jnp.where(active, free_amp, 0.0),
            jnp.where(active, free_adp, 0.0),
            jnp.where(active, free_atp, 0.0),
            jnp.where(active, free_protein, total_protein),
            jnp.where(active, protein_amp, 0.0),
            jnp.where(active, protein_atp, 0.0),
            jnp.where(active, protein_amp_atp, 0.0),
            jnp.where(active, protein_adp, 0.0),
            jnp.where(active, protein_adp2, 0.0),
        )

    @staticmethod
    def equilibrium_species(
        protein_molar,
        adp_molar,
        temperature_k,
        *,
        delta_g_amp,
        delta_g_atp,
        delta_delta_g_amp_atp,
        delta_g_adp,
        delta_delta_g_adp,
        log_adp_dismutation_keq,
    ):
        """Equilibrium species for an AdK+ADP series.

        Returns ``{free_amp, free_adp, free_atp, free_protein, protein_amp, protein_atp,
        protein_amp_atp, protein_adp, protein_adp2}``, vectorized over ``protein_molar`` and
        ``adp_molar`` totals.
        """
        ka_amp = association_constant_from_delta_g(delta_g_amp, temperature_k)
        ka_atp = association_constant_from_delta_g(delta_g_atp, temperature_k)
        amp_atp_coupling = jnp.exp(-beta_mol_per_kcal(temperature_k) * delta_delta_g_amp_atp)
        ka_amp_atp = ka_amp * ka_atp * amp_atp_coupling
        ka_adp_first = association_constant_from_delta_g(delta_g_adp, temperature_k)
        ka_adp_second = association_constant_from_delta_g(delta_g_adp + delta_delta_g_adp, temperature_k)
        keq = jnp.exp(log_adp_dismutation_keq)
        protein = jnp.atleast_1d(jnp.asarray(protein_molar, dtype=float))
        adp = jnp.atleast_1d(jnp.asarray(adp_molar, dtype=float))
        (
            free_amp,
            free_adp,
            free_atp,
            free_protein,
            protein_amp,
            protein_atp,
            protein_amp_atp,
            protein_adp,
            protein_adp2,
        ) = jax.vmap(
            lambda p, lig: AdKADPSteadyStateModel._species(
                p, lig, ka_amp, ka_atp, ka_amp_atp, ka_adp_first, ka_adp_second, keq
            )
        )(protein, adp)
        return {
            "free_amp": free_amp,
            "free_adp": free_adp,
            "free_atp": free_atp,
            "free_protein": free_protein,
            "protein_amp": protein_amp,
            "protein_atp": protein_atp,
            "protein_amp_atp": protein_amp_atp,
            "protein_adp": protein_adp,
            "protein_adp2": protein_adp2,
        }


@dataclass(frozen=True)
class CooperativeBindingModel:
    """Sequential ``n_sites``-site cooperative model (default two sites).

    ``n_sites`` identical, interacting ligand sites on one receptor. The binding polynomial uses
    **microscopic** stepwise association constants with explicit binomial statistical factors::

        z = sum_{j=0}^{n} C(n, j) * (prod_{i=1..j} ka_i) * x^j,    ka_i = exp(-delta_g_i / RT),

    so the fraction of receptor with ``j`` ligands bound is ``C(n,j)(prod ka_i) x^j / z`` and zero
    free-energy cooperativity (all ``ka_i`` equal) factorizes to ``(1 + ka x)^n`` -- independent
    identical sites. ``delta_g_1 = delta_g`` is the first intrinsic step and the later steps are given
    as offsets, ``delta_g_i = delta_g + delta_delta_g_i``; each step carries its own enthalpy.

    **Parameter API (and backward compatibility).** For the default ``n_sites = 2`` the parameters are
    exactly the historical names and the math is byte-for-byte identical to the original two-site
    model: ``delta_g``, ``delta_delta_g`` (= ``delta_g_2 - delta_g_1``), ``delta_h_first``,
    ``delta_h_second``. For ``n_sites > 2`` the per-step parameters are the indexed names
    ``delta_delta_g_2 .. delta_delta_g_n`` and ``delta_h_1 .. delta_h_n`` (``delta_g`` is still step 1).
    The static methods infer the site count from whichever parameter names are present, so every
    existing call -- ``CooperativeBindingModel.expected_heats(...)`` /
    ``MODEL_REGISTRY['cooperative'].equilibrium_species(...)`` with the two-site names -- keeps working
    unchanged. ``equilibrium_species`` likewise returns the legacy ``singly_bound`` / ``doubly_bound``
    keys for two sites and ``bound_1 .. bound_n`` for more.

    Two no-cooperativity nulls are available as flags (:meth:`heat_kwargs` expands the reduced
    parameters to the full per-step set, and the model builders use it):

    - ``equivalent_sites`` -- all sites forced **fully equal** (every ``delta_delta_g_i = 0`` *and* all
      step enthalpies equal), fit with only ``delta_g`` and ``delta_h``.
    - ``equal_affinity`` -- only the **free-energy** cooperativity is removed (every
      ``delta_delta_g_i = 0``) while the step enthalpies stay free. For two sites the corresponding
      Bayes factor nests inside the full model with ``delta_delta_g`` as the single extra parameter.

    This mirrors the ``racemic`` flag on ``EnantiomericMixtureBindingModel``.

    - ``distinct_sites`` -- **n independent, non-identical sites** (default off). The binomial,
      identical-site partition function ``prod(1 + ka x)`` with statistical factors is replaced by the
      explicit product over distinct sites; the same parameter names are reused with reinterpreted
      meaning: ``delta_g`` = site-1 free energy and ``delta_delta_g`` / ``delta_delta_g_i`` = the
      site-i *offset* (ΔG_i − ΔG_1) rather than a cooperativity, and ``delta_h_*`` are per-site
      enthalpies. ``equilibrium_species`` then returns the **microstates** -- ``apo_protein`` and one
      ``bound_<sites>`` per occupied-site subset (e.g. ``bound_1``, ``bound_2``, ``bound_12`` for two
      sites) -- so a modality that can tell the sites apart (WAXS) resolves each distinct bound pattern;
      ITC heats sum over per-site occupancies and are unaffected. Sites are independent (no site-site
      cooperativity in this form). The identical-site path (``distinct_sites=False``) is byte-for-byte
      unchanged.
    """

    name: str = "cooperative"
    n_sites: int = 2
    equivalent_sites: bool = False
    equal_affinity: bool = False
    distinct_sites: bool = False

    def heat_kwargs(self, thermodynamics) -> dict:
        """Return the ``expected_heats`` thermodynamic kwargs from the sampled parameters.

        Identity for the full model; ``equivalent_sites`` maps ``{delta_g, delta_h}`` and
        ``equal_affinity`` maps ``{delta_g, delta_h_*}`` to the full per-step kwargs with every
        free-energy cooperativity zeroed. For two sites the output uses the legacy names
        (``delta_delta_g``, ``delta_h_first``, ``delta_h_second``); for more sites the indexed names.
        """
        if not (self.equivalent_sites or self.equal_affinity):
            return dict(thermodynamics)
        n = self.n_sites
        out: dict = {"delta_g": thermodynamics["delta_g"]}
        if n == 2:
            out["delta_delta_g"] = 0.0
        else:
            for i in range(2, n + 1):
                out[f"delta_delta_g_{i}"] = 0.0
        if self.equivalent_sites:
            delta_h = thermodynamics["delta_h"]
            step_enthalpies = [delta_h] * n
        else:  # equal_affinity: the step enthalpies stay free
            step_enthalpies = (
                [thermodynamics["delta_h_first"], thermodynamics["delta_h_second"]]
                if n == 2
                else [thermodynamics[f"delta_h_{i}"] for i in range(1, n + 1)]
            )
        if n == 2:
            out["delta_h_first"], out["delta_h_second"] = step_enthalpies
        else:
            for i, value in enumerate(step_enthalpies, start=1):
                out[f"delta_h_{i}"] = value
        return out

    @staticmethod
    def _stepwise_delta_g(thermodynamics) -> list:
        """Intrinsic stepwise binding free energies ``[delta_g_1, ..., delta_g_n]`` from the kwargs.

        Two-site legacy form ``{delta_g, delta_delta_g}`` -> ``[delta_g, delta_g + delta_delta_g]``;
        general form ``{delta_g, delta_delta_g_2, ...}`` -> ``[delta_g, delta_g + delta_delta_g_2, ...]``.
        """
        delta_g = thermodynamics["delta_g"]
        if "delta_delta_g" in thermodynamics:
            return [delta_g, delta_g + thermodynamics["delta_delta_g"]]
        offsets = [0.0]
        index = 2
        while f"delta_delta_g_{index}" in thermodynamics:
            offsets.append(thermodynamics[f"delta_delta_g_{index}"])
            index += 1
        return [delta_g + offset for offset in offsets]

    @staticmethod
    def _stepwise_delta_h(thermodynamics) -> list:
        """Stepwise binding enthalpies ``[delta_h_1, ..., delta_h_n]`` from the kwargs.

        Two-site legacy form uses ``delta_h_first`` / ``delta_h_second``; general form uses
        ``delta_h_1 .. delta_h_n``.
        """
        if "delta_h_first" in thermodynamics or "delta_h_second" in thermodynamics:
            return [thermodynamics["delta_h_first"], thermodynamics["delta_h_second"]]
        enthalpies = []
        index = 1
        while f"delta_h_{index}" in thermodynamics:
            enthalpies.append(thermodynamics[f"delta_h_{index}"])
            index += 1
        return enthalpies

    @staticmethod
    def _species(total_receptor, total_ligand, association_constants):
        """Equilibrium ``(free_ligand, apo, [bound_1, ..., bound_n])`` for ``n`` interacting sites.

        ``association_constants`` is the length-``n`` sequence of intrinsic stepwise constants ``ka_i``;
        the binding polynomial carries the binomial statistical factors ``C(n, j)``. Free ligand is
        found by bisection (monotone on ``[0, total_ligand]``) then one Newton step so the result keeps
        the exact implicit-function gradient -- the bisection bracket comparisons would block autodiff.
        """
        n = len(association_constants)
        # Polynomial coefficients a_j = C(n, j) * prod_{i<=j} ka_i (with a_0 = 1).
        coefficients = [jnp.asarray(1.0)]
        running = jnp.asarray(1.0)
        for j in range(1, n + 1):
            running = running * association_constants[j - 1]
            coefficients.append(comb(n, j) * running)

        def occupancy_terms(free_ligand):
            terms = [coefficients[j] * free_ligand ** j for j in range(1, n + 1)]
            partition = coefficients[0] + sum(terms)
            return terms, partition

        def mass_balance(free_ligand):
            terms, partition = occupancy_terms(free_ligand)
            bound_ligand = sum(j * terms[j - 1] for j in range(1, n + 1)) / partition
            return free_ligand + total_receptor * bound_ligand - total_ligand

        def body(_i, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            balance = mass_balance(mid)
            return jnp.where(balance > 0.0, jnp.asarray([lo, mid]), jnp.asarray([mid, hi]))

        bounds = jax.lax.fori_loop(0, 80, body, jnp.asarray([0.0, jnp.maximum(total_ligand, 0.0)]))
        free_ligand = jax.lax.stop_gradient(0.5 * (bounds[0] + bounds[1]))
        balance = mass_balance(free_ligand)
        balance_slope = jax.grad(mass_balance)(free_ligand)
        free_ligand = free_ligand - balance / balance_slope

        terms, partition = occupancy_terms(free_ligand)
        apo = total_receptor / partition
        bound = [total_receptor * terms[j - 1] / partition for j in range(1, n + 1)]
        return free_ligand, apo, bound

    @staticmethod
    def _distinct_microstate_sites(n: int) -> list:
        """0-based occupied-site tuples for every non-empty microstate, ordered by bitmask 1..2^n-1."""
        return [tuple(i for i in range(n) if mask & (1 << i)) for mask in range(1, 1 << n)]

    @staticmethod
    def _free_ligand_distinct(total_receptor, total_ligand, association_constants):
        """Free ligand for ``n`` INDEPENDENT distinct sites: ``x + R * sum_i theta_i = L`` with
        ``theta_i = ka_i x / (1 + ka_i x)``. Bisection (monotone on ``[0, L]``) + one Newton step, so the
        result keeps the exact implicit-function gradient (matching :meth:`_species`)."""
        ka = association_constants
        n = len(ka)

        def mass_balance(free_ligand):
            bound = sum(ka[i] * free_ligand / (1.0 + ka[i] * free_ligand) for i in range(n))
            return free_ligand + total_receptor * bound - total_ligand

        def body(_i, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            return jnp.where(mass_balance(mid) > 0.0, jnp.asarray([lo, mid]), jnp.asarray([mid, hi]))

        bounds = jax.lax.fori_loop(0, 80, body, jnp.asarray([0.0, jnp.maximum(total_ligand, 0.0)]))
        free_ligand = jax.lax.stop_gradient(0.5 * (bounds[0] + bounds[1]))
        return free_ligand - mass_balance(free_ligand) / jax.grad(mass_balance)(free_ligand)

    @staticmethod
    def _species_distinct(total_receptor, total_ligand, association_constants):
        """``(free_ligand, apo, [microstate populations])`` for ``n`` independent distinct sites.

        The microstate list is ordered by :meth:`_distinct_microstate_sites`; microstate ``S`` has
        receptor concentration ``R * prod_{i in S}(ka_i x) / z`` with ``z = prod_i (1 + ka_i x)``, and the
        populations + ``apo`` sum to ``R``.
        """
        ka = association_constants
        n = len(ka)
        x = CooperativeBindingModel._free_ligand_distinct(total_receptor, total_ligand, association_constants)
        partition = 1.0 + ka[0] * x
        for i in range(1, n):
            partition = partition * (1.0 + ka[i] * x)
        apo = total_receptor / partition
        microstates = []
        for mask in range(1, 1 << n):
            weight = total_receptor
            for i in range(n):
                if mask & (1 << i):
                    weight = weight * (ka[i] * x)
            microstates.append(weight / partition)
        return x, apo, microstates

    def expected_heats(
        self,
        injection_volumes_liter,
        *,
        cell_volume_liter,
        cell_concentration_molar,
        syringe_concentration_molar,
        temperature_k,
        heat_offset,
        **thermodynamics,
    ):
        """Per-injection heats. ``thermodynamics`` carries the stepwise free energies + enthalpies in
        either the two-site legacy names or the indexed ``n``-site names (see the class docstring)."""
        stepwise_delta_g = CooperativeBindingModel._stepwise_delta_g(thermodynamics)
        stepwise_delta_h = CooperativeBindingModel._stepwise_delta_h(thermodynamics)
        association_constants = [association_constant_from_delta_g(g, temperature_k) for g in stepwise_delta_g]
        n = len(association_constants)
        volumes = jnp.asarray(injection_volumes_liter)

        if self.distinct_sites:
            # independent distinct sites: total relative enthalpy = R * sum_i theta_i * dH_i, so the heat
            # tracks the change in per-site bound receptor concentration (no occupancy/binomial step).
            def step_distinct(carry, injection_volume):
                total_receptor, total_ligand = carry[0], carry[1]
                previous = carry[2:]  # per-site bound receptor concentration from the prior injection
                dilution = 1.0 - injection_volume / cell_volume_liter
                total_receptor = total_receptor * dilution
                total_ligand = total_ligand * dilution + syringe_concentration_molar * (injection_volume / cell_volume_liter)
                free = CooperativeBindingModel._free_ligand_distinct(total_receptor, total_ligand, association_constants)
                bound = [total_receptor * association_constants[i] * free / (1.0 + association_constants[i] * free) for i in range(n)]
                delta_mol = sum(stepwise_delta_h[i] * (bound[i] - dilution * previous[i]) for i in range(n))
                heat = cell_volume_liter * delta_mol * MICROCALORIES_PER_KCAL + heat_offset
                return (total_receptor, total_ligand, *bound), heat

            carry0 = (cell_concentration_molar, 0.0) + (0.0,) * n
            _carry, heats = jax.lax.scan(step_distinct, carry0, volumes)
            return heats

        def step(carry, injection_volume):
            total_receptor, total_ligand = carry[0], carry[1]
            previous = carry[2:]  # cumulative "at least k+1 bound" from the prior injection
            dilution = 1.0 - injection_volume / cell_volume_liter
            total_receptor = total_receptor * dilution
            total_ligand = total_ligand * dilution + syringe_concentration_molar * (injection_volume / cell_volume_liter)
            _free, _apo, bound = CooperativeBindingModel._species(total_receptor, total_ligand, association_constants)
            at_least = [sum(bound[k:]) for k in range(n)]  # receptor concentration with >= k+1 ligands
            delta_mol = sum(stepwise_delta_h[k] * (at_least[k] - dilution * previous[k]) for k in range(n))
            heat = cell_volume_liter * delta_mol * MICROCALORIES_PER_KCAL + heat_offset
            return (total_receptor, total_ligand, *at_least), heat

        carry0 = (cell_concentration_molar, 0.0) + (0.0,) * n
        _carry, heats = jax.lax.scan(step, carry0, volumes)
        return heats

    def equilibrium_species(self, protein_molar, ligand_molar, temperature_k, **thermodynamics):
        """Equilibrium species, vectorized over the ``(protein, ligand)`` totals.

        Identical sites: ``{free_ligand, apo_protein, ...}`` with bound states keyed ``singly_bound`` /
        ``doubly_bound`` for two sites (backward compatible) and ``bound_1 .. bound_n`` for more.
        ``distinct_sites``: the bound states are the **microstates** ``bound_<sites>`` (e.g. ``bound_1``,
        ``bound_2``, ``bound_12``). Exposes the model's species to other modalities (e.g. the WAXS MCR).
        """
        stepwise_delta_g = CooperativeBindingModel._stepwise_delta_g(thermodynamics)
        association_constants = [association_constant_from_delta_g(g, temperature_k) for g in stepwise_delta_g]
        n = len(association_constants)
        protein = jnp.atleast_1d(jnp.asarray(protein_molar, dtype=float))
        ligand = jnp.atleast_1d(jnp.asarray(ligand_molar, dtype=float))
        if self.distinct_sites:
            free, apo, microstates = jax.vmap(
                lambda p, lig: CooperativeBindingModel._species_distinct(p, lig, association_constants)
            )(protein, ligand)
            species = {"free_ligand": free, "apo_protein": apo}
            for sites, population in zip(CooperativeBindingModel._distinct_microstate_sites(n), microstates):
                species["bound_" + "".join(str(i + 1) for i in sites)] = population
            return species
        free, apo, bound = jax.vmap(
            lambda p, lig: CooperativeBindingModel._species(p, lig, association_constants)
        )(protein, ligand)
        species = {"free_ligand": free, "apo_protein": apo}
        if n == 2:
            species["singly_bound"], species["doubly_bound"] = bound[0], bound[1]
        else:
            for k in range(n):
                species[f"bound_{k + 1}"] = bound[k]
        return species

    def enthalpy_density(self, receptor_molar, ligand_molar, temperature_k, **thermodynamics):
        """Equilibrium enthalpy density (kcal/L) relative to apo, vectorized over the ``(receptor,
        ligand)`` totals.

        For a *preformed-dimer* receptor the totals are dimer concentrations. Exposes the per-condition
        density so a likelihood can predict heats for an externally supplied concentration schedule
        (e.g. the sequential TrpR phosphate titration).
        """
        stepwise_delta_g = CooperativeBindingModel._stepwise_delta_g(thermodynamics)
        stepwise_delta_h = CooperativeBindingModel._stepwise_delta_h(thermodynamics)
        association_constants = [association_constant_from_delta_g(g, temperature_k) for g in stepwise_delta_g]
        n = len(association_constants)
        receptor = jnp.atleast_1d(jnp.asarray(receptor_molar, dtype=float))
        ligand = jnp.atleast_1d(jnp.asarray(ligand_molar, dtype=float))

        if self.distinct_sites:
            def density_distinct(total_receptor, total_ligand):
                free = CooperativeBindingModel._free_ligand_distinct(total_receptor, total_ligand, association_constants)
                return sum(
                    stepwise_delta_h[i] * (total_receptor * association_constants[i] * free / (1.0 + association_constants[i] * free))
                    for i in range(n)
                )

            return jax.vmap(density_distinct)(receptor, ligand)

        def density(total_receptor, total_ligand):
            _free, _apo, bound = CooperativeBindingModel._species(total_receptor, total_ligand, association_constants)
            return sum(stepwise_delta_h[k] * sum(bound[k:]) for k in range(n))

        return jax.vmap(density)(receptor, ligand)


@dataclass(frozen=True)
class DimerizationCooperativeBindingModel:
    """Dimerizing receptor plus cooperative ligand binding to the dimer.

    Reaction convention:

    M + M <-> D
    D + L <-> DL
    DL + L <-> DL2

    The receptor total concentration is monomer-equivalent. `delta_g_dimer`
    defines the standard dimer association free energy. `delta_g_binding` is
    the first ligand-binding step on the dimer and `delta_delta_g_binding`
    shifts the second binding step.
    """

    name: str = "dimerization_cooperative"

    @staticmethod
    def _species(total_protein, total_ligand, k_dim, ka1, ka2):
        def dimer_species_for_ligand(free_ligand):
            z = 1.0 + ka1 * free_ligand + ka1 * ka2 * free_ligand * free_ligand
            a = 2.0 * k_dim * z
            free_monomer = jnp.where(
                a > 0.0,
                (-1.0 + jnp.sqrt(jnp.maximum(1.0 + 4.0 * a * total_protein, 1.0))) / (2.0 * a),
                total_protein,
            )
            dimer = k_dim * free_monomer * free_monomer
            singly = dimer * ka1 * free_ligand
            doubly = singly * ka2 * free_ligand
            return free_monomer, dimer, singly, doubly

        def ligand_balance(free_ligand):
            _m, _d, singly, doubly = dimer_species_for_ligand(free_ligand)
            return free_ligand + singly + 2.0 * doubly - total_ligand

        def body(_i, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            return jnp.where(ligand_balance(mid) > 0.0, jnp.asarray([lo, mid]), jnp.asarray([mid, hi]))

        # Bisection is robust but its autodiff gradient is wrong (the bracket comparisons
        # block gradient flow), which corrupts NUTS and MAP gradients. Solve for the value
        # with bisection under stop_gradient, then take one Newton step so the result carries
        # the exact implicit-function gradient (matching CooperativeBindingModel).
        bounds = jax.lax.fori_loop(0, 80, body, jnp.asarray([0.0, jnp.maximum(total_ligand, 0.0)]))
        free_ligand = jax.lax.stop_gradient(0.5 * (bounds[0] + bounds[1]))
        balance = ligand_balance(free_ligand)
        balance_slope = jax.grad(ligand_balance)(free_ligand)
        free_ligand = free_ligand - balance / balance_slope
        free_monomer, dimer, singly, doubly = dimer_species_for_ligand(free_ligand)
        return free_monomer, free_ligand, dimer, singly, doubly

    @staticmethod
    def _enthalpy_density(total_protein, total_ligand, k_dim, ka1, ka2, h_dim, h1, h2):
        _m, _l, dimer, singly, doubly = DimerizationCooperativeBindingModel._species(
            total_protein, total_ligand, k_dim, ka1, ka2
        )
        return h_dim * (dimer + singly + doubly) + h1 * (singly + doubly) + h2 * doubly

    @staticmethod
    def enthalpy_density(
        protein_molar,
        ligand_molar,
        temperature_k,
        *,
        delta_g_dimer,
        delta_g_binding,
        delta_delta_g_binding,
        delta_h_dimer,
        delta_h_first,
        delta_h_second,
    ):
        """Equilibrium enthalpy density (kcal/L) at monomer-equivalent ``protein_molar`` and total
        ``ligand_molar``, vectorized over the ``(protein, ligand)`` totals.

        Unlike :meth:`expected_heats` (which scans a single self-contained titration), this exposes the
        per-condition enthalpy density directly, so a likelihood can predict heats for an externally
        supplied concentration schedule -- e.g. the sequential, multi-segment TrpR phosphate titration,
        where the integrated heat of injection ``i`` is ``V_cell * (H_after - f_i * H_before)``.
        """
        k_dim = association_constant_from_delta_g(delta_g_dimer, temperature_k)
        ka1 = association_constant_from_delta_g(delta_g_binding, temperature_k)
        ka2 = association_constant_from_delta_g(delta_g_binding + delta_delta_g_binding, temperature_k)
        protein = jnp.atleast_1d(jnp.asarray(protein_molar, dtype=float))
        ligand = jnp.atleast_1d(jnp.asarray(ligand_molar, dtype=float))
        return jax.vmap(
            lambda p, lig: DimerizationCooperativeBindingModel._enthalpy_density(
                p, lig, k_dim, ka1, ka2, delta_h_dimer, delta_h_first, delta_h_second
            )
        )(protein, ligand)

    @staticmethod
    def expected_heats(
        injection_volumes_liter,
        *,
        cell_volume_liter,
        cell_concentration_molar,
        syringe_concentration_molar,
        temperature_k,
        delta_g_dimer,
        delta_g_binding,
        delta_delta_g_binding,
        delta_h_dimer,
        delta_h_first,
        delta_h_second,
        heat_offset,
    ):
        k_dim = association_constant_from_delta_g(delta_g_dimer, temperature_k)
        ka1 = association_constant_from_delta_g(delta_g_binding, temperature_k)
        ka2 = association_constant_from_delta_g(delta_g_binding + delta_delta_g_binding, temperature_k)
        volumes = jnp.asarray(injection_volumes_liter)
        h0 = DimerizationCooperativeBindingModel._enthalpy_density(
            cell_concentration_molar, 0.0, k_dim, ka1, ka2, delta_h_dimer, delta_h_first, delta_h_second
        )

        def step(carry, injection_volume):
            total_protein, total_ligand, previous_density = carry
            dilution = 1.0 - injection_volume / cell_volume_liter
            total_protein = total_protein * dilution
            total_ligand = total_ligand * dilution + syringe_concentration_molar * (injection_volume / cell_volume_liter)
            density = DimerizationCooperativeBindingModel._enthalpy_density(
                total_protein, total_ligand, k_dim, ka1, ka2, delta_h_dimer, delta_h_first, delta_h_second
            )
            heat = cell_volume_liter * (density - dilution * previous_density) * MICROCALORIES_PER_KCAL + heat_offset
            return (total_protein, total_ligand, density), heat

        (_protein, _ligand, _density), heats = jax.lax.scan(
            step, (cell_concentration_molar, 0.0, h0), volumes
        )
        return heats

    @staticmethod
    def equilibrium_species(
        protein_molar,
        ligand_molar,
        temperature_k,
        *,
        delta_g_dimer,
        delta_g_binding,
        delta_delta_g_binding,
    ):
        """Equilibrium ``{free_ligand, free_monomer, dimer, singly_bound, doubly_bound}`` concentrations.

        Vectorized over the ``(protein, ligand)`` totals (``_species`` vmapped over conditions). The
        protein total is monomer-equivalent. Exposes the model's species to other modalities.
        """
        k_dim = association_constant_from_delta_g(delta_g_dimer, temperature_k)
        ka1 = association_constant_from_delta_g(delta_g_binding, temperature_k)
        ka2 = association_constant_from_delta_g(delta_g_binding + delta_delta_g_binding, temperature_k)
        protein = jnp.atleast_1d(jnp.asarray(protein_molar, dtype=float))
        ligand = jnp.atleast_1d(jnp.asarray(ligand_molar, dtype=float))
        monomer, free_ligand, dimer, singly, doubly = jax.vmap(
            lambda p, lig: DimerizationCooperativeBindingModel._species(p, lig, k_dim, ka1, ka2)
        )(protein, ligand)
        return {
            "free_ligand": free_ligand,
            "free_monomer": monomer,
            "dimer": dimer,
            "singly_bound": singly,
            "doubly_bound": doubly,
        }


@dataclass(frozen=True)
class DimerizationMonomerCooperativeBindingModel:
    """Dimerizing receptor with cooperative ligand binding to the dimer *and* ligand binding to the monomer.

    This extends :class:`DimerizationCooperativeBindingModel` with an ``M + L <-> ML`` channel, giving
    the coupled monomer-ligand/dimer two-site model:

    M + M  <-> D
    M + L  <-> ML
    D + L  <-> DL
    DL + L <-> DL2

    The receptor total concentration is monomer-equivalent. ``delta_g_dimer`` is the standard dimer
    association free energy; ``delta_g_binding`` is the first ligand-binding step on the dimer and
    ``delta_delta_g_binding`` shifts the second. Monomer ligand affinity is parameterized as weaker
    than (or equal to) the first dimer site via ``delta_g_monomer = delta_g_binding +
    delta_delta_g_monomer`` with ``delta_delta_g_monomer >= 0`` (so ``k_m <= ka1``). This is the
    free-energy analogue of the deterministic ``k_m = k1 / 10**gap`` construction, but as a signed
    offset suited to a soft non-negative prior rather than a hard bound (the deterministic fit railed
    the gap at its floor).

    Enthalpy density:

    ``H = h_m * [ML] + h_dim * ([D] + [DL] + [DL2]) + h1 * ([DL] + [DL2]) + h2 * [DL2]``
    """

    name: str = "dimerization_monomer_cooperative"

    @staticmethod
    def _species(total_protein, total_ligand, k_dim, ka1, ka2, k_m):
        def species_for_free_ligand(free_ligand):
            z_m = 1.0 + k_m * free_ligand
            z_d = 1.0 + ka1 * free_ligand + ka1 * ka2 * free_ligand * free_ligand
            a = 2.0 * k_dim * z_d
            # Protein mass balance is a * M^2 + z_m * M - P_tot = 0 (the M + ML + 2*(D+DL+DL2) sum),
            # so free monomer is the positive root; fall back to M = P_tot / z_m when no dimer forms.
            free_monomer = jnp.where(
                a > 0.0,
                (-z_m + jnp.sqrt(jnp.maximum(z_m * z_m + 4.0 * a * total_protein, 0.0))) / (2.0 * a),
                total_protein / z_m,
            )
            monomer_ligand = k_m * free_monomer * free_ligand
            dimer = k_dim * free_monomer * free_monomer
            singly = dimer * ka1 * free_ligand
            doubly = singly * ka2 * free_ligand
            return free_monomer, monomer_ligand, dimer, singly, doubly

        def ligand_balance(free_ligand):
            _m, monomer_ligand, _d, singly, doubly = species_for_free_ligand(free_ligand)
            return free_ligand + monomer_ligand + singly + 2.0 * doubly - total_ligand

        def body(_i, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            return jnp.where(ligand_balance(mid) > 0.0, jnp.asarray([lo, mid]), jnp.asarray([mid, hi]))

        # Same numerical pattern as DimerizationCooperativeBindingModel: bisection is robust but its
        # autodiff gradient is wrong (the bracket comparisons block gradient flow), which corrupts NUTS
        # and MAP gradients. Solve for the value with bisection under stop_gradient, then take one Newton
        # step so the result carries the exact implicit-function gradient.
        bounds = jax.lax.fori_loop(0, 80, body, jnp.asarray([0.0, jnp.maximum(total_ligand, 0.0)]))
        free_ligand = jax.lax.stop_gradient(0.5 * (bounds[0] + bounds[1]))
        balance = ligand_balance(free_ligand)
        balance_slope = jax.grad(ligand_balance)(free_ligand)
        free_ligand = free_ligand - balance / balance_slope
        free_monomer, monomer_ligand, dimer, singly, doubly = species_for_free_ligand(free_ligand)
        return free_monomer, free_ligand, monomer_ligand, dimer, singly, doubly

    @staticmethod
    def _enthalpy_density(total_protein, total_ligand, k_dim, ka1, ka2, k_m, h_dim, h1, h2, h_m):
        _m, _l, monomer_ligand, dimer, singly, doubly = DimerizationMonomerCooperativeBindingModel._species(
            total_protein, total_ligand, k_dim, ka1, ka2, k_m
        )
        return (
            h_dim * (dimer + singly + doubly)
            + h1 * (singly + doubly)
            + h2 * doubly
            + h_m * monomer_ligand
        )

    @staticmethod
    def enthalpy_density(
        protein_molar,
        ligand_molar,
        temperature_k,
        *,
        delta_g_dimer,
        delta_g_binding,
        delta_delta_g_binding,
        delta_delta_g_monomer,
        delta_h_dimer,
        delta_h_first,
        delta_h_second,
        delta_h_monomer,
    ):
        """Equilibrium enthalpy density (kcal/L), vectorized over the monomer-equivalent ``protein_molar``
        and total ``ligand_molar`` totals; see :meth:`DimerizationCooperativeBindingModel.enthalpy_density`.
        """
        k_dim = association_constant_from_delta_g(delta_g_dimer, temperature_k)
        ka1 = association_constant_from_delta_g(delta_g_binding, temperature_k)
        ka2 = association_constant_from_delta_g(delta_g_binding + delta_delta_g_binding, temperature_k)
        k_m = association_constant_from_delta_g(delta_g_binding + delta_delta_g_monomer, temperature_k)
        protein = jnp.atleast_1d(jnp.asarray(protein_molar, dtype=float))
        ligand = jnp.atleast_1d(jnp.asarray(ligand_molar, dtype=float))
        return jax.vmap(
            lambda p, lig: DimerizationMonomerCooperativeBindingModel._enthalpy_density(
                p, lig, k_dim, ka1, ka2, k_m, delta_h_dimer, delta_h_first, delta_h_second, delta_h_monomer
            )
        )(protein, ligand)

    @staticmethod
    def expected_heats(
        injection_volumes_liter,
        *,
        cell_volume_liter,
        cell_concentration_molar,
        syringe_concentration_molar,
        temperature_k,
        delta_g_dimer,
        delta_g_binding,
        delta_delta_g_binding,
        delta_delta_g_monomer,
        delta_h_dimer,
        delta_h_first,
        delta_h_second,
        delta_h_monomer,
        heat_offset,
    ):
        k_dim = association_constant_from_delta_g(delta_g_dimer, temperature_k)
        ka1 = association_constant_from_delta_g(delta_g_binding, temperature_k)
        ka2 = association_constant_from_delta_g(delta_g_binding + delta_delta_g_binding, temperature_k)
        k_m = association_constant_from_delta_g(delta_g_binding + delta_delta_g_monomer, temperature_k)
        volumes = jnp.asarray(injection_volumes_liter)
        h0 = DimerizationMonomerCooperativeBindingModel._enthalpy_density(
            cell_concentration_molar, 0.0, k_dim, ka1, ka2, k_m,
            delta_h_dimer, delta_h_first, delta_h_second, delta_h_monomer,
        )

        def step(carry, injection_volume):
            total_protein, total_ligand, previous_density = carry
            dilution = 1.0 - injection_volume / cell_volume_liter
            total_protein = total_protein * dilution
            total_ligand = total_ligand * dilution + syringe_concentration_molar * (injection_volume / cell_volume_liter)
            density = DimerizationMonomerCooperativeBindingModel._enthalpy_density(
                total_protein, total_ligand, k_dim, ka1, ka2, k_m,
                delta_h_dimer, delta_h_first, delta_h_second, delta_h_monomer,
            )
            heat = cell_volume_liter * (density - dilution * previous_density) * MICROCALORIES_PER_KCAL + heat_offset
            return (total_protein, total_ligand, density), heat

        (_protein, _ligand, _density), heats = jax.lax.scan(
            step, (cell_concentration_molar, 0.0, h0), volumes
        )
        return heats

    @staticmethod
    def equilibrium_species(
        protein_molar,
        ligand_molar,
        temperature_k,
        *,
        delta_g_dimer,
        delta_g_binding,
        delta_delta_g_binding,
        delta_delta_g_monomer,
    ):
        """Equilibrium ``{free_ligand, free_monomer, monomer_ligand, dimer, singly_bound, doubly_bound}``.

        Vectorized over the ``(protein, ligand)`` totals (``_species`` vmapped over conditions). The
        protein total is monomer-equivalent. Exposes the model's species to other modalities.
        """
        k_dim = association_constant_from_delta_g(delta_g_dimer, temperature_k)
        ka1 = association_constant_from_delta_g(delta_g_binding, temperature_k)
        ka2 = association_constant_from_delta_g(delta_g_binding + delta_delta_g_binding, temperature_k)
        k_m = association_constant_from_delta_g(delta_g_binding + delta_delta_g_monomer, temperature_k)
        protein = jnp.atleast_1d(jnp.asarray(protein_molar, dtype=float))
        ligand = jnp.atleast_1d(jnp.asarray(ligand_molar, dtype=float))
        monomer, free_ligand, monomer_ligand, dimer, singly, doubly = jax.vmap(
            lambda p, lig: DimerizationMonomerCooperativeBindingModel._species(p, lig, k_dim, ka1, ka2, k_m)
        )(protein, ligand)
        return {
            "free_ligand": free_ligand,
            "free_monomer": monomer,
            "monomer_ligand": monomer_ligand,
            "dimer": dimer,
            "singly_bound": singly,
            "doubly_bound": doubly,
        }


@dataclass(frozen=True)
class EnantiomericMixtureBindingModel:
    """Competitive binding of an enantiomeric mixture to one receptor.

    This implements the racemic-mixture (RM) and enantiomeric-mixture (EM)
    models of Nguyen et al. 2022. The titrant is a mixture of two ligands
    (enantiomers) that compete for a single receptor site with possibly
    different free energies and enthalpies. ``rho`` is the mole fraction of
    ligand 1 in the syringe, so the syringe contains ``rho * [L]_s`` of ligand 1
    and ``(1 - rho) * [L]_s`` of ligand 2.

    By convention ligand 1 is the higher-affinity enantiomer: ``delta_g1`` is its
    binding free energy and ``delta_delta_g = delta_g2 - delta_g1 >= 0`` shifts
    the second, weaker binder. ``delta_h1`` and ``delta_h2`` are the binding
    enthalpies of ligand 1 and ligand 2.

    The boolean ``racemic`` flag controls how the NumPyro model treats ``rho``.
    When ``racemic`` is True the mixture is assumed to be a 1:1 racemate and the
    composition is fixed at ``rho = 0.5`` (the RM model). When False, ``rho`` is a
    free parameter sampled on ``(0, 1)`` (the EM model). ``expected_heats`` always
    takes ``rho`` explicitly; the flag only changes the model builder in
    ``inference.build_numpyro_model``.

    Equilibrium concentrations use the analytic competitive-binding solution of
    Wang (1995), which has exact (closed-form) gradients suited to NUTS.
    """

    name: str = "enantiomeric_mixture"
    racemic: bool = True

    @staticmethod
    def _complex_concentrations(kd1, kd2, total_receptor, total_ligand1, total_ligand2):
        """Bound concentrations of two competing ligands (Wang 1995).

        REF: Zhi-Xin Wang, "An exact mathematical expression for describing
        competitive binding of two different ligands to a protein molecule",
        FEBS Letters 360 (1995) 111-114. All inputs and outputs are molar.
        """
        a = kd1 + kd2 + total_ligand1 + total_ligand2 - total_receptor
        b = kd2 * (total_ligand1 - total_receptor) + kd1 * (total_ligand2 - total_receptor) + kd1 * kd2
        c = -kd1 * kd2 * total_receptor
        d = jnp.sqrt(jnp.maximum(a * a - 3.0 * b, 0.0))
        cos_arg = jnp.clip((-2.0 * a**3 + 9.0 * a * b - 27.0 * c) / (2.0 * d**3), -1.0, 1.0)
        theta = jnp.arccos(cos_arg)
        x = 2.0 * d * jnp.cos(theta / 3.0) - a
        rl1 = total_ligand1 * x / (3.0 * kd1 + x)
        rl2 = total_ligand2 * x / (3.0 * kd2 + x)
        return rl1, rl2

    @staticmethod
    def expected_heats(
        injection_volumes_liter,
        *,
        cell_volume_liter,
        cell_concentration_molar,
        syringe_concentration_molar,
        temperature_k,
        rho,
        delta_g1,
        delta_delta_g,
        delta_h1,
        delta_h2,
        heat_offset,
    ):
        kd1 = dissociation_constant_from_delta_g(delta_g1, temperature_k)
        kd2 = dissociation_constant_from_delta_g(delta_g1 + delta_delta_g, temperature_k)
        volumes = jnp.asarray(injection_volumes_liter)

        def step(carry, injection_volume):
            dilution_cumulative, previous_rl1, previous_rl2 = carry
            dilution = 1.0 - injection_volume / cell_volume_liter
            dilution_cumulative = dilution_cumulative * dilution
            total_receptor = cell_concentration_molar * dilution_cumulative
            total_ligand = syringe_concentration_molar * (1.0 - dilution_cumulative)
            rl1, rl2 = EnantiomericMixtureBindingModel._complex_concentrations(
                kd1, kd2, total_receptor, rho * total_ligand, (1.0 - rho) * total_ligand
            )
            delta_rl1_mol = cell_volume_liter * (rl1 - dilution * previous_rl1)
            delta_rl2_mol = cell_volume_liter * (rl2 - dilution * previous_rl2)
            heat = (
                delta_h1 * delta_rl1_mol + delta_h2 * delta_rl2_mol
            ) * MICROCALORIES_PER_KCAL + heat_offset
            return (dilution_cumulative, rl1, rl2), heat

        (_dilution, _rl1, _rl2), heats = jax.lax.scan(step, (1.0, 0.0, 0.0), volumes)
        return heats


MODEL_REGISTRY = {
    "two_component": TwoComponentBindingModel(),
    "cooperative": CooperativeBindingModel(),
    "cooperative_equivalent_sites": CooperativeBindingModel(
        name="cooperative_equivalent_sites", equivalent_sites=True
    ),
    "cooperative_equal_affinity": CooperativeBindingModel(
        name="cooperative_equal_affinity", equal_affinity=True
    ),
    "cooperative_distinct": CooperativeBindingModel(name="cooperative_distinct", distinct_sites=True),
    "dimerization_cooperative": DimerizationCooperativeBindingModel(),
    "dimerization_monomer_cooperative": DimerizationMonomerCooperativeBindingModel(),
    "racemic_mixture": EnantiomericMixtureBindingModel(name="racemic_mixture", racemic=True),
    "enantiomeric_mixture": EnantiomericMixtureBindingModel(name="enantiomeric_mixture", racemic=False),
}
