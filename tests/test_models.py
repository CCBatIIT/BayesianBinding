from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from pytest import approx

from bayesian_binding.data import load_dat
from bayesian_binding.constants import beta_mol_per_kcal
from bayesian_binding.inference import build_numpyro_model
from bayesian_binding.models import (
    AdKADPSteadyStateModel,
    CooperativeBindingModel,
    DimerizationCooperativeBindingModel,
    DimerizationMonomerCooperativeBindingModel,
    EnantiomericMixtureBindingModel,
    TwoComponentBindingModel,
    association_constant_from_delta_g,
)


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
FIXTURES = REPO / "tests" / "fixtures"


def test_two_component_expected_heats_shape():
    experiment = load_dat(FIXTURES / "simple.DAT")
    q = TwoComponentBindingModel.expected_heats(
        experiment.injection_volumes_liter,
        cell_volume_liter=experiment.cell_volume_liter,
        cell_concentration_molar=experiment.cell_concentration_molar,
        syringe_concentration_molar=experiment.syringe_concentration_molar,
        temperature_k=experiment.temperature_k,
        delta_g=-4.0,
        delta_h=-3.0,
        heat_offset=0.0,
    )
    assert q.shape == experiment.heats_microcalorie.shape
    assert np.all(np.isfinite(np.asarray(q)))


def test_cooperative_two_site_default_matches_explicit_polynomial():
    # The default n_sites=2 must stay byte-identical to the historical 1 + 2 ka1 x + ka1 ka2 x^2 model.
    temperature_k = 279.15
    delta_g, delta_delta_g = -8.0, -1.0
    ka1 = float(association_constant_from_delta_g(delta_g, temperature_k))
    ka2 = float(association_constant_from_delta_g(delta_g + delta_delta_g, temperature_k))
    protein, ligand = 4.0e-5, 6.0e-5
    free, apo, bound = CooperativeBindingModel._species(protein, ligand, [ka1, ka2])
    z = 1.0 + 2.0 * ka1 * free + ka1 * ka2 * free**2
    assert float(apo) == approx(protein / z, rel=1e-9)
    assert float(bound[0]) == approx(protein * 2.0 * ka1 * free / z, rel=1e-9)  # singly
    assert float(bound[1]) == approx(protein * ka1 * ka2 * free**2 / z, rel=1e-9)  # doubly
    # Legacy equilibrium_species keys preserved for two sites.
    species = CooperativeBindingModel.equilibrium_species(
        jnp.array([protein]), jnp.array([ligand]), temperature_k, delta_g=delta_g, delta_delta_g=delta_delta_g
    )
    assert set(species) == {"free_ligand", "apo_protein", "singly_bound", "doubly_bound"}


def test_cooperative_four_site_independent_limit_matches_langmuir():
    # Four independent identical sites: all stepwise intrinsic dG equal (delta_delta_g_i = 0). The
    # binomial statistical factors make mean occupancy 4 * ka * x / (1 + ka * x).
    temperature_k = 279.15
    intrinsic_dg = -7.0
    ka = float(association_constant_from_delta_g(intrinsic_dg, temperature_k))
    constants = [association_constant_from_delta_g(intrinsic_dg, temperature_k)] * 4
    protein = 3.0e-5
    for ligand_total in (1.0e-5, 5.0e-5, 1.2e-4, 3.0e-4):
        free, apo, bound = CooperativeBindingModel._species(protein, ligand_total, constants)
        occupancy = float(sum((k + 1) * bound[k] for k in range(4)) / protein)
        expected = float(4.0 * ka * free / (1.0 + ka * free))
        assert occupancy == approx(expected, rel=1e-6)
        assert float(apo + sum(bound)) == approx(protein, rel=1e-9)  # populations sum to total
        assert float(free + sum((k + 1) * bound[k] for k in range(4))) == approx(ligand_total, rel=1e-7)


def test_cooperative_four_site_expected_heats_shape_and_gradient():
    experiment = load_dat(FIXTURES / "simple.DAT")
    kwargs = dict(
        cell_volume_liter=experiment.cell_volume_liter,
        cell_concentration_molar=experiment.cell_concentration_molar,
        syringe_concentration_molar=experiment.syringe_concentration_molar,
        temperature_k=experiment.temperature_k,
        delta_g=-9.0,
        delta_delta_g_2=1.0,
        delta_delta_g_3=2.5,
        delta_delta_g_4=3.5,
        delta_h_1=-5.0,
        delta_h_2=-4.0,
        delta_h_3=-3.0,
        delta_h_4=-2.0,
        heat_offset=0.0,
    )
    four_site = CooperativeBindingModel(n_sites=4)
    q = four_site.expected_heats(experiment.injection_volumes_liter, **kwargs)
    assert q.shape == experiment.heats_microcalorie.shape
    assert np.all(np.isfinite(np.asarray(q)))

    def loss(delta_g):
        local = dict(kwargs, delta_g=delta_g)
        return jnp.sum(four_site.expected_heats(experiment.injection_volumes_liter, **local) ** 2)

    grad = float(jax.grad(loss)(-9.0))  # implicit-gradient solver must give a finite, nonzero gradient
    assert np.isfinite(grad) and grad != 0.0


def test_cooperative_four_site_equilibrium_species_columns():
    species = CooperativeBindingModel(n_sites=4).equilibrium_species(
        jnp.array([3.0e-5, 3.0e-5]),
        jnp.array([5.0e-5, 1.5e-4]),
        279.15,
        delta_g=-9.0,
        delta_delta_g_2=1.0,
        delta_delta_g_3=2.5,
        delta_delta_g_4=3.5,
    )
    assert set(species) == {"free_ligand", "apo_protein", "bound_1", "bound_2", "bound_3", "bound_4"}
    stacked = np.stack([np.asarray(species[name]) for name in species])
    assert stacked.shape == (6, 2)
    assert np.all(np.isfinite(stacked)) and np.all(stacked >= -1e-12)


def test_adk_adp_steady_state_mass_balance_and_reaction_quotient():
    protein = jnp.array([0.0, 100e-6, 400e-6])
    adp = jnp.array([1.0e-3, 1.0e-3, 4.0e-3])
    keq = 0.8
    params = dict(
        delta_g_amp=-5.0,
        delta_g_atp=-6.0,
        delta_delta_g_amp_atp=-1.0,
        delta_g_adp=-5.5,
        delta_delta_g_adp=-0.5,
        log_adp_dismutation_keq=np.log(keq),
    )
    sp = AdKADPSteadyStateModel.equilibrium_species(
        protein,
        adp,
        298.15,
        **params,
    )
    nucleotide_total = (
        sp["free_amp"]
        + sp["free_adp"]
        + sp["free_atp"]
        + sp["protein_amp"]
        + sp["protein_atp"]
        + 2.0 * sp["protein_amp_atp"]
        + sp["protein_adp"]
        + 2.0 * sp["protein_adp2"]
    )
    amp_total = sp["free_amp"] + sp["protein_amp"] + sp["protein_amp_atp"]
    atp_total = sp["free_atp"] + sp["protein_atp"] + sp["protein_amp_atp"]
    protein_total = (
        sp["free_protein"]
        + sp["protein_amp"]
        + sp["protein_atp"]
        + sp["protein_amp_atp"]
        + sp["protein_adp"]
        + sp["protein_adp2"]
    )
    quotient = sp["free_amp"] * sp["free_atp"] / (sp["free_adp"] * sp["free_adp"])
    np.testing.assert_allclose(np.asarray(nucleotide_total), np.asarray(adp), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(np.asarray(amp_total), np.asarray(atp_total), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(np.asarray(protein_total), np.asarray(protein), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(np.asarray(quotient), keq, rtol=1e-5, atol=1e-10)
    beta = beta_mol_per_kcal(298.15)
    b_amp_atp = np.exp(
        -beta * (params["delta_g_amp"] + params["delta_g_atp"] + params["delta_delta_g_amp_atp"])
    )
    b_adp2 = np.exp(-beta * (2.0 * params["delta_g_adp"] + params["delta_delta_g_adp"]))
    expected_bound_ratio = keq * b_amp_atp / b_adp2
    observed_bound_ratio = sp["protein_amp_atp"][1:] / sp["protein_adp2"][1:]
    np.testing.assert_allclose(
        np.asarray(observed_bound_ratio), expected_bound_ratio, rtol=1e-5, atol=1e-10
    )


def test_adk_adp_steady_state_reduces_to_analytic_no_protein_case():
    adp = jnp.array([1.0e-3, 4.0e-3])
    keq = 0.25
    sp = AdKADPSteadyStateModel.equilibrium_species(
        jnp.zeros_like(adp),
        adp,
        298.15,
        delta_g_amp=-5.0,
        delta_g_atp=-6.0,
        delta_delta_g_amp_atp=-1.0,
        delta_g_adp=-5.5,
        delta_delta_g_adp=-0.5,
        log_adp_dismutation_keq=np.log(keq),
    )
    ratio = np.sqrt(keq)
    product = ratio * np.asarray(adp) / (1.0 + 2.0 * ratio)
    np.testing.assert_allclose(np.asarray(sp["free_amp"]), product, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(np.asarray(sp["free_atp"]), product, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(np.asarray(sp["free_adp"]), np.asarray(adp) - 2.0 * product, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(
            sp["protein_amp"]
            + sp["protein_atp"]
            + sp["protein_amp_atp"]
            + sp["protein_adp"]
            + sp["protein_adp2"]
        ),
        0.0,
        atol=1e-14,
    )


def test_cooperative_expected_heats_shape():
    volumes = jnp.ones(10) * 1.0e-6
    q = CooperativeBindingModel.expected_heats(
        volumes,
        cell_volume_liter=1.4e-3,
        cell_concentration_molar=50e-6,
        syringe_concentration_molar=500e-6,
        temperature_k=298.15,
        delta_g=-7.0,
        delta_delta_g=-1.0,
        delta_h_first=-8.0,
        delta_h_second=-5.0,
        heat_offset=0.0,
    )
    assert q.shape == (10,)
    assert np.all(np.isfinite(np.asarray(q)))


def test_dimerization_expected_heats_shape():
    volumes = jnp.ones(10) * 2.0e-6
    q = DimerizationCooperativeBindingModel.expected_heats(
        volumes,
        cell_volume_liter=1.4e-3,
        cell_concentration_molar=40e-6,
        syringe_concentration_molar=800e-6,
        temperature_k=298.15,
        delta_g_dimer=-7.0,
        delta_g_binding=-6.0,
        delta_delta_g_binding=-0.5,
        delta_h_dimer=-3.0,
        delta_h_first=-8.0,
        delta_h_second=-5.0,
        heat_offset=0.0,
    )
    assert q.shape == (10,)
    assert np.all(np.isfinite(np.asarray(q)))


def test_dimerization_monomer_expected_heats_shape():
    volumes = jnp.ones(10) * 2.0e-6
    q = DimerizationMonomerCooperativeBindingModel.expected_heats(
        volumes,
        cell_volume_liter=1.4e-3,
        cell_concentration_molar=40e-6,
        syringe_concentration_molar=800e-6,
        temperature_k=298.15,
        delta_g_dimer=-7.0,
        delta_g_binding=-6.0,
        delta_delta_g_binding=-0.5,
        delta_delta_g_monomer=1.0,
        delta_h_dimer=-3.0,
        delta_h_first=-8.0,
        delta_h_second=-5.0,
        delta_h_monomer=-6.0,
        heat_offset=0.0,
    )
    assert q.shape == (10,)
    assert np.all(np.isfinite(np.asarray(q)))


def test_dimerization_monomer_reduces_to_dimerization_when_monomer_unbinds():
    """As the monomer site is made arbitrarily weak (large delta_delta_g_monomer, so k_m -> 0) the
    monomer-binding model must coincide with the plain dimerization-cooperative model. This is the
    nesting that the (b) vs (c) Bayes factor exploits."""
    volumes = jnp.ones(12) * 3.0e-6
    common = dict(
        cell_volume_liter=1.4229e-3,
        cell_concentration_molar=7.6e-6,
        syringe_concentration_molar=20e-6,
        temperature_k=298.15,
        heat_offset=0.0,
    )
    shared = dict(delta_g_dimer=-8.0, delta_g_binding=-7.0, delta_delta_g_binding=-0.5)
    enth = dict(delta_h_dimer=-3.0, delta_h_first=-8.0, delta_h_second=-5.0)
    q_monomer = DimerizationMonomerCooperativeBindingModel.expected_heats(
        volumes, **common, **shared, delta_delta_g_monomer=40.0, **enth, delta_h_monomer=-6.0
    )
    q_dimer = DimerizationCooperativeBindingModel.expected_heats(
        volumes, **common, **shared, **enth
    )
    np.testing.assert_allclose(np.asarray(q_monomer), np.asarray(q_dimer), rtol=1e-6, atol=1e-9)


def test_dimerization_monomer_mass_balance_closes():
    """Equilibrium species reproduce the monomer-equivalent protein and total-ligand inputs."""
    protein = jnp.array([5e-6, 7.6e-6, 1.0e-5])
    ligand = jnp.array([1e-6, 5e-6, 2e-5])
    sp = DimerizationMonomerCooperativeBindingModel.equilibrium_species(
        protein, ligand, 298.15,
        delta_g_dimer=-8.0, delta_g_binding=-7.0, delta_delta_g_binding=-0.5, delta_delta_g_monomer=1.0,
    )
    protein_recon = sp["free_monomer"] + sp["monomer_ligand"] + 2.0 * (
        sp["dimer"] + sp["singly_bound"] + sp["doubly_bound"]
    )
    ligand_recon = sp["free_ligand"] + sp["monomer_ligand"] + sp["singly_bound"] + 2.0 * sp["doubly_bound"]
    np.testing.assert_allclose(np.asarray(protein_recon), np.asarray(protein), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(np.asarray(ligand_recon), np.asarray(ligand), rtol=1e-6, atol=1e-12)


def test_dimerization_monomer_registry_instance():
    from bayesian_binding.models import MODEL_REGISTRY

    assert MODEL_REGISTRY["dimerization_monomer_cooperative"].name == "dimerization_monomer_cooperative"


def test_enantiomeric_mixture_expected_heats_shape():
    volumes = jnp.ones(10) * 1.0e-6
    q = EnantiomericMixtureBindingModel.expected_heats(
        volumes,
        cell_volume_liter=1.3513e-3,
        cell_concentration_molar=50e-6,
        syringe_concentration_molar=1000e-6,
        temperature_k=300.0,
        rho=0.45,
        delta_g1=-11.0,
        delta_delta_g=4.0,
        delta_h1=-7.0,
        delta_h2=-2.0,
        heat_offset=0.0,
    )
    assert q.shape == (10,)
    assert np.all(np.isfinite(np.asarray(q)))


def test_enantiomeric_mixture_registry_instances():
    from bayesian_binding.models import MODEL_REGISTRY

    assert MODEL_REGISTRY["racemic_mixture"].racemic is True
    assert MODEL_REGISTRY["enantiomeric_mixture"].racemic is False


def test_cooperative_null_variants_heat_kwargs():
    """The two cooperative nulls expand their reduced parameters correctly; full model is identity."""
    from bayesian_binding.models import MODEL_REGISTRY

    full = MODEL_REGISTRY["cooperative"]
    equiv = MODEL_REGISTRY["cooperative_equivalent_sites"]
    affin = MODEL_REGISTRY["cooperative_equal_affinity"]
    assert equiv.equivalent_sites and not equiv.equal_affinity
    assert affin.equal_affinity and not affin.equivalent_sites
    # equal_affinity: delta_delta_g fixed to 0, the two step enthalpies kept distinct.
    assert affin.heat_kwargs({"delta_g": -7.0, "delta_h_first": -9.0, "delta_h_second": -5.0}) == {
        "delta_g": -7.0, "delta_delta_g": 0.0, "delta_h_first": -9.0, "delta_h_second": -5.0
    }
    # equivalent_sites: delta_delta_g = 0 AND the enthalpies tied.
    assert equiv.heat_kwargs({"delta_g": -7.0, "delta_h": -6.0}) == {
        "delta_g": -7.0, "delta_delta_g": 0.0, "delta_h_first": -6.0, "delta_h_second": -6.0
    }
    # full model: identity.
    assert full.heat_kwargs(
        {"delta_g": -7.0, "delta_delta_g": -1.0, "delta_h_first": -9.0, "delta_h_second": -5.0}
    )["delta_delta_g"] == -1.0


def test_cooperative_equal_affinity_allows_distinct_enthalpies():
    """At delta_delta_g = 0, distinct step enthalpies give different heats than equal ones, so the
    equal-affinity null is genuinely distinct from the equivalent-sites null."""
    from bayesian_binding.models import MODEL_REGISTRY

    volumes = jnp.ones(12) * 5.0e-6
    common = dict(cell_volume_liter=0.2052e-3, cell_concentration_molar=1.0e-4,
                  syringe_concentration_molar=2.0e-3, temperature_k=298.15, heat_offset=0.0)
    coop = MODEL_REGISTRY["cooperative"]
    q_distinct = np.asarray(coop.expected_heats(
        volumes, delta_g=-7.0, delta_delta_g=0.0, delta_h_first=-9.0, delta_h_second=-5.0, **common))
    q_equal = np.asarray(coop.expected_heats(
        volumes, delta_g=-7.0, delta_delta_g=0.0, delta_h_first=-7.0, delta_h_second=-7.0, **common))
    assert np.all(np.isfinite(q_distinct))
    assert not np.allclose(q_distinct, q_equal)


def test_enantiomeric_mixture_reduces_to_two_component():
    # Two identical enantiomers (delta_delta_g = 0, equal enthalpies) at a 50:50
    # ratio are equivalent to a single ligand at the full syringe concentration,
    # so the competitive-binding heats must match the two-component model.
    volumes = jnp.ones(12) * 5.0e-6
    common = dict(
        cell_volume_liter=1.434e-3,
        cell_concentration_molar=1.0e-4,
        syringe_concentration_molar=1.0e-3,
        temperature_k=298.15,
        heat_offset=3.0,
    )
    q_mixture = EnantiomericMixtureBindingModel.expected_heats(
        volumes, rho=0.5, delta_g1=-9.0, delta_delta_g=0.0, delta_h1=-2.5, delta_h2=-2.5, **common
    )
    q_two_component = TwoComponentBindingModel.expected_heats(
        volumes, delta_g=-9.0, delta_h=-2.5, **common
    )
    np.testing.assert_allclose(
        np.asarray(q_mixture), np.asarray(q_two_component), rtol=1e-5, atol=1e-2
    )


def test_build_numpyro_model_closure():
    experiment = load_dat(FIXTURES / "simple.DAT")
    model = build_numpyro_model(experiment, model_name="two_component")
    assert callable(model)


# --- Equilibrium-species (concentration) checks for the regression-tested binding models ---
# These validate the species that the regression tests exercise end to end: mass balance, limiting
# cases, the law of mass action, and (for the bisection-solved cooperative model) the implicit-function
# gradient that NUTS relies on.

_T = 298.15


def test_two_component_equilibrium_species_mass_balance_and_law_of_mass_action():
    protein = jnp.array([0.0, 50e-6, 200e-6])
    ligand = jnp.array([100e-6, 100e-6, 50e-6])
    sp = TwoComponentBindingModel.equilibrium_species(protein, ligand, _T, delta_g=-8.0)
    np.testing.assert_allclose(np.asarray(sp["free_protein"] + sp["complex"]), np.asarray(protein), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(np.asarray(sp["free_ligand"] + sp["complex"]), np.asarray(ligand), rtol=1e-6, atol=1e-12)
    assert float(sp["complex"][0]) == 0.0  # no protein -> no complex
    assert np.all(np.asarray(sp["complex"]) >= -1e-15)
    # Law of mass action: free_protein * free_ligand / complex = Kd.
    kd = float(np.exp(-8.0 * beta_mol_per_kcal(_T)))
    ratio = np.asarray(sp["free_protein"][1:] * sp["free_ligand"][1:] / sp["complex"][1:])
    np.testing.assert_allclose(ratio, kd, rtol=1e-5)


def test_cooperative_equilibrium_species_mass_balance_and_limits():
    protein = jnp.array([50e-6, 100e-6])
    ligand = jnp.array([20e-6, 400e-6])
    sp = CooperativeBindingModel.equilibrium_species(protein, ligand, _T, delta_g=-7.0, delta_delta_g=-1.0)
    protein_recon = sp["apo_protein"] + sp["singly_bound"] + sp["doubly_bound"]
    ligand_recon = sp["free_ligand"] + sp["singly_bound"] + 2.0 * sp["doubly_bound"]
    np.testing.assert_allclose(np.asarray(protein_recon), np.asarray(protein), rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(np.asarray(ligand_recon), np.asarray(ligand), rtol=1e-5, atol=1e-12)
    for key in ("apo_protein", "singly_bound", "doubly_bound", "free_ligand"):
        assert np.all(np.asarray(sp[key]) >= -1e-12)
    # No ligand -> all apo.
    sp0 = CooperativeBindingModel.equilibrium_species(protein, jnp.zeros_like(ligand), _T, delta_g=-7.0, delta_delta_g=-1.0)
    np.testing.assert_allclose(np.asarray(sp0["apo_protein"]), np.asarray(protein), rtol=1e-6, atol=1e-12)


def test_cooperative_equilibrium_species_implicit_gradient_matches_finite_difference():
    """The free-ligand solve uses an implicit-function gradient (stop_gradient + one Newton step), so
    autodiff through equilibrium_species must match a finite difference -- a plain bisection would not."""
    protein = jnp.array([60e-6])
    ligand = jnp.array([120e-6])

    def total_bound(delta_g):
        sp = CooperativeBindingModel.equilibrium_species(protein, ligand, _T, delta_g=delta_g, delta_delta_g=-1.0)
        return jnp.sum(sp["singly_bound"] + 2.0 * sp["doubly_bound"])

    grad = float(jax.grad(total_bound)(-7.0))
    eps = 1e-4
    fd = (float(total_bound(-7.0 + eps)) - float(total_bound(-7.0 - eps))) / (2 * eps)
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, fd, rtol=2e-2)


def test_dimerization_cooperative_equilibrium_species_mass_balance():
    protein = jnp.array([10e-6, 40e-6])
    ligand = jnp.array([5e-6, 50e-6])
    sp = DimerizationCooperativeBindingModel.equilibrium_species(
        protein, ligand, _T, delta_g_dimer=-8.0, delta_g_binding=-7.0, delta_delta_g_binding=-0.5
    )
    protein_recon = sp["free_monomer"] + 2.0 * (sp["dimer"] + sp["singly_bound"] + sp["doubly_bound"])
    ligand_recon = sp["free_ligand"] + sp["singly_bound"] + 2.0 * sp["doubly_bound"]
    np.testing.assert_allclose(np.asarray(protein_recon), np.asarray(protein), rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(np.asarray(ligand_recon), np.asarray(ligand), rtol=1e-5, atol=1e-12)


def test_enantiomeric_mixture_wang_complex_concentrations():
    """Wang (1995) competitive-binding solution: bound within totals, receptor conserved, and each
    ligand obeys the law of mass action."""
    kd1 = float(np.exp(-9.0 * beta_mol_per_kcal(_T)))
    kd2 = float(np.exp(-5.0 * beta_mol_per_kcal(_T)))  # weaker competitor
    receptor, ligand1, ligand2 = 50e-6, 40e-6, 60e-6
    rl1, rl2 = EnantiomericMixtureBindingModel._complex_concentrations(kd1, kd2, receptor, ligand1, ligand2)
    rl1, rl2 = float(rl1), float(rl2)
    assert 0.0 <= rl1 <= ligand1 and 0.0 <= rl2 <= ligand2
    assert rl1 + rl2 <= receptor + 1e-12
    free_receptor = receptor - rl1 - rl2
    np.testing.assert_allclose(free_receptor * (ligand1 - rl1) / rl1, kd1, rtol=1e-5)
    np.testing.assert_allclose(free_receptor * (ligand2 - rl2) / rl2, kd2, rtol=1e-5)
