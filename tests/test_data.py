from pathlib import Path

import pytest

from bayesian_binding.data import load_dat, load_itc, parse_itc_raw


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
FIXTURES = REPO / "tests" / "fixtures"


def test_load_dat_fixture():
    experiment = load_dat(FIXTURES / "simple.DAT")
    assert experiment.n_injections == 4
    assert experiment.cell_concentration_molar > 0.0
    assert experiment.syringe_concentration_molar > 0.0
    assert experiment.heats_microcalorie[0] < 0.0


def test_parse_itc_fixture():
    metadata, blocks = parse_itc_raw(FIXTURES / "simple.itc")
    assert metadata["number_of_injections"] == 3
    assert len(blocks) == 3
    assert blocks[0].volume_liter > 0.0


def test_load_itc_fixture_with_raw_integration():
    experiment = load_itc(FIXTURES / "simple.itc")
    assert experiment.n_injections == 3
    assert experiment.heats_microcalorie.shape == experiment.injection_volumes_liter.shape


def test_load_mg_edta_dat_when_parent_project_is_available():
    path = PROJECT / "data" / "Mg-EDTA" / "Mg1EDTAp1a.DAT"
    if not path.exists():
        pytest.skip("Parent Binding project data are not available.")
    experiment = load_dat(path)
    assert experiment.n_injections == 23
