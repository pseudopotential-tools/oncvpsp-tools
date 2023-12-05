"""Tests for the ONCVPSP classes."""

from pathlib import Path

import pytest

from oncvpsp_tools import ONCVPSPInput

oncv_directory = Path(__file__).parent / "oncvpsp"


def test_oncv_input_pedantic():
    """A line-by-line test for Ge.oncv.in."""
    inp = ONCVPSPInput.from_file(oncv_directory / "Ge.oncv.in")

    """
    Checking the atom section
    # atsym, z, nc, nv, iexc   psfile
        Ge  32.0   5   3   3   psp8
    """
    assert inp.atom.atsym == "Ge"
    assert inp.atom.z == 32.0
    assert inp.atom.nc == 5
    assert inp.atom.nv == 3
    assert inp.atom.iexc == 3
    assert inp.atom.psfile == "psp8"

    """
    Checking the configuration section
    # n, l, f  (nc+nv lines)
        1    0    2.0
        2    0    2.0
        2    1    6.0
        3    0    2.0
        3    1    6.0
        3    2   10.0
        4    0    2.0
        4    1    2.0
    """
    nlfs = [
        [1, 0, 2.0],
        [2, 0, 2.0],
        [2, 1, 6.0],
        [3, 0, 2.0],
        [3, 1, 6.0],
        [3, 2, 10.0],
        [4, 0, 2.0],
        [4, 1, 2.0],
    ]
    for i, (n, l, f) in enumerate(nlfs):
        assert inp.reference_configuration[i].n == n
        assert inp.reference_configuration[i].l == l
        assert inp.reference_configuration[i].f == f

    """
    Checking the lmax section
    # lmax
        2
    """
    assert inp.lmax == 2

    """
    Checking the optimization section
    # l, rc, ep, ncon, nbas, qcut  (lmax+1 lines, l's must be in order)
        0    2.60   -0.00    4    8    5.00
        1    2.60   -0.00    4    8    5.20
        2    2.00    0.00    4    9    8.40
    """
    for i, values in enumerate(
        [[0, 2.60, -0.00, 4, 8, 5.00], [1, 2.60, -0.00, 4, 8, 5.20], [2, 2.00, 0.00, 4, 9, 8.40]]
    ):
        assert inp.optimization[i].l == values[0]
        assert inp.optimization[i].rc == values[1]
        assert inp.optimization[i].ep == values[2]
        assert inp.optimization[i].ncon == values[3]
        assert inp.optimization[i].nbas == values[4]
        assert inp.optimization[i].qcut == values[5]

    """
    Checking the local potential section
    # lloc, lpopt, rc(5), dvloc0
        4    5    2.0    0.0
    """
    assert inp.local_potential.lloc == 4
    assert inp.local_potential.lpopt == 5
    assert inp.local_potential.rc == 2.0
    assert inp.local_potential.dvloc0 == 0.0

    """
    Checking the projectors section
    # l, nproj, debl
        0    2    1.50
        1    2    1.50
        2    2    1.50
    """
    for i, (l, nproj, debl) in enumerate([[0, 2, 1.50], [1, 2, 1.50], [2, 2, 1.50]]):
        assert inp.vkb_projectors[i].l == l
        assert inp.vkb_projectors[i].nproj == nproj
        assert inp.vkb_projectors[i].debl == debl

    """
    Checking the model core charge section
    # icmod, fcfact, (rcfact)
        0    0.25
    """
    assert inp.model_core_charge.icmod == 0
    assert inp.model_core_charge.fcfact == 0.25
    assert inp.model_core_charge.rcfact is None

    """
    Checking the log derivative analysis section
    # epsh1, epsh2, depsh
       -2.0  2.0  0.02
    """
    assert inp.log_derivative_analysis.epsh1 == -2.0
    assert inp.log_derivative_analysis.epsh2 == 2.0
    assert inp.log_derivative_analysis.depsh == 0.02

    """
    Checking the output grid section
    # rlmax, drl
        4.0  0.01
    """
    assert inp.output_grid.rlmax == 4.0
    assert inp.output_grid.drl == 0.01

    """
    Checking the test configurations
    # ncnf
        4
    #   n    l    f
        3
        3    2   10.00
        4    0    1.00
        4    1    2.00
    #
        3
        3    2   10.00
        4    0    2.00
        4    1    1.00
    #
        3
        3    2   10.00
        4    0    1.00
        4    1    1.00
    #
        3
        3    2   10.00
        4    0    1.00
        4    2    1.00
    #
    """
    reference_configs = [
        [[3, 2, 10.00], [4, 0, 1.00], [4, 1, 2.00]],
        [[3, 2, 10.00], [4, 0, 2.00], [4, 1, 1.00]],
        [[3, 2, 10.00], [4, 0, 1.00], [4, 1, 1.00]],
        [[3, 2, 10.00], [4, 0, 1.00], [4, 2, 1.00]],
    ]
    assert len(inp.test_configurations) == 4
    for config, reference_config in zip(inp.test_configurations, reference_configs):
        assert len(config) == len(reference_config)
        for subshell, (n, l, f) in zip(config, reference_config):
            assert subshell.n == n
            assert subshell.l == l
            assert subshell.f == f


@pytest.mark.parametrize("filename", oncv_directory.glob("*.in"))
def test_oncv_input_roundtrip(filename):
    """Test creating a :class:`ONCVPSPInput` object from file, writing it back to disk, and reading it again."""
    oncv = ONCVPSPInput.from_file(filename)
    oncv.to_file(filename.with_suffix(".rewritten.in"))
    oncv2 = ONCVPSPInput.from_file(filename.with_suffix(".rewritten.in"))
    assert oncv == oncv2