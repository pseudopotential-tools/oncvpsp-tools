"""Tests for the ONCVPSP classes."""

from pathlib import Path

import pytest

from oncvpsp_tools import ONCVPSPInput, ONCVPSPOutput

oncv_directory = Path(__file__).parent / "oncvpsp"


@pytest.mark.parametrize("filename", oncv_directory.glob("*.in"))
def test_oncv_input(filename):
    """Test creating a :class:`ONCVPSPInput` object from an ONCVPSP input file."""
    ONCVPSPInput.from_file(filename)


@pytest.mark.parametrize("filename", oncv_directory.glob("*.in"))
def test_oncv_input_roundtrip(filename):
    """Test creating a :class:`ONCVPSPInput` object from file, writing it back to disk, and reading it again."""
    oncv = ONCVPSPInput.from_file(filename)
    oncv.to_file(filename.with_suffix(".rewritten.in"))
    oncv2 = ONCVPSPInput.from_file(filename.with_suffix(".rewritten.in"))
    assert oncv == oncv2


@pytest.mark.parametrize("filename", oncv_directory.glob("*.out"))
def test_oncv_output(filename):
    """Test creating a :class:`ONCVPSPOutput` object from an ONCVPSP input file."""
    oncvo = ONCVPSPOutput.from_file(filename)
    oncvo.charge_densities.plot()
