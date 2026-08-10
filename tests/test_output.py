"""Tests for the ONCVPSP classes."""

from pathlib import Path

import pytest

from oncvpsp_tools import ONCVPSPOutput

oncv_directory = Path(__file__).parent / "oncvpsp"


@pytest.mark.parametrize("filename", oncv_directory.glob("*.out"))
def test_oncv_output(filename):
    """Test creating a :class:`ONCVPSPOutput` object from an ONCVPSP input file."""
    oncvo = ONCVPSPOutput.from_file(filename)
    oncvo.charge_densities.plot()


@pytest.mark.parametrize("filename", oncv_directory.glob("*.out"))
def test_oncv_output_roundtrip(filename):
    """Test writing a :class:`ONCVPSPOutput` object to disk and reading it back in."""
    oncvo = ONCVPSPOutput.from_file(filename)
    oncvo.to_file(filename.with_suffix(".rewritten.out"))
    oncvo2 = ONCVPSPOutput.from_file(filename.with_suffix(".rewritten.out"))
    assert oncvo == oncvo2
