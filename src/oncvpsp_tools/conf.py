"""Class for handling ONCVPSP input file with customized confining potential."""

from dataclasses import dataclass

from .utils import sanitize
from .input import ONCVPSPInput, ONCVPSPEntry


@dataclass(repr=False)
class ONCVPSPConfiningPotential(ONCVPSPEntry):
    """Class for the confining potential block in an ONCVPSP input file."""

    depth: float
    r: float
    beta: float


class ONCVPSPInputWithConf(ONCVPSPInput):
    """
    Class for handling ONCVPSP input file with customized confining potential.

    Takes the same arguments as :class:`ONCVPSPInput` with the additional keyword

    :param confining_potential: The confining potential block of the input file
    :type confining_potential: :class:`ONCVPSPConfiningPotential`

    """

    def __init__(self, *args, confining_potential=ONCVPSPConfiningPotential(0.0, 0.0, 0.0), **kwargs):
        super().__init__(*args, **kwargs)
        self.confining_potential = confining_potential

    def to_str(self):
        """Convert the input file to a string."""
        return super().to_str() + "\n# CONFINING POTENTIAL\n" + self.confining_potential.to_str()
    
    @classmethod
    def from_str(cls, content):
        """Read the input file from a string."""
        out = super(ONCVPSPInputWithConf, cls).from_str(content)
        content=[l for l in content.split("\n") if l.strip() and not l.strip().startswith('#')]
        iconf = 1 + out.atom.nc + out.atom.nv + 1 + out.lmax + 1 + 1 + out.lmax + 1 + 1 + 1 + 1 + 1 + sum([len(c) + 1 for c in out.test_configurations])
        if len(content) > iconf:
            out.confining_potential = ONCVPSPConfiningPotential(*[sanitize(v) for v in content[iconf].split()])
        return out

