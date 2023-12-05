"""Classes for handling ONCVPSP input files."""

import subprocess
from collections import UserList
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from .utils import sanitize


class ONCVPSPEntry:
    """Generic class for an entry in an ONCVPSP input file."""

    @property
    def columns(self) -> str:
        """Return the column headers for the entry."""
        return "# " + " ".join([f"{str(k): >8}" for k in self.__dict__.keys()])[2:]

    @property
    def content(self) -> str:
        """Return the content of the entry."""
        return " ".join([f"{str(v): >8}" for v in self.__dict__.values() if v is not None])

    def to_str(self) -> str:
        """Return the text representation of the entry."""
        return f"{self.columns}\n{self.content}"

    def __repr__(self):
        """Make the repr of the entry very simple."""
        return str(self.__dict__)


T = TypeVar("T")


class ONCVPSPList(UserList, Generic[T]):
    """Generic class for an entry in an ONCVPSP input file that contains multiple elements and emulates a list."""

    def __init__(self, data):
        """Create an :class:`ONCVPSPList` object."""
        super().__init__(data)

    @property
    def columns(self) -> str:
        """Return the column headers for the list."""
        if self.data and isinstance(self.data[0], ONCVPSPEntry):
            return self.data[0].columns
        else:
            return ""

    def to_str(self, print_length=False) -> str:
        """Return the text representation of the list."""
        out = []
        if print_length:
            out.append(f"{len(self): >8}")
        out.append(self.columns)
        out += [
            d.to_str(print_length) if isinstance(d, ONCVPSPList) else d.content for d in self.data
        ]
        return "\n".join(out)


@dataclass(repr=False)
class ONCVPSPAtom(ONCVPSPEntry):
    """Class for the atom block in an ONCVPSP input file."""

    atsym: str
    z: float
    nc: int
    nv: int
    iexc: int
    psfile: str


@dataclass(repr=False)
class ONCVPSPConfigurationSubshell(ONCVPSPEntry):
    """Class for a subshell in an ONCVPSP configuration block."""

    n: int
    l: int
    f: float
    energy: float = -0.0


@dataclass(repr=False)
class ONCVPSPOptimizationChannel(ONCVPSPEntry):
    """Class for an optimization channel in an ONCVPSP input file."""

    l: int
    rc: float
    ep: float
    ncon: int
    nbas: int
    qcut: float


@dataclass(repr=False)
class ONCVPSPLocalPotential(ONCVPSPEntry):
    """Class for the local potential block in an ONCVPSP input file."""

    lloc: int
    lpopt: int
    rc: float
    dvloc0: float


@dataclass(repr=False)
class ONCVPSPVKBProjector(ONCVPSPEntry):
    """Class for a VKB projector in an ONCVPSP input file."""

    l: int
    nproj: int
    debl: float


@dataclass(repr=False)
class ONCVPSPModelCoreCharge(ONCVPSPEntry):
    """Class for the model core charge block in an ONCVPSP input file."""

    icmod: int
    fcfact: float
    rcfact: Optional[float] = None


@dataclass(repr=False)
class ONCVPSPLogDerivativeAnalysis(ONCVPSPEntry):
    """Class for the log derivative analysis block in an ONCVPSP input file."""

    epsh1: float
    epsh2: float
    depsh: float


@dataclass(repr=False)
class ONCVPSPOutputGrid(ONCVPSPEntry):
    """Class for the output grid block in an ONCVPSP input file."""

    rlmax: float
    drl: float


@dataclass
class ONCVPSPInput:
    """Class for the contents of an ONCVPSP input file.

    The :class:`ONCVPSPInput` class is a dataclass that helps a user interact with input files for ``oncvpsp.x``.

    Typically, a user will create an :class:`ONCVPSPInput` object from an input file using the :meth:`from_file` method,
    as follows ::

        from oncvpsp_tools import ONCVPSPInput
        inp = ONCVPSPInput.from_file("path/to/input.in")

    :class:`ONCVPSPInput` objects -- being a :class:`dataclass` -- have the same attributes as the input parameters
    (listed below). Use these to interact with the contents of the input file.

    :param atom: The atom block of the input file
    :type atom: :class:`ONCVPSPAtom`
    :param reference_configuration: The reference configuration block of the input file
    :type reference_configuration: :class:`ONCVPSPList[ONCVPSPConfigurationSubshell]`
    :param lmax: The lmax block of the input file
    :type lmax: int
    :param optimization: The optimization block of the input file
    :type optimization: :class:`ONCVPSPList[ONCVPSPOptimizationChannel]`
    :param local_potential: The local potential block of the input file
    :type local_potential: :class:`ONCVPSPLocalPotential`
    :param vkb_projectors: The VKB projectors block of the input file
    :type vkb_projectors: :class:`ONCVPSPList[ONCVPSPVKBProjector]`
    :param model_core_charge: The model core charge block of the input file
    :type model_core_charge: :class:`ONCVPSPModelCoreCharge`
    :param log_derivative_analysis: The log derivative analysis block of the input file
    :type log_derivative_analysis: :class:`ONCVPSPLogDerivativeAnalysis`
    :param output_grid: The output grid block of the input file
    :type output_grid: :class:`ONCVPSPOutputGrid`
    :param test_configurations: The test configurations block of the input file
    :type test_configurations: :class:`ONCVPSPList[ONCVPSPList[ONCVPSPConfigurationSubshell]]`

    """

    atom: ONCVPSPAtom
    reference_configuration: ONCVPSPList[ONCVPSPConfigurationSubshell]
    lmax: int
    optimization: ONCVPSPList[ONCVPSPOptimizationChannel]
    local_potential: ONCVPSPLocalPotential
    vkb_projectors: ONCVPSPList[ONCVPSPVKBProjector]
    model_core_charge: ONCVPSPModelCoreCharge
    log_derivative_analysis: ONCVPSPLogDerivativeAnalysis
    output_grid: ONCVPSPOutputGrid
    test_configurations: ONCVPSPList[ONCVPSPList[ONCVPSPConfigurationSubshell]]

    @classmethod
    def from_file(cls, filename: str):
        """Create an :class:`ONCVPSPInput` object from an ONCVPSP input file."""
        with open(filename, "r") as f:
            txt = f.read()
        return cls.from_str(txt)

    @classmethod
    def from_str(cls, txt: str):
        """Create an :class:`ONCVPSPInput` object from a string."""
        lines = [line.strip() for line in txt.split("\n")]

        content = [line for line in lines if not line.startswith("#") and line]

        # atom
        atom = ONCVPSPAtom(*[sanitize(v) for v in content[0].split()])

        # reference configuration
        ntot = atom.nc + atom.nv
        reference_configuration: ONCVPSPList[ONCVPSPConfigurationSubshell] = ONCVPSPList(
            [
                ONCVPSPConfigurationSubshell(*[sanitize(v) for v in line.split()])
                for line in content[1 : ntot + 1]
            ]
        )

        # lmax
        lmax = int(content[ntot + 1])

        # optimization
        istart = ntot + 2
        iend = istart + lmax + 1
        optimization: ONCVPSPList[ONCVPSPOptimizationChannel] = ONCVPSPList(
            [
                ONCVPSPOptimizationChannel(*[sanitize(v) for v in line.split()])
                for line in content[istart:iend]
            ]
        )

        # local potential
        local_potential = ONCVPSPLocalPotential(*[sanitize(v) for v in content[iend].split()])

        # VKB projectors
        istart = iend + 1
        iend = istart + lmax + 1
        vkb: ONCVPSPList[ONCVPSPVKBProjector] = ONCVPSPList(
            [
                ONCVPSPVKBProjector(*[sanitize(v) for v in line.split()])
                for line in content[istart:iend]
            ]
        )

        # model core charge
        mcc = ONCVPSPModelCoreCharge(*[sanitize(v) for v in content[iend].split()])
        iend += 1

        # log derivative analysis
        log_derivative_analysis = ONCVPSPLogDerivativeAnalysis(
            *[sanitize(v) for v in content[iend].split()]
        )
        iend += 1

        # output grid
        output_grid = ONCVPSPOutputGrid(*[sanitize(v) for v in content[iend].split()])
        iend += 1

        # test configurations
        ncvf = int(content[iend])
        iend += 1
        test_configs: ONCVPSPList[ONCVPSPList[ONCVPSPConfigurationSubshell]] = ONCVPSPList([])
        for _ in range(ncvf):
            nv = int(content[iend])
            istart = iend + 1
            iend = istart + nv
            test_configs.append(
                ONCVPSPList(
                    [
                        ONCVPSPConfigurationSubshell(*[sanitize(v) for v in line.split()])
                        for line in content[istart:iend]
                    ],
                )
            )

        return cls(
            atom,
            reference_configuration,
            lmax,
            optimization,
            local_potential,
            vkb,
            mcc,
            log_derivative_analysis,
            output_grid,
            test_configs,
        )

    def to_str(self) -> str:
        """Return the text representation of the ONCVPSP input file."""
        return "\n".join(
            [
                "# ATOM AND REFERENCE CONFIGURATION",
                self.atom.to_str(),
                self.reference_configuration.to_str(),
                "# PSEUDOPOTENTIAL AND OPTIMIZATION",
                "#   lmax",
                f"{self.lmax: >8}",
                self.optimization.to_str(),
                "# LOCAL POTENTIAL",
                self.local_potential.to_str(),
                "# VANDERBILT-KLEINMAN-BYLANDER PROJECTORS",
                self.vkb_projectors.to_str(),
                "# MODEL CORE CHARGE",
                self.model_core_charge.to_str(),
                "# LOG DERIVATIVE ANALYSIS",
                self.log_derivative_analysis.to_str(),
                "# OUTPUT GRID",
                self.output_grid.to_str(),
                "# TEST CONFIGURATIONS",
                "# ncnf",
                self.test_configurations.to_str(print_length=True),
            ]
        ).replace("\n\n", "\n")

    def to_file(self, filename: str):
        """Write the ONCVPSP input file to disk."""
        with open(filename, "w") as f:
            f.write(self.to_str())

    def run(self, oncvpsp_command="oncvpso.x"):
        """Run the ONCVPSP executable and return the output."""
        from oncvpsp_tools.output import ONCVPSPOutput

        # Write the input file
        self.to_file("tmp.oncvpsp.in")

        # Run oncvpsp.x
        with open("tmp.oncvpsp.in", "r") as input_file:
            result = subprocess.run(
                oncvpsp_command, stdin=input_file, capture_output=True, shell=True, text=True
            )

        # Parse and return the result
        try:
            return ONCVPSPOutput.from_str(result.stdout)
        except Exception:
            output_file = "tmp.oncvpsp.out"
            with open(output_file, "w") as f:
                f.write(result.stdout)
            raise ValueError(f"ONCVPSP failed; inspect the output ({output_file})")
