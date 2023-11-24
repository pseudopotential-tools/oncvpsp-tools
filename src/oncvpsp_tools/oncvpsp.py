"""Classes for handling ONCVPSP input and output files."""

from collections import UserList
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, Iterable, List, Optional, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np

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

    def __init__(self, data, print_length=False):
        """Create an :class:`ONCVPSPList` object."""
        self.print_length = print_length
        super().__init__(data)

    @property
    def columns(self) -> str:
        """Return the column headers for the list."""
        if self.data:
            return self.data[0].columns
        else:
            return ""

    @property
    def content(self) -> str:
        """Return the content of the list."""
        out = ""
        if self.print_length:
            out = f"{len(self): >8}\n"
        out += "\n".join([d.content for d in self.data])
        return out

    def to_str(self) -> str:
        """Return the text representation of the list."""
        return f"{self.columns}\n{self.content}"


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

    The :class:`ONCVPSPInput` class is a dataclass that helps a user interact with input files for ``oncvpsp.x``, which
    can sometimes be found in the ``info`` block of ``UPF`` files.
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

        content = [line for line in lines if not line.startswith("#")]

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
            istart = iend + 1
            iend = istart + atom.nv
            test_configs.append(
                ONCVPSPList(
                    [
                        ONCVPSPConfigurationSubshell(*[sanitize(v) for v in line.split()])
                        for line in content[istart:iend]
                    ],
                    print_length=True,
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
                f"{len(self.test_configurations): >8}",
                self.test_configurations.to_str(),
            ]
        )

    def to_file(self, filename: str):
        """Write the ONCVPSP input file to disk."""
        with open(filename, "w") as f:
            f.write(self.to_str())


@dataclass
class ONCVPSPOutputData:
    """Generic class for storing data from an ONCVPSP output file."""

    x: np.ndarray
    y: np.ndarray
    xlabel: str = "radius ($a_0$)"
    info: Dict[str, Any] = field(default_factory=dict)

    def plot(self, ax=None, **kwargs):
        """Plot the data."""
        if ax is None:
            _, ax = plt.subplots()
        if "label" not in kwargs:
            kwargs["label"] = ", ".join([f"{k}={v}" for k, v in self.info.items()])
        if (
            "ls" not in kwargs
            and "linestyle" not in kwargs
            and self.info.get("kind", None) == "pseudo"
        ):
            kwargs["ls"] = "--"
        ax.plot(self.x, self.y, **kwargs)
        ax.set_xlabel(self.xlabel)
        ax.set_xlim([self.x.min(), self.x.max()])
        ax.legend()
        return ax

    @classmethod
    def from_str(cls, string: str, identifier: str, xcol: int, ycol: int, **kwargs):
        """Create an :class:`ONCVPSPOutputData` object from a string."""
        relevant_lines = [
            line.strip().split()
            for line in string.split("\n")
            if line.strip().startswith(identifier)
        ]

        x = np.array([float(line[xcol]) for line in relevant_lines])
        y = np.array([float(line[ycol]) for line in relevant_lines])

        return cls(x, y, **kwargs)

    @classmethod
    def from_file(cls, filename: Union[Path, str], identifier: str, xcol: int, ycol: int, **kwargs):
        """Create an :class:`ONCVPSPOutputData` object from a file."""
        filename = Path(filename)

        with open(filename, "r") as f:
            lines = f.read()

        return cls.from_str(lines, identifier, xcol, ycol, **kwargs)


class ONCVPSPOutputDataList(UserList, Generic[T]):
    """Generic class for a list of ONCVPSPOutputData objects, with a few extra functionalities."""

    label: str

    def __init__(self, data, label: str = ""):
        """Create an :class:`ONCVPSPOutputDataList` object."""
        super().__init__(data)
        self.label = label

    def plot(self, ax=None, kwargs_list: Optional[List[Dict[str, Any]]] = None, **kwargs):
        """Plot all the data in the list."""
        if kwargs_list is None:
            kwargs_list = [{} for _ in self.data]
        for i, (data, specific_kwargs) in enumerate(zip(self.data, kwargs_list)):
            # Make the colors match for entries that only differ by info['kind']
            if ax and "color" not in specific_kwargs and "color" not in kwargs:
                # Get the previous colors used and the matching info dictionaries
                colors = [line.get_color() for line in ax.get_lines()[-i:]]
                infos = [{k: v for k, v in d.info.items() if k != "kind"} for d in self.data[:i]]

                # Use the same color if the dictionaries match (ignoring the 'kind' key)
                for info, color in zip(infos, colors):
                    if info == {k: v for k, v in data.info.items() if k != "kind"}:
                        specific_kwargs["color"] = color
                        break

            ax = data.plot(ax, **specific_kwargs, **kwargs)
        ax.set_title(self.label)

        # Set xlimits to the largest range of x values
        ax.set_xlim([min([d.x.min() for d in self.data]), max([d.x.max() for d in self.data])])

        return ax

    @classmethod
    def from_str(
        cls,
        label: str,
        string: str,
        identifiers,
        xcol: int,
        ycols: Iterable[int],
        kwargs_list: Optional[List[Dict[str, Any]]] = None,
    ):
        """Create an :class:`ONCVPSPOutputDataList` object from a string."""
        if kwargs_list is None:
            kwargs_list = [{} for _ in identifiers]
        oncvlist = cls(
            [
                ONCVPSPOutputData.from_str(string, identifier, xcol, ycol, **kwargs)
                for identifier, ycol, kwargs in zip(identifiers, ycols, kwargs_list)
            ]
        )
        oncvlist.label = label
        return oncvlist


@dataclass
class ONCVPSPOutput:
    """Class for all of the contents of an ONCVPSP output file.

    The :class:`ONCVPSPOutput` class is a dataclass that helps a user interact with output files from ``oncvpsp.x``
    """

    content: str
    input: ONCVPSPInput
    semilocal_ion_pseudopotentials: ONCVPSPOutputDataList[ONCVPSPOutputData]
    local_pseudopotential: ONCVPSPOutputData
    charge_densities: ONCVPSPOutputDataList[ONCVPSPOutputData]
    wavefunctions: ONCVPSPOutputDataList[ONCVPSPOutputData]
    arctan_log_derivatives: ONCVPSPOutputDataList[ONCVPSPOutputData]
    projectors: ONCVPSPOutputDataList[ONCVPSPOutputData]
    energy_error: ONCVPSPOutputDataList[ONCVPSPOutputData]

    @classmethod
    def from_file(cls, filename: str):
        """Create an :class:`ONCVPSPOutput` object from an ONCVPSP output file."""
        with open(filename, "r") as f:
            content = f.read()

        splitcontent = content.split("\n")

        # ONCVPSP input
        istart = splitcontent.index("# ATOM AND REFERENCE CONFIGURATION")
        input = ONCVPSPInput.from_str("\n".join(splitcontent[istart:]))

        # Semilocal ion pseudopotentials
        slp_kwargs = [{"info": {"l": l}} for l in range(input.lmax + 1)]
        semilocal_ion_pseudopotentials = ONCVPSPOutputDataList.from_str(
            "semilocal ion pseudopotentials",
            content,
            ["!p" for _ in range(input.lmax + 1)],
            1,
            range(3, input.lmax + 4),
            slp_kwargs,
        )

        # Local pseudopotential
        local_pseudopotential = ONCVPSPOutputData.from_str(content, "!L", 1, 2)

        # Charge densities
        cd_kwargs = [{"info": {"rho": rho}} for rho in ["C", "M", "V"]]
        charge_densities = ONCVPSPOutputDataList.from_str(
            "charge densities", content, ["!r ", "!r ", "!r "], 1, [2, 3, 4], cd_kwargs
        )

        # Pseudo and real wavefunctions
        il_pairs = sorted(
            list(
                set(
                    [
                        line.strip().split()[1]
                        for line in splitcontent
                        if line.strip().startswith("&")
                    ]
                )
            )
        )
        kinds = ["full", "pseudo"]
        kwargs = [
            {"info": {"kind": kind, "i": int(il[0]), "l": int(il[1])}}
            for il in il_pairs
            for kind in kinds
        ]
        identifiers = ["&    " + il for il in il_pairs for _ in kinds]
        ycols = [kind_col for _ in range(len(il_pairs)) for kind_col in [3, 4]]
        wavefunctions = ONCVPSPOutputDataList.from_str(
            "wavefunctions", content, identifiers, 2, ycols, kwargs
        )

        # Arctan log derivatives
        identifiers = [f"!      {l}" for l in range(4) for kind in kinds]
        ycols = [kind_col for _ in range(4) for kind_col in [3, 4]]
        kwargs = [{"info": {"kind": kind, "l": l}} for l in range(4) for kind in kinds]
        arctan_log_derivatives = ONCVPSPOutputDataList.from_str(
            "arctan log derivatives", content, identifiers, 2, ycols, kwargs
        )

        # Projectors
        ls = [proj.l for proj in input.vkb_projectors for _ in range(proj.nproj)]
        identifiers = [f"!J     {l}" for l in ls]
        ycols = [x + 3 for proj in input.vkb_projectors for x in range(proj.nproj)]
        kwargs = [
            {"info": {"i": i, "l": proj.l}}
            for proj in input.vkb_projectors
            for i in range(proj.nproj)
        ]
        projectors = ONCVPSPOutputDataList.from_str(
            "projectors", content, identifiers, 2, ycols, kwargs
        )

        # Energy error per electron
        identifiers = [f"!C     {l}" for l in range(input.lmax + 1)]
        eepe_kwargs = [
            {"info": {"l": l}, "xlabel": "cutoff energy (Ha)"} for l in range(input.lmax + 1)
        ]
        eepe = ONCVPSPOutputDataList.from_str(
            "energy error per electron",
            content,
            identifiers,
            2,
            [3 for _ in identifiers],
            eepe_kwargs,
        )

        return cls(
            content,
            input,
            semilocal_ion_pseudopotentials,
            local_pseudopotential,
            charge_densities,
            wavefunctions,
            arctan_log_derivatives,
            projectors,
            eepe,
        )

    @property
    def upf(self) -> str:
        """Return the UPF part of the ONCVPSP output file."""
        flines = self.content.split("\n")

        [istart] = [flines.index(x) for x in flines if "<UPF" in x]
        [iend] = [flines.index(x) for x in flines if "</UPF" in x]
        return "\n".join(flines[istart : iend + 1])

    def to_upfdict(self):
        """Return the UPF part of the ONCVPSP output file as a UPFDict."""
        from upf_tools import UPFDict

        return UPFDict.from_str(self.upf)
