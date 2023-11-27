"""Classes for handling ONCVPSP output files."""

from collections import UserList
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, Iterable, List, Optional, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np

from .input import ONCVPSPInput

T = TypeVar("T")


@dataclass
class ONCVPSPOutputData:
    """Generic class for storing data from an ONCVPSP output file."""

    x: np.ndarray
    y: np.ndarray
    xlabel: str = "radius ($a_0$)"
    label: str = ""
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
        if self.label:
            ax.set_title(self.label)
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
        ax.legend()
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
    """Class for the contents of an ``oncvpsp.x`` output file.

    The :class:`ONCVPSPOutput` class is a dataclass that helps a user interact with output files from ``oncvpsp.x``.
    Typically, a user will not create a :class:`ONCVPSPOutput` object directly, but rather use the class method
    :meth:`from_file` as follows::

        from oncvpsp_tools import ONCVPSPOutput
        output = ONCVPSPOutput.from_file("path/to/output")

    :class:`ONCVPSPOutput` objects -- being a :class:`dataclass` -- have the same attributes as the input parameters
    (listed below). Use these to interact with the contents of the output file. For example, to plot the semilocal ion
    pseudopotentials::

        output.semilocal_ion_pseudopotentials.plot()

    :param content: the entire content of the output file
    :type content:  str
    :param input:   the input file used to generate the output
    :type input:    :class:`ONCVPSPInput`
    :param semilocal_ion_pseudopotentials: the semilocal ion pseudopotentials
    :type semilocal_ion_pseudopotentials:   :class:`ONCVPSPOutputDataList`
    :param local_pseudopotential: the local pseudopotential
    :type local_pseudopotential:   :class:`ONCVPSPOutputData`
    :param charge_densities: the charge densities
    :type charge_densities:   :class:`ONCVPSPOutputDataList`
    :param wavefunctions: the pseudoatomic wavefunctions
    :type wavefunctions:   :class:`ONCVPSPOutputDataList`
    :param arctan_log_derivatives: the arctan log derivatives
    :type arctan_log_derivatives:   :class:`ONCVPSPOutputDataList`
    :param projectors: the projectors
    :type projectors:   :class:`ONCVPSPOutputDataList`
    :param energy_error: the energy error
    :type energy_error:   :class:`ONCVPSPOutputDataList`
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
    def from_str(cls, content: str):
        """Create an :class:`ONCVPSPOutput` object from a string."""
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
        local_pseudopotential = ONCVPSPOutputData.from_str(
            content, "!L", 1, 2, label="local pseudopotential"
        )

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

    @classmethod
    def from_file(cls, filename: str):
        """Create an :class:`ONCVPSPOutput` object from an ONCVPSP output file."""
        with open(filename, "r") as f:
            content = f.read()

        return cls.from_str(content)

    def to_upf(self) -> str:
        """Return the UPF part of the ONCVPSP output file."""
        flines = self.content.split("\n")

        [istart] = [flines.index(x) for x in flines if "<UPF" in x]
        [iend] = [flines.index(x) for x in flines if "</UPF" in x]
        return "\n".join(flines[istart : iend + 1])
