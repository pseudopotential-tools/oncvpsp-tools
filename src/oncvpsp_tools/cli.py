# -*- coding: utf-8 -*-

"""Command line interface for :mod:`oncvpsp_tools`.

Why does this file exist, and why not put this in ``__main__``? You might be tempted to import things from ``__main__``
later, but that will cause problems--the code will get executed twice:

- When you run ``python3 -m oncvpsp_tools`` python will execute``__main__.py`` as a script.
  That means there won't be any ``oncvpsp_tools.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
  there's no ``oncvpsp_tools.__main__`` in ``sys.modules``.

.. seealso:: https://click.palletsprojects.com/en/8.1.x/setuptools/#setuptools-integration
"""

import logging

import click
import matplotlib.pyplot as plt

__all__ = [
    "main",
]

logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def main():
    """CLI for oncvpsp_tools."""


if __name__ == "__main__":
    main()

valid_toplots = [
    "arctan_log_derivatives",
    "charge_densities",
    "energy_error",
    "local_pseudopotential",
    "projectors",
    "semilocal_ion_pseudopotentials",
    "wavefunctions",
]


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("toplot", type=click.Choice(["all"] + valid_toplots), default="all")
@click.option(
    "--tofile", is_flag=True, default=False, help="Write to file instead of displaying onscreen"
)
def plot(filename, toplot, tofile):
    """Plot the contents of the provided ONCVPSP output file."""
    if toplot == "all":
        toplot = valid_toplots
    else:
        toplot = [toplot]

    from oncvpsp_tools import ONCVPSPOutput

    out = ONCVPSPOutput.from_file(filename)

    for name in toplot:
        obj = getattr(out, name)
        obj.plot()
        if tofile:
            plt.savefig(f"{filename}_{name}.png", format="png")
        else:
            plt.show()
