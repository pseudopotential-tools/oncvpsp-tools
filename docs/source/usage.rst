Usage
=====

API
---

`oncvpsp-tools` implements the contents of `oncvpsp.x` input and output files as Python dataclasses. These are documented below

.. autoclass:: oncvpsp_tools.input.ONCVPSPInput
    :members:

.. autoclass:: oncvpsp_tools.output.ONCVPSPOutput
    :members:

CLI
---

`oncvpsp-tools` provides a command-line interface for plotting the contents of an ONCVPSP output file. For example::

    oncvpsp_tools plot /path/to/file

For more options, see `oncvpsp_tools plot --help`
