"""Miscellaneous utilities for oncvpsp-tools."""

import json
from typing import Any


def sanitize(value: str) -> Any:
    """Convert an arbitrary string to an int/float/bool if it appears to be one of these."""
    try:
        value = json.loads(value)
    except json.decoder.JSONDecodeError:
        pass
    return value
