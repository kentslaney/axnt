"""axnt: Implicit state management for JAX functions and models."""

from .stateful import (
    cml_state_specs,
    implicit,
    managed,
    restores,
)

__version__ = "0.1.0"

__all__ = [
    "cml_state_specs",
    "implicit",
    "managed",
    "restores",
]

