"""axnt: Implicit state management for JAX functions and models."""

from .stateful import (
    Context,
    Decorator,
    implicit,
    managed,
    restores,
)

__version__ = "0.1.0"

__all__ = [
    "Context",
    "Decorator",
    "implicit",
    "managed",
    "restores",
]
