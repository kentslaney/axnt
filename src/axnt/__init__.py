"""axnt: Implicit state management for JAX functions and models."""

from .stateful import (
    implicit,
    managed,
    restores,
    unwrap,
)

__version__ = "0.1.0"

__all__ = [
    "implicit",
    "managed",
    "restores",
    "unwrap",
]
