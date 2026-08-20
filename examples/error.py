"""Demonstration of JAX trace boundary detection and error handling in axnt.

When a stateful function using @restores is nested inside an unmanaged trace
boundary (such as @jax.jit without @implicit), axnt detects the missing state
boundary and raises a NameError indicating the origin trace and missing state.
"""

import jax
import jax.numpy as jnp
from axnt import implicit, restores


@restores(momentum=jnp.ones(()))
def block(x):
    global momentum
    momentum += x
    return x + momentum


@jax.jit
def unmanaged_wrapper(x):
    return block(x)


@jax.jit(donate_argnames=["state"])
@implicit(argname="state")
def exported(x):
    return unmanaged_wrapper(x)


def main():
    print("Testing unmanaged trace boundary detection (expecting NameError)...")
    try:
        exported(None, 2)
        print("Unexpected: call succeeded without error.")
    except NameError as e:
        print(f"Successfully caught expected NameError:\n  {e}")


if __name__ == "__main__":
    main()
