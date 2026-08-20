from dataclasses import dataclass

import jax
import jax.numpy as jnp
from axnt import implicit, restores

momentum: jax.Array


@restores(momentum=jnp.ones(()))
def block(x):
    global momentum
    momentum += x
    return x + momentum


@jax.jit(donate_argnames=["state"])
@implicit(argname="state")
def exported(x):
    return block(x)


@jax.tree_util.register_dataclass
@dataclass
class Model:
    @jax.jit(donate_argnames=["state"])
    @implicit("state")
    def exported(self, x):
        assert self.__class__ == __class__, f"{self.__class__} != {__class__}"
        return block(x)


def main():
    model = Model()
    output = [None] * 5

    state, output[0] = exported(None, 2)
    state, output[1] = exported(state, 3)
    state, output[2] = model.exported(state, 2)
    state, output[3] = Model.exported(model, state, 3)
    state, output[4] = Model.exported(self=model, state=state, x=4)

    print("Output:", jnp.asarray(output))
    return state, jnp.asarray(output)


if __name__ == "__main__":
    main()
