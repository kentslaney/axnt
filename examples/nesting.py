import jax
import jax.numpy as jnp
from axnt import implicit, managed, restores


@restores(momentum1=jnp.ones(()))
def block1(x):
    global momentum1
    momentum1 += x
    return x + momentum1


@restores(momentum2=jnp.zeros(()))
def block2(x):
    global momentum2
    momentum2 += x
    return x + momentum2


@restores(momentum3=jnp.zeros(()))
def block3(x):
    global momentum3
    momentum3 += x
    return x + momentum3


@managed("branch")
@jax.jit
@implicit
def nested(x):
    res = block1(x)
    return res


@managed
@jax.jit
@implicit
def flat(x):
    res = block2(x)
    return res


@jax.jit
@implicit
def exported(x):
    res = nested(..., x)
    res += flat(..., x)
    res += block3(x)
    return res


def main():
    output = [None] * 5

    state, output[0] = exported(None, 2)
    state, output[1] = exported(state, 3)
    state, output[2] = flat(state, 2)
    state, output[3] = flat(None, 3)
    state, output[4] = exported(state, 4)

    print("Output:", jnp.asarray(output))
    return state, jnp.asarray(output)


if __name__ == "__main__":
    main()
