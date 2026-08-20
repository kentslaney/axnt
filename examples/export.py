"""CoreML Export for Stateful Models using axnt and stablehlo-coreml.

Demonstrates exporting an axnt stateful JAX model to Apple Core ML format
using the `states` mapping and `StateSpec` interface supported by stablehlo-coreml.
"""

import warnings
import jax
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
from jax import export
import jax.numpy as jnp

from axnt import implicit, restores, state_specs

try:
    import coremltools as ct
    from stablehlo_coreml import DEFAULT_HLO_PIPELINE, StateSpec, convert
    _COREML_AVAILABLE = True
except ImportError:
    _COREML_AVAILABLE = False


@restores(momentum1=jnp.ones((), dtype=jnp.float16))
def block1(x):
    global momentum1
    momentum1 += x
    return x + momentum1


@restores(momentum2=jnp.zeros((), dtype=jnp.float16))
def block2(x):
    global momentum2
    momentum2 += x
    return x + momentum2


@implicit
def exported(x):
    return block1(x) + block2(x)


def export_demo():
    input_shapes = (jnp.zeros((), dtype=jnp.float32),)

    # 1. Obtain state shape from the state structure after the first call
    state_shape, _ = jax.eval_shape(exported, None, *input_shapes)
    print("State Shape after first call:", state_shape)

    # 2. Export JAX function to StableHLO module
    # JAX exports (state, *inputs) -> (new_state, result)
    context = jax_mlir.make_ir_context()
    jax_exported = export.export(jax.jit(exported))(state_shape, *input_shapes)
    hlo_module = ir.Module.parse(jax_exported.mlir_module(), context=context)
    print("\nStableHLO Module:\n", hlo_module)


    if not _COREML_AVAILABLE:
        print("\nNote: Install `stablehlo-coreml` and `coremltools` to run Core ML conversion:")
        print("  pip install stablehlo-coreml coremltools\n")
        return

    # 3. Convert StableHLO to MIL with state mapping generated automatically by axnt
    # Argument numbers and indices are handled internally without leaking into userspace
    mil_program = convert(
        hlo_module,
        minimum_deployment_target=ct.target.iOS18,
        states=exported.state_specs(StateSpec),
    )
    print("\nMIL Program:\n", mil_program)

    # 4. Convert MIL program to Core ML model
    cml_model = ct.convert(
        mil_program,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        pass_pipeline=DEFAULT_HLO_PIPELINE,
    )
    print("\nCore ML Model:\n", cml_model)

    # 5. Inference with in-place Core ML state
    state = cml_model.make_state()
    # Write any non-zero initial values to the Core ML state
    for k, v in exported.check_non_zero_defaults().items():
        state.write_state(k, float(v))


    y1 = cml_model.predict({"x": 2.0}, state=state)
    print("Step 1 Output:", y1, "Momentum1 state:", state.read_state("momentum1"))

    y2 = cml_model.predict({"x": 3.0}, state=state)
    print("Step 2 Output:", y2, "Momentum1 state:", state.read_state("momentum1"))


if __name__ == "__main__":
    export_demo()
