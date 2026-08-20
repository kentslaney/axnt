"""Experimental CoreML Export for Stateful Models using axnt.

Note:
    stablehlo-coreml interface for stateful models has changed in recent releases,
    so this export script is preserved as a reference implementation for CoreML /
    MIL stateful translation context.
"""

import jax
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
from jax import export
import jax.numpy as jnp

from axnt import implicit, restores

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from stablehlo_coreml.ops_register import register_stablehlo_op
    from stablehlo_coreml.converter import StableHloConverter, DEFAULT_HLO_PIPELINE, register_optimizations
    from stablehlo_coreml.translation_context import TranslationContext
    from jaxlib.mlir.dialects.stablehlo import CustomCallOp
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


if _COREML_AVAILABLE:
    class MilInjector(StableHloConverter):
        def process_block(self, context, block):
            self.process_block = super().process_block
            return list(self.patch(*super().process_block(context, block)))

        def patch(self, *outputs):
            return outputs


    # TODO: support for multiple functions (process_block hook breaks and fn names)
    class StatefulIO(MilInjector):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.internal_io = {}
            self.external_io = {}

        @register_stablehlo_op
        def op_custom_call(self, context: TranslationContext, op: CustomCallOp):
            call_target = op.call_target_name.value
            if call_target == "ffi_read":
                placeholder = op.inputs[0]
                hlo_func = placeholder.owner.owner
                with hlo_func.context:
                    attrs = hlo_func.arg_attrs[placeholder.arg_number]
                mapping = int(attrs['tf.aliasing_output'])
                assert placeholder.arg_number == mapping
                key = op.attributes["backend_config"].value
                arg = placeholder.get_name()
                state = context[arg]
                self.internal_io[mapping] = state
                self.external_io[key] = arg
                context.add_result(op.result, mb.read_state(input=state))
            else:
                return super().op_custom_call(context, op)

        def patch(self, *a):
            for k, v in self.internal_io.items():
                mb.coreml_update_state(state=v, value=a[k])
            return [x for i, x in enumerate(a) if i not in self.internal_io]


def export_demo():
    if not _COREML_AVAILABLE:
        print("stablehlo_coreml or coremltools not available in current environment.")
        return

    context = jax_mlir.make_ir_context()
    input_shapes = (jnp.zeros(()),)
    state_shape = exported(..., *input_shapes)[0]
    jax_exported = export.export(
        jax.jit(exported, donate_argnames=["state"]),
        disabled_checks=[
            export.DisabledSafetyCheck.custom_call("ffi_read")
        ],
    )(state_shape, *input_shapes)
    hlo_module = ir.Module.parse(jax_exported.mlir_module(), context=context)

    print("StableHLO Module:\n", hlo_module)

    converter = StatefulIO(opset_version=ct.target.iOS18)
    mil_program = converter.convert(hlo_module)
    print("MIL Program:\n", mil_program)

    # TODO: rename StableHLO-generated arguments
    # TODO: StateType doesn't support a default value
    # PyTorch's register_buffer initialization is ignored and make_state gives zeros
    # add/subtract on read/write and warn on non-zero offset to consider using
    #   write_state
    print("External IO:", converter.external_io, "Defaults:", exported.defaults)

    register_optimizations()
    cml_model = ct.convert(
        mil_program,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        pass_pipeline=DEFAULT_HLO_PIPELINE,
    )
    print("CoreML Model:\n", cml_model)


if __name__ == "__main__":
    export_demo()
