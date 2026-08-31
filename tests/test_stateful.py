from dataclasses import dataclass
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from axnt import implicit, managed, restores, unwrap





class TestStatefulBasic(unittest.TestCase):
    def test_single_state_accumulation(self):
        @restores(momentum=jnp.ones(()))
        def block(x):
            global momentum
            momentum += x
            return x + momentum

        @implicit(argname="state")
        def step(x):
            return block(x)

        state, y1 = step(None, 2.0)
        # initial momentum = 1.0; momentum becomes 3.0; y1 = 2 + 3 = 5.0
        np.testing.assert_allclose(float(y1), 5.0)
        np.testing.assert_allclose(float(state.momentum), 3.0)

        state, y2 = step(state, 3.0)
        # momentum becomes 3.0 + 3.0 = 6.0; y2 = 3 + 6 = 9.0
        np.testing.assert_allclose(float(y2), 9.0)
        np.testing.assert_allclose(float(state.momentum), 6.0)

    def test_jit_compilation(self):
        @restores(count=jnp.zeros(()))
        def block(x):
            global count
            count += x
            return count

        @jax.jit(donate_argnames=["state"])
        @implicit("state")
        def step(x):
            return block(x)

        state, out1 = step(None, 5.0)
        np.testing.assert_allclose(float(out1), 5.0)

        state, out2 = step(state, 10.0)
        np.testing.assert_allclose(float(out2), 15.0)

    def test_dataclass_methods(self):
        @restores(acc=jnp.ones(()))
        def block(x):
            global acc
            acc += x
            return x + acc

        @jax.tree_util.register_dataclass
        @dataclass
        class Counter:
            scale: float = 2.0

            @jax.jit(donate_argnames=["state"])
            @implicit("state")
            def forward(self, x):
                return block(x) * self.scale

        c = Counter(scale=2.0)
        state, y1 = c.forward(None, 2.0)
        # acc: 1 + 2 = 3; (2 + 3) * 2 = 10
        np.testing.assert_allclose(float(y1), 10.0)

        state, y2 = Counter.forward(c, state, 3.0)
        # acc: 3 + 3 = 6; (3 + 6) * 2 = 18
        np.testing.assert_allclose(float(y2), 18.0)


class TestStatefulComposition(unittest.TestCase):
    def test_nested_and_flat_managed(self):
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
            return block1(x)

        @managed
        @jax.jit
        @implicit
        def flat(x):
            return block2(x)

        @jax.jit
        @implicit
        def exported(x):
            res = nested(..., x)
            res += flat(..., x)
            res += block3(x)
            return res

        output = [None] * 5
        state, output[0] = exported(None, 2.0)
        state, output[1] = exported(state, 3.0)
        state, output[2] = flat(state, 2.0)
        state, output[3] = flat(None, 3.0)
        state, output[4] = exported(state, 4.0)

        results = [float(x) for x in output]
        expected = [13.0, 25.0, 9.0, 6.0, 28.0]
        np.testing.assert_allclose(results, expected)


class TestStatefulErrors(unittest.TestCase):
    def test_missing_trace_boundary_raises_name_error(self):
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

        with self.assertRaises(NameError) as ctx:
            exported(None, 2.0)

        self.assertIn("missing 'state'", str(ctx.exception))

    def test_multiple_restores_raises_runtime_error(self):
        @restores(count=jnp.zeros(()))
        def block_a(x):
            global count
            count += x
            return count

        @restores(count=jnp.zeros(()))
        def block_b(x):
            global count
            count += x
            return count

        @implicit("state")
        def step(x):
            return block_a(x) + block_b(x)

        with self.assertRaises(RuntimeError) as ctx:
            step(None, 1.0)

        self.assertIn("already restored in this trace", str(ctx.exception))


class TestStateSpecsAndDefaults(unittest.TestCase):
    def test_defaults_and_initial_state(self):
        @restores(momentum=jnp.ones((), dtype=jnp.float16), bias=jnp.zeros((), dtype=jnp.float32))
        def block(x):
            global momentum, bias
            momentum += x
            bias += 1.0
            return momentum + bias

        @implicit("state")
        def step(x):
            return block(x)

        # Before any step or after dry shape eval, initial_state reflects declared defaults
        _ = jax.eval_shape(step, None, jnp.zeros(()))
        init = step.initial_state
        self.assertIsNotNone(init)
        np.testing.assert_allclose(float(init.momentum), 1.0)
        np.testing.assert_allclose(float(init.bias), 0.0)

        # Mock StateSpec for testing state_specs mapping
        class MockStateSpec:
            def __init__(self, output=None, name=None):
                self.output = output
                self.name = name

        # Flat dictionary mapping for single-function export
        specs = step.cml_state_specs(StateSpec=MockStateSpec)
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].name, "momentum")
        self.assertEqual(specs[0].output, 0)
        self.assertEqual(specs[1].name, "bias")
        self.assertEqual(specs[1].output, 1)

        # Multi-function export with specified function name
        named_specs = step.cml_state_specs(StateSpec=MockStateSpec, function_name="main")
        self.assertIn("main", named_specs)
        self.assertEqual(named_specs["main"][0].name, "momentum")


class TestBranchStateSpecs(unittest.TestCase):
    """State declared through `@managed("branch")` has to export like it runs."""

    @staticmethod
    def leaf_paths(tree):
        paths, _ = jax.tree_util.tree_flatten_with_path(tree)
        return [jax.tree_util.keystr(path) for path, _ in paths]

    @staticmethod
    def pipeline():
        @restores(nested1=jnp.ones(()))
        def inner1(x):
            global nested1
            nested1 += x
            return nested1

        @restores(nested2=jnp.zeros(()))
        def inner2(x):
            global nested2
            nested2 += x
            return nested2

        @restores(merged=jnp.zeros(()))
        def sibling(x):
            global merged
            merged += x
            return merged

        # `@managed` outside `@jax.jit` is the documented composition order, and
        # the branch holds more than one key so field and leaf counts differ.
        @managed("branch")
        @jax.jit
        @implicit
        def sub(x):
            return inner1(x) + inner2(x)

        @implicit
        def root(x):
            return sub(..., x) + sibling(x)

        return root

    def test_defaults_reach_through_jit(self):
        root = self.pipeline()
        state, _ = root(None, 1.0)

        # `@managed` sees a PjitFunction, not the Decorator holding `defaults`;
        # reading only the outermost object contributes nothing and the specs
        # come back empty with no error.
        init = root.initial_state
        self.assertIsNotNone(init)
        self.assertIn("branch", init._fields)
        self.assertEqual(len(root.state_leaves()), 3)

    def test_initial_state_matches_running_state(self):
        root = self.pipeline()
        state, _ = root(None, 1.0)

        # Same leaves in the same order, or the spec numbering below describes
        # a state layout the traced function does not produce.
        self.assertEqual(
            self.leaf_paths(root.initial_state), self.leaf_paths(state)
        )
        self.assertEqual(
            self.leaf_paths(state), [".branch.nested1", ".branch.nested2", ".merged"]
        )

    def test_specs_numbered_by_leaf_not_field(self):
        class MockStateSpec:
            def __init__(self, output=None, name=None):
                self.output = output
                self.name = name

        root = self.pipeline()
        _ = root(None, 1.0)

        # Two top-level fields over three leaves: numbering by field would bind
        # "merged" to output 1, which is really the branch's second key.
        self.assertEqual(len(root.initial_state._fields), 2)
        specs = root.cml_state_specs(StateSpec=MockStateSpec, warn_non_zero=False)
        self.assertEqual(
            [(specs[i].output, specs[i].name) for i in sorted(specs)],
            [(0, "nested1"), (1, "nested2"), (2, "merged")],
        )

    def test_check_non_zero_defaults_flattens_branches(self):
        root = self.pipeline()
        _ = root(None, 1.0)

        # A branch contributes a mapping rather than an array; calling
        # jnp.asarray on it raises, and cml_state_specs does so by default.
        non_zero = root.check_non_zero_defaults()
        self.assertEqual(set(non_zero), {"nested1"})
        np.testing.assert_allclose(float(non_zero["nested1"]), 1.0)

    def test_flat_managed_unchanged(self):
        @restores(only=jnp.ones(()))
        def block(x):
            global only
            only += x
            return only

        @managed
        @jax.jit
        @implicit
        def child(x):
            return block(x)

        @implicit
        def root(x):
            return child(..., x)

        state, _ = root(None, 1.0)
        self.assertEqual(self.leaf_paths(root.initial_state), [".only"])
        self.assertEqual(self.leaf_paths(state), [".only"])
        self.assertEqual([n for n, _ in root.state_leaves()], ["only"])


class TestExportBoundary(unittest.TestCase):
    """Constraints that only show up on the way to Core ML."""

    def test_unwrap_reaches_the_boundary_through_jit(self):
        @restores(carried=jnp.ones(()))
        def block(x):
            global carried
            carried += x
            return carried

        @jax.jit
        @implicit
        def step(x):
            return block(x)

        state, _ = step(None, 1.0)

        # `@jax.jit` returns a PjitFunction, which forwards neither attribute
        # access nor the state accessors behind it.
        self.assertFalse(hasattr(step, "initial_state"))
        boundary = unwrap(step)
        self.assertIsNotNone(boundary.initial_state)
        self.assertEqual([n for n, _ in boundary.state_leaves()], ["carried"])

        with self.assertRaises(TypeError):
            unwrap(lambda x: x)

    def test_non_float_state_warns(self):
        @restores(cursor=jnp.zeros((), dtype=jnp.int32), value=jnp.zeros(()))
        def block(x):
            global cursor, value
            cursor = cursor + 1
            value = value + x
            return value

        @implicit
        def step(x):
            return block(x)

        _ = jax.eval_shape(step, None, jnp.zeros(()))

        # Core ML states are floating point; an int32 state is rejected at
        # conversion, so it is worth saying so before the conversion runs.
        self.assertEqual(set(step.check_non_float_defaults()), {"cursor"})
        with self.assertWarns(UserWarning) as caught:
            step.cml_state_specs(StateSpec=lambda output, name: (output, name))
        self.assertTrue(
            any("floating point" in str(w.message) for w in caught.warnings)
        )

    def test_written_but_unread_state_needs_keep_unused(self):
        from jax import export
        from jax._src.interpreters import mlir as jax_mlir
        from jax._src.lib.mlir import ir

        @restores(read_and_written=jnp.zeros((4, 2)), written_only=jnp.zeros((3, 2)))
        def block(x):
            global read_and_written, written_only
            written_only = x + jnp.zeros((3, 2))       # never read: dead argument
            read_and_written = read_and_written + 1.0
            return jnp.sum(read_and_written)

        @implicit
        def step(x):
            return block(x)

        shape, _ = jax.eval_shape(step, None, jnp.zeros(()))

        def hlo_arity(keep_unused):
            exported = export.export(jax.jit(step, keep_unused=keep_unused))(
                shape, jnp.zeros(())
            )
            module = ir.Module.parse(
                exported.mlir_module(), context=jax_mlir.make_ir_context()
            )
            op = next(
                o for o in module.body.operations
                if "main" in str(getattr(o, "name", ""))
            )
            return len(op.body.blocks[0].arguments), len(exported.in_avals)

        # Without keep_unused, jit drops the dead state input from the lowered
        # signature while in_avals still reports it, so the state inputs stop
        # lining up with the outputs cml_state_specs numbers.
        dropped, declared = hlo_arity(False)
        self.assertEqual(declared, 3)
        self.assertEqual(dropped, 2)

        kept, declared = hlo_arity(True)
        self.assertEqual(kept, declared)


if __name__ == "__main__":
    unittest.main()





