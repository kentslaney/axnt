from dataclasses import dataclass
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from axnt import cml_state_specs, implicit, managed, restores




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

        # Top-level helper function
        helper_specs = cml_state_specs(step, StateSpec=MockStateSpec)
        self.assertEqual(len(helper_specs), 2)


if __name__ == "__main__":
    unittest.main()




