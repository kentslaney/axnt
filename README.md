# axnt

Implicit state management for JAX functions, methods, and model pipelines.

`axnt` allows you to write natural, stateful computation blocks in JAX using `@restores` and `@implicit` decorators, automatically transforming global/local state mutations into pure, functional state-passing pipelines compatible with `@jax.jit`, custom tracers, and compilation boundaries.

---

## Installation

```bash
pip install axnt
```

---

## Key Features

- **`@restores(**defaults)`**: Declare state variables and default initial values. Variables are temporarily bound into module globals for the scope of the function and saved back into the state container upon return.
- **`@implicit(argname="state")`**: Wrap pure or JIT-compiled functions to automatically accept and return explicit state tuples/dictionaries.
- **`@managed` & `@managed("branch")`**: Compose stateful child functions within parent stateful functions (both flat and hierarchical namespaces) using `...` (Ellipsis) delegation.
- **JAX JIT & Tracer Boundary Safety**: Detects when JAX tracers escape or cross untracked function boundaries, raising actionable `NameError` diagnostics.

---

## Quick Start

### Basic Function State

```python
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
def step(x):
    return block(x)

# First run: state initialized automatically from defaults
state, out1 = step(None, 2.0)  # out1: 5.0, state.momentum: 3.0

# Subsequent runs: pass previous state
state, out2 = step(state, 3.0) # out2: 9.0, state.momentum: 6.0
```

### Dataclass & Method Usage

```python
from dataclasses import dataclass
import jax
from axnt import implicit, restores

@jax.tree_util.register_dataclass
@dataclass
class Model:
    @jax.jit(donate_argnames=["state"])
    @implicit("state")
    def forward(self, x):
        return block(x)

model = Model()
state, out = model.forward(None, 2.0)
state, out = Model.forward(model, state, 3.0)
```

### Nested & Hierarchical Composition

```python
from axnt import implicit, managed, restores

@managed("subsystem")
@jax.jit
@implicit
def sub_block(x):
    return block1(x)

@jax.jit
@implicit
def parent_pipeline(x):
    # Delegate state lookup/updates to subsystem branch
    return sub_block(..., x)
```

---

## State Safety & Design

### Preventing Branching Side-Effects (Single Restore per Trace)

In pure functional frameworks like JAX, transformations like `@jax.jit` trace computations into a pure computational graph. If the same named state variable were restored and mutated multiple times within the same execution trace, the functional execution order could diverge from Python's imperative evaluation order, causing lost updates or branching side-effects.

To guarantee functional correctness, `axnt` enforces a **single restore per state key per trace**. Attempting to restore the same key multiple times within one trace raises a `RuntimeError`.

### Eliminating State Threading ("Prop Drilling")

In standard JAX pipelines, keeping track of state across deep model hierarchies usually requires **state threading** (or "prop drilling"): manually accepting, passing, and returning state objects through every intermediate function layer, even when those layers only serve to orchestrate child blocks.

`axnt` eliminates this boilerplate:
- **Single Common Ancestor**: All stateful components within a pipeline share a single root boundary marked by `@implicit`.
- **Automatic State Routing**: Subcomponents declare their own local state dependencies with `@restores(...)`, while `@managed` / `@managed("branch")` automatically route the corresponding state slices through the call tree via `...` (Ellipsis) delegation.
- **Clean Signatures**: Intermediate functions remain completely oblivious to the internal state of their subcomponents.

### Model Export & Core ML State Mapping

When exporting models to Apple Core ML via `stablehlo-coreml`, `axnt` automates state mapping without leaking low-level argument indices into userspace:
- **`exported.initial_state` & `exported.defaults`**: Access declared default tensor initializations directly, without running a forward pass with dummy zeros.
- **`exported.cml_state_specs()`**: Automatically constructs the `states` mapping for `stablehlo_coreml.convert`, matching each state variable to its output position.


```python
from stablehlo_coreml import convert

mil_program = convert(
    hlo_module,
    minimum_deployment_target=ct.target.iOS18,
    states=exported.cml_state_specs(),
)
```






---

## Examples

Check the `examples/` directory:
- `examples/trial.py`: Standalone function and dataclass method use cases.
- `examples/nesting.py`: Hierarchical (`@managed("branch")`) and flat (`@managed`) state composition.
- `examples/error.py`: Trace boundary verification and missing state error diagnostics.
- `examples/export.py`: Experimental CoreML / MIL export integration reference.

---

## License

CC0
