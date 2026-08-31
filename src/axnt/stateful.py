import functools
import inspect
from collections import namedtuple
from types import MethodType

import jax
import jax.extend.core
import jax.numpy as jnp

jax.tree_util.register_static(type(Ellipsis))



class Context:
    scope = None

    def __init__(self, name, state):
        self.defaults = {}
        self.closure = {} if state is None else state
        self.name = name

    def __enter__(self):
        self.parent = __class__.scope
        if __class__.scope is not None:
            self.locked = set(__class__.scope.locked)
        else:
            self.locked = set()
        __class__.scope = self
        return self

    def __getitem__(self, key):
        return self.closure[key] if self.starting else getattr(self.closure, key)

    def __setitem__(self, key, value):
        if self.starting:
            self.closure[key] = value
        else:
            if hasattr(self.closure, "_fields") and key not in self.closure._fields:
                d = self.closure._asdict()
                d[key] = value
                self.closure = namedtuple(type(self.closure).__name__, d.keys())(**d)
            else:
                self.closure = self.closure._replace(**{key: value})

    def register(self, key, default):
        self.defaults[key] = default
        return default

    @functools.cached_property
    def starting(self):
        return isinstance(self.closure, dict)

    @property
    def serializable(self):
        if self.starting:
            return namedtuple(self.name, self.closure.keys())(**self.closure)
        return self.closure

    def _asdict(self):
        if self.starting:
            return self.closure
        return self.closure._asdict()

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                trace = jax.extend.core.find_top_trace(())
                for k, v in self._asdict().items():
                    if isinstance(v, jax.core.Tracer):
                        if v._trace != trace:
                            src = v._trace.frame.debug_info.func_src_info
                            raise NameError(
                                f"implicit '{k}' type {v} was produced by a trace "
                                f"missing '{self.name}' for {src}"
                            )
        finally:
            __class__.scope = self.parent


def _walk_wrapped(f):
    """Yield a callable and everything it wraps, via `__wrapped__`.

    `functools.wraps` sets `__wrapped__`, and `jax.jit` exposes it too, so this
    reaches the `Decorator` under a stack of transformations. Wrappers do not
    forward attribute access, so reading only the outermost object silently
    finds nothing.
    """
    seen = set()
    while f is not None and id(f) not in seen:
        seen.add(id(f))
        yield f
        f = getattr(f, "__wrapped__", None)


def unwrap(fn):
    """Return the `@implicit` boundary behind a wrapped callable.

    `@jax.jit` returns a `PjitFunction`, so `jitted.initial_state` raises
    `AttributeError` even though the state it describes is right there.
    `unwrap(jitted).initial_state` reaches it.
    """
    for f in _walk_wrapped(fn):
        if isinstance(f, Decorator):
            return f
    raise TypeError(f"{fn!r} is not an axnt @implicit function")


def _unwrap_defaults(f):
    """Reach a callable's declared defaults through jit and other wrappers.

    `@managed` sits outside `@jax.jit`, so the child it holds is a
    `PjitFunction` rather than the `Decorator` that owns `defaults`.
    """
    for wrapped in _walk_wrapped(f):
        defaults = getattr(wrapped, "defaults", None)
        if isinstance(defaults, dict):
            return defaults
    return None


def _leaf_name(key):
    """Name of a single pytree path entry, whatever key type produced it."""
    for attr in ("name", "key", "idx"):
        if hasattr(key, attr):
            return str(getattr(key, attr))
    return str(key)


def restores(**contained):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*a, **kw):
            assert Context.scope is not None, "missing boundary for implicits"
            namespace = f.__globals__
            restored_keys = []
            try:
                for k in contained:
                    assert k not in namespace
                    if k in Context.scope.locked:
                        raise RuntimeError(
                            f"The state '{k}' was already restored in this trace. "
                            "Multiple restores cause branching side-effects in JAX."
                        )
                    Context.scope.locked.add(k)
                    is_missing = not Context.scope.starting and not hasattr(Context.scope.closure, k)
                    namespace[k] = (
                        Context.scope.register(k, contained[k])
                        if (Context.scope.starting or is_missing)
                        else Context.scope[k]
                    )
                    restored_keys.append(k)
                result = f(*a, **kw)
                for k in restored_keys:
                    Context.scope[k] = namespace[k].astype(contained[k].dtype)
                return result
            finally:
                for k in restored_keys:
                    if k in namespace:
                        del namespace[k]

        return wrapper

    return decorator


class Decorator:
    def __init__(self, f, argname):
        self.f, self.argname = f, argname
        functools.update_wrapper(self, f)
        self.prebound = (
            f.__qualname__.rsplit(".", 2)[-2:-1] not in ([], ["<locals>"])
        )

    @functools.partial(property, None)
    def prebound(self, value):
        sig = inspect.signature(self.f)
        params = list(sig.parameters.values())
        index = int(value)
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if (
                index < len(params)
                and params[index].kind == inspect.Parameter.POSITIONAL_ONLY
            )
            else inspect.Parameter.KEYWORD_ONLY
            if (index > 0 and params[0].kind == inspect.Parameter.KEYWORD_ONLY)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        params.insert(index, inspect.Parameter(self.argname, kind))
        self.__signature__ = sig.replace(parameters=params)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return MethodType(self, instance)

    def __call__(*args, **kwargs):
        self, args = args[0], args[1:]
        sig = self.__signature__
        start = next(iter(sig.parameters.values())).kind
        bound = sig.bind(*args, **kwargs)
        if start == inspect.Parameter.POSITIONAL_ONLY:
            index = list(sig.parameters).index(self.argname)
            state = bound.args[index]
            args = bound.args[:index] + bound.args[index + 1 :]
        else:
            state, args = bound.arguments.pop(self.argname), bound.args
        with Context(self.argname, state) as scope:
            result = self.f(*args, **bound.kwargs)
            if scope.defaults:
                self.defaults = scope.defaults
            return result if scope is None else (scope.serializable, result)

    @staticmethod
    def _nest(name, mapping):
        """Build a namedtuple from a defaults mapping, recursing into branches.

        `@managed("branch")` contributes the child's defaults as a nested
        mapping, and the running state holds a namedtuple in that position, so
        the declared state has to nest the same way or its leaves will not line
        up with the ones the traced function returns.
        """
        fields = {
            k: Decorator._nest(k, v) if isinstance(v, dict) else v
            for k, v in mapping.items()
        }
        return namedtuple(name, fields.keys())(**fields)

    @property
    def initial_state(self):
        """Construct the initial state namedtuple directly from declared default initializers."""
        if hasattr(self, "defaults"):
            return self._nest(self.argname, self.defaults)
        return None

    def state_leaves(self):
        """Declared state as (name, default) per exported output, in leaf order.

        `jax.export` emits one output per pytree leaf, so this — not the
        top-level fields — is the numbering Core ML states have to match.
        """
        init = self.initial_state
        if init is None:
            return []
        paths, _ = jax.tree_util.tree_flatten_with_path(init)
        return [(_leaf_name(path[-1]), leaf) for path, leaf in paths]

    def check_non_float_defaults(self):
        """State keys whose declared dtype Core ML cannot hold.

        Core ML states are floating point; an integer state is rejected at
        conversion, so counters and cursors have to be stored as floats.
        """
        return {
            name: jnp.asarray(default).dtype
            for name, default in self.state_leaves()
            if not jnp.issubdtype(jnp.asarray(default).dtype, jnp.floating)
        }

    def check_non_zero_defaults(self):
        """Check and return any state keys that have non-zero default initializations."""
        return {
            name: default for name, default in self.state_leaves()
            if not jnp.all(jnp.asarray(default) == 0)
        }

    def cml_state_specs(self, StateSpec=None, function_name=None, warn_non_zero=True):
        """Generate stablehlo-coreml states mapping without exposing argument numbers in userspace."""
        if StateSpec is None:
            from stablehlo_coreml import StateSpec


        if warn_non_zero:
            non_zero = self.check_non_zero_defaults()
            if non_zero:
                import warnings

                warnings.warn(
                    f"Non-zero state defaults detected for keys: {list(non_zero.keys())}. "
                    "Core ML allocates state tensors to zero by default via `make_state()`. "
                    "Ensure non-zero defaults are written with `state.write_state(key, val)` "
                    "prior to inference.",
                    UserWarning,
                    stacklevel=2,
                )

            non_float = self.check_non_float_defaults()
            if non_float:
                import warnings

                warnings.warn(
                    f"Non-floating-point state defaults detected: "
                    f"{ {k: str(v) for k, v in non_float.items()} }. "
                    "Core ML states must be floating point and conversion will "
                    "reject these; store counters and cursors as floats.",
                    UserWarning,
                    stacklevel=2,
                )

        specs = {
            i: StateSpec(output=i, name=name)
            for i, (name, _) in enumerate(self.state_leaves())
        }

        if function_name is not None:
            return {function_name: specs}
        return specs




def implicit(argname):
    def decorator(f):
        return Decorator(f, argname)

    if not isinstance(argname, str):
        return Decorator(argname, "state")
    return decorator


def managed(arg=None):
    branch = arg if isinstance(arg, str) else None

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*a, **kw):
            is_delegated = (
                (len(a) > 0 and a[0] is Ellipsis)
                or (kw.get("state") is Ellipsis)
            )
            if not is_delegated:
                return f(*a, **kw)

            scope = Context.scope
            if scope is None:
                return f(*a, **kw)

            if branch is not None:
                sub_state = (
                    scope.closure.get(branch, None)
                    if scope.starting
                    else getattr(scope.closure, branch, None)
                )
            else:
                sub_state = (
                    (scope.closure if scope.closure else None)
                    if scope.starting
                    else scope.closure
                )

            if len(a) > 0 and a[0] is Ellipsis:
                new_a = (sub_state,) + a[1:]
                new_kw = kw
            else:
                new_a = a
                new_kw = dict(kw)
                new_kw["state"] = sub_state

            out = f(*new_a, **new_kw)

            if isinstance(out, tuple) and len(out) == 2:
                new_sub_state, result = out
            else:
                return out

            if branch is not None:
                if scope.starting:
                    scope.closure[branch] = new_sub_state
                    sub_defaults = _unwrap_defaults(f)
                    if sub_defaults is not None:
                        scope.defaults[branch] = sub_defaults
                else:
                    if hasattr(scope.closure, "_fields") and branch not in scope.closure._fields:
                        d = scope.closure._asdict()
                        d[branch] = new_sub_state
                        scope.closure = namedtuple(type(scope.closure).__name__, d.keys())(**d)
                    else:
                        scope.closure = scope.closure._replace(**{branch: new_sub_state})
            else:
                if scope.starting:
                    if hasattr(new_sub_state, "_asdict"):
                        scope.closure.update(new_sub_state._asdict())
                    elif isinstance(new_sub_state, dict):
                        scope.closure.update(new_sub_state)
                    sub_defaults = _unwrap_defaults(f)
                    if isinstance(sub_defaults, dict):
                        scope.defaults.update(sub_defaults)
                else:
                    items = new_sub_state._asdict() if hasattr(new_sub_state, "_asdict") else new_sub_state
                    if hasattr(scope.closure, "_fields"):
                        d = scope.closure._asdict()
                        d.update(items)
                        scope.closure = namedtuple(type(scope.closure).__name__, d.keys())(**d)
                    else:
                        scope.closure = scope.closure._replace(**items)

            return result

        return wrapper

    if not isinstance(arg, str) and arg is not None:
        f = arg
        branch = None
        return decorator(f)
    return decorator



