import functools
import inspect
from collections import namedtuple
from types import MethodType

import jax
import jax.extend.core

jax.tree_util.register_static(type(Ellipsis))


class Context:
    scope = None

    def __init__(self, name, state):
        self.closure = {} if state is None else state
        self.name = name

    def __enter__(self):
        self.parent = __class__.scope
        if __class__.scope is not None:
            self.locked = __class__.scope.locked
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
        trace = jax.extend.core.find_top_trace(())
        for k, v in self._asdict().items():
            if isinstance(v, jax.core.Tracer):
                if v._trace != trace:
                    src = v._trace.frame.debug_info.func_src_info
                    raise NameError(
                        f"implicit '{k}' type {v} was produced by a trace "
                        f"missing '{self.name}' for {src}"
                    )
        __class__.scope = self.parent


def restores(**contained):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*a, **kw):
            assert Context.scope is not None, "missing boundary for implicits"
            namespace = f.__globals__
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
            result = f(*a, **kw)
            for k in contained:
                Context.scope[k] = namespace[k].astype(contained[k].dtype)
                del namespace[k]
            return result

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
            return result if scope is None else (scope.serializable, result)


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
