"""@smith.jit: launch a kernel over a grid, kernel[grid](*args, **constexprs).

With plain Python lists the kernel runs once per program in Python.
With Metal buffers the kernel is traced into Metal source, compiled, and run on the GPU.
"""
import contextlib
import ctypes
import inspect
import itertools
import sys

from . import viz
from .ops import _ctx, Pointer
from .trace import trace_kernel


class constexpr:
    """Annotation marker: `BLOCK: sth.constexpr = 256`."""


def _wrap(arg):
    if hasattr(arg, "__setitem__") and hasattr(arg, "__len__"):
        return Pointer(arg, 0)
    return arg


def _uses_metal(args, kwargs):
    if sys.platform != "darwin":
        return False
    from .compiler import Buffer
    return any(isinstance(value, Buffer) for value in (*args, *kwargs.values()))


@contextlib.contextmanager
def batch():
    if sys.platform != "darwin":
        yield None
        return
    from .compiler import batch as device_batch
    with device_batch() as active:
        yield active


def synchronize():
    if sys.platform == "darwin":
        from .compiler import synchronize as synchronize_device
        synchronize_device()


class JITFunction:
    def __init__(self, function):
        self.function = function
        self.signature = inspect.signature(function)
        self.programs = {}
        self.__name__ = function.__name__
        self.__doc__ = function.__doc__

    def __call__(self, *args, **kwargs):
        raise TypeError(f"launch '{self.__name__}' with {self.__name__}[grid](...)")

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return self._launch(grid, args, kwargs)
        return launch

    def _launch(self, grid, args, kwargs, fast_math=True):
        if _uses_metal(args, kwargs):
            return self._launch_metal(grid, args, kwargs, fast_math)
        g = tuple(grid(kwargs) if callable(grid) else grid)
        g = g + (1,) * (3 - len(g))
        wrapped = [_wrap(arg) for arg in args]
        _ctx["grid"] = g
        for pid in itertools.product(*(range(n) for n in g)):
            _ctx["pid"] = pid
            self.function(*wrapped, **kwargs)

    def _program(self, values, constexprs):
        key = tuple((name, value if name in constexprs else getattr(value, "dtype", type(value)))
                    for name, value in values.items())
        if key not in self.programs:
            with viz.span("trace", self.__name__) as active:
                self.programs[key] = trace_kernel(self.function, values, constexprs)
                active.key = (self.programs[key].name, self.programs[key].source)
        viz.record_program(self.programs[key])
        return self.programs[key]

    def _bind(self, args, kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        constexprs = {name: value for name, value in bound.arguments.items()
                      if self.signature.parameters[name].annotation is constexpr}
        return bound.arguments, constexprs

    def trace(self, *args, **kwargs):
        values, constexprs = self._bind(args, kwargs)
        return self._program(values, constexprs)

    def _launch_metal(self, grid, args, kwargs, fast_math=True):
        from .compiler import Buffer, compile_msl, launch
        values, constexprs = self._bind(args, kwargs)
        program = self._program(values, constexprs)
        kernel = compile_msl(program.source, program.name, fast_math)

        arguments = []
        for name in program.runtime_names:
            value = values[name]
            if isinstance(value, Buffer):
                arguments.append(value)
            else:
                arguments.append(ctypes.c_int32(value) if type(value) is int else ctypes.c_float(value))

        g = tuple(grid(constexprs) if callable(grid) else grid)
        g = g + (1,) * (3 - len(g))
        return launch(kernel, arguments, (g[0] * program.lanes, g[1], g[2]), (program.lanes, 1, 1))


def jit(function):
    return JITFunction(function)
