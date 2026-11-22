import ctypes
import math

from . import dtype as dtype_module
from .dtype import DType
from .jit import JITFunction
from .render import render_statements
from .trace import BufferSpec

SMITH_NAMES = {
    "float32": ("float32",),
    "float16": ("float16", "half"),
    "bfloat16": ("bfloat16", "bfloat"),
    "int8": ("int8", "char"),
    "int16": ("int16", "short"),
    "int32": ("int32",),
    "int64": ("int64", "long"),
    "uint8": ("uint8", "uchar"),
    "uint16": ("uint16", "ushort"),
    "uint32": ("uint32",),
    "uint64": ("uint64", "ulong"),
    "bool": ("bool_",),
}
MLX_ATTRIBUTES = {"bool": "bool_"}


def _mlx():
    import mlx.core as mx
    return mx


def _mlx_name(dtype):
    return str(dtype).rsplit(".", 1)[-1]


def smith_dtype(mlx_dtype) -> DType:
    name = _mlx_name(mlx_dtype)
    for candidate in SMITH_NAMES.get(name, ()):
        found = getattr(dtype_module, candidate, None)
        if isinstance(found, DType):
            return found
    raise TypeError(f"mlx dtype {name} has no smith dtype yet")


def mlx_dtype(dtype: DType):
    mx = _mlx()
    found = getattr(mx, MLX_ATTRIBUTES.get(dtype.name, dtype.name), None)
    if found is None:
        raise TypeError(f"smith dtype {dtype.name} has no mlx dtype")
    return found


def mlx_leaf(node):
    if node.op == "argument":
        return f"{node.arg}[0]"
    if node.op == "buffer":
        return node.arg
    if node.op == "program_id":
        return f"int(threadgroup_position_in_grid.{'xyz'[node.arg]})"
    if node.op == "program_count":
        return f"int(threadgroups_per_grid.{'xyz'[node.arg]})"
    if node.op == "lane":
        return "int(thread_position_in_threadgroup.x)"
    raise ValueError(f"cannot render leaf {node.op}")


def buffer_from_mlx(array):
    from .compiler import Buffer
    mx = _mlx()
    dtype = smith_dtype(array.dtype)
    mx.eval(array)
    storage = array if dtype.ctype is not None else array.view(mx.uint16)
    raw = bytes(memoryview(storage))
    buffer = Buffer(dtype, array.size)
    ctypes.memmove(buffer.contents_ptr(), raw, buffer.nbytes)
    return buffer


def mlx_from_buffer(buffer, shape):
    from .compiler import synchronize
    mx = _mlx()
    synchronize()
    shape = tuple(shape)
    if math.prod(shape) != buffer.n:
        raise ValueError(f"shape {shape} does not match a buffer of {buffer.n} elements")
    raw = ctypes.string_at(buffer.contents_ptr(), buffer.nbytes)
    if buffer.dtype.ctype is None:
        array = mx.array(memoryview(raw).cast("H")).view(mlx_dtype(buffer.dtype))
    else:
        array = mx.array(memoryview(raw).cast(buffer.dtype.ctype._type_))
    return array.reshape(shape)


class MLXKernel:
    def __init__(self, jit_function):
        self.jit_function = jit_function
        self.__name__ = jit_function.__name__
        self.__doc__ = jit_function.__doc__
        self.kernels = {}
        self.bodies = {}

    def __call__(self, *args, **kwargs):
        raise TypeError(f"launch '{self.__name__}' with {self.__name__}[grid](...)")

    def __getitem__(self, grid):
        def launch(*args, init_value=None, **kwargs):
            return self._launch(grid, args, kwargs, init_value)
        return launch

    def _kernel(self, body, input_names, output_names, header=""):
        key = (body, input_names, output_names, header)
        if key not in self.kernels:
            self.kernels[key] = _mlx().fast.metal_kernel(
                name=self.__name__,
                input_names=list(input_names),
                output_names=list(output_names),
                source=body,
                header=header,
            )
        return self.kernels[key]

    def _launch(self, grid, args, kwargs, init_value):
        mx = _mlx()
        values, constexprs = self.jit_function._bind(args, kwargs)
        arrays = {name: value for name, value in values.items()
                  if name not in constexprs and isinstance(value, mx.array)}

        traced = {}
        for name, value in values.items():
            if name in constexprs:
                traced[name] = value
            elif name in arrays:
                traced[name] = BufferSpec(smith_dtype(value.dtype))
            elif type(value) in (int, float):
                traced[name] = value
            else:
                raise TypeError(f"unsupported kernel argument {name!r} of type {type(value).__name__}")
        program = self.jit_function._program(traced, constexprs)

        input_names, output_names = [], []
        for name in program.runtime_names:
            if name not in arrays:
                continue
            if name in program.loaded and name in program.stored:
                raise ValueError(f"buffer {name!r} is both loaded and stored: in/out buffers are not supported, "
                                 "pass separate arrays")
            if name in program.stored:
                output_names.append(name)
            elif name in program.loaded:
                input_names.append(name)
        if not output_names:
            raise ValueError(f"kernel {self.__name__!r} stores to no buffer")

        scalar_names = [name for name in program.runtime_names if name not in arrays]
        inputs = [arrays[name] for name in input_names]
        for name in scalar_names:
            scalar = values[name]
            inputs.append(mx.array([scalar], dtype=mx.int32 if type(scalar) is int else mx.float32))
        all_input_names = tuple(input_names + scalar_names)

        group = tuple(grid(constexprs) if callable(grid) else grid)
        group = group + (1,) * (3 - len(group))
        lanes = program.lanes
        if program not in self.bodies:
            self.bodies[program] = "\n".join(render_statements(program.graph, leaf=mlx_leaf))
        body = self.bodies[program]
        header = "#include <metal_simdgroup_matrix>\n" if program.uses_matrices else ""
        kernel = self._kernel(body, all_input_names, tuple(output_names), header)

        options = {} if init_value is None else {"init_value": init_value}
        results = kernel(
            inputs=inputs,
            grid=(group[0] * lanes, group[1], group[2]),
            threadgroup=(lanes, 1, 1),
            output_shapes=[arrays[name].shape for name in output_names],
            output_dtypes=[arrays[name].dtype for name in output_names],
            **options,
        )
        return results[0] if len(results) == 1 else tuple(results)


def kernel(jit_function) -> MLXKernel:
    if not isinstance(jit_function, JITFunction):
        raise TypeError("mlx_bridge.kernel expects a @smith.jit function")
    return MLXKernel(jit_function)
