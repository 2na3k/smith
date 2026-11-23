import contextlib
import ctypes
import ctypes.util
import functools
import sys
import time

from . import viz


class CompileError(RuntimeError):
    pass


_objc = ctypes.CDLL(ctypes.util.find_library("objc"))
_objc.objc_getClass.restype = ctypes.c_void_p
_objc.objc_getClass.argtypes = [ctypes.c_char_p]
_objc.sel_registerName.restype = ctypes.c_void_p
_objc.sel_registerName.argtypes = [ctypes.c_char_p]
_objc.objc_autoreleasePoolPush.restype = ctypes.c_void_p
_objc.objc_autoreleasePoolPush.argtypes = []
_objc.objc_autoreleasePoolPop.restype = None
_objc.objc_autoreleasePoolPop.argtypes = [ctypes.c_void_p]
_Metal = ctypes.CDLL(ctypes.util.find_library("Metal"))
_Metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
_Metal.MTLCreateSystemDefaultDevice.argtypes = []


@functools.cache
def _selector(name):
    return _objc.sel_registerName(name.encode())


@functools.cache
def _sender(restype, argtypes):
    return ctypes.cast(_objc.objc_msgSend, ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes))


def _send(obj, sel: str, *args, restype=ctypes.c_void_p, argtypes=()):
    return _sender(restype, tuple(argtypes))(obj, _selector(sel), *args)


def _release(handle) -> None:
    if handle and not sys.is_finalizing():
        _send(handle, "release", restype=None)


@contextlib.contextmanager
def _autorelease_pool():
    pool = _objc.objc_autoreleasePoolPush()
    try:
        yield
    finally:
        _objc.objc_autoreleasePoolPop(pool)


def _nsstring(s: str) -> int:
    cls = _objc.objc_getClass(b"NSString")
    return _send(cls, "stringWithUTF8String:", s.encode(), argtypes=(ctypes.c_char_p,))


def _from_nsstring(ns) -> str:
    return _send(ns, "UTF8String", restype=ctypes.c_char_p).decode()


def _error_text(err) -> str:
    return _from_nsstring(_send(err, "localizedDescription"))


@functools.cache
def metal_device():
    device = _Metal.MTLCreateSystemDefaultDevice()
    if not device:
        raise CompileError("no Metal device available (macOS with Metal required)")
    return device


class Kernel:
    def __init__(self, name: str, pipeline: int, source: str = None, compile_seconds: float = None):
        self.name = name
        self.pipeline = pipeline
        self.source = source
        self.compile_seconds = compile_seconds
        self.width = None
        self.limit = None

    def maximum_threads(self) -> int:
        if self.limit is None:
            self.limit = _send(self.pipeline, "maxTotalThreadsPerThreadgroup", restype=ctypes.c_ulong)
        return self.limit

    def execution_width(self) -> int:
        if self.width is None:
            self.width = _send(self.pipeline, "threadExecutionWidth", restype=ctypes.c_ulong)
        return self.width

    def __del__(self):
        _release(self.pipeline)

    def __repr__(self) -> str:
        return f"Kernel({self.name!r})"


_cache: dict[tuple[str, str], Kernel] = {}


def compile_msl(source: str, name: str) -> Kernel:
    key = (source, name)
    if key in _cache:
        viz.record_compile(name, source, _cache[key].compile_seconds)
        return _cache[key]

    device = metal_device()
    owned = []
    started = time.perf_counter()
    with viz.span("compile", name, (name, source)), _autorelease_pool():
        try:
            err = ctypes.c_void_p()
            library = _send(device, "newLibraryWithSource:options:error:", _nsstring(source), None, ctypes.byref(err),
                            argtypes=(ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)))
            if not library:
                raise CompileError(_error_text(err.value) if err.value else "newLibraryWithSource failed")
            owned.append(library)

            function = _send(library, "newFunctionWithName:", _nsstring(name), argtypes=(ctypes.c_void_p,))
            if not function:
                raise CompileError(f"no function named {name!r} in source")
            owned.append(function)

            err = ctypes.c_void_p()
            pipeline = _send(device, "newComputePipelineStateWithFunction:error:", function, ctypes.byref(err),
                             argtypes=(ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)))
            if not pipeline:
                raise CompileError(_error_text(err.value) if err.value else "pipeline creation failed")
        finally:
            for handle in owned:
                _release(handle)

    kernel = Kernel(name, pipeline, source, time.perf_counter() - started)
    _cache[key] = kernel
    viz.record_compile(name, source, kernel.compile_seconds)
    return kernel


MTLResourceStorageModeShared = 0


class MTLSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_ulong), ("height", ctypes.c_ulong), ("depth", ctypes.c_ulong)]


@functools.cache
def _queue():
    return _send(metal_device(), "newCommandQueue")


class Buffer:
    def __init__(self, dtype, n: int):
        self.handle = None
        self.dtype, self.n = dtype, n
        self.nbytes = dtype.nbytes(n)
        self.handle = _send(metal_device(), "newBufferWithLength:options:", self.nbytes,
                            MTLResourceStorageModeShared, argtypes=(ctypes.c_ulong, ctypes.c_ulong))
        if not self.handle:
            raise CompileError("newBufferWithLength failed")

    def __del__(self):
        _release(self.handle)

    @classmethod
    def from_list(cls, dtype, values):
        values = list(values)
        buffer = cls(dtype, len(values))
        raw = dtype.pack(values)
        ctypes.memmove(buffer.contents_ptr(), raw, len(raw))
        return buffer

    def contents_ptr(self) -> int:
        return _send(self.handle, "contents")

    def to_list(self) -> list:
        synchronize()
        return self.dtype.unpack(ctypes.string_at(self.contents_ptr(), self.nbytes), self.n)


def _triple(size):
    size = (size,) if isinstance(size, int) else tuple(size)
    return size + (1,) * (3 - len(size))


_inflight: list = []
_observers: list = []
_batch = None

STATUS_COMPLETED = 4
STATUS_ERROR = 5
PRUNE_THRESHOLD = 32


class Submission:
    def __init__(self, owner=None):
        self.owner = owner
        self.command_buffer = _send(_queue(), "commandBuffer")
        self.encoder = _send(self.command_buffer, "computeCommandEncoder")
        _send(self.command_buffer, "retain", restype=None)
        _send(self.encoder, "retain", restype=None)
        self.launches = []
        self.committed = False
        self.pipeline = None
        self.bindings = {}

    def encode(self, kernel, arguments, grid, group):
        encoder = self.encoder
        if self.pipeline is not kernel.pipeline:
            _send(encoder, "setComputePipelineState:", kernel.pipeline, restype=None, argtypes=(ctypes.c_void_p,))
            self.pipeline = kernel.pipeline
        for i, argument in enumerate(arguments):
            if isinstance(argument, Buffer):
                if self.bindings.get(i) is not argument:
                    _send(encoder, "setBuffer:offset:atIndex:", argument.handle, 0, i, restype=None,
                          argtypes=(ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong))
                    self.bindings[i] = argument
            else:
                raw = bytes(argument)
                if self.bindings.get(i) != raw:
                    _send(encoder, "setBytes:length:atIndex:", ctypes.addressof(argument), ctypes.sizeof(argument), i,
                          restype=None, argtypes=(ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong))
                    self.bindings[i] = raw
        _send(encoder, "dispatchThreads:threadsPerThreadgroup:", MTLSize(*grid), MTLSize(*group),
              restype=None, argtypes=(MTLSize, MTLSize))

    def commit(self):
        if self.committed:
            return
        self.committed = True
        if self.owner is not None and self.owner.current is self:
            self.owner.current = None
        _send(self.encoder, "endEncoding", restype=None)
        _send(self.command_buffer, "commit", restype=None)
        now = time.perf_counter()
        for event in self.launches:
            event.submitted = now
        _release(self.encoder)
        _release(self.command_buffer)
        self.bindings.clear()


class Batch:
    def __init__(self):
        self.current = None
        self.submissions = []

    def flush(self):
        if self.current is not None:
            self.current.commit()


class Launch:
    def __init__(self, submission, kernel: Kernel, grid, group, encode_started=None):
        self.command_buffer = submission.command_buffer
        _send(self.command_buffer, "retain", restype=None)
        self.submission = submission
        self.encode_started = encode_started
        self.encode_ended = None
        self.kernel = kernel
        self.grid = grid
        self.group = group
        self.submitted = None
        self.error_text = None
        self.waited = False

    def __del__(self):
        _release(self.command_buffer)

    @property
    def batched(self) -> bool:
        return len(self.submission.launches) > 1

    def _status(self) -> int:
        return _send(self.command_buffer, "status", restype=ctypes.c_ulong)

    @property
    def done(self) -> bool:
        return self.waited or (self.submission.committed and self._status() >= STATUS_COMPLETED)

    @property
    def failed(self) -> bool:
        return self.error_text is not None or (self.submission.committed and self._status() == STATUS_ERROR)

    def wait(self) -> "Launch":
        if not self.waited:
            self.submission.commit()
            if self._status() < STATUS_COMPLETED:
                with viz.span("wait", self.kernel.name, (self.kernel.name, self.kernel.source), self):
                    _send(self.command_buffer, "waitUntilCompleted", restype=None)
            else:
                _send(self.command_buffer, "waitUntilCompleted", restype=None)
            self.waited = True
            with _autorelease_pool():
                error = _send(self.command_buffer, "error")
                if error:
                    self.error_text = _error_text(error)
        if self.error_text:
            raise CompileError(self.error_text)
        return self

    @property
    def gpu_interval(self):
        self.wait()
        start = _send(self.command_buffer, "GPUStartTime", restype=ctypes.c_double)
        end = _send(self.command_buffer, "GPUEndTime", restype=ctypes.c_double)
        return start, end

    @property
    def gpu_time(self) -> float:
        start, end = self.gpu_interval
        return (end - start) / len(self.submission.launches)

    def __repr__(self) -> str:
        return f"Launch({self.kernel.name!r}, grid={self.grid}, group={self.group}, done={self.done})"


def add_launch_observer(callback) -> None:
    _observers.append(callback)


def remove_launch_observer(callback) -> None:
    _observers.remove(callback)


def synchronize() -> None:
    if _batch is not None:
        _batch.flush()
    pending = list(_inflight)
    _inflight.clear()
    failure = None
    for event in pending:
        try:
            event.wait()
        except CompileError as error:
            failure = failure or error
    if failure:
        raise failure


@contextlib.contextmanager
def batch():
    global _batch
    if _batch is not None:
        raise RuntimeError("batches cannot be nested")
    _batch = Batch()
    try:
        yield _batch
    finally:
        finished, _batch = _batch, None
        finished.flush()


def launch(kernel: Kernel, arguments: list, grid, group=None, wait: bool = False) -> Launch:
    grid = _triple(grid)
    maximum = kernel.maximum_threads()
    group = _triple(group) if group is not None else (min(grid[0], maximum), 1, 1)
    threads_per_group = group[0] * group[1] * group[2]
    if threads_per_group > maximum:
        raise CompileError(f"threadgroup of {threads_per_group} threads exceeds the limit of {maximum} for {kernel.name}")

    encode_started = time.perf_counter()
    with _autorelease_pool():
        if _batch is not None:
            if _batch.current is None:
                _batch.current = Submission(_batch)
            submission = _batch.current
        else:
            submission = Submission()
        submission.encode(kernel, arguments, grid, group)
        event = Launch(submission, kernel, grid, group, encode_started)
        submission.launches.append(event)
        event.encode_ended = time.perf_counter()
        if _batch is None:
            submission.commit()

    if len(_inflight) >= PRUNE_THRESHOLD:
        _inflight[:] = [item for item in _inflight if not item.done or item.failed]
    _inflight.append(event)
    for observer in list(_observers):
        observer(event)
    if wait:
        event.wait()
    return event
