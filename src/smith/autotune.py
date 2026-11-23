import contextlib
import ctypes
import itertools
import json
import os
import statistics
import tempfile

from . import viz
from .jit import JITFunction, _uses_metal, synchronize


class Config:
    def __init__(self, **values):
        self.values = values

    def __eq__(self, other):
        return isinstance(other, Config) and self.values == other.values

    def __hash__(self):
        return hash(tuple(sorted(self.values.items())))

    def __repr__(self):
        inner = ", ".join(f"{name}={value!r}" for name, value in self.values.items())
        return f"Config({inner})"


def configs_from_grid(options):
    names = list(options)
    combinations = itertools.product(*(options[name] for name in names))
    return [Config(**dict(zip(names, combination))) for combination in combinations]


def autotune(configs, key, restore=(), warmup=2, repeat=5, fast_math=True, finalists=3):
    def decorator(jit_function):
        return Autotuner(jit_function, configs, key, restore=restore, warmup=warmup, repeat=repeat,
                         fast_math=fast_math, finalists=finalists)
    return decorator


def _cache_path(kernel_name):
    directory = os.environ.get("SMITH_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "smith")
    return os.path.join(directory, f"autotune-{kernel_name}.json")


def _read_json(path):
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path, data):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".autotune-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(temporary)
        raise


def _failure_types():
    from .compiler import CompileError
    return (TypeError, ValueError, CompileError)


class Autotuner:
    def __init__(self, kernel, configs, key, restore=(), warmup=2, repeat=5, fast_math=True, finalists=3):
        if not isinstance(kernel, JITFunction):
            raise TypeError("autotune must decorate a @smith.jit function")
        if not configs:
            raise ValueError("autotune needs at least one config")
        if repeat < 1 or warmup < 0 or finalists < 1:
            raise ValueError("autotune needs repeat >= 1, warmup >= 0 and finalists >= 1")
        unknown = [name for name in key if name not in kernel.signature.parameters]
        if unknown:
            raise ValueError(f"autotune key names unknown parameters: {unknown}")
        self.kernel = kernel
        self.configs = list(configs)
        self.key = tuple(key)
        self.restore = tuple(restore)
        self.warmup = warmup
        self.repeat = repeat
        self.fast_math = fast_math
        self.finalists = finalists
        self.memory = {}
        self.best_config = None
        self.results = []
        self.__name__ = kernel.__name__

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return self._launch(grid, args, kwargs)
        return launch

    def _launch(self, grid, args, kwargs):
        values, _ = self.kernel._bind(args, kwargs)
        if not _uses_metal(args, kwargs):
            return self._run(grid, args, kwargs, values, self.configs[0])

        from .compiler import Buffer
        for name in self.restore:
            if not isinstance(values.get(name), Buffer):
                raise TypeError(f"restore entry {name!r} must name a buffer argument")

        text = self._key_text(values)
        path = _cache_path(self.kernel.__name__)
        if text not in self.memory:
            record = _read_json(path).get(text)
            cached = self._valid_record(record)
            if not cached:
                with viz.phase("tuning"):
                    record = self._tune(grid, args, kwargs, values)
                entries = _read_json(path)
                entries[text] = record
                _write_json(path, entries)
            self.memory[text] = record
            viz.record_autotune(self.kernel.__name__, text, record, cached)

        record = self.memory[text]
        self.best_config = Config(**record["config"])
        self.results = record["results"]
        return self._run(grid, args, kwargs, values, self.best_config)

    def _key_text(self, values):
        from .compiler import Buffer
        parameters = {name: values[name] for name in self.key}
        dtypes = {name: value.dtype.name for name, value in values.items() if isinstance(value, Buffer)}
        return json.dumps({"key": parameters, "dtypes": dtypes, "fast_math": self.fast_math}, sort_keys=True, default=repr)

    def _valid_record(self, record):
        if not isinstance(record, dict) or not isinstance(record.get("config"), dict):
            return False
        if not isinstance(record.get("results"), list):
            return False
        return any(Config(**record["config"]) == config for config in self.configs)

    def _run(self, grid, args, kwargs, values, config):
        merged = {**values, **config.values}
        resolved = (lambda _meta: grid(merged)) if callable(grid) else grid
        return self.kernel._launch(resolved, args, {**kwargs, **config.values}, self.fast_math)

    def _tune(self, grid, args, kwargs, values):
        from .compiler import Buffer, CompileError
        failures = _failure_types()
        synchronize()
        saved = [(values[name], ctypes.string_at(values[name].contents_ptr(), values[name].nbytes))
                 for name in self.restore if isinstance(values[name], Buffer)]

        results = []
        for config in self.configs:
            try:
                seconds = self._measure(grid, args, kwargs, values, config, saved, self.repeat)
            except failures as error:
                results.append({"config": config.values, "seconds": None, "refined": None, "error": str(error)})
            else:
                results.append({"config": config.values, "seconds": seconds, "refined": None, "error": None})

        valid = [result for result in results if result["seconds"] is not None]
        if not valid:
            self._restore_buffers(saved)
            reasons = "; ".join(f"{result['config']}: {result['error']}" for result in results)
            raise CompileError(f"no valid config for {self.kernel.__name__}: {reasons}")
        finalists = sorted(valid, key=lambda result: result["seconds"])[:self.finalists]
        for result in finalists:
            result["refined"] = self._measure(grid, args, kwargs, values, Config(**result["config"]), saved, self.repeat * 3)
        self._restore_buffers(saved)
        best = min(finalists, key=lambda result: result["refined"])
        return {"config": best["config"], "results": results}

    def _measure(self, grid, args, kwargs, values, config, saved, repeat):
        samples = []
        for run in range(self.warmup + repeat):
            self._restore_buffers(saved)
            event = self._run(grid, args, kwargs, values, config)
            if run < self.warmup:
                event.wait()
            else:
                samples.append(event.gpu_time)
        return statistics.median(samples)

    def _restore_buffers(self, saved):
        for buffer, raw in saved:
            ctypes.memmove(buffer.contents_ptr(), raw, len(raw))
