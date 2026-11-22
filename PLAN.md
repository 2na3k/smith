# smith plan

A Triton/TileLang-style kernel DSL in pure Python that compiles to Metal. No dependencies (stdlib + ctypes).
MLX is an optional bridge only, imported lazily.

## Pipeline (already built on main)

```
@sth.jit function
  -> trace.py     run once with Symbols, building a typed graph        (ops.py dispatches to the tracer)
  -> ir.py        Graph / Node, hash-consed, typed, casts inserted
  -> passes.py    simplify: identities, constant folding, mask folding
  -> render.py    render_statements(graph, leaf) -> lines; render_metal -> full MSL source
  -> compiler.py  compile_msl -> Kernel (pipeline), Buffer, launch -> Launch (async), synchronize
```

Key facts:
- `launch(...)` is asynchronous. It returns a `Launch` with `.done`, `.wait()`, `.gpu_time` (seconds, from GPUStartTime/GPUEndTime).
- `synchronize()` waits for everything in flight. `Buffer.to_list()` synchronizes automatically.
- `compiler.add_launch_observer(callback)` / `remove_launch_observer(callback)`: callback receives each `Launch` at submit time.
- `JITFunction.trace(*args, **kwargs) -> Program` traces without launching. Arguments for pointer parameters may be
  real `Buffer`s or `smith.trace.BufferSpec(dtype)` placeholders. `JITFunction.programs` caches traced programs.
- `Program` has: `source`, `name`, `lanes`, `runtime_names`, `graph`, `buffers` (name -> DType), `scalars` (name -> DType),
  `loaded` / `stored` (sets of buffer parameter names).
- Execution model: one threadgroup per program, `lanes` threads per program. `program_id(axis)` is the threadgroup
  position, `arange` is the thread index inside the threadgroup. Total threads = grid[0] * lanes in x.
- Interpreter path: kernels called with plain Python lists still run in pure Python (ops.py).

## Conventions (all work must follow)

- No comments and no docstrings in code you write.
- Full words for names: `pointer`, `buffer`, `offset`, `device`, `library`, `function`, `command_buffer`, `encoder`.
  Never abbreviate those. Keep existing names elsewhere as they are (`pid`, `gid`, `n`, `i`, `dtype`, `msl`, ...).
- Standard library and ctypes only. MLX may only be imported lazily inside the MLX bridge module.
- Tests use `unittest`. Metal tests are guarded with `@unittest.skipUnless(sys.platform == "darwin", ...)`.
- Run tests: `PYTHONPATH=src python3 -m unittest discover -s tests`
- Do not edit `src/smith/__init__.py` (the orchestrator wires exports). Do not edit files owned by another stream.
  If you need a change in a file you do not own, describe it in your final report instead.
- Commit your work on your own branch inside your worktree when done (small, clear commit message). Do not push.
- Final report: files added/changed, test command and result, deviations from this plan, requested changes elsewhere.

## Streams

### A. dtypes (branch work/dtypes)
Owns: `src/smith/dtype.py`, `tests/test_dtype.py`.
Limited edits: `ir.py` (`RANK`, `promote` only), `passes.py` (`_wrap`, `_convert`, `_convertible`, `_is_value` only),
`render.py` (`_literal` only), `compiler.py` (`Buffer.__init__`, `from_list`, `to_list` only).
Deliver:
- New dtypes: float16 (`half`), bfloat16 (`bfloat`), int8 (`char`), int16 (`short`), int64 (`long`),
  uint8 (`uchar`), uint16 (`ushort`), uint64 (`ulong`). Keep `float32`, `int32`, `uint32`, `bool_` working unchanged.
  float64 is not supported by Metal: provide a clear error if someone asks for it.
- `DType` gains `kind` ("float" | "int" | "uint" | "bool") and `bits`; `ctype` may be `None` for half and bfloat16.
  Add `pack(values) -> bytes` and `unpack(raw, count) -> list` on `DType` (struct format `e` for half; bfloat16 via
  float32 bit pattern, round-to-nearest-even) and use them in `Buffer.from_list` / `to_list`.
- Promotion: widen within a kind, integer with float gives the float type, float16 with bfloat16 gives float32; document in tests.
- Rendering of literals for every dtype (`1.5h`, `2u`, `3l`, bfloat via cast), wraparound constant folding per bit width.
- Metal tests: an add kernel per dtype family compiles and runs correctly (bfloat16 needs macOS 14+; skip if compile fails).

### B. profiler and layout visualization (branch work/profiler)
Owns (new files only): `src/smith/profiler.py`, `src/smith/layout.py`, `tests/test_profiler.py`, `tests/test_layout.py`,
`examples/profile_add.py`, `examples/layout_add.py`.
Deliver:
- `Profiler` context manager using `add_launch_observer`: records kernel name, grid, group, submit time, gpu_time.
  `report()` returns a text table (count, total/mean/min/max GPU ms per kernel); `to_dict()` for JSON.
  Estimate bytes moved per launch from the graph (loads and stores times dtype size times active threads) and report GB/s.
- A pure-Python reference evaluator for a `Program` graph: given grid, scalar arguments and (optionally) buffer contents,
  compute for every thread (program, lane) the offset and mask of each load/store. No Metal required.
- Layout viz: ASCII map of which (program, lane) touches each element of a buffer (with overlap counts and untouched
  elements), a standalone HTML file (inline SVG, no scripts or external assets) of the same mapping, and
  `graph_to_dot(program)` (Graphviz DOT text).
- Access analysis per load/store: lane stride (1 means coalesced), overlapping writes between threads, out-of-range
  offsets when the buffer length is known. Return structured warnings.

### C. autotuning (branch work/autotune)
Owns: `src/smith/autotune.py`, `tests/test_autotune.py`, `examples/autotune_add.py`.
Deliver:
- `autotune(configs, key, restore=(), warmup=2, repeat=5)` decorator stacked over `@jit`. `Config(**constexprs)` holds
  constexpr values; helper `configs_from_grid({"BLOCK": [32, 64, 128]})`.
- `Autotuner` mirrors `JITFunction`: `kernel[grid](*args, **kwargs)`; the grid callable receives the chosen config merged
  with kwargs. First call for a key benchmarks every valid config using `Launch.gpu_time` (median of `repeat`, after
  `warmup`), skips configs that fail to trace/compile or exceed the threadgroup limit, picks the fastest, then launches it.
- Cache per key (values of the `key` parameters plus buffer dtypes) in memory and in a JSON file under
  `$SMITH_CACHE_DIR` (default `~/.cache/smith`), written atomically. Expose `.best_config` and `.results`.
- `restore` names buffer parameters whose contents are saved before benchmarking and restored after, so in/out
  buffers are not corrupted by repeated runs.
- List arguments (interpreter path) just run the first config without tuning.

### D. MLX bridge (branch work/mlx)
Owns: `src/smith/mlx_bridge.py`, `tests/test_mlx_bridge.py`, `examples/mlx_add.py`, `pyproject.toml` (optional extra `mlx`), `uv.lock`.
MLX is installed in `/Users/lechithanh/workspace/smith/.venv` (python 3.12, mlx 0.32.4): run MLX tests with
`PYTHONPATH=<your worktree>/src /Users/lechithanh/workspace/smith/.venv/bin/python -m unittest ...`.
Deliver:
- `kernel(jit_function)` returns an `MLXKernel`. `add_mlx[grid](x, y, out, n, BLOCK=4)` takes `mx.array`s for pointer
  parameters. Use `jit_function.trace(...)` with `BufferSpec` (dtype mapped from the mlx dtype), then `Program.loaded` /
  `Program.stored` to decide roles: loaded-only -> input, stored -> output (shape and dtype taken from the array passed,
  contents ignored). A buffer both loaded and stored raises a clear error.
- Generate the body with `render.render_statements(program.graph, leaf=mlx_leaf)` where `mlx_leaf` maps
  `program_id` -> `int(threadgroup_position_in_grid.x)`, `lane` -> `int(thread_position_in_threadgroup.x)`,
  `program_count` -> `int(threadgroups_per_grid.x)`, scalar `argument` -> `name[0]` (scalars are passed as 1-element
  arrays), `buffer` -> its name. Run through `mx.fast.metal_kernel` (lazy, runs in MLX's own stream, no C++ changes, no copies).
  `grid=(programs*lanes, ...)`, `threadgroup=(lanes, 1, 1)`. Optional `init_value=` for outputs with masked tails.
  Returns one `mx.array` or a tuple in output parameter order.
- Copy fallback so smith's own `launch` can run on MLX data: `buffer_from_mlx(array)` and `mlx_from_buffer(buffer, shape)`
  via the buffer protocol (unified memory; evaluate the array first; bfloat16 through a uint16 view).
- dtype mapping by name tolerant of dtypes that do not exist yet (stream A adds more).
- Tests: correctness against `mx.add`, composition inside MLX graphs (`mx.sum(add_mlx(...))`), `mx.compile` if it works,
  scalar arguments, masked tail, error on in/out buffers. Skip when mlx is not importable.
- Never import mlx at smith import time.

### E. integration (orchestrator)
Merge branches, wire exports in `__init__.py` (`autotune`, `Config`, `Profiler`, ...), cross-feature tests
(mlx -> smith kernel -> mlx with float16, autotuned kernel under the profiler), README with the architecture.
