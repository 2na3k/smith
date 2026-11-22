import sys
from dataclasses import dataclass

from .ir import live_nodes
from .layout import evaluate

GIGA = 1e9


@dataclass
class LaunchRecord:
    name: str
    grid: tuple
    group: tuple
    submitted: float
    timing: object
    bytes_moved: object
    bytes_exact: bool

    @property
    def gpu_time(self):
        return self.timing.gpu_time


def _memory_nodes(program):
    live = live_nodes(program.graph)
    return [node for node in program.graph.nodes
            if node.index in live and node.op in ("load", "store")]


def estimate_bytes(program, dispatch, scalars=None, buffers=None):
    dispatch = tuple(dispatch) + (1,) * (3 - len(dispatch))
    if scalars is None:
        threads = dispatch[0] * dispatch[1] * dispatch[2]
        per_thread = sum(node.inputs[0].dtype.size for node in _memory_nodes(program))
        return per_thread * threads, False
    grid = (dispatch[0] // program.lanes, dispatch[1], dispatch[2])
    evaluation = evaluate(program, grid, scalars, buffers)
    total = sum(program.buffers[access.buffer].size for access in evaluation.accesses if access.active is not False)
    return total, True


class Profiler:
    def __init__(self):
        self.records = []
        self.tracked = []
        self._compiler = None

    def track(self, program, scalars=None, buffers=None):
        self.tracked.append((program, scalars, buffers))
        return self

    def __enter__(self):
        if sys.platform != "darwin":
            raise RuntimeError("Profiler records Metal launches and needs macOS")
        from . import compiler
        self._compiler = compiler
        compiler.add_launch_observer(self._observe)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._compiler.remove_launch_observer(self._observe)
        if exc_type is None:
            self._compiler.synchronize()
        return False

    def _match(self, launch):
        from .compiler import CompileError, compile_msl
        for entry in self.tracked:
            program = entry[0]
            if program.name != launch.kernel.name:
                continue
            try:
                kernel = compile_msl(program.source, program.name)
            except CompileError:
                continue
            if kernel is launch.kernel:
                return entry
        return None

    def _observe(self, launch):
        bytes_moved, exact = None, False
        entry = self._match(launch)
        if entry is not None:
            program, scalars, buffers = entry
            try:
                bytes_moved, exact = estimate_bytes(program, launch.grid, scalars, buffers)
            except ValueError:
                bytes_moved, exact = estimate_bytes(program, launch.grid)
        self.records.append(LaunchRecord(
            launch.kernel.name, tuple(launch.grid), tuple(launch.group), launch.submitted,
            launch, bytes_moved, exact))

    def summary(self):
        groups = {}
        for record in self.records:
            groups.setdefault(record.name, []).append(record)
        rows = []
        for name, group in groups.items():
            times = [record.gpu_time for record in group]
            total = sum(times)
            moved = [record.bytes_moved for record in group]
            bytes_total = None if any(value is None for value in moved) else sum(moved)
            rows.append({
                "name": name,
                "launches": len(group),
                "total_seconds": total,
                "mean_seconds": total / len(group),
                "min_seconds": min(times),
                "max_seconds": max(times),
                "bytes_moved": bytes_total,
                "bytes_exact": all(record.bytes_exact for record in group),
                "gb_per_s": None if bytes_total is None or total == 0 else bytes_total / total / GIGA,
            })
        return rows

    def report(self):
        rows = self.summary()
        if not rows:
            return "no launches recorded"
        header = (f"{'kernel':<20} {'count':>6} {'total ms':>10} {'mean ms':>10} {'min ms':>10} "
                  f"{'max ms':>10} {'bytes':>14} {'GB/s':>9}")
        lines = [header, "-" * len(header)]
        for row in rows:
            moved = "-" if row["bytes_moved"] is None else f"{row['bytes_moved']:,}"
            if row["bytes_moved"] is not None and not row["bytes_exact"]:
                moved += "*"
            rate = "-" if row["gb_per_s"] is None else f"{row['gb_per_s']:.2f}"
            lines.append(f"{row['name']:<20} {row['launches']:>6} {row['total_seconds'] * 1e3:>10.3f} "
                         f"{row['mean_seconds'] * 1e3:>10.3f} {row['min_seconds'] * 1e3:>10.3f} "
                         f"{row['max_seconds'] * 1e3:>10.3f} {moved:>14} {rate:>9}")
        if any(row["bytes_moved"] is not None and not row["bytes_exact"] for row in rows):
            lines.append("* upper bound: every thread assumed active (pass scalars to track for exact counts)")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "kernels": self.summary(),
            "launches": [
                {
                    "name": record.name,
                    "grid": list(record.grid),
                    "group": list(record.group),
                    "submitted": record.submitted,
                    "gpu_seconds": record.gpu_time,
                    "bytes_moved": record.bytes_moved,
                    "bytes_exact": record.bytes_exact,
                }
                for record in self.records
            ],
        }
