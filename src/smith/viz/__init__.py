import atexit
import contextlib
import os
import re
import subprocess
import sys
import time
import weakref
from collections import deque

from ..ir import live_nodes
from . import opprofile

LAUNCH_LIMIT = 4000
SPAN_LIMIT = 20000
GAP_THRESHOLD_MS = 0.005
HIDDEN_ARGUMENTS = ("load", "store", "loop")


class _State:
    def __init__(self):
        self.enabled = False
        self.observing = False
        self.phase = "run"
        self.origin = time.perf_counter()
        self.version = 0
        self.launch_counter = 0
        self.programs = []
        self.programs_by_key = {}
        self.launches = deque(maxlen=LAUNCH_LIMIT)
        self.spans = deque(maxlen=SPAN_LIMIT)
        self.autotunes = []
        self.hardware = None


state = _State()


def _argument(node):
    if node.arg is None or node.op in HIDDEN_ARGUMENTS:
        return None
    if node.op == "matrix_load":
        return "transposed" if node.arg[0] else None
    return str(node.arg)


def graph_json(graph, roots=()):
    live = live_nodes(graph, roots)
    nodes = []
    for node in graph.nodes:
        if node.index not in live:
            continue
        entry = {"id": node.index, "op": node.op, "arg": _argument(node),
                 "dtype": node.dtype.name if node.dtype is not None else None,
                 "inputs": [source.index for source in node.inputs]}
        if node.op == "loop":
            body = node.arg
            entry["body"] = {**graph_json(body.graph, body.outputs), "index": body.index.index,
                             "carried": [n.index for n in body.carried], "outputs": [n.index for n in body.outputs]}
        nodes.append(entry)
    return {"nodes": nodes, "total": len(graph.nodes)}


def op_counts(graph):
    counts = {}
    for node in graph["nodes"]:
        counts[node["op"]] = counts.get(node["op"], 0) + 1
        if "body" in node:
            for op, count in op_counts(node["body"]).items():
                counts[op] = counts.get(op, 0) + count
    return counts


def enabled():
    return state.enabled


def enable():
    if state.enabled:
        return
    state.enabled = True
    if sys.platform == "darwin" and not state.observing:
        from .. import compiler
        compiler.add_launch_observer(_observe)
        state.observing = True


def disable():
    state.enabled = False
    if state.observing:
        from .. import compiler
        compiler.remove_launch_observer(_observe)
        state.observing = False


def reset():
    state.programs.clear()
    state.programs_by_key.clear()
    state.launches.clear()
    state.spans.clear()
    state.autotunes.clear()
    state.launch_counter = 0
    state.origin = time.perf_counter()
    state.version += 1


@contextlib.contextmanager
def phase(name):
    previous = state.phase
    state.phase = name
    try:
        yield
    finally:
        state.phase = previous


def record_program(program):
    if not state.enabled:
        return
    key = (program.name, program.source)
    if key in state.programs_by_key:
        return
    traced, simplified = graph_json(program.traced), graph_json(program.graph)
    state.programs_by_key[key] = len(state.programs)
    state.programs.append({
        "id": len(state.programs),
        "phase": state.phase,
        "name": program.name,
        "lanes": program.lanes,
        "constexprs": {name: repr(value) for name, value in program.constexprs.items()},
        "buffers": {name: dtype.name for name, dtype in program.buffers.items()},
        "scalars": {name: dtype.name for name, dtype in program.scalars.items()},
        "loaded": sorted(program.loaded),
        "stored": sorted(program.stored),
        "uses_matrices": program.uses_matrices,
        "source": program.source,
        "compile_ms": None,
        "stages": {"traced": traced, "simplified": simplified},
        "op_counts": {"traced": op_counts(traced), "simplified": op_counts(simplified)},
        "op_profile": opprofile.profile(simplified),
    })
    state.version += 1


def record_compile(name, source, seconds):
    if not state.enabled:
        return
    index = state.programs_by_key.get((name, source))
    if index is not None and state.programs[index]["compile_ms"] is None and seconds is not None:
        state.programs[index]["compile_ms"] = seconds * 1e3
        state.version += 1


def record_autotune(kernel, key, record, cached):
    if not state.enabled:
        return
    state.autotunes.append({"kernel": kernel, "key": key, "best": record["config"], "results": record["results"],
                            "cached": cached})
    state.version += 1


def _observe(launch):
    state.launch_counter += 1
    state.launches.append((state.launch_counter, launch, state.phase))
    state.version += 1


class _Span:
    __slots__ = ("kind", "label", "start", "key", "launch")

    def __init__(self, kind, label, key=None, launch=None):
        self.kind = kind
        self.label = label
        self.key = key
        self.launch = weakref.ref(launch) if launch is not None else None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        state.spans.append((self.kind, self.label, self.start, time.perf_counter(), state.phase, self.key, self.launch))
        state.version += 1
        return False


class _Unrecorded:
    key = None


_NO_SPAN = contextlib.nullcontext(_Unrecorded())


def span(kind, label, key=None, launch=None):
    return _Span(kind, label, key, launch) if state.enabled else _NO_SPAN


def hardware():
    if state.hardware is None:
        cores, chip = None, None
        try:
            output = subprocess.run(["ioreg", "-rc", "AGXAccelerator", "-d1"], capture_output=True, text=True, timeout=5).stdout
            match = re.search(r'"gpu-core-count"\s*=\s*(\d+)', output)
            cores = int(match.group(1)) if match else None
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            pass
        state.hardware = {"chip": chip, "gpu_cores": cores, "simd_width": 32}
    return state.hardware


def occupancy(grid, group, simd_width, cores):
    groups, per_group = 1, 1
    for extent, size in zip(grid, group):
        groups *= -(-extent // size)
        per_group *= size
    simds = -(-per_group // simd_width)
    info = {"groups": groups, "threads_per_group": per_group, "simd_groups": simds,
            "simd_efficiency": per_group / (simds * simd_width)}
    if cores and groups:
        full, extra = divmod(groups, cores)
        loads = [full + (1 if index < extra else 0) for index in range(cores)]
        info.update({"idle_cores": loads.count(0), "core_efficiency": (groups / cores) / -(-groups // cores),
                     "core_loads": loads})
    return info


BATCH_ID_OFFSET = 1_000_000


def _homogeneous(launches):
    return len({(launch.kernel.name, launch.kernel.source) for launch in launches}) == 1


def _launch_rows(base):
    cores = hardware()["gpu_cores"]
    rows, batch_ids, batches = [], {}, {}
    for number, launch, phase_name in list(state.launches):
        submission = launch.submission
        batched = len(submission.launches) > 1
        encode_start = (launch.encode_started - base) * 1e3
        if batched:
            encode_end = (launch.encode_ended - base) * 1e3
            batch_ids.setdefault(id(submission), BATCH_ID_OFFSET + len(batch_ids))
            batches.setdefault(id(submission), []).append((number, launch, phase_name))
        else:
            encode_end = (launch.submitted - base) * 1e3 if launch.submitted is not None else encode_start
        row = {"id": number, "name": launch.kernel.name, "phase": phase_name,
               "program": state.programs_by_key.get((launch.kernel.name, launch.kernel.source)),
               "grid": list(launch.grid), "group": list(launch.group), "start": None, "end": None,
               "encode_start": encode_start, "encode_end": encode_end, "encode_ms": encode_end - encode_start,
               "submit": (launch.submitted - base) * 1e3 if launch.submitted is not None else None}
        row.update(occupancy(launch.grid, launch.group, launch.kernel.execution_width(), cores))
        if batched:
            row["member_of"] = batch_ids[id(submission)]
        elif launch.done and not launch.failed:
            start, end = launch.gpu_interval
            row["start"], row["end"] = (start - base) * 1e3, (end - base) * 1e3
            row["ms"] = row["end"] - row["start"]
            row["queue_ms"] = row["start"] - row["submit"]
        rows.append(row)
    for key, members in batches.items():
        launches = [item[1] for item in members]
        first, last = launches[0], launches[-1]
        submission = first.submission
        if not (submission.committed and first.done and not first.failed):
            continue
        start, end = first.gpu_interval
        size = len(submission.launches)
        names = sorted({launch.kernel.name for launch in launches})
        same = _homogeneous(launches)
        commit_start = (last.encode_ended - base) * 1e3
        submit = (first.submitted - base) * 1e3
        row = {"id": batch_ids[key], "name": f"batch x{size} ({', '.join(names)})", "phase": members[0][2],
               "program": state.programs_by_key.get((first.kernel.name, first.kernel.source)) if same else None,
               "grid": list(first.grid), "group": list(first.group), "batch_size": size,
               "start": (start - base) * 1e3, "end": (end - base) * 1e3, "encode_start": commit_start,
               "encode_end": submit, "encode_ms": submit - commit_start, "submit": submit}
        row["ms"] = row["end"] - row["start"]
        row["queue_ms"] = row["start"] - submit
        row.update(occupancy(first.grid, first.group, first.kernel.execution_width(), cores))
        if not same:
            for name in ("core_loads", "idle_cores", "core_efficiency"):
                row.pop(name, None)
        rows.append(row)
    rows.sort(key=lambda row: row["encode_start"])
    return rows


def _span_rows(base):
    numbers = {id(launch): number for number, launch, _ in list(state.launches)}
    rows = []
    for kind, label, start, end, phase_name, key, launch_ref in list(state.spans):
        row = {"kind": kind, "label": label, "start": (start - base) * 1e3, "end": (end - base) * 1e3,
               "ms": (end - start) * 1e3, "phase": phase_name, "program": state.programs_by_key.get(key),
               "launch": None, "finished_before": None, "notify_lag": None}
        launch = launch_ref() if launch_ref is not None else None
        if launch is not None:
            row["launch"] = numbers.get(id(launch))
            if launch.done and not launch.failed:
                finished = (launch.gpu_interval[1] - base) * 1e3
                row["finished_before"] = row["start"] - finished
                row["notify_lag"] = row["end"] - finished
        rows.append(row)
    return rows


def _overlap(start, end, other_start, other_end):
    return max(0.0, min(end, other_end) - max(start, other_start))


def _gaps(rows, spans):
    timed = sorted((row for row in rows if row.get("ms") is not None), key=lambda row: row["start"])
    gaps = []
    for before, after in zip(timed, timed[1:]):
        if before["phase"] == "tuning" or after["phase"] == "tuning":
            continue
        start, end = before["end"], after["start"]
        length = end - start
        if length <= GAP_THRESHOLD_MS:
            continue
        causes = {}
        for item in spans:
            if item["phase"] == "run":
                covered = _overlap(start, end, item["start"], item["end"])
                if covered > 0:
                    causes[item["kind"]] = causes.get(item["kind"], 0.0) + covered
        encoded = _overlap(start, end, after["encode_start"], after["submit"])
        if encoded > 0:
            causes["encode"] = causes.get("encode", 0.0) + encoded
        waiting = max(0.0, after["start"] - max(after["submit"], start))
        if waiting > 0:
            causes["queue"] = causes.get("queue", 0.0) + min(waiting, length)
        explained = sum(causes.values())
        if explained < length:
            causes["other host code"] = length - explained
        gaps.append({"from": before["id"], "to": after["id"], "name": after["name"], "start": start, "ms": length,
                     "causes": {kind: min(value, length) for kind, value in sorted(causes.items(), key=lambda item: -item[1])}})
    return gaps


def snapshot():
    launches = list(state.launches)
    spans = list(state.spans)
    candidates = [launch.encode_started for _, launch, _ in launches if launch.encode_started is not None]
    candidates += [launch.submitted for _, launch, _ in launches if launch.submitted is not None] + [item[2] for item in spans]
    base = min(candidates) if candidates else state.origin
    rows = _launch_rows(base)
    span_rows = _span_rows(base)
    gaps = _gaps(rows, span_rows)
    programs = []
    for entry in state.programs:
        own = [row for row in rows if row["program"] == entry["id"] and row["phase"] == "run" and row.get("ms") is not None]
        times = [row["ms"] / row.get("batch_size", 1) for row in own for _ in range(row.get("batch_size", 1))]
        shape = own[-1] if own else None
        programs.append({
            "id": entry["id"], "name": entry["name"], "lanes": entry["lanes"], "constexprs": entry["constexprs"],
            "compile_ms": entry["compile_ms"], "launches": len(times),
            "tuning_launches": sum(1 for row in rows if row["program"] == entry["id"] and row["phase"] == "tuning" and "batch_size" not in row),
            "candidate": entry["phase"] == "tuning" and not times,
            "total_ms": sum(times), "mean_ms": sum(times) / len(times) if times else None,
            "min_ms": min(times) if times else None, "max_ms": max(times) if times else None,
            "groups": shape["groups"] if shape else None,
            "idle_cores": shape.get("idle_cores") if shape else None,
            "core_efficiency": shape.get("core_efficiency") if shape else None,
            "simd_efficiency": shape["simd_efficiency"] if shape else None,
            "nodes": {name: len(graph["nodes"]) for name, graph in entry["stages"].items()},
            "op_profile": None if entry["phase"] == "tuning" and not times else entry["op_profile"],
        })
    run = [row for row in rows if row["phase"] == "run" and row.get("ms") is not None]
    busy = sum(row["ms"] for row in run)
    idle = sum(gap["ms"] for gap in gaps)
    queue = [row["queue_ms"] for row in run]
    stats = {"gpu_busy_ms": busy, "idle_ms": idle, "duty_cycle": busy / (busy + idle) if busy + idle else None,
             "mean_queue_ms": sum(queue) / len(queue) if queue else None,
             "encode_ms": sum(row["encode_ms"] for row in rows if row["phase"] == "run")}
    spans_end = [row["end"] for row in rows if row.get("end") is not None]
    return {"version": state.version, "programs": programs, "launches": rows, "spans": span_rows, "gaps": gaps,
            "stats": stats, "hardware": hardware(), "autotunes": list(state.autotunes),
            "span_ms": max(spans_end) if spans_end else 0.0}


def program_detail(index):
    if 0 <= index < len(state.programs):
        return state.programs[index]
    return None


def serve(port=None, open_browser=True, block=True):
    from .server import serve as run
    return run(port, open_browser, block)


def _serve_at_exit():
    try:
        serve(block=True)
    except KeyboardInterrupt:
        pass


if os.environ.get("SMITH_VIZ", "0") not in ("", "0"):
    enable()
    atexit.register(_serve_at_exit)
