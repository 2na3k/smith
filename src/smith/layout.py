import html
import itertools
import math
import operator
import string
from dataclasses import dataclass, field

from .dtype import bool_, float32, int32, uint32
from .ir import BINARY, live_nodes

GLYPHS = string.digits + string.ascii_letters
PALETTE = ("#4e79a7", "#f28e2b", "#59a14f", "#edc948", "#76b7b2",
           "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#86bcb6")
UNTOUCHED_COLOR = "#f4f4f4"
SHARED_COLOR = "#8c8c8c"
CONFLICT_COLOR = "#d62728"
STORE_COLOR = "#ffd8a8"
PLAIN_OPERATORS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "and": operator.and_,
    "or": operator.or_,
    "shl": operator.lshift,
    "shr": operator.rshift,
    "less": operator.lt,
    "less_equal": operator.le,
    "greater": operator.gt,
    "greater_equal": operator.ge,
}


@dataclass(frozen=True)
class Access:
    kind: str
    buffer: str
    node: int
    program: tuple
    lane: int
    offset: object
    active: object
    value: object


@dataclass
class Evaluation:
    program: object
    grid: tuple
    scalars: dict
    lengths: dict
    accesses: list


@dataclass
class AccessSummary:
    kind: str
    buffer: str
    node: int
    lane_stride: object
    coalesced: object


@dataclass
class AccessWarning:
    kind: str
    buffer: str
    node: object
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class Analysis:
    summaries: list
    warnings: list

    @property
    def clean(self):
        return not self.warnings


@dataclass
class ElementMap:
    buffer: str
    length: int
    programs: list
    threads: list
    writers: list


def _triple(size):
    size = (size,) if isinstance(size, int) else tuple(size)
    return size + (1,) * (3 - len(size))


def _wrap(value, dtype):
    if dtype == int32:
        return (value + 2**31) % 2**32 - 2**31
    if dtype == uint32:
        return value % 2**32
    return value


def _convert(value, dtype):
    if value is None:
        return None
    if dtype == float32:
        return float(value)
    if dtype == bool_:
        return bool(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        value = int(value)
    return _wrap(int(value), dtype)


def _divide(left, right):
    if right == 0:
        if left == 0 or math.isnan(left):
            return math.nan
        return math.copysign(math.inf, left) * math.copysign(1.0, right)
    return left / right


def _fmod(left, right):
    try:
        return math.fmod(left, right)
    except ValueError:
        return math.nan


def _truncating_quotient(left, right):
    quotient = abs(left) // abs(right)
    return quotient if (left >= 0) == (right >= 0) else -quotient


def _binary(op, left, right, dtype):
    if left is None or right is None:
        return None
    if op in PLAIN_OPERATORS:
        return _convert(PLAIN_OPERATORS[op](left, right), dtype)
    if op == "truediv":
        return _divide(left, right)
    if dtype == float32:
        if op == "floordiv":
            quotient = _divide(left, right)
            return float(math.floor(quotient)) if math.isfinite(quotient) else quotient
        return _fmod(left, right)
    if right == 0:
        return None
    quotient = _truncating_quotient(left, right)
    if op == "floordiv":
        return _convert(quotient, dtype)
    return _convert(left - right * quotient, dtype)


def _read(contents, name, offset):
    if offset is None:
        return None
    data = contents.get(name)
    if data is None or not 0 <= offset < len(data):
        return None
    return data[offset]


def _run_thread(nodes, program_id, lane, grid, scalars, contents):
    values = {}
    accesses = []
    for node in nodes:
        op = node.op
        inputs = node.inputs
        if op == "const":
            values[node.index] = node.arg
        elif op == "argument":
            values[node.index] = _convert(scalars[node.arg], node.dtype)
        elif op == "program_id":
            values[node.index] = program_id[node.arg]
        elif op == "program_count":
            values[node.index] = grid[node.arg]
        elif op == "lane":
            values[node.index] = lane
        elif op == "cast":
            values[node.index] = _convert(values.get(inputs[0].index), node.dtype)
        elif op == "neg":
            value = values.get(inputs[0].index)
            values[node.index] = None if value is None else _convert(-value, node.dtype)
        elif op in BINARY:
            values[node.index] = _binary(op, values.get(inputs[0].index), values.get(inputs[1].index), node.dtype)
        elif op == "load":
            name = inputs[0].arg
            offset = values.get(inputs[1].index)
            if len(inputs) == 4:
                mask = values.get(inputs[2].index)
                other = values.get(inputs[3].index)
                active = None if mask is None else bool(mask)
            else:
                active, other = True, None
            if active is None:
                value = None
            elif active:
                value = _read(contents, name, offset)
            else:
                value = other
            values[node.index] = value
            accesses.append(Access("load", name, node.index, program_id, lane, offset, active, value))
        elif op == "store":
            name = inputs[0].arg
            offset = values.get(inputs[1].index)
            value = values.get(inputs[2].index)
            mask = values.get(inputs[3].index) if len(inputs) == 4 else True
            active = None if mask is None else bool(mask)
            accesses.append(Access("store", name, node.index, program_id, lane, offset, active, value))
    return accesses


def evaluate(program, grid, scalars=None, buffers=None):
    grid = _triple(grid)
    scalars = dict(scalars or {})
    missing = [name for name in program.scalars if name not in scalars]
    if missing:
        raise ValueError(f"missing scalar arguments for {program.name}: {', '.join(missing)}")
    contents, lengths = {}, {}
    for name in program.buffers:
        given = (buffers or {}).get(name)
        if given is None:
            contents[name], lengths[name] = None, None
        elif isinstance(given, int):
            contents[name], lengths[name] = None, given
        else:
            contents[name] = list(given)
            lengths[name] = len(contents[name])
    live = live_nodes(program.graph)
    nodes = [node for node in program.graph.nodes if node.index in live]
    if any(node.op == "loop" for node in nodes):
        raise NotImplementedError("layout analysis does not support kernels with loops yet")
    accesses = []
    for program_id in itertools.product(*(range(size) for size in grid)):
        for lane in range(program.lanes):
            accesses.extend(_run_thread(nodes, program_id, lane, grid, scalars, contents))
    return Evaluation(program, grid, scalars, lengths, accesses)


def _lane_strides(group):
    lanes_by_program = {}
    for access in group:
        if access.offset is None or access.active is False:
            continue
        lanes_by_program.setdefault(access.program, {})[access.lane] = access.offset
    strides = set()
    for lanes in lanes_by_program.values():
        for lane, offset in lanes.items():
            if lane + 1 in lanes:
                strides.add(lanes[lane + 1] - offset)
    return strides


def analyze(evaluation):
    summaries, warnings = [], []
    groups = {}
    for access in evaluation.accesses:
        groups.setdefault(access.node, []).append(access)
    for node, group in groups.items():
        first = group[0]
        strides = _lane_strides(group)
        stride = next(iter(strides)) if len(strides) == 1 else None
        coalesced = stride == 1 if stride is not None else None
        summaries.append(AccessSummary(first.kind, first.buffer, node, stride, coalesced))
        if len(strides) > 1:
            warnings.append(AccessWarning(
                "irregular_stride", first.buffer, node,
                f"{first.kind} of {first.buffer} has lane-dependent offsets (strides {sorted(strides)})",
                {"strides": sorted(strides)}))
        elif stride is not None and abs(stride) > 1:
            warnings.append(AccessWarning(
                "strided", first.buffer, node,
                f"{first.kind} of {first.buffer} advances {stride} elements per lane: not coalesced",
                {"stride": stride}))
        length = evaluation.lengths.get(first.buffer)
        if length is not None:
            outside = [a.offset for a in group if a.offset is not None and a.active is not False
                       and not 0 <= a.offset < length]
            if outside:
                warnings.append(AccessWarning(
                    "out_of_range", first.buffer, node,
                    f"{first.kind} of {first.buffer} reaches {len(outside)} offsets outside length {length}",
                    {"length": length, "count": len(outside), "min_offset": min(outside), "max_offset": max(outside)}))

    writers = {}
    for access in evaluation.accesses:
        if access.kind != "store" or access.offset is None or access.active is False:
            continue
        writers.setdefault((access.buffer, access.offset), set()).add((access.program, access.lane))
    conflicts = {}
    for (buffer, offset), threads in writers.items():
        if len(threads) > 1:
            conflicts.setdefault(buffer, []).append((offset, len(threads)))
    for buffer in sorted(conflicts):
        entries = sorted(conflicts[buffer])
        warnings.append(AccessWarning(
            "overlapping_writes", buffer, None,
            f"{len(entries)} elements of {buffer} are written by several threads",
            {"count": len(entries), "sample_offsets": [offset for offset, _ in entries[:16]],
             "max_writers": max(count for _, count in entries)}))
    return Analysis(summaries, warnings)


def _flat_program(program_id, grid):
    flat = 0
    for position, size in zip(program_id, grid):
        flat = flat * size + position
    return flat


def element_map(evaluation, buffer, length=None):
    if length is None:
        length = evaluation.lengths.get(buffer)
    touched = {}
    for access in evaluation.accesses:
        if access.buffer != buffer or access.offset is None or access.active is False:
            continue
        programs, threads, writers = touched.setdefault(access.offset, (set(), set(), set()))
        thread = (access.program, access.lane)
        programs.add(_flat_program(access.program, evaluation.grid))
        threads.add(thread)
        if access.kind == "store":
            writers.add(thread)
    if length is None:
        length = max((offset for offset in touched if offset >= 0), default=-1) + 1
    programs, threads, writers = [], [], []
    for offset in range(length):
        touching, touching_threads, storing = touched.get(offset, (set(), set(), set()))
        programs.append(tuple(sorted(touching)))
        threads.append(len(touching_threads))
        writers.append(len(storing))
    return ElementMap(buffer, length, programs, threads, writers)


def _program_glyph(programs):
    if not programs:
        return "."
    if len(programs) > 1:
        return "*"
    return GLYPHS[programs[0]] if programs[0] < len(GLYPHS) else "+"


def _count_glyph(count):
    if count == 0:
        return "."
    return str(count) if count < 10 else "+"


def _rows(length, width, glyphs):
    if not length:
        return ["  (empty)"]
    digits = len(str(length - 1))
    return [f"{start:>{digits}} | {''.join(glyphs[start:start + width])}" for start in range(0, length, width)]


def ascii_map(evaluation, buffer, length=None, width=32):
    emap = element_map(evaluation, buffer, length)
    lines = [f"buffer {buffer}: {emap.length} elements, grid {evaluation.grid}, "
             f"{evaluation.program.lanes} lanes per program",
             "program owning each element ('.' untouched, '*' several programs):"]
    lines += _rows(emap.length, width, [_program_glyph(programs) for programs in emap.programs])
    lines.append("threads touching each element (overlap count, '+' above 9):")
    lines += _rows(emap.length, width, [_count_glyph(count) for count in emap.threads])
    untouched = sum(1 for count in emap.threads if count == 0)
    shared = sum(1 for count in emap.threads if count > 1)
    conflicting = sum(1 for count in emap.writers if count > 1)
    lines.append(f"untouched {untouched}, shared {shared}, written by several threads {conflicting}")
    return "\n".join(lines)


def _fill(programs, writers):
    if not programs:
        return UNTOUCHED_COLOR
    if writers > 1:
        return CONFLICT_COLOR
    if len(programs) > 1:
        return SHARED_COLOR
    return PALETTE[programs[0] % len(PALETTE)]


def _legend(y):
    entries = [(UNTOUCHED_COLOR, "untouched"), (PALETTE[0], "one program (colors per program)"),
               (SHARED_COLOR, "several programs"), (CONFLICT_COLOR, "written by several threads")]
    parts = []
    for position, (color, label) in enumerate(entries):
        x = position * 230
        parts.append(f'<rect x="{x}" y="{y}" width="14" height="14" fill="{color}" stroke="#555"/>')
        parts.append(f'<text x="{x + 20}" y="{y + 11}" font-size="12" fill="#222">{html.escape(label)}</text>')
    return parts


def layout_html(evaluation, buffer, length=None, width=32, path=None):
    emap = element_map(evaluation, buffer, length)
    cell, gap, top = 22, 2, 8
    rows = max(1, -(-emap.length // width))
    svg_width = width * (cell + gap)
    legend_y = top + rows * (cell + gap) + 12
    svg_height = legend_y + 24
    parts = []
    for offset in range(emap.length):
        row, column = divmod(offset, width)
        x = column * (cell + gap)
        y = top + row * (cell + gap)
        title = html.escape(f"element {offset}: programs {list(emap.programs[offset])}, "
                            f"threads {emap.threads[offset]}, writers {emap.writers[offset]}")
        fill = _fill(emap.programs[offset], emap.writers[offset])
        parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#999">'
                     f"<title>{title}</title></rect>")
        parts.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" font-size="9" text-anchor="middle" '
                     f'fill="#222">{offset}</text>')
    parts += _legend(legend_y)
    title = html.escape(f"{evaluation.program.name}: {buffer} ({emap.length} elements)")
    summary = html.escape(f"grid {evaluation.grid}, {evaluation.program.lanes} lanes per program")
    body = "\n".join(parts)
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>body {{ font-family: Helvetica, Arial, sans-serif; margin: 16px; }}</style>
</head>
<body>
<h2>{title}</h2>
<p>{summary}</p>
<svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
{body}
</svg>
</body>
</html>
"""
    if path is not None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(document)
    return document


def _quote(text):
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _node_label(node):
    head = f"%{node.index} {node.op}"
    if node.arg is not None:
        head += f" [{node.arg}]"
    parts = [_quote(head)]
    if node.dtype is not None:
        parts.append(_quote(node.dtype.name))
    return "\\n".join(parts)


def graph_to_dot(program):
    graph = program.graph
    live = live_nodes(graph)
    nodes = [node for node in graph.nodes if node.index in live]
    lines = ["digraph kernel {", "  rankdir=TB;", '  node [shape=box, fontname="Helvetica"];']
    for node in nodes:
        style = f' style=filled fillcolor="{STORE_COLOR}"' if node.op == "store" else ""
        lines.append(f'  n{node.index} [label="{_node_label(node)}"{style}];')
    for node in nodes:
        for item in node.inputs:
            lines.append(f"  n{item.index} -> n{node.index};")
    lines.append("}")
    return "\n".join(lines) + "\n"
