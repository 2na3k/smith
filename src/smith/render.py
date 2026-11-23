import itertools
import math

from .dtype import bool_, float16, float32
from .ir import LEAVES, MATRIX, live_nodes, walk

SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "truediv": "/",
    "and": "&",
    "or": "|",
    "shl": "<<",
    "shr": ">>",
    "less": "<",
    "less_equal": "<=",
    "greater": ">",
    "greater_equal": ">=",
}


LITERAL_SUFFIXES = {"int32": "", "uint32": "u", "int64": "l", "uint64": "ul"}


def _float32_text(value):
    if math.isnan(value):
        return "NAN"
    if math.isinf(value):
        return "INFINITY" if value > 0 else "(-INFINITY)"
    text = f"{value!r}f"
    return f"({text})" if value < 0 else text


def _literal(node):
    value, dtype = node.arg, node.dtype
    if dtype == bool_:
        return "true" if value else "false"
    if dtype.kind == "float":
        if dtype == float32:
            return _float32_text(value)
        if dtype == float16 and math.isfinite(value):
            return f"({value!r}h)" if value < 0 else f"{value!r}h"
        return f"{dtype.msl}({_float32_text(value)})"
    if dtype.name in LITERAL_SUFFIXES:
        if dtype.name == "int64" and value == -2**63:
            return "(-9223372036854775807l - 1l)"
        text = f"{value}{LITERAL_SUFFIXES[dtype.name]}"
    else:
        text = f"{dtype.msl}({value})"
    return f"({text})" if value < 0 else text


def metal_leaf(node):
    if node.op in ("argument", "buffer"):
        return node.arg
    if node.op == "shared":
        return node.arg[0]
    if node.op == "program_id":
        return f"int(program_id.{'xyz'[node.arg]})"
    if node.op == "program_count":
        return f"int(program_count.{'xyz'[node.arg]})"
    if node.op == "lane":
        return "int(lane_id.x)"
    raise ValueError(f"cannot render leaf {node.op}")


def _reference(node, names, leaf):
    if node.op == "const":
        return _literal(node)
    if node.op in LEAVES:
        return leaf(node)
    return names[node.index]


def _expression(node, names, leaf):
    op = node.op
    arguments = [_reference(i, names, leaf) for i in node.inputs]
    if op in SYMBOLS:
        return f"({arguments[0]} {SYMBOLS[op]} {arguments[1]})"
    if op in ("maximum", "minimum"):
        return f"{op[:3]}({arguments[0]}, {arguments[1]})"
    if op in ("exp", "log", "sqrt", "rsqrt"):
        return f"{op}({arguments[0]})"
    if op == "matrix_element":
        return f"{arguments[0]}.thread_elements()[{node.arg}]"
    if op == "shuffle_xor":
        return f"simd_shuffle_xor({arguments[0]}, ushort({node.arg}))"
    if op == "where":
        return f"({arguments[0]} ? {arguments[1]} : {arguments[2]})"
    if op == "floordiv":
        if node.dtype.kind == "float":
            return f"floor({arguments[0]} / {arguments[1]})"
        return f"({arguments[0]} / {arguments[1]})"
    if op == "mod":
        if node.dtype.kind == "float":
            return f"fmod({arguments[0]}, {arguments[1]})"
        return f"({arguments[0]} % {arguments[1]})"
    if op == "neg":
        return f"(-{arguments[0]})"
    if op == "cast":
        return f"{node.dtype.msl}({arguments[0]})"
    if op == "load":
        element = f"{arguments[0]}[{arguments[1]}]"
        if len(arguments) == 2:
            return element
        return f"({arguments[2]} ? {element} : {arguments[3]})"
    raise ValueError(f"cannot render {op}")


def _store(node, names, leaf):
    buffer, offset, value, *mask = [_reference(i, names, leaf) for i in node.inputs]
    assignment = f"{buffer}[{offset}] = {value};"
    return f"if ({mask[0]}) {{ {assignment} }}" if mask else assignment


COPY_TYPES = {32: "uint", 64: "uint2", 128: "uint4"}


def _copy(node, names, leaf):
    destination, destination_offset, source, source_offset = [_reference(i, names, leaf) for i in node.inputs]
    vector = COPY_TYPES[node.inputs[0].dtype.bits * node.arg]
    return (f"*(threadgroup {vector}*)(&{destination}[{destination_offset}]) = "
            f"*(device const {vector}*)(&{source}[{source_offset}]);")


def _inner_leaf(loop_id, capture_references, outer_leaf):
    def leaf(node):
        if node.op == "capture":
            return capture_references[node.arg]
        if node.op == "loop_index":
            return f"index_{loop_id}"
        if node.op == "carried":
            return f"carried_{loop_id}_{node.arg}"
        return outer_leaf(node)

    return leaf


def _matrix_lines(node, variable, names, leaf):
    arguments = [_reference(i, names, leaf) for i in node.inputs]
    declaration = f"{node.dtype.msl} {variable}"
    if node.op == "matrix_fill":
        return [f"{declaration} = make_filled_simdgroup_matrix<{node.dtype.element.msl}, 8, 8>({arguments[0]});"]
    if node.op == "matrix_load":
        transpose = "true" if node.arg[0] else "false"
        return [f"{declaration};",
                f"simdgroup_load({variable}, {arguments[0]} + {arguments[1]}, {arguments[2]}, ulong2(0, 0), {transpose});"]
    if node.op == "matrix_mma":
        return [f"{declaration};", f"simdgroup_multiply_accumulate({variable}, {arguments[0]}, {arguments[1]}, {arguments[2]});"]
    return [f"{declaration};", f"{variable}.thread_elements()[0] = {arguments[0]};",
            f"{variable}.thread_elements()[1] = {arguments[1]};"]


def _loop(node, names, leaf, counter):
    body = node.arg
    loop_id = next(counter)
    carried_count = len(body.carried)
    start = _reference(node.inputs[0], names, leaf)
    end = _reference(node.inputs[1], names, leaf)
    initial = [_reference(n, names, leaf) for n in node.inputs[2:2 + carried_count]]
    captures = [_reference(n, names, leaf) for n in node.inputs[2 + carried_count:]]
    lines = [f"{variable.dtype.msl} carried_{loop_id}_{slot} = {value};"
             for slot, (variable, value) in enumerate(zip(body.carried, initial))]
    inner_leaf = _inner_leaf(loop_id, captures, leaf)
    inner_lines, inner_names = _render(body.graph, inner_leaf, counter, body.outputs)
    lines.append(f"for (int index_{loop_id} = {start}; index_{loop_id} < {end}; index_{loop_id}++) {{")
    lines.extend(f"    {line}" for line in inner_lines)
    updates = [(slot, variable.dtype.msl, _reference(output, inner_names, inner_leaf))
               for slot, (variable, output) in enumerate(zip(body.carried, body.outputs))]
    lines.extend(f"    {dtype} next_{loop_id}_{slot} = {value};" for slot, dtype, value in updates)
    lines.extend(f"    carried_{loop_id}_{slot} = next_{loop_id}_{slot};" for slot, _, _ in updates)
    lines.append("}")
    return loop_id, lines


def _render(graph, leaf, counter, roots=()):
    live = live_nodes(graph, roots)
    names = {}
    loops = {}
    lines = []
    for node in graph.nodes:
        if node.index not in live or node.op in LEAVES:
            continue
        if node.op == "store":
            lines.append(_store(node, names, leaf))
        elif node.op == "barrier":
            lines.append("threadgroup_barrier(mem_flags::mem_threadgroup);")
        elif node.op == "copy":
            lines.append(_copy(node, names, leaf))
        elif node.op == "loop":
            loops[node.index], loop_lines = _loop(node, names, leaf, counter)
            lines.extend(loop_lines)
        elif node.op == "loop_result":
            names[node.index] = f"carried_{loops[node.inputs[0].index]}_{node.arg}"
        elif node.op in MATRIX and node.op != "matrix_element":
            variable = f"value_{next(counter)}"
            names[node.index] = variable
            lines.extend(_matrix_lines(node, variable, names, leaf))
        else:
            variable = f"value_{next(counter)}"
            names[node.index] = variable
            lines.append(f"{node.dtype.msl} {variable} = {_expression(node, names, leaf)};")
    return lines, names


def shared_declarations(graph):
    seen = {}
    for node in walk(graph):
        if node.op == "shared":
            seen[node.arg[0]] = f"threadgroup {node.dtype.msl} {node.arg[0]}[{node.arg[1]}];"
    return list(seen.values())


def render_statements(graph, leaf=metal_leaf):
    return shared_declarations(graph) + _render(graph, leaf, itertools.count(1))[0]


def render_metal(graph, name, parameters):
    lines = render_statements(graph)
    signature = ",\n    ".join(parameters)
    body = "\n".join(f"    {line}" for line in lines)
    return f"#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\nkernel void {name}(\n    {signature}) {{\n{body}\n}}\n"
