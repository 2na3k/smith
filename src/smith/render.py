import math

from .dtype import bool_, float16, float32
from .ir import LEAVES, live_nodes

SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "truediv": "/",
    "and": "&",
    "or": "|",
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
    if op == "floordiv":
        if node.dtype == float32:
            return f"floor({arguments[0]} / {arguments[1]})"
        return f"({arguments[0]} / {arguments[1]})"
    if op == "mod":
        if node.dtype == float32:
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


def render_statements(graph, leaf=metal_leaf):
    live = live_nodes(graph)
    names = {}
    lines = []
    for node in graph.nodes:
        if node.index not in live or node.op in LEAVES or node.op == "const":
            continue
        if node.op == "store":
            lines.append(_store(node, names, leaf))
            continue
        variable = f"value_{len(names) + 1}"
        names[node.index] = variable
        lines.append(f"{node.dtype.msl} {variable} = {_expression(node, names, leaf)};")
    return lines


def render_metal(graph, name, parameters):
    lines = render_statements(graph)
    signature = ",\n    ".join(parameters)
    body = "\n".join(f"    {line}" for line in lines)
    return f"#include <metal_stdlib>\nusing namespace metal;\nkernel void {name}(\n    {signature}) {{\n{body}\n}}\n"
