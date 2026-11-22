import math
import operator

from .dtype import bool_, float32
from .ir import BINARY, COMPARISONS, SHIFTS, Graph, LoopBody

FOLDABLE = {
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


def _wrap(value, dtype):
    modulus = 1 << dtype.bits
    value %= modulus
    if dtype.kind == "int" and value >= modulus // 2:
        value -= modulus
    return value


def _convert(value, dtype):
    if dtype.kind == "float":
        return dtype.unpack(dtype.pack([value]), 1)[0]
    if dtype == bool_:
        return bool(value)
    return _wrap(int(value), dtype)


def _convertible(value, dtype):
    return not (dtype.kind in ("int", "uint") and isinstance(value, float) and not math.isfinite(value))


def _is_value(node, value):
    return node.op == "const" and node.dtype != bool_ and node.arg == value


def _is_true(node):
    return node.op == "const" and node.dtype == bool_ and node.arg is True


def _is_false(node):
    return node.op == "const" and node.dtype == bool_ and node.arg is False


def _simplify_binary(graph, op, dtype, inputs):
    left, right = inputs
    if (left.op == "const" and right.op == "const" and left.dtype.kind != "float" and op in FOLDABLE
            and (op not in SHIFTS or 0 <= right.arg < left.dtype.bits)):
        return graph.node("const", (), _convert(FOLDABLE[op](left.arg, right.arg), dtype), dtype)
    if op == "add":
        if _is_value(right, 0) and left.dtype == dtype:
            return left
        if _is_value(left, 0) and right.dtype == dtype:
            return right
    elif op == "sub":
        if _is_value(right, 0) and left.dtype == dtype:
            return left
    elif op == "mul":
        if _is_value(right, 1) and left.dtype == dtype:
            return left
        if _is_value(left, 1) and right.dtype == dtype:
            return right
        if dtype != float32 and (_is_value(left, 0) or _is_value(right, 0)):
            return graph.node("const", (), 0, dtype)
    elif op in SHIFTS:
        if _is_value(right, 0) and left.dtype == dtype:
            return left
        if _is_value(left, 0):
            return graph.node("const", (), 0, dtype)
    elif op == "floordiv":
        if dtype != float32 and _is_value(right, 1):
            return left
    return None


def _rewrite(graph, node, inputs):
    op, dtype = node.op, node.dtype
    if op == "cast":
        (source,) = inputs
        if source.dtype == dtype:
            return source
        if source.op == "const" and _convertible(source.arg, dtype):
            return graph.node("const", (), _convert(source.arg, dtype), dtype)
    elif op == "neg":
        (value,) = inputs
        if value.op == "neg":
            return value.inputs[0]
        if value.op == "const" and value.dtype != float32:
            return graph.node("const", (), _convert(-value.arg, dtype), dtype)
    elif op in BINARY:
        folded = _simplify_binary(graph, op, dtype, inputs)
        if folded is not None:
            return folded
    elif op == "matrix_element":
        (matrix,) = inputs
        if matrix.op == "matrix_make":
            return matrix.inputs[node.arg]
    elif op == "where":
        condition, left, right = inputs
        if _is_true(condition):
            return left
        if _is_false(condition):
            return right
    elif op == "loop":
        body = node.arg
        inner, replaced = _simplify(body.graph)
        simplified = LoopBody(inner, replaced[body.index.index], [replaced[n.index] for n in body.carried],
                              [replaced[n.index] for n in body.outputs])
        simplified.captures = list(inputs[2 + len(body.carried):])
        return graph.node("loop", inputs, simplified, None, unique=True)
    elif op == "load":
        if len(inputs) == 4 and _is_true(inputs[2]):
            inputs = inputs[:2]
    elif op == "store":
        if len(inputs) == 4:
            if _is_true(inputs[3]):
                inputs = inputs[:3]
            elif _is_false(inputs[3]):
                return None
    return graph.node(op, inputs, node.arg, dtype, unique=op == "store")


def _simplify(graph):
    result = Graph()
    result.version = graph.version
    replacements = {}
    for node in graph.nodes:
        inputs = tuple(replacements[i.index] for i in node.inputs)
        replacements[node.index] = _rewrite(result, node, inputs)
    return result, replacements


def simplify(graph):
    return _simplify(graph)[0]
