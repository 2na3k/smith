import math
import operator

from .dtype import bool_, float32, int32, uint32
from .ir import BINARY, COMPARISONS, Graph

FOLDABLE = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "and": operator.and_,
    "or": operator.or_,
    "less": operator.lt,
    "less_equal": operator.le,
    "greater": operator.gt,
    "greater_equal": operator.ge,
}


def _wrap(value, dtype):
    if dtype == int32:
        return (value + 2**31) % 2**32 - 2**31
    return value & 0xFFFFFFFF


def _convert(value, dtype):
    if dtype == float32:
        return float(value)
    if dtype == bool_:
        return bool(value)
    return _wrap(int(value), dtype)


def _convertible(value, dtype):
    return not (dtype in (int32, uint32) and isinstance(value, float) and not math.isfinite(value))


def _is_value(node, value):
    return node.op == "const" and node.dtype != bool_ and node.arg == value


def _is_true(node):
    return node.op == "const" and node.dtype == bool_ and node.arg is True


def _is_false(node):
    return node.op == "const" and node.dtype == bool_ and node.arg is False


def _simplify_binary(graph, op, dtype, inputs):
    left, right = inputs
    if left.op == "const" and right.op == "const" and left.dtype != float32 and op in FOLDABLE:
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


def simplify(graph):
    result = Graph()
    result.version = graph.version
    replacements = {}
    for node in graph.nodes:
        inputs = tuple(replacements[i.index] for i in node.inputs)
        replacements[node.index] = _rewrite(result, node, inputs)
    return result
