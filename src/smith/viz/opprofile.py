CATEGORIES = {
    "global load": {"cost": 12.0, "what": "Reads from device memory. Latency is hidden by other SIMD groups, so the real cost depends on occupancy and coalescing."},
    "vector copy": {"cost": 14.0, "what": "A 4 to 16 byte vector copy per thread from device memory into threadgroup memory."},
    "global store": {"cost": 12.0, "what": "Writes to device memory."},
    "matrix load": {"cost": 16.0, "what": "simdgroup_load of an 8x8 tile straight from device memory."},
    "shared memory": {"cost": 3.0, "what": "Threadgroup memory reads, writes and tile loads from it."},
    "barrier": {"cost": 10.0, "what": "threadgroup_barrier: every SIMD group in the threadgroup waits here for the slowest one."},
    "matrix multiply": {"cost": 16.0, "what": "simdgroup_multiply_accumulate on 8x8 tiles, the matrix units' work."},
    "matrix registers": {"cost": 1.0, "what": "Filling, splitting and building matrix fragments in registers."},
    "special math": {"cost": 4.0, "what": "exp, log, sqrt, rsqrt: several times the cost of a plain add."},
    "float arithmetic": {"cost": 1.0, "what": "Float add, subtract, multiply, negate, divide."},
    "index math": {"cost": 1.0, "what": "Integer address and loop arithmetic."},
    "select / convert": {"cost": 1.0, "what": "Comparisons, where, min/max, casts and boolean logic."},
    "shuffle": {"cost": 2.0, "what": "simd_shuffle_xor: data exchange between lanes of one SIMD group."},
}
OP_COST = {"truediv": 4.0, "floordiv": 4.0, "mod": 4.0}
FREE = ("const", "argument", "buffer", "shared", "program_id", "program_count", "lane", "capture", "loop_index",
        "carried", "loop_result", "loop")
SPECIAL = ("exp", "log", "sqrt", "rsqrt")
SELECT = ("where", "cast", "maximum", "minimum", "less", "less_equal", "greater", "greater_equal", "and", "or")
FLOAT_KINDS = ("float16", "bfloat16", "float32")
NODE_LIMIT = 24
SEQUENCE_LIMIT = 80


def _category(node, by_id):
    op = node["op"]
    if op in FREE:
        return None
    first = by_id.get(node["inputs"][0]) if node["inputs"] else None
    in_shared = first is not None and first["op"] == "shared"
    if op in ("load", "store"):
        return "shared memory" if in_shared else ("global load" if op == "load" else "global store")
    if op == "matrix_load":
        return "shared memory" if in_shared else "matrix load"
    if op == "copy":
        return "vector copy"
    if op == "barrier":
        return "barrier"
    if op == "matrix_mma":
        return "matrix multiply"
    if op in ("matrix_fill", "matrix_element", "matrix_make"):
        return "matrix registers"
    if op in SPECIAL:
        return "special math"
    if op == "shuffle_xor":
        return "shuffle"
    if op in SELECT:
        return "select / convert"
    if node["dtype"] in FLOAT_KINDS:
        return "float arithmetic"
    return "index math"


def _node_cost(node, category):
    return OP_COST.get(node["op"], CATEGORIES[category]["cost"])


def _value(node_id, by_id):
    node = by_id.get(node_id)
    if node is None:
        return None
    op = node["op"]
    if op == "const":
        try:
            return int(float(node["arg"]))
        except (TypeError, ValueError):
            return None
    inputs = [_value(source, by_id) for source in node["inputs"]]
    if op == "cast":
        return inputs[0]
    if op == "minimum":
        known = [value for value in inputs if value is not None]
        return min(known) if known else None
    if any(value is None for value in inputs):
        return None
    if op == "add":
        return inputs[0] + inputs[1]
    if op == "sub":
        return inputs[0] - inputs[1]
    if op == "mul":
        return inputs[0] * inputs[1]
    if op == "floordiv" and inputs[1]:
        return inputs[0] // inputs[1]
    return None


def _trips(node, by_id):
    start, end = _value(node["inputs"][0], by_id), _value(node["inputs"][1], by_id)
    if start is None or end is None:
        return 1, True
    exact = all(by_id[source]["op"] != "minimum" for source in node["inputs"][:2])
    return max(0, end - start), not exact


def _section(graph, label, trips, estimated):
    by_id = {node["id"]: node for node in graph["nodes"]}
    items, run = [], None
    for node in graph["nodes"]:
        if node["op"] == "loop":
            run = None
            count, guess = _trips(node, by_id)
            items.append(_section(node["body"], f"loop %{node['id']}", count, guess))
            continue
        category = _category(node, by_id)
        if category is None:
            continue
        cost = _node_cost(node, category)
        if run is None or run["category"] != category:
            run = {"type": "run", "category": category, "count": 0, "cost": 0.0, "ops": {}, "nodes": []}
            items.append(run)
        run["count"] += 1
        run["cost"] += cost
        run["ops"][node["op"]] = run["ops"].get(node["op"], 0) + 1
        if len(run["nodes"]) < NODE_LIMIT:
            run["nodes"].append(node["id"])
    cost = sum(item["cost"] * (item["trips"] if item["type"] == "loop" else 1) for item in items)
    return {"type": "loop", "label": label, "trips": trips, "estimated": estimated, "cost": cost, "items": items}


def _phase(label, multiplier, trips, estimated):
    return {"label": label, "multiplier": multiplier, "trips": trips, "estimated": estimated, "cost": 0.0, "count": 0,
            "categories": {}, "ops": {}, "sequence": []}


def _phases(section, multiplier, label, trips, estimated, out):
    emitted = []

    def start():
        return _phase(label, multiplier, trips, estimated)

    def finish(phase):
        if phase["count"]:
            emitted.append(phase)
            out.append(phase)

    current = start()
    for item in section["items"]:
        if item["type"] == "loop":
            finish(current)
            _phases(item, multiplier * item["trips"], f"{item['label']} x{item['trips']}", item["trips"], item["estimated"], out)
            current = start()
            continue
        sequence = current["sequence"]
        if sequence and sequence[-1][0] == item["category"]:
            sequence[-1][1] += item["count"]
            sequence[-1][2] += item["cost"]
        elif len(sequence) < SEQUENCE_LIMIT:
            sequence.append([item["category"], item["count"], item["cost"]])
        weight = item["cost"] * multiplier
        entry = current["categories"].setdefault(item["category"], {"cost": 0.0, "count": 0})
        entry["cost"] += weight
        entry["count"] += item["count"] * multiplier
        current["cost"] += weight
        current["count"] += item["count"] * multiplier
        for op, count in item["ops"].items():
            current["ops"][op] = current["ops"].get(op, 0) + count * multiplier
    finish(current)
    if len(emitted) > 1:
        for number, phase in enumerate(emitted, 1):
            phase["label"] = f"{label} part {number}"


def profile(graph):
    root = _section(graph, "kernel", 1, False)
    phases = []
    _phases(root, 1, "kernel", 1, False, phases)
    total = sum(phase["cost"] for phase in phases)
    totals = {}
    for phase in phases:
        phase["dominant"] = max(phase["categories"], key=lambda name: phase["categories"][name]["cost"])
        for name, entry in phase["categories"].items():
            summed = totals.setdefault(name, {"cost": 0.0, "count": 0})
            summed["cost"] += entry["cost"]
            summed["count"] += entry["count"]
    return {"total": total, "phases": phases, "categories": totals, "weights": CATEGORIES}
