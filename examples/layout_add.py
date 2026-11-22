import os
import sys
import tempfile

import smith as sth
from smith import dtype
from smith.layout import analyze, ascii_map, evaluate, graph_to_dot, layout_html
from smith.trace import BufferSpec


@sth.jit
def add(x_pointer, y_pointer, out_pointer, n, BLOCK: sth.constexpr = 4):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    y = sth.load(y_pointer + offset, mask=mask)
    sth.store(out_pointer + offset, x + y, mask=mask)


if __name__ == "__main__":
    n = 10
    block = 4
    grid = (sth.cdiv(n, block),)
    spec = BufferSpec(dtype.float32)
    program = add.trace(spec, spec, spec, n, BLOCK=block)
    evaluation = evaluate(program, grid, {"n": n}, {"x_pointer": n, "y_pointer": n, "out_pointer": n})

    print(f"kernel {program.name}: grid {grid}, {program.lanes} lanes per program, {len(evaluation.accesses)} accesses")
    print()
    print(ascii_map(evaluation, "out_pointer"))
    print()

    analysis = analyze(evaluation)
    for summary in analysis.summaries:
        print(f"{summary.kind:<5} {summary.buffer:<12} lane stride {summary.lane_stride} coalesced {summary.coalesced}")
    print("warnings: none" if analysis.clean else f"warnings: {analysis.warnings}")

    directory = tempfile.gettempdir()
    html_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(directory, "smith_layout_add.html")
    dot_path = os.path.splitext(html_path)[0] + ".dot"
    layout_html(evaluation, "out_pointer", path=html_path)
    with open(dot_path, "w", encoding="utf-8") as handle:
        handle.write(graph_to_dot(program))
    print()
    print(f"wrote {html_path}")
    print(f"wrote {dot_path}")
