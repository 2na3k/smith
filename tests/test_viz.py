import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

import smith as sth
from smith import dtype, viz
from smith.trace import BufferSpec
from smith.viz.server import make_server

if sys.platform == "darwin":
    from smith.compiler import Buffer


@sth.jit
def add(x_pointer, y_pointer, out_pointer, n, BLOCK: sth.constexpr = 64):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    sth.store(out_pointer + offset, sth.load(x_pointer + offset, mask=mask) + sth.load(y_pointer + offset, mask=mask), mask=mask)


@sth.jit
def row_sum(x_pointer, out_pointer, columns, BLOCK: sth.constexpr = 8):
    row = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    (total,) = sth.loop(columns, (0.0,), lambda column, acc: (acc + sth.load(x_pointer + row * columns + column),))
    sth.store(out_pointer + row, total)


def fetch(server, path):
    url = f"http://127.0.0.1:{server.server_address[1]}{path}"
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.read()


class Recording(unittest.TestCase):
    def setUp(self):
        viz.disable()
        viz.reset()
        viz.enable()

    def tearDown(self):
        viz.disable()
        viz.reset()


class TestRecording(Recording):
    def trace_add(self, block=64):
        spec = BufferSpec(dtype.float32)
        return add.trace(spec, spec, spec, 10, BLOCK=block)

    def test_traced_programs_are_recorded_with_both_ir_stages(self):
        program = self.trace_add()
        detail = viz.program_detail(0)
        self.assertEqual(detail["name"], "add")
        self.assertEqual(detail["source"], program.source)
        self.assertEqual(detail["constexprs"], {"BLOCK": "64"})
        self.assertEqual(detail["buffers"], {"x_pointer": "float32", "y_pointer": "float32", "out_pointer": "float32"})
        self.assertEqual(detail["loaded"], ["x_pointer", "y_pointer"])
        self.assertEqual(detail["stored"], ["out_pointer"])
        self.assertGreater(len(detail["stages"]["traced"]["nodes"]), len(detail["stages"]["simplified"]["nodes"]))

    def test_simplify_savings_show_up_in_op_counts(self):
        self.trace_add()
        counts = viz.program_detail(0)["op_counts"]
        self.assertGreater(counts["traced"]["cast"], counts["simplified"].get("cast", 0))

    def test_loop_bodies_are_nested_in_the_graph(self):
        spec = BufferSpec(dtype.float32)
        row_sum.trace(spec, spec, 4)
        nodes = viz.program_detail(0)["stages"]["simplified"]["nodes"]
        loop = next(node for node in nodes if node["op"] == "loop")
        self.assertIn("load", [node["op"] for node in loop["body"]["nodes"]])
        self.assertEqual(len(loop["body"]["carried"]), 1)
        self.assertGreater(viz.program_detail(0)["op_counts"]["simplified"]["load"], 0)

    def test_only_live_nodes_are_included(self):
        self.trace_add()
        stage = viz.program_detail(0)["stages"]["traced"]
        self.assertLessEqual(len(stage["nodes"]), stage["total"])

    def test_identical_programs_are_recorded_once(self):
        self.trace_add()
        self.trace_add()
        self.assertEqual(len(viz.snapshot()["programs"]), 1)
        self.trace_add(block=32)
        self.assertEqual(len(viz.snapshot()["programs"]), 2)

    def test_nothing_is_recorded_when_disabled(self):
        viz.disable()
        self.trace_add(block=16)
        self.assertEqual(viz.snapshot()["programs"], [])

    def test_unknown_program_is_none(self):
        self.assertIsNone(viz.program_detail(0))
        self.assertIsNone(viz.program_detail(-1))

    def test_tuning_phase_marks_candidate_programs(self):
        with viz.phase("tuning"):
            self.trace_add(block=16)
        self.trace_add(block=32)
        programs = viz.snapshot()["programs"]
        self.assertEqual([program["candidate"] for program in programs], [True, False])

    def test_autotune_results_are_recorded(self):
        record = {"config": {"BLOCK": 64}, "results": [{"config": {"BLOCK": 64}, "seconds": 0.001, "error": None}]}
        viz.record_autotune("add", "{}", record, False)
        entry = viz.snapshot()["autotunes"][0]
        self.assertEqual(entry["best"], {"BLOCK": 64})
        self.assertFalse(entry["cached"])


class TestOccupancy(unittest.TestCase):
    def test_every_core_gets_work(self):
        info = viz.occupancy((1024, 4, 1), (256, 1, 1), 32, 16)
        self.assertEqual((info["groups"], info["idle_cores"], info["core_efficiency"]), (16, 0, 1.0))
        self.assertEqual(info["core_loads"], [1] * 16)

    def test_fewer_threadgroups_than_cores_leaves_cores_idle(self):
        info = viz.occupancy((512, 4, 1), (256, 1, 1), 32, 16)
        self.assertEqual(info["groups"], 8)
        self.assertEqual(info["idle_cores"], 8)
        self.assertEqual(info["core_efficiency"], 0.5)
        self.assertEqual(info["core_loads"], [1] * 8 + [0] * 8)

    def test_a_partial_last_round_lowers_efficiency(self):
        info = viz.occupancy((20 * 64, 1, 1), (64, 1, 1), 32, 16)
        self.assertEqual(info["groups"], 20)
        self.assertEqual(info["core_loads"], [2] * 4 + [1] * 12)
        self.assertAlmostEqual(info["core_efficiency"], (20 / 16) / 2)

    def test_threadgroups_that_are_not_a_multiple_of_the_simd_width(self):
        info = viz.occupancy((65, 1, 1), (65, 1, 1), 32, 16)
        self.assertEqual(info["simd_groups"], 3)
        self.assertAlmostEqual(info["simd_efficiency"], 65 / 96)

    def test_unknown_core_count_still_reports_simd_use(self):
        info = viz.occupancy((64, 1, 1), (32, 1, 1), 32, None)
        self.assertNotIn("core_loads", info)
        self.assertEqual(info["groups"], 2)
        self.assertEqual(info["simd_efficiency"], 1.0)

    def test_hardware_description(self):
        hardware = viz.hardware()
        self.assertEqual(hardware["simd_width"], 32)
        self.assertTrue(hardware["gpu_cores"] is None or hardware["gpu_cores"] > 0)


class TestGaps(unittest.TestCase):
    def row(self, number, start, end, phase="run", encode=None):
        encode = start - 0.05 if encode is None else encode
        return {"id": number, "name": "kernel", "phase": phase, "start": start, "end": end, "ms": end - start,
                "encode_start": encode - 0.02, "submit": encode}

    def span(self, kind, start, end, phase="run"):
        return {"kind": kind, "label": "x", "start": start, "end": end, "ms": end - start, "phase": phase}

    def test_gap_causes_come_from_overlapping_host_spans(self):
        rows = [self.row(1, 0.0, 1.0), self.row(2, 2.0, 3.0, encode=1.95)]
        gaps = viz._gaps(rows, [self.span("trace", 1.1, 1.5)])
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0]["ms"], 1.0)
        self.assertAlmostEqual(gaps[0]["causes"]["trace"], 0.4)
        self.assertGreater(gaps[0]["causes"]["other host code"], 0)

    def test_gaps_next_to_tuning_launches_are_ignored(self):
        rows = [self.row(1, 0.0, 1.0), self.row(2, 5.0, 6.0, phase="tuning"), self.row(3, 9.0, 10.0)]
        self.assertEqual(viz._gaps(rows, []), [])

    def test_tiny_gaps_are_ignored(self):
        rows = [self.row(1, 0.0, 1.0), self.row(2, 1.002, 2.0)]
        self.assertEqual(viz._gaps(rows, []), [])

    def test_queue_wait_is_a_cause(self):
        rows = [self.row(1, 0.0, 1.0), self.row(2, 2.0, 3.0, encode=1.2)]
        gaps = viz._gaps(rows, [])
        self.assertAlmostEqual(gaps[0]["causes"]["queue"], 0.8)


class TestSpans(Recording):
    def test_spans_are_recorded_with_their_phase(self):
        with viz.span("trace", "demo"):
            pass
        with viz.phase("tuning"):
            with viz.span("compile", "demo"):
                pass
        spans = viz.snapshot()["spans"]
        self.assertEqual([(item["kind"], item["phase"]) for item in spans], [("trace", "run"), ("compile", "tuning")])
        self.assertTrue(all(item["ms"] >= 0 for item in spans))

    def test_spans_are_ignored_when_disabled(self):
        viz.disable()
        with viz.span("trace", "demo"):
            pass
        self.assertEqual(viz.snapshot()["spans"], [])

    def test_spans_are_attributed_to_the_kernel_not_just_its_name(self):
        spec = BufferSpec(dtype.float32)
        first = add.trace(spec, spec, spec, 10, BLOCK=16)
        second = add.trace(spec, spec, spec, 10, BLOCK=32)
        self.assertEqual(first.name, second.name)
        with viz.span("wait", "add", (first.name, first.source)):
            pass
        with viz.span("wait", "add", (second.name, second.source)):
            pass
        waits = [item for item in viz.snapshot()["spans"] if item["kind"] == "wait"]
        self.assertEqual([item["program"] for item in waits], [0, 1])

    def test_wait_spans_report_how_long_after_the_kernel_finished_the_call_returned(self):
        class FakeLaunch:
            done = True
            failed = False

            def __init__(self, end):
                self.end = end
                self.kernel = type("K", (), {"name": "k", "source": "s", "execution_width": lambda self: 32})()
                self.submitted = time.perf_counter()
                self.encode_started = self.submitted
                self.encode_ended = self.submitted
                self.submission = type("S", (), {"launches": [None], "committed": True})()
                self.grid, self.group = (1, 1, 1), (1, 1, 1)

            @property
            def gpu_interval(self):
                return self.end - 0.001, self.end

        launch = FakeLaunch(time.perf_counter())
        viz._observe(launch)
        time.sleep(0.002)
        with viz.span("wait", "k", ("k", "s"), launch):
            time.sleep(0.003)
        row = next(item for item in viz.snapshot()["spans"] if item["kind"] == "wait")
        self.assertEqual(row["launch"], 1)
        self.assertGreater(row["finished_before"], 1.5)
        self.assertGreater(row["notify_lag"], row["finished_before"] + 2.5)

    def test_wait_spans_without_a_launch_have_no_lag_figures(self):
        with viz.span("wait", "k"):
            pass
        row = viz.snapshot()["spans"][0]
        self.assertIsNone(row["launch"])
        self.assertIsNone(row["notify_lag"])

    def test_a_span_without_a_known_kernel_has_no_program(self):
        with viz.span("wait", "mystery", ("mystery", "source")):
            pass
        self.assertIsNone(viz.snapshot()["spans"][0]["program"])

    def test_a_span_key_can_be_set_inside_the_block_even_when_disabled(self):
        viz.disable()
        with viz.span("trace", "demo") as active:
            active.key = ("demo", "source")

    def test_traced_spans_get_their_kernels_program(self):
        @sth.jit
        def named(x_pointer, out_pointer, BLOCK: sth.constexpr = 8):
            lane = sth.arange(0, BLOCK)
            sth.store(out_pointer + lane, sth.load(x_pointer + lane))

        spec = BufferSpec(dtype.float32)
        named.trace(spec, spec)
        trace = next(item for item in viz.snapshot()["spans"] if item["kind"] == "trace" and item["label"] == "named")
        self.assertEqual(trace["program"], 0)

    def test_tracing_a_kernel_records_a_trace_span(self):
        @sth.jit
        def fresh(x_pointer, out_pointer, BLOCK: sth.constexpr = 8):
            lane = sth.arange(0, BLOCK)
            sth.store(out_pointer + lane, sth.load(x_pointer + lane))

        spec = BufferSpec(dtype.float32)
        fresh.trace(spec, spec)
        self.assertIn(("trace", "fresh"), [(item["kind"], item["label"]) for item in viz.snapshot()["spans"]])


class TestServer(Recording):
    def setUp(self):
        super().setUp()
        spec = BufferSpec(dtype.float32)
        add.trace(spec, spec, spec, 10, BLOCK=64)
        self.server = make_server(0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        super().tearDown()

    def test_page_is_served(self):
        status, body = fetch(self.server, "/")
        self.assertEqual(status, 200)
        self.assertIn(b"smith viz", body)
        self.assertNotIn(b"http://", body.replace(b"http://www.w3.org", b""))

    def test_state_endpoint(self):
        state = json.loads(fetch(self.server, "/api/state")[1])
        self.assertEqual(state["programs"][0]["name"], "add")
        self.assertIn("version", state)
        self.assertEqual(state["launches"], [])

    def test_program_endpoint(self):
        detail = json.loads(fetch(self.server, "/api/program/0")[1])
        self.assertIn("kernel void add(", detail["source"])
        self.assertEqual(set(detail["stages"]), {"traced", "simplified"})

    def test_missing_resources_are_404(self):
        for path in ("/api/program/7", "/api/program/x", "/nope"):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                fetch(self.server, path)
            self.assertEqual(caught.exception.code, 404, path)

    def test_busy_port_falls_back_to_the_next_one(self):
        taken = self.server.server_address[1]
        other = make_server(taken, tries=5)
        try:
            self.assertNotEqual(other.server_address[1], taken)
        finally:
            other.server_close()


@unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
class TestMetalRecording(Recording):
    def test_launches_carry_gpu_timing_and_their_program(self):
        n = 256
        x = Buffer.from_list(dtype.float32, [1.0] * n)
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        for _ in range(3):
            add[(4,)](x, x, out, n, BLOCK=64)
        sth.synchronize()
        state = viz.snapshot()
        self.assertEqual(len(state["launches"]), 3)
        launch = state["launches"][0]
        self.assertEqual(launch["program"], 0)
        self.assertEqual(launch["grid"], [256, 1, 1])
        self.assertEqual(launch["group"], [64, 1, 1])
        self.assertGreater(launch["ms"], 0)
        program = state["programs"][0]
        self.assertEqual(program["launches"], 3)
        self.assertIsNotNone(program["compile_ms"])
        self.assertLessEqual(program["min_ms"], program["mean_ms"])

    def test_launch_rows_split_host_and_gpu_time(self):
        n = 1 << 16
        x = Buffer.from_list(dtype.float32, [1.0] * n)
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        for _ in range(3):
            add[(n // 64,)](x, x, out, n, BLOCK=64)
        sth.synchronize()
        state = viz.snapshot()
        for row in state["launches"]:
            self.assertGreater(row["encode_ms"], 0)
            self.assertGreaterEqual(row["queue_ms"], 0)
            self.assertAlmostEqual(row["queue_ms"], row["start"] - row["submit"])
            self.assertGreaterEqual(row["start"], row["submit"])
            self.assertEqual(row["groups"], n // 64)
            self.assertEqual(row["simd_efficiency"], 1.0)
        kinds = {item["kind"] for item in state["spans"]}
        self.assertTrue({"trace"} <= kinds or state["programs"][0]["compile_ms"] is not None)
        self.assertGreater(state["stats"]["gpu_busy_ms"], 0)
        self.assertGreater(state["stats"]["encode_ms"], 0)

    def test_batched_launches_become_one_gpu_row_plus_encode_only_members(self):
        n = 1 << 16
        x = Buffer.from_list(dtype.float32, [1.0] * n)
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        with sth.batch():
            for _ in range(4):
                add[(n // 64,)](x, x, out, n, BLOCK=64)
        sth.synchronize()
        state = viz.snapshot()
        members = [row for row in state["launches"] if "member_of" in row]
        batches = [row for row in state["launches"] if "batch_size" in row]
        self.assertEqual(len(members), 4)
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(batch["batch_size"], 4)
        self.assertTrue(all(row["member_of"] == batch["id"] and "ms" not in row for row in members))
        self.assertGreater(batch["ms"], 0)
        self.assertGreaterEqual(batch["queue_ms"], 0)
        self.assertEqual(batch["program"], members[0]["program"])
        for before, after in zip(members, members[1:]):
            self.assertLessEqual(before["encode_end"], after["encode_start"] + 1e-6)
        self.assertGreaterEqual(batch["encode_start"], members[-1]["encode_end"] - 1e-6)
        program = state["programs"][0]
        self.assertEqual(program["launches"], 4)
        self.assertAlmostEqual(program["mean_ms"] * 4, batch["ms"])

    def test_launches_still_in_an_open_batch_are_listed_without_gpu_rows(self):
        n = 1 << 16
        x = Buffer.from_list(dtype.float32, [1.0] * n)
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        with sth.batch():
            add[(n // 64,)](x, x, out, n, BLOCK=64)
            add[(n // 64,)](x, x, out, n, BLOCK=64)
            state = viz.snapshot()
            self.assertEqual(len([row for row in state["launches"] if "member_of" in row]), 2)
            self.assertEqual([row for row in state["launches"] if "batch_size" in row], [])

    def test_gpu_timestamps_share_the_host_clock(self):
        n = 1 << 16
        x = Buffer.from_list(dtype.float32, [1.0] * n)
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        add[(n // 64,)](x, x, out, n, BLOCK=64).wait()
        row = viz.snapshot()["launches"][-1]
        self.assertLess(row["queue_ms"], 50.0)

    def test_autotuning_is_recorded_and_its_candidates_are_marked(self):
        tuned = sth.autotune(sth.configs_from_grid({"BLOCK": [32, 64]}), key=["n"], warmup=1, repeat=2)(add)
        n = 256
        x = Buffer.from_list(dtype.float32, [1.0] * n)
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"SMITH_CACHE_DIR": directory}):
            tuned[lambda meta: (n // meta["BLOCK"],)](x, x, out, n)
            tuned[lambda meta: (n // meta["BLOCK"],)](x, x, out, n)
        sth.synchronize()
        state = viz.snapshot()
        self.assertEqual(len(state["autotunes"]), 1)
        self.assertEqual(len(state["autotunes"][0]["results"]), 2)
        self.assertGreater(sum(1 for row in state["launches"] if row["phase"] == "tuning"), 0)
        compiles = [item for item in state["spans"] if item["kind"] == "compile" and item["phase"] == "run"]
        self.assertTrue(all(item["program"] is not None for item in compiles))
        winners = [program for program in state["programs"] if not program["candidate"]]
        self.assertEqual(len(winners), 1)
        self.assertGreater(winners[0]["launches"], 0)


class TestPage(unittest.TestCase):
    def script(self):
        html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "smith", "viz", "index.html"), encoding="utf-8").read()
        return re.search(r"<script>(.*)</script>", html, re.S).group(1)

    def test_page_has_the_expected_views(self):
        script = self.script()
        for name in ("overviewHtml", "timelineHtml", "autotuneHtml", "kernelHtml", "launchDetail", "coreGrid", "gapsTable",
                     "buildChart", "runsHtml", "coreStrip"):
            self.assertIn(f"function {name}(", script)

    def test_same_named_kernels_are_told_apart_by_their_differing_settings(self):
        script = self.script()
        self.assertIn("function displayName(", script)
        self.assertGreaterEqual(script.count("displayName("), 4)

    def test_kernel_pages_have_a_runs_tab_with_the_core_chart(self):
        script = self.script()
        self.assertIn('["runs", "Runs + cores"]', script)
        self.assertNotIn('["launches", "Launches"]', script)

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_javascript_parses(self):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(self.script())
        try:
            result = subprocess.run(["node", "--check", handle.name], capture_output=True, text=True)
        finally:
            os.unlink(handle.name)
        self.assertEqual(result.returncode, 0, result.stderr)


class TestEnvironmentSwitch(unittest.TestCase):
    def test_smith_viz_serves_after_the_script_finishes(self):
        script = (
            "import smith as sth\n"
            "from smith import dtype\n"
            "from smith.trace import BufferSpec\n"
            "@sth.jit\n"
            "def copy(x_pointer, out_pointer, BLOCK: sth.constexpr = 8):\n"
            "    lane = sth.arange(0, BLOCK)\n"
            "    sth.store(out_pointer + lane, sth.load(x_pointer + lane))\n"
            "copy.trace(BufferSpec(dtype.float32), BufferSpec(dtype.float32))\n"
            "print('script done', flush=True)\n"
        )
        environment = {**os.environ, "SMITH_VIZ": "1", "SMITH_VIZ_NO_BROWSER": "1", "SMITH_VIZ_PORT": "0",
                       "PYTHONPATH": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")}
        process = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, env=environment)
        try:
            self.assertEqual(process.stdout.readline().strip(), "script done")
            line = process.stdout.readline().strip()
            self.assertTrue(line.startswith("smith viz: http://127.0.0.1:"), line)
            url = line.split()[2]
            with urllib.request.urlopen(url + "/api/state", timeout=5) as response:
                state = json.loads(response.read())
            self.assertEqual([program["name"] for program in state["programs"]], ["copy"])
        finally:
            process.terminate()
            process.communicate(timeout=10)
            process.stdout.close()


if __name__ == "__main__":
    unittest.main()
