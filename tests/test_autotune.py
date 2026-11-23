import contextlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import smith as sth
from smith import dtype
from smith.autotune import Autotuner, Config, configs_from_grid

if sys.platform == "darwin":
    from smith import compiler
    from smith.compiler import Buffer


@sth.jit
def add(x_pointer, y_pointer, out_pointer, n, BLOCK: sth.constexpr = 4):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    y = sth.load(y_pointer + offset, mask=mask)
    sth.store(out_pointer + offset, x + y, mask=mask)


@sth.jit
def accumulate(x_pointer, out_pointer, n, BLOCK: sth.constexpr = 4):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    previous = sth.load(out_pointer + offset, mask=mask)
    sth.store(out_pointer + offset, previous + x, mask=mask)


class TestConfigs(unittest.TestCase):
    def test_configs_from_grid_is_the_cartesian_product(self):
        configs = configs_from_grid({"BLOCK": [32, 64], "WARPS": [1, 2]})
        self.assertEqual(configs, [Config(BLOCK=32, WARPS=1), Config(BLOCK=32, WARPS=2),
                                   Config(BLOCK=64, WARPS=1), Config(BLOCK=64, WARPS=2)])

    def test_config_equality_compares_values(self):
        self.assertEqual(Config(BLOCK=4), Config(BLOCK=4))
        self.assertNotEqual(Config(BLOCK=4), Config(BLOCK=8))
        self.assertEqual(len({Config(BLOCK=4), Config(BLOCK=4)}), 1)


class CacheDirectoryCase(unittest.TestCase):
    def setUp(self):
        self.cache_directory = tempfile.mkdtemp(prefix="smith-autotune-test-")
        self.addCleanup(shutil.rmtree, self.cache_directory, True)
        patcher = mock.patch.dict(os.environ, {"SMITH_CACHE_DIR": self.cache_directory})
        patcher.start()
        self.addCleanup(patcher.stop)

    def cache_file(self, kernel_name):
        return os.path.join(self.cache_directory, f"autotune-{kernel_name}.json")


class TestAutotuneInterpreter(CacheDirectoryCase):
    def test_list_arguments_run_first_config_without_tuning(self):
        n = 10
        x = [float(i) for i in range(n)]
        y = [10.0 * i for i in range(n)]
        out = [0.0] * n
        tuner = Autotuner(add, configs_from_grid({"BLOCK": [2, 4]}), key=("n",))
        tuner[lambda meta: (sth.cdiv(n, meta["BLOCK"]),)](x, y, out, n)
        self.assertEqual(out, [a + b for a, b in zip(x, y)])
        self.assertEqual(tuner.memory, {})
        self.assertEqual(os.listdir(self.cache_directory), [])


@unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
class TestAutotuneMetal(CacheDirectoryCase):
    def make_tuner(self, kernel, block_sizes, **options):
        options.setdefault("warmup", 1)
        options.setdefault("repeat", 2)
        return Autotuner(kernel, configs_from_grid({"BLOCK": block_sizes}), key=("n",), **options)

    def test_picks_a_config_and_results_are_correct(self):
        n = 10
        x = Buffer.from_list(dtype.float32, [float(i) for i in range(n)])
        y = Buffer.from_list(dtype.float32, [10.0 * i for i in range(n)])
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        tuner = self.make_tuner(add, [2, 4, 8])
        tuner[lambda meta: (sth.cdiv(n, meta["BLOCK"]),)](x, y, out, n)
        self.assertEqual(out.to_list(), [11.0 * i for i in range(n)])
        self.assertIn(tuner.best_config, configs_from_grid({"BLOCK": [2, 4, 8]}))
        self.assertEqual(len(tuner.results), 3)
        self.assertTrue(all(result["seconds"] is not None and result["seconds"] >= 0.0
                            for result in tuner.results))

    def test_fast_math_is_part_of_the_cache_key_and_reaches_the_compiler(self):
        n = 16
        x = Buffer.from_list(dtype.float32, [1.0] * n)
        y = Buffer.from_list(dtype.float32, [2.0] * n)
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        grid = lambda meta: (sth.cdiv(n, meta["BLOCK"]),)
        modes = []
        original = compiler.compile_msl

        def spy(source, name, fast_math=True):
            modes.append(fast_math)
            return original(source, name, fast_math)

        with mock.patch("smith.compiler.compile_msl", spy):
            fast = self.make_tuner(add, [2, 4])
            precise = self.make_tuner(add, [2, 4], fast_math=False)
            fast[grid](x, y, out, n)
            precise[grid](x, y, out, n)
        self.assertIn(True, modes)
        self.assertIn(False, modes)
        self.assertEqual(out.to_list(), [3.0] * n)
        with open(self.cache_file("add")) as handle:
            self.assertEqual(len(json.load(handle)), 2)

    def test_finalists_are_remeasured_and_decide_the_winner(self):
        n = 16
        x = Buffer.from_list(dtype.float32, [1.0] * n)
        y = Buffer.from_list(dtype.float32, [2.0] * n)
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        tuner = self.make_tuner(add, [2, 4, 8, 16], finalists=2)
        tuner[lambda meta: (sth.cdiv(n, meta["BLOCK"]),)](x, y, out, n)
        refined = [result for result in tuner.results if result["refined"] is not None]
        self.assertEqual(len(refined), 2)
        self.assertEqual(tuner.best_config.values, min(refined, key=lambda result: result["refined"])["config"])
        self.assertEqual(sorted(result["seconds"] for result in tuner.results)[:2],
                         sorted(result["seconds"] for result in refined))

    def test_finalists_must_be_positive(self):
        with self.assertRaises(ValueError):
            Autotuner(add, configs_from_grid({"BLOCK": [2]}), key=("n",), finalists=0)

    def test_cache_hit_does_not_rebenchmark(self):
        n = 16
        x = Buffer.from_list(dtype.float32, [1.0] * n)
        y = Buffer.from_list(dtype.float32, [2.0] * n)
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        tuner = self.make_tuner(add, [2, 4, 8])
        grid = lambda meta: (sth.cdiv(n, meta["BLOCK"]),)
        tuner[grid](x, y, out, n)
        seen = []
        observer = seen.append
        compiler.add_launch_observer(observer)
        try:
            tuner[grid](x, y, out, n)
            sth.synchronize()
        finally:
            compiler.remove_launch_observer(observer)
        self.assertEqual(len(seen), 1)
        self.assertEqual(out.to_list(), [3.0] * n)

    def test_json_cache_is_reused_by_a_fresh_autotuner(self):
        n = 16
        x = Buffer.from_list(dtype.float32, [1.0] * n)
        y = Buffer.from_list(dtype.float32, [2.0] * n)
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        grid = lambda meta: (sth.cdiv(n, meta["BLOCK"]),)
        first = self.make_tuner(add, [2, 4, 8])
        first[grid](x, y, out, n)
        with open(self.cache_file("add")) as handle:
            entries = json.load(handle)
        self.assertEqual(len(entries), 1)

        fresh = self.make_tuner(add, [2, 4, 8])
        seen = []
        observer = seen.append
        compiler.add_launch_observer(observer)
        try:
            fresh[grid](x, y, out, n)
            sth.synchronize()
        finally:
            compiler.remove_launch_observer(observer)
        self.assertEqual(len(seen), 1)
        self.assertEqual(fresh.best_config, first.best_config)
        self.assertEqual(out.to_list(), [3.0] * n)

    def test_invalid_config_is_skipped(self):
        n = 10
        x = Buffer.from_list(dtype.float32, [float(i) for i in range(n)])
        y = Buffer.from_list(dtype.float32, [1.0] * n)
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        tuner = self.make_tuner(add, [4096, 4, 8])
        tuner[lambda meta: (sth.cdiv(n, meta["BLOCK"]),)](x, y, out, n)
        self.assertNotEqual(tuner.best_config, Config(BLOCK=4096))
        self.assertEqual(tuner.results[0]["config"], {"BLOCK": 4096})
        self.assertIsNone(tuner.results[0]["seconds"])
        self.assertIsNotNone(tuner.results[0]["error"])
        self.assertEqual(out.to_list(), [float(i) + 1.0 for i in range(n)])

    def test_restore_protects_an_in_out_buffer(self):
        n = 8
        x = Buffer.from_list(dtype.float32, [1.0] * n)
        out = Buffer.from_list(dtype.float32, [10.0] * n)
        tuner = Autotuner(accumulate, configs_from_grid({"BLOCK": [2, 4]}), key=("n",),
                          restore=("out_pointer",), warmup=1, repeat=3)
        tuner[lambda meta: (sth.cdiv(n, meta["BLOCK"]),)](x, out, n)
        self.assertEqual(out.to_list(), [11.0] * n)

    def test_restore_must_name_a_buffer(self):
        n = 4
        x = Buffer.from_list(dtype.float32, [1.0] * n)
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        tuner = Autotuner(accumulate, configs_from_grid({"BLOCK": [2]}), key=("n",), restore=("n",))
        with self.assertRaises(TypeError):
            tuner[(1,)](x, out, n)


if __name__ == "__main__":
    unittest.main()
