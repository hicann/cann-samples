# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU regression tests: python -m unittest discover -s scripts."""

import ast
import contextlib
import csv
import importlib.util
import io
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from verify_result import CLOSE_TOLERANCES, bf16_to_fp32, compare


class ComparisonTest(unittest.TestCase):
    def test_cli_defaults_to_isclose_and_accepts_override(self):
        script = Path(__file__).with_name("verify_result.py")
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "output"
            output.mkdir()
            for name in ("numerator", "logit_max", "exp_sum"):
                np.array([1e-6], dtype=np.float32).tofile(output / f"golden_prepare_{name}.bin")
                np.array([1.1e-6], dtype=np.float32).tofile(output / f"npu_prepare_{name}.bin")
            args = [sys.executable, str(script), "--root", root, "--mode", "prepare",
                    "--t", "1", "--s", "1", "--d", "1"]
            default = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(default.returncode, 0, default.stdout + default.stderr)
            self.assertIn("standard=isclose", default.stdout)
            strict = subprocess.run(args + ["--compare", "stat_rel_err"], capture_output=True, text=True)
            self.assertEqual(strict.returncode, 1, strict.stdout + strict.stderr)
            self.assertIn("standard=stat_rel_err", strict.stdout)

    def test_old_transformer_checkout_is_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "ops-transformer").mkdir()
            fake_script = Path(root) / "samples/scripts/test_verify_result.py"
            with mock.patch(__name__ + ".__file__", str(fake_script)):
                with self.assertRaisesRegex(unittest.SkipTest, "lacks BlockAttnRes test assets"):
                    TransformerAlignmentTest.setUpClass()

    def check(self, actual, expected, dtype="float32", method="stat_rel_err", stage="update"):
        with contextlib.redirect_stdout(io.StringIO()):
            return compare(f"{stage}/h", np.asarray(actual, dtype=np.float32),
                           np.asarray(expected, dtype=np.float32), dtype, method)

    def test_mean_and_max_are_both_required(self):
        self.assertTrue(self.check([1.00005], [1.0]))
        self.assertFalse(self.check([1.0002], [1.0]))  # Mean fails, max passes.
        actual = np.ones(1000, dtype=np.float32)
        actual[0] = 1.002  # Mean passes, max fails.
        self.assertFalse(self.check(actual, np.ones_like(actual)))

    def test_bf16_uses_output_dtype(self):
        expected = bf16_to_fp32(np.array([0x3E9A], dtype=np.uint16))
        adjacent = bf16_to_fp32(np.array([0x3E9B], dtype=np.uint16))
        corrupted = bf16_to_fp32(np.array([0x3EAA], dtype=np.uint16))
        self.assertTrue(self.check(adjacent, expected, "bfloat16"))
        self.assertFalse(self.check(adjacent, expected, "float32"))
        self.assertTrue(np.allclose(corrupted, expected, atol=0.03, rtol=0.03))
        self.assertFalse(self.check(corrupted, expected, "bfloat16"))

    def test_isclose_pair_is_rtol_ptol_not_atol(self):
        self.assertFalse(self.check([0.0001], [0], method="isclose"))
        actual = np.ones(2000, dtype=np.float32)
        actual[0] = 2
        self.assertTrue(self.check(actual, np.ones_like(actual), method="isclose"))
        actual[:3] = 2
        self.assertFalse(self.check(actual, np.ones_like(actual), method="isclose"))
        self.assertTrue(self.check([0.04], [0], method="isclose", stage="prepare"))
        self.assertFalse(self.check([0.06], [0], method="isclose", stage="prepare"))

    def test_special_values_and_empty_match_ttk(self):
        for method in ("stat_rel_err", "isclose"):
            self.assertTrue(self.check([np.nan, np.inf, -np.inf, -0.0],
                                       [np.nan, np.inf, -np.inf, 0.0], method=method))
            self.assertFalse(self.check([np.inf], [-np.inf], method=method))
            self.assertFalse(self.check([np.nan], [0], method=method))
            self.assertTrue(self.check([], [], method=method))
            self.assertFalse(self.check([0], [], method=method))
        self.assertFalse(self.check([1e-8], [0]))


class TransformerAlignmentTest(unittest.TestCase):
    @staticmethod
    def load_case(root, folder, name, dtype=np.float32):
        return np.fromfile(Path(root) / folder / (name + ".bin"), dtype=dtype)

    @classmethod
    def setUpClass(cls):
        cls.scripts = Path(__file__).resolve().parent
        # Optional source-checkout tests; installed samples remain standalone.
        cls.transformer = next((p / "ops-transformer" for p in cls.scripts.parents
                                if (p / "ops-transformer").is_dir()), None)
        if cls.transformer is None:
            raise unittest.SkipTest("sibling ops-transformer checkout is unavailable")
        required = [
            "block_attn_res_prepare/tests/st/arch35/ttk_kernel_block_attn_res_prepare.csv",
            "block_attn_res_update/tests/assets/spec.py",
            *[f"block_attn_res_update/tests/st/arch35/ttk_{mode}_block_attn_res_update_st.csv"
              for mode in ("kernel", "aclnn", "e2e")],
        ]
        missing = [name for name in required if not (cls.transformer / "attention" / name).is_file()]
        if missing:
            raise unittest.SkipTest("ops-transformer checkout lacks BlockAttnRes test assets: " + ", ".join(missing))

    def test_all_st_csv_tolerances(self):
        for stage, count in (("prepare", 3), ("update", 2)):
            paths = list((self.transformer / "attention" / f"block_attn_res_{stage}" /
                          "tests/st/arch35").glob("*.csv"))
            self.assertTrue(paths)
            rtol, ptol, atol = CLOSE_TOLERANCES[stage]
            for path in paths:
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertTrue(rows)
                for row in rows:
                    self.assertEqual(ast.literal_eval(row["precision_tolerances"]),
                                     tuple((rtol, ptol) for _ in range(count)), path.name)
                    absolute = ast.literal_eval(row["absolute_precision"])
                    self.assertEqual(absolute, tuple(atol for _ in range(count))
                                     if isinstance(absolute, tuple) else atol, path.name)

    def test_update_golden_matches_transformer(self):
        path = self.transformer / "attention/block_attn_res_update/tests/assets/spec.py"
        spec = importlib.util.spec_from_file_location("transformer_update_spec", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for d, seed in itertools.product((63, 64, 65, 128, 129, 512), (0, 42)):
            with self.subTest(d=d, seed=seed), tempfile.TemporaryDirectory() as root:
                self.check_update_golden(module, root, d, seed)

    def check_update_golden(self, module, root, d, seed):
        subprocess.run([sys.executable, str(self.scripts / "gen_data.py"),
                        "--output", root, "--t", "3", "--n", "4", "--s", "2",
                        "--d", str(d), "--valid-blocks", "3", "--slot", "1",
                        "--seed", str(seed)], check=True)
        partial, h = module.BlockAttnResUpdateTestSpec.golden(
            self.load_case(root, "input", "partial_block").reshape(3, d),
            bf16_to_fp32(self.load_case(root, "input", "delta", np.uint16)).reshape(3, d),
            self.load_case(root, "input", "pseudo_query").reshape(2, d)[1],
            self.load_case(root, "output", "golden_prepare_numerator").reshape(2, 3, d)[1],
            self.load_case(root, "output", "golden_prepare_logit_max").reshape(2, 3)[1],
            self.load_case(root, "output", "golden_prepare_exp_sum").reshape(2, 3)[1])
        np.testing.assert_array_equal(partial.ravel(), self.load_case(root, "output", "golden_update_partial"))
        np.testing.assert_array_equal(h.ravel(), bf16_to_fp32(
            self.load_case(root, "output", "golden_update_h", np.uint16)))


if __name__ == "__main__":
    unittest.main()
