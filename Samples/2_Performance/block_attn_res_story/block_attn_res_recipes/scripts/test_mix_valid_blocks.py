# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""UINT64 clamping regression.

Run CPU coverage with unittest discovery. To also run on NPU, set
BLOCK_ATTN_RES_PREPARE_EXE to the installed block_attn_res_prepare executable.
The NPU tests run sequentially and regenerate that executable's input/output files.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


BOUNDARY_VALUES = (1 << 63, (1 << 64) - 1)
OUTPUTS = ("numerator", "logit_max", "exp_sum")


class ValidBlocksGoldenTest(unittest.TestCase):
    def generate_case(self, root, valid):
        generator = Path(__file__).with_name("gen_data.py")
        subprocess.run([sys.executable, str(generator), "--output", str(root),
                        "--t", "32", "--n", "8", "--s", "24", "--d", "512",
                        "--valid-blocks", str(valid), "--slot", "0"], check=True)
        stored = np.fromfile(root / "input/valid_blocks.bin", dtype=np.uint64)
        self.assertEqual(stored.size, 1)
        self.assertEqual(int(stored[0]), valid)
        return [np.fromfile(root / "output" / f"golden_prepare_{name}.bin", dtype=np.float32)
                for name in OUTPUTS]

    def check_clamped_case(self, root, valid, baseline):
        for actual, expected in zip(self.generate_case(root, valid), baseline):
            np.testing.assert_array_equal(actual, expected)

    def test_invalid_arguments_rejected_before_generation(self):
        generator = Path(__file__).with_name("gen_data.py")
        cases = (("--t", "0"), ("--s", "0"), ("--n", "0"), ("--n", "65"),
                 ("--d", "0"), ("--d", "8193"), ("--valid-blocks", "-1"),
                 ("--valid-blocks", str(1 << 64)), ("--slot", "2"),
                 ("--eps", "nan"), ("--eps", "0"))
        with tempfile.TemporaryDirectory() as root:
            command = [sys.executable, str(generator), "--output", root,
                       "--t", "2", "--n", "8", "--s", "2", "--d", "64",
                       "--valid-blocks", "8", "--slot", "0"]
            for option, value in cases:
                with self.subTest(option=option, value=value):
                    result = subprocess.run([*command, option, value], capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("ValueError", result.stderr)
                    self.assertFalse((Path(root) / "input").exists())

    def test_uint64_inputs_clamp_to_n(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self.generate_case(root, 8)
            self.assertTrue(np.any(baseline[0] != 0))
            for valid in BOUNDARY_VALUES:
                with self.subTest(valid_blocks=valid):
                    self.check_clamped_case(root, valid, baseline)


@unittest.skipUnless(os.environ.get("BLOCK_ATTN_RES_PREPARE_EXE"),
                     "set BLOCK_ATTN_RES_PREPARE_EXE to run Mix tests on NPU")
class MixValidBlocksDeviceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(os.environ["BLOCK_ATTN_RES_PREPARE_EXE"]).resolve(strict=True)
        cls.root = cls.binary.parent
        cls.baseline = cls.run_mix(8)
        if not np.any(cls.baseline[0] != 0):
            raise AssertionError("valid_blocks=N must produce a non-empty numerator")

    @classmethod
    def run_mix(cls, valid):
        result = subprocess.run([str(cls.binary), "32", "8", "24", "512", "--template", "mix",
                                 "--valid-blocks", str(valid), "--warmup", "0", "--repeat", "1"],
                                capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise AssertionError(f"Mix failed for valid_blocks={valid}:\n{result.stdout}\n{result.stderr}")
        if "prepare template=mix" not in result.stdout:
            raise AssertionError("test did not execute the Mix template")
        stored = np.fromfile(cls.root / "input/valid_blocks.bin", dtype=np.uint64)
        if stored.size != 1 or int(stored[0]) != valid:
            raise AssertionError("runtime valid_blocks input was not preserved as UINT64")
        return [np.fromfile(cls.root / "output" / f"npu_prepare_{name}.bin", dtype=np.float32)
                for name in OUTPUTS]

    def assert_clamped(self, valid):
        for name, actual, expected in zip(OUTPUTS, self.run_mix(valid), self.baseline):
            np.testing.assert_array_equal(actual, expected, err_msg=f"{name}: valid_blocks={valid}")

    def test_mix_uint64_high_bit(self):
        self.assert_clamped(BOUNDARY_VALUES[0])

    def test_mix_uint64_max(self):
        self.assert_clamped(BOUNDARY_VALUES[1])


if __name__ == "__main__":
    unittest.main()
