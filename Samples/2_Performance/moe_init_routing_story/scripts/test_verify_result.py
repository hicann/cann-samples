#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Ethan Zou
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import shutil
import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS = {
    "expanded_x": ("expaned_x.bin", "result_expanded_x.bin", np.float32, [1.0, 2.0]),
    "row_idx": ("expanded_row_idx.bin", "result_expanded_row_idx.bin", np.int32, [0, 1]),
    "token_count": ("expert_token_count.bin", "result_expert_token_count.bin", np.int64, [1] * 8),
}


class MoeResultVerificationTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="cann-moe-verify-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        (self.root / "cmake").mkdir()
        self.scripts = self.root / "Samples/2_Performance/moe_init_routing_story/scripts"
        self.scripts.mkdir(parents=True)
        for name in ("verify_result.py", "gen_data.py"):
            shutil.copyfile(SCRIPT_DIR / name, self.scripts / name)
        self.data = self.root / "build/Samples/2_Performance/moe_init_routing_story"
        self.data.mkdir(parents=True)
        self.reset_artifacts()

    def reset_artifacts(self):
        for golden, actual, dtype, values in ARTIFACTS.values():
            content = np.asarray(values, dtype=dtype).tobytes()
            (self.data / golden).write_bytes(content)
            (self.data / actual).write_bytes(content)

    def run_verifier(self, expected, verbose=False):
        before = {p.name: p.read_bytes() for p in self.data.glob("*.bin")}
        command = [sys.executable, str(self.scripts / "verify_result.py")]
        if verbose:
            command.append("--verbose")
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        self.assertEqual({p.name: p.read_bytes() for p in self.data.glob("*.bin")}, before)
        self.assertNotIn("Traceback", result.stderr)
        return result

    def test_output_lengths_match_reference(self):
        for name, (golden, actual, dtype, values) in ARTIFACTS.items():
            for count in (0, len(values) - 1, len(values) + 1):
                with self.subTest(artifact=name, count=count):
                    self.reset_artifacts()
                    data = values[:count] if count <= len(values) else values + [0]
                    (self.data / actual).write_bytes(np.asarray(data, dtype=dtype).tobytes())
                    self.run_verifier(1)

    def test_partial_elements_are_rejected_in_either_file(self):
        for name, (golden, actual, dtype, values) in ARTIFACTS.items():
            for filename in (golden, actual):
                with self.subTest(artifact=name, filename=filename):
                    self.reset_artifacts()
                    path = self.data / filename
                    path.write_bytes(path.read_bytes() + b"x")
                    self.run_verifier(1)

    def test_nonfinite_float_reference_or_result_is_rejected(self):
        golden, actual, dtype, values = ARTIFACTS["expanded_x"]
        for invalid, target in product((np.nan, np.inf, -np.inf), ("actual", "golden", "both")):
            with self.subTest(value=invalid, target=target):
                self.reset_artifacts()
                content = np.asarray([invalid, values[1]], dtype=dtype).tobytes()
                if target in ("golden", "both"):
                    (self.data / golden).write_bytes(content)
                if target in ("actual", "both"):
                    (self.data / actual).write_bytes(content)
                self.run_verifier(1)

    def test_minimum_signed_integer_is_not_equal_to_zero(self):
        for name in ("row_idx", "token_count"):
            golden, actual, dtype, values = ARTIFACTS[name]
            for verbose in (False, True):
                with self.subTest(artifact=name, verbose=verbose):
                    self.reset_artifacts()
                    reference = np.zeros(len(values), dtype=dtype)
                    result = reference.copy()
                    result[0] = np.iinfo(dtype).min
                    (self.data / golden).write_bytes(reference.tobytes())
                    (self.data / actual).write_bytes(result.tobytes())
                    self.run_verifier(1, verbose)

    def test_matching_finite_and_empty_routed_data_are_preserved(self):
        self.run_verifier(0)
        for name in ("expanded_x", "row_idx"):
            golden, actual, _, _ = ARTIFACTS[name]
            (self.data / golden).write_bytes(b"")
            (self.data / actual).write_bytes(b"")
        self.run_verifier(0)

    def test_actual_data_generator_outputs_remain_compatible(self):
        generated = subprocess.run(
            [sys.executable, str(self.scripts / "gen_data.py"), "-n", "4", "-k", "2", "-c", "3"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)
        for golden, actual, _, _ in ARTIFACTS.values():
            shutil.copyfile(self.data / golden, self.data / actual)
        self.run_verifier(0)


if __name__ == "__main__":
    unittest.main()
