#!/usr/bin/env python3
# Copyright (c) 2026 Ethan_Zou
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host CLI regression checks for invalid msopprof duration values."""

import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path

SCRIPT = Path(__file__).with_name("summarize_msopprof.py")
CSV_SUFFIX = "OPPROF_test/OpBasicInfo.csv"


class TestProfileDurationValidation(unittest.TestCase):
    def run_summary(self, profiles, missing=False, bom=False):
        with tempfile.TemporaryDirectory(prefix="cann-msopprof-耗时-") as directory:
            root = Path(directory) / "profile results"
            if not missing:
                root.mkdir()
            for name, content in profiles.items():
                path = root / name / CSV_SUFFIX
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8-sig" if bom else "utf-8")
            before = {path: path.read_bytes() for path in root.rglob("*.csv")}
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(root)],
                capture_output=True,
                text=True,
                check=False,
            )
            after = {path: path.read_bytes() for path in root.rglob("*.csv")}
            self.assertEqual(after, before)
            return result

    def test_invalid_numeric_durations_fail_in_every_run_position(self):
        values_and_positions = product(
            ("nan", "NaN", "inf", "-inf", "-1", "-0.000001", "1e309"),
            ("only", "first", "middle", "last"),
        )
        for value, position in values_and_positions:
            with self.subTest(value=value, position=position):
                values = [value]
                if position != "only":
                    values = ["1", "3"]
                    values.insert({"first": 0, "middle": 1, "last": 2}[position], value)
                profiles = {
                    f"0_baseline/run{index}": f"Task Duration(us)\n{duration}\n"
                    for index, duration in enumerate(values, 1)
                }
                result = self.run_summary(profiles)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("finite, non-negative", result.stderr)
                self.assertIn("OpBasicInfo.csv", result.stderr)
                self.assertNotIn("0_baseline\t", result.stdout)

    def test_valid_duration_and_median_controls(self):
        cases = [
            (["0", "-0"], "0.000000,-0.000000", "0.000000", False),
            (["1.25"], "1.250000", "1.250000", False),
            (["3", "1", "2"], "3.000000,1.000000,2.000000", "2.000000", False),
            (
                ["8", "2", "4", "6"],
                "8.000000,2.000000,4.000000,6.000000",
                "5.000000",
                False,
            ),
            (["1e-3", "4e2", "0"], "0.001000,400.000000,0.000000", "0.001000", True),
        ]
        for values, runs, median, bom in cases:
            with self.subTest(values=values, bom=bom):
                profiles = {
                    f"0_baseline/run{index}": f"Task Duration(us)\n{duration}\n"
                    for index, duration in enumerate(values, 1)
                }
                result = self.run_summary(profiles, bom=bom)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, "")
                self.assertEqual(
                    result.stdout,
                    f"stage\truns_us\tmedian_us\n0_baseline\t{runs}\t{median}\n",
                )

    def test_existing_invalid_schema_and_missing_data_controls(self):
        contents = [
            "Other column\n1\n",
            "Task Duration(us)\n",
            "Task Duration(us)\n1\n2\n",
            'Task Duration(us)\n""\n',
            "Task Duration(us)\nnot-a-number\n",
        ]
        for content in contents:
            with self.subTest(content=content):
                result = self.run_summary({"0_baseline/run1": content})
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("0_baseline\t", result.stdout)
        for missing in (False, True):
            with self.subTest(missing=missing):
                result = self.run_summary({}, missing=missing)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
