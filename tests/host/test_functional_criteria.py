# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run the real manifest runner with small host-side commands."""

from pathlib import Path
import subprocess
import sys

import pytest
import yaml


def run_manifest(manifest):
    root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [
            sys.executable,
            str(root / ".ci/run_ci_functional.py"),
            "--manifest",
            str(manifest),
            "--artifacts-dir",
            "",
        ],
        cwd=root,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )


@pytest.mark.parametrize(
    ("criteria", "expected_exit", "message"),
    [
        ({"all_steps_exit_zer0": True}, 1, "unknown pass_criteria"),
        ({"all_steps_exit_zero": True}, 1, "exit code is 7"),
        ({"exit_code": 7}, 0, "[PASSED]"),
        ("invalid", 1, "pass_criteria must be a mapping"),
        ({"stdout_contains": "ab"}, 1, "stdout_contains must be a list of strings"),
        (
            {"stdout_not_contains": None},
            1,
            "stdout_not_contains must be a list of strings",
        ),
        (
            {"final_stdout_contains": [7]},
            1,
            "final_stdout_contains must be a list of strings",
        ),
        ({"all_steps_exit_zero": "false"}, 1, "all_steps_exit_zero must be a boolean"),
        ({"exit_code": True}, 1, "exit_code must be an integer"),
        ({"exit_code": "7"}, 1, "exit_code must be an integer"),
        ({"exit_code": None}, 1, "exit_code must be an integer"),
    ],
)
def test_manifest_criteria_are_not_silently_ignored(
    tmp_path, criteria, expected_exit, message
):
    manifest = tmp_path / "manifest.yaml"
    marker = tmp_path / "step_ran"
    manifest.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "samples": [
                    {
                        "id": "host_fixture",
                        "steps": [
                            {
                                "name": "exit_seven",
                                "cwd": ".",
                                "cmd": [
                                    sys.executable,
                                    "-c",
                                    "from pathlib import Path; import sys; "
                                    "Path(sys.argv[1]).touch(); raise SystemExit(7)",
                                    str(marker),
                                ],
                            }
                        ],
                        "pass_criteria": criteria,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = run_manifest(manifest)
    assert result.returncode == expected_exit, result.stderr
    assert message in result.stdout
    assert marker.exists() is (
        criteria in ({"all_steps_exit_zero": True}, {"exit_code": 7})
    )


@pytest.mark.parametrize("steps", [None, []])
def test_setup_only_sample_does_not_pass_or_run_setup(tmp_path, steps):
    manifest = tmp_path / "manifest.yaml"
    marker = tmp_path / "setup_ran"
    sample = {
        "id": "setup_only",
        "setup": [
            {
                "name": "prepare",
                "cwd": ".",
                "cmd": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
                    str(marker),
                ],
            }
        ],
        "pass_criteria": {"all_steps_exit_zero": True},
    }
    if steps is not None:
        sample["steps"] = steps
    manifest.write_text(
        yaml.safe_dump({"version": 1, "samples": [sample]}), encoding="utf-8"
    )
    result = run_manifest(manifest)
    assert result.returncode == 1, result.stderr
    assert "steps must be a non-empty list" in result.stdout
    assert not marker.exists()


def test_valid_typed_criteria_keep_expected_nonzero_exit(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    sample = {
        "id": "valid_criteria",
        "steps": [
            {
                "name": "expected_failure",
                "cwd": ".",
                "cmd": [
                    sys.executable,
                    "-c",
                    "print('expected output'); raise SystemExit(7)",
                ],
            }
        ],
        "pass_criteria": {
            "exit_code": 7,
            "all_steps_exit_zero": False,
            "stdout_contains": ["expected"],
            "stdout_not_contains": ["unexpected"],
            "final_stdout_contains": ["expected output"],
        },
    }
    manifest.write_text(
        yaml.safe_dump({"version": 1, "samples": [sample]}), encoding="utf-8"
    )
    result = run_manifest(manifest)
    assert result.returncode == 0, result.stderr
    assert "[PASSED] valid_criteria" in result.stdout
