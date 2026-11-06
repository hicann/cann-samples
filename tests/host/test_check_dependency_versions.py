# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run the actual dependency checker against installed host packages."""

from importlib.metadata import version
from pathlib import Path
import shutil
import subprocess
import sys

from packaging.version import Version
import pytest


NUMPY_MAJOR = Version(version("numpy")).major


@pytest.mark.parametrize(
    ("requirement", "expected"),
    [
        (f"numpy>={NUMPY_MAJOR + 1}", 1),
        (f"numpy>={NUMPY_MAJOR},<{NUMPY_MAJOR + 1}", 0),
        (f"numpy=={version('numpy')}", 0),
        ("samples_nonexistent_package_8329>=1", 1),
        ('samples_nonexistent_package_8329>=1; python_version < "0"', 0),
    ],
)
def test_real_checker_respects_installed_versions(tmp_path, requirement, expected):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = Path(__file__).resolve().parents[2] / "scripts/check_env.py"
    shutil.copyfile(source, scripts / "check_env.py")
    (tmp_path / "requirements.txt").write_text(requirement + "\n", encoding="utf-8")
    program = (
        "import importlib.util,sys; "
        'spec=importlib.util.spec_from_file_location("checker", sys.argv[1]); '
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        "sys.exit(0 if m.check_python_deps() else 1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(scripts / "check_env.py")],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    assert result.returncode == expected, result.stdout + result.stderr
