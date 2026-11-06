# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check architecture and argument forwarding by the real wrapper script."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize(
    ("arch", "expected"), [(None, "dav-3510"), ("dav-2201", "dav-2201")]
)
def test_functional_wrapper_passes_requested_architecture(tmp_path, arch, expected):
    root = tmp_path / "repo"
    (root / ".ci").mkdir(parents=True)
    (root / "bin").mkdir()
    source = Path(__file__).resolve().parents[2] / ".ci/run_ci_functional.sh"
    shutil.copyfile(source, root / ".ci/run_ci_functional.sh")
    (root / ".ci/build.sh").write_text(
        '#!/bin/bash\nprintf "%s\\n" "${1:-dav-3510}" > build-arch\n', encoding="utf-8"
    )
    fake_python = root / "bin/python3"
    fake_python.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > runner-args\n', encoding="utf-8"
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(root / "bin") + os.pathsep + env["PATH"]
    if arch is None:
        env.pop("NPU_ARCH", None)
    else:
        env["NPU_ARCH"] = arch
    result = subprocess.run(
        [
            shutil.which("bash") or "bash",
            ".ci/run_ci_functional.sh",
            "--sample",
            "vector_add",
        ],
        cwd=root,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert (root / "build-arch").read_text(encoding="utf-8").strip() == expected
    args = (root / "runner-args").read_text(encoding="utf-8").splitlines()
    assert args[-2:] == ["--sample", "vector_add"]
