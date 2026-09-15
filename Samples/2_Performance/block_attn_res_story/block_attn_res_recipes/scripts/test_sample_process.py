# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Linux host regression tests; no CANN installation or NPU is required."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform.startswith("linux"), "POSIX launcher requires Linux")
class PythonLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            raise unittest.SkipTest("a C++ compiler is required")
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.binary = cls.root / "launcher"
        source = cls.root / "launcher.cpp"
        source.write_text('''#include "sample_process.h"
#include <cstdlib>
int main(int argc, char** argv) {
    const char* value = std::getenv("LD_LIBRARY_PATH");
    const std::string before = value == nullptr ? "" : value;
    const int result = BlockAttnResStory::RunPython(
        std::vector<std::string>(argv + 1, argv + argc));
    value = std::getenv("LD_LIBRARY_PATH");
    if (before != (value == nullptr ? "" : value)) return 124;
    return result < 0 ? 125 : result;
}
''', encoding="utf-8")
        include = Path(__file__).resolve().parents[1] / "include"
        subprocess.run([compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror",
                        "-I", str(include), str(source), "-o", str(cls.binary)], check=True)
        # Control PATH so the launcher uses the same Python as the test runner.
        (cls.root / "python3").symlink_to(sys.executable)

    def launch(self, *args, path=None):
        env = dict(os.environ, PATH=str(self.root) if path is None else path,
                   LD_LIBRARY_PATH="/unused/library/path", SAMPLE_PROCESS_TEST="preserved")
        return subprocess.run([str(self.binary), *args], cwd=self.root, env=env,
                              capture_output=True, text=True, timeout=10)

    def test_paths_and_arguments_are_literal(self):
        directory = self.root / 'space $(touch injected_dollar) `touch injected_backtick` "quote\';semi\nline'
        directory.mkdir()
        script = directory / "probe.py"
        script.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(sys.argv[1], 'result.json').write_text(json.dumps([sys.argv, "
            "os.environ.get('LD_LIBRARY_PATH'), os.environ.get('SAMPLE_PROCESS_TEST')]))\n",
            encoding="utf-8")
        result = self.launch(str(script), str(directory), "", "--root", str(directory))
        self.assertEqual(result.returncode, 0, result.stderr)
        argv, library_path, inherited = json.loads((directory / "result.json").read_text())
        self.assertEqual(argv, [str(script), str(directory), "", "--root", str(directory)])
        self.assertIsNone(library_path)
        self.assertEqual(inherited, "preserved")
        self.assertFalse((self.root / "injected_dollar").exists())
        self.assertFalse((self.root / "injected_backtick").exists())

    def test_nonzero_exit_is_propagated(self):
        result = self.launch("-c", "raise SystemExit(7)")
        self.assertEqual(result.returncode, 7)
        self.assertIn("exit code=7", result.stderr)

    def test_missing_python_is_reported(self):
        result = self.launch("-c", "pass", path="/nonexistent/python/path")
        self.assertEqual(result.returncode, 125)
        self.assertIn("cannot start python3", result.stderr)

    def test_signal_termination_is_reported(self):
        result = self.launch("-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)")
        self.assertEqual(result.returncode, 125)
        self.assertIn("terminated by signal=", result.stderr)


if __name__ == "__main__":
    unittest.main()
