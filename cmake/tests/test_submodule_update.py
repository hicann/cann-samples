#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host integration checks of real CMake targets and local Git submodules."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CMAKE_DIR = Path(__file__).resolve().parents[1]


class SubmoduleUpdateTest(unittest.TestCase):
    def setUp(self):
        self.git = shutil.which("git")
        self.cmake = shutil.which("cmake")
        if not self.git or not self.cmake:
            self.skipTest("Host Git and CMake are required")
        self.temp_dir = tempfile.TemporaryDirectory(prefix="cann submodule test ")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.events = self.root / "events"
        self.events.mkdir()
        self.env = os.environ.copy()
        self.env.update(GIT_ALLOW_PROTOCOL="file", GIT_CONFIG_NOSYSTEM="1")
        self.create_git_proxy()
        self.source = self.root / "source"
        self.init_repo(self.source)
        (self.source / "cmake").mkdir()
        for path in CMAKE_DIR.glob("*.cmake"):
            shutil.copyfile(path, self.source / "cmake" / path.name)
        (self.source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.16)\nproject(host_probe LANGUAGES NONE)\n"
            "include(cmake/shmem.cmake)\ninclude(cmake/tensor_api.cmake)\n"
            "add_custom_target(host_both ALL DEPENDS "
            "cann_samples_shmem_dependencies cann_samples_tensor_api_dependencies)\n"
        )
        for name in ("shmem", "asc-devkit"):
            remote = self.root / name
            self.init_repo(remote)
            (remote / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.16)\nproject(host_dependency LANGUAGES NONE)\n"
                'set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "" FORCE)\n'
                "install(FILES CMakeLists.txt DESTINATION share/host_dependency)\n"
            )
            self.run_command([self.git, "add", "."], remote)
            self.run_command([self.git, "commit", "-qm", "Host dependency fixture"], remote)
            self.run_command([self.git, "submodule", "add", "-q", str(remote), "third_party/" + name], self.source)
        self.run_command([self.git, "add", "."], self.source)
        self.run_command([self.git, "commit", "-qm", "Host superproject fixture"], self.source)
        self.run_command([self.git, "submodule", "deinit", "-f", "--all"], self.source)
        shutil.rmtree(self.source / ".git/modules")
        self.build = self.root / "build"
        self.run_command(
            [self.cmake, "-S", str(self.source), "-B", str(self.build), f"-DGIT_EXECUTABLE={self.proxy}"], self.root
        )

    def create_git_proxy(self):
        self.proxy = self.root / "git proxy"
        self.proxy.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, subprocess, sys, time\n"
            f"events = pathlib.Path({str(self.events)!r})\n"
            "tracked = sys.argv[1:3] == ['submodule', 'update']\n"
            "start = time.monotonic_ns() if tracked else None\n"
            "if tracked: time.sleep(0.2)\n"
            f"result = subprocess.run([{self.git!r}, *sys.argv[1:]], check=False)\n"
            "if tracked:\n"
            "    record = {'start': start, 'end': time.monotonic_ns(), 'args': sys.argv[1:]}\n"
            "    (events / f'{os.getpid()}.json').write_text(json.dumps(record))\n"
            "sys.exit(result.returncode)\n"
        )
        self.proxy.chmod(0o700)

    def run_command(self, command, cwd):
        result = subprocess.run(command, cwd=cwd, env=self.env, capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def init_repo(self, path):
        path.mkdir()
        self.run_command([self.git, "init", "-q", "-b", "master"], path)
        self.run_command([self.git, "config", "user.name", "Host Test"], path)
        self.run_command([self.git, "config", "user.email", "host-test@example.invalid"], path)

    def assert_serial_updates(self):
        records = [json.loads(path.read_text()) for path in self.events.glob("*.json")]
        self.assertEqual(len(records), 2)
        intervals = sorted((record["start"], record["end"]) for record in records)
        self.assertLessEqual(intervals[0][1], intervals[1][0], "Git submodule updates overlapped")
        for name in ("shmem", "asc-devkit"):
            self.assertTrue((self.source / "third_party" / name / "CMakeLists.txt").is_file())
        shmem = next(record for record in records if record["args"][-1] == "third_party/shmem")
        tensor_api = next(record for record in records if record["args"][-1] == "third_party/asc-devkit")
        self.assertIn("--recursive", shmem["args"])
        self.assertNotIn("--recursive", tensor_api["args"])

    def test_parallel_targets_serialize_initialization_and_release_lock(self):
        for phase in ("initialization", "already initialized"):
            with self.subTest(phase=phase):
                for path in self.events.glob("*.json"):
                    path.unlink()
                self.run_command([self.cmake, "--build", str(self.build), "--parallel", "2"], self.root)
                self.assert_serial_updates()

    def test_linked_worktree_uses_common_git_directory(self):
        linked = self.root / "linked source"
        self.run_command([self.git, "worktree", "add", "--detach", str(linked), "HEAD"], self.source)
        self.source = linked
        self.build = self.root / "linked build"
        self.assertTrue((linked / ".git").is_file())
        self.run_command(
            [self.cmake, "-S", str(linked), "-B", str(self.build), f"-DGIT_EXECUTABLE={self.proxy}"], self.root
        )
        self.run_command([self.cmake, "--build", str(self.build), "--parallel", "2"], self.root)
        self.assert_serial_updates()

    def test_git_failure_is_reported_and_releases_lock(self):
        command = [
            self.cmake,
            f"-DSOURCE_DIR={self.source}",
            f"-DGIT_EXECUTABLE={self.proxy}",
            "-DSUBMODULE_PATH=missing/path",
            "-P",
            str(self.source / "cmake/update_submodule.cmake"),
        ]
        result = subprocess.run(command, cwd=self.root, env=self.env, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Failed to update Git submodule missing/path", result.stderr)
        for path in self.events.glob("*.json"):
            path.unlink()
        self.run_command([self.cmake, "--build", str(self.build), "--parallel", "2"], self.root)
        self.assert_serial_updates()


if __name__ == "__main__":
    unittest.main()
