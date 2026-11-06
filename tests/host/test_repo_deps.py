#!/usr/bin/env python3
# Copyright (c) 2026 We1chan.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host-only checks for the repository dependency preflight."""

import importlib.util
import logging
import shutil
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class RepoDepsTest(unittest.TestCase):
    def setUp(self):
        if not shutil.which("git"):
            self.skipTest("Git is required")
        self.temp = tempfile.TemporaryDirectory(prefix="cann-repo-deps-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        checker = self.root / "scripts" / "check_env.py"
        shutil.copyfile(REPO_ROOT / "scripts" / "check_env.py", checker)
        (self.root / ".gitmodules").write_text(
            '[submodule "shmem"]\n\tpath = third_party/shmem\n'
            '[submodule "asc-devkit"]\n\tpath = third_party/asc-devkit\n',
            encoding="utf-8",
        )
        spec = importlib.util.spec_from_file_location("check_repo_deps_test", checker)
        self.checker = importlib.util.module_from_spec(spec)
        logger = logging.getLogger("check_env")
        previous_handlers = tuple(logger.handlers)
        previous_disabled = logger.disabled
        spec.loader.exec_module(self.checker)
        for handler in logger.handlers:
            if handler not in previous_handlers:
                self.addCleanup(logger.removeHandler, handler)
        logger.disabled = True
        self.addCleanup(setattr, logger, "disabled", previous_disabled)

    def populate(self, name):
        directory = self.root / "third_party" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "CMakeLists.txt").write_text("# dependency fixture\n")

    def test_missing_parent_is_not_a_success(self):
        self.assertFalse(self.checker.check_repo_deps())

    def test_missing_one_declared_submodule_is_not_a_success(self):
        self.populate("shmem")
        self.assertFalse(self.checker.check_repo_deps())

    def test_empty_and_git_only_directories_are_not_initialized(self):
        self.populate("shmem")
        other = self.root / "third_party" / "asc-devkit"
        other.mkdir()
        self.assertFalse(self.checker.check_repo_deps())
        (other / ".git").write_text("gitdir: fixture\n")
        self.assertFalse(self.checker.check_repo_deps())

    def test_populated_declared_submodules_pass(self):
        self.populate("shmem")
        self.populate("asc-devkit")
        (self.root / "third_party" / "unrelated").mkdir()
        self.assertTrue(self.checker.check_repo_deps())

    def test_new_submodule_with_space_in_path_is_checked(self):
        self.populate("shmem")
        self.populate("asc-devkit")
        with (self.root / ".gitmodules").open("a", encoding="utf-8") as modules:
            modules.write('\n[submodule "extra"]\n\tpath = third_party/extra dependency\n')
        self.assertFalse(self.checker.check_repo_deps())
        self.populate("extra dependency")
        self.assertTrue(self.checker.check_repo_deps())

    def test_missing_metadata_cannot_report_success(self):
        self.populate("shmem")
        self.populate("asc-devkit")
        (self.root / ".gitmodules").unlink()
        self.assertFalse(self.checker.check_repo_deps())

    def test_empty_declared_path_returns_failure(self):
        self.populate("shmem")
        self.populate("asc-devkit")
        with (self.root / ".gitmodules").open("a", encoding="utf-8") as modules:
            modules.write('\n[submodule "empty"]\n\tpath =\n')
        self.assertFalse(self.checker.check_repo_deps())

    def test_non_ascii_submodule_path(self):
        self.populate("shmem")
        self.populate("asc-devkit")
        with (self.root / ".gitmodules").open("a", encoding="utf-8") as modules:
            modules.write('\n[submodule "extra"]\n\tpath = third_party/依赖\n')
        self.assertFalse(self.checker.check_repo_deps())
        self.populate("依赖")
        self.assertTrue(self.checker.check_repo_deps())


if __name__ == "__main__":
    unittest.main()
