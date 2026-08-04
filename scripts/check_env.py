#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import importlib
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
WARN = "\033[33m[WARN]\033[0m"
INFO = "\033[36m[INFO]\033[0m"


class _ColorFormatter(logging.Formatter):
    def format(self, record):
        if record.msg == "SEP":
            return "=" * 60
        return super().format(record)


_logger = logging.getLogger("check_env")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_ColorFormatter("%(message)s"))
_logger.addHandler(_handler)
_logger.setLevel(logging.INFO)
_logger.propagate = False


def _try_find_command(name):
    return shutil.which(name)


def _parse_version(text):
    match = re.search(r"(\d+(?:\.\d+)+)", text or "")
    if not match:
        return None
    return tuple(int(p) for p in match.group(1).split("."))


def _older_than(version, min_version):
    n = max(len(version), len(min_version))
    version = version + (0,) * (n - len(version))
    min_version = min_version + (0,) * (n - len(min_version))
    return version < min_version


def check_command(cmd_name, min_version=None, version_args=None):
    cmd_path = _try_find_command(cmd_name)
    if cmd_path is None:
        _logger.info(f"  {FAIL} `{cmd_name}` not found in PATH")
        return False
    if min_version is None:
        _logger.info(f"  {PASS} `{cmd_name}` found at {cmd_path}")
        return True
    args = version_args or ["--version"]
    try:
        result = subprocess.run([cmd_path] + args, capture_output=True, text=True, timeout=5)
        version_text = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        _logger.warning(f"  {WARN} `{cmd_name}` found at {cmd_path} but version query timed out")
        return True
    except Exception:
        _logger.warning(f"  {WARN} `{cmd_name}` found at {cmd_path} but failed to query version")
        return True
    version = _parse_version(version_text)
    min_ver = _parse_version(min_version)
    if version is None:
        _logger.warning(f"  {WARN} `{cmd_name}` version could not be parsed from: "
                        f"{version_text.strip()!r}; skipping version comparison")
        return True
    if min_ver is None:
        _logger.warning(f"  {WARN} `{cmd_name}` min_version {min_version!r} could not be parsed; "
                        "skipping version comparison")
        return True
    version_str = ".".join(map(str, version))
    if _older_than(version, min_ver):
        _logger.info(f"  {FAIL} `{cmd_name}` found at {cmd_path}  "
                     f"(version: {version_str}, required >= {min_version})")
        return False
    _logger.info(f"  {PASS} `{cmd_name}` found at {cmd_path}  (version: {version_str})")
    return True


def check_cann_env():
    _logger.info(f"\n{INFO} Checking CANN environment ...")
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_home:
        _logger.info(f"  {FAIL} ASCEND_HOME_PATH is not set")
        _logger.info(f"     -> Please run: source <install_path>/ascend-toolkit/set_env.sh")
        return False
    _logger.info(f"  {PASS} ASCEND_HOME_PATH = {ascend_home}")
    set_env_sh = os.path.join(ascend_home, "set_env.sh")
    if not os.path.isfile(set_env_sh):
        set_env_sh = os.path.join(os.path.dirname(ascend_home), "ascend-toolkit", "set_env.sh")
    if os.path.isfile(set_env_sh):
        _logger.info(f"  {PASS} set_env.sh found at {set_env_sh}")
    else:
        _logger.warning(f"  {WARN} ASCEND_HOME_PATH is set but set_env.sh not found at expected location")
    ld_library = os.environ.get("LD_LIBRARY_PATH", "")
    if "ascend" in ld_library.lower():
        _logger.info(f"  {PASS} LD_LIBRARY_PATH includes Ascend paths")
    else:
        _logger.warning(f"  {WARN} LD_LIBRARY_PATH may not include Ascend libraries. "
                        "Ensure set_env.sh has been sourced")
    return True


def check_python_deps():
    _logger.info(f"\n{INFO} Checking Python dependencies ...")
    repo_root = Path(__file__).resolve().parent.parent
    req_file = repo_root / "requirements.txt"
    if not req_file.is_file():
        _logger.warning(f"  {WARN} requirements.txt not found at {req_file}")
        return False
    missing = []
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
            pkg_spec = line.strip()
            try:
                importlib.import_module(pkg_name.replace("-", "_"))
            except ImportError:
                missing.append(pkg_spec)
    if missing:
        _logger.info(f"  {FAIL} {len(missing)} package(s) missing: {', '.join(missing)}")
        _logger.info(f"     -> Please run: pip3 install -r {req_file}")
        return False
    _logger.info(f"  {PASS} All packages in requirements.txt can be imported")
    return True


def check_repo_deps():
    _logger.info(f"\n{INFO} Checking repository dependencies ...")
    repo_root = Path(__file__).resolve().parent.parent
    third_party = repo_root / "third_party"
    has_missing = False
    if third_party.is_dir():
        for entry in sorted(third_party.iterdir()):
            if not entry.is_dir():
                continue
            if any(entry.iterdir()):
                _logger.info(f"  {PASS} third_party/{entry.name}/ is populated")
            else:
                has_missing = True
                _logger.info(f"  {FAIL} third_party/{entry.name}/ exists but is empty")
    if has_missing:
        _logger.info(f"     -> Please run: git submodule update --init --recursive")
        return False
    return True


def check_npu_runtime():
    _logger.info(f"\n{INFO} Checking NPU runtime environment ...")
    npu_smi = _try_find_command("npu-smi")
    if npu_smi is None:
        _logger.warning(f"  {WARN} npu-smi not found. Cannot verify NPU devices.")
        _logger.warning("     -> A runnable NPU is required to execute compiled samples, "
                        "but not required for compilation")
        return True
    _logger.info(f"  {PASS} npu-smi found at {npu_smi}")
    _logger.info("     -> Run `npu-smi info` to check NPU device status")
    return True


def main():
    _logger.log(logging.INFO, "SEP")
    _logger.info("  CANN Samples - Pre-build Environment Check")
    _logger.log(logging.INFO, "SEP")
    _logger.info(f"\n{INFO} Checking basic commands ...")
    ok = True
    ok &= check_command("cmake", min_version="3.16.0", version_args=["--version"])
    ok &= check_command("python3", min_version="3.10.0", version_args=["--version"])
    ok &= check_command("pip3")
    ok &= check_command("zip")
    ok &= check_command("git", version_args=["--version"])
    ok &= check_cann_env()
    ok &= check_python_deps()
    ok &= check_repo_deps()
    ok &= check_npu_runtime()
    _logger.info("\n")
    _logger.log(logging.INFO, "SEP")
    if ok:
        _logger.info(f"  {PASS} All checks passed. You are ready to build samples!")
    else:
        _logger.info(f"  {FAIL} Some checks failed. Please fix the issues above before building.")
    _logger.log(logging.INFO, "SEP")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
