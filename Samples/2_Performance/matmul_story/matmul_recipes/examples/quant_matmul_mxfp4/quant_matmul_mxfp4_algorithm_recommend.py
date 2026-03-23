# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


KERNEL_TIME_PATTERN = re.compile(r"Kernel elapsed time:\s*([0-9]+(?:\.[0-9]+)?)\s+us")


@dataclass(frozen=True)
class Candidate:
    """One installed executable that can participate in recommendation."""

    label: str
    executable_name: str


@dataclass
class CandidateResult:
    """Execution record used for compatibility filtering and final ranking."""

    label: str
    executable_path: Path
    kernel_time_us: Optional[float]
    return_code: int
    output: str

    @property
    def succeeded(self) -> bool:
        return self.return_code == 0 and self.kernel_time_us is not None


def print_usage(program_name: str) -> None:
    print(f"Usage: {program_name} m k n")
    print("Args:")
    print("  m: row of matrix A")
    print("  k: shared dimension of A and B")
    print("  n: col of matrix B")
    print(f"Example: {program_name} 1024 4096 2048")


def parse_positive_uint64(arg: str, name: str) -> int:
    if not arg.isdigit():
        raise ValueError(f"{name} must be a positive integer")
    value = int(arg)
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def parse_arguments(argv: List[str]) -> tuple[int, int, int]:
    if len(argv) >= 2 and argv[1] in ("-h", "--help"):
        print_usage(Path(argv[0]).name)
        raise SystemExit(0)
    if len(argv) != 4:
        raise ValueError("Expected exactly 3 arguments: m k n")

    m = parse_positive_uint64(argv[1], "m")
    k = parse_positive_uint64(argv[2], "k")
    n = parse_positive_uint64(argv[3], "n")
    if k % 2 != 0:
        raise ValueError("k must be an even number")
    return m, k, n


def resolve_story_root(script_path: Path) -> Path:
    # The recommendation helper lives three levels below `matmul_story`.
    return script_path.resolve().parents[3]


def resolve_executable(script_dir: Path, executable_name: str) -> Path:
    # Support both Windows (`.exe`) and POSIX executable layouts so the same
    # recommendation script works in different sample environments.
    direct_path = script_dir / executable_name
    if direct_path.exists():
        return direct_path

    windows_path = script_dir / f"{executable_name}.exe"
    if windows_path.exists():
        return windows_path

    raise FileNotFoundError(f"Executable not found: {executable_name}")


def discover_candidates(script_dir: Path) -> List[Candidate]:
    candidates: List[Candidate] = []
    seen_names = set()
    script_stem = Path(__file__).stem

    # Treat every executable in the install directory as a candidate so newly
    # added algorithms are picked up without editing this helper again.
    for entry in sorted(script_dir.iterdir(), key=lambda item: item.name):
        if not entry.is_file():
            continue

        is_windows_executable = entry.suffix.lower() == ".exe"
        is_posix_executable = entry.suffix == "" and os.access(entry, os.X_OK)
        if not (is_windows_executable or is_posix_executable):
            continue

        executable_name = entry.stem if is_windows_executable else entry.name
        if executable_name == script_stem:
            continue
        if executable_name in seen_names:
            continue

        label = executable_name
        candidates.append(Candidate(label=label, executable_name=executable_name))
        seen_names.add(executable_name)

    return candidates


def run_command(command: List[str], workdir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=workdir, text=True, capture_output=True, check=False)


def resolve_gen_data_script(script_dir: Path) -> Path:
    # The installed recommendation helper expects `gen_data.py` to be colocated
    # with the candidate executables so every tool sees the same input/output layout.
    script_path = script_dir / "gen_data.py"
    if script_path.exists():
        return script_path

    raise FileNotFoundError(f"gen_data.py was not found in {script_dir}")


def generate_input(script_dir: Path, m: int, k: int, n: int) -> None:
    # The recommendation must compare all candidates on the same generated
    # dataset, so input generation is centralized here.
    script_path = resolve_gen_data_script(script_dir)
    result = run_command([sys.executable, str(script_path), str(m), str(k), str(n)], script_path.parent)
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        raise RuntimeError(f"Failed to generate input data.\n{output}")


def parse_kernel_time_us(output: str) -> Optional[float]:
    match = KERNEL_TIME_PATTERN.search(output)
    return float(match.group(1)) if match else None


def run_candidate(script_dir: Path, story_root: Path, candidate: Candidate, m: int, k: int, n: int) -> CandidateResult:
    # Each candidate executable is run against the same generated input so the
    # ranking compares kernel time under identical data and shape conditions.
    executable_path = resolve_executable(script_dir, candidate.executable_name)
    # The executable resolves its own data paths relative to its install
    # directory, so the working directory only needs to be a stable project path.
    result = run_command([str(executable_path), str(m), str(k), str(n)], story_root)
    output = result.stdout + result.stderr
    return CandidateResult(
        label=candidate.label,
        executable_path=executable_path,
        kernel_time_us=parse_kernel_time_us(output),
        return_code=result.returncode,
        output=output,
    )


def print_ranking(results: List[CandidateResult]) -> None:
    # Failed or unsupported executables are filtered out before ranking so the
    # printed list only contains compatible algorithms.
    ranked_results = sorted(
        [item for item in results if item.succeeded],
        key=lambda item: item.kernel_time_us if item.kernel_time_us is not None else float("inf"),
    )

    print("\n[Recommended Algorithm Ranking]")
    for index, result in enumerate(ranked_results, start=1):
        print(f"  {index}. {result.label}")

    if not ranked_results:
        print("  No compatible algorithm found for the current shape.")
        return
    print("  Note: Only algorithms that support the current shape are listed.")


def main(argv: List[str]) -> int:
    try:
        m, k, n = parse_arguments(argv)
    except ValueError as error:
        print(f"ERROR: {error}")
        print_usage(Path(argv[0]).name)
        return 1

    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    story_root = resolve_story_root(script_path)
    candidates = discover_candidates(script_dir)
    if not candidates:
        print(f"ERROR: No executable files were found in {script_dir}")
        return 1

    try:
        generate_input(script_dir, m, k, n)
    except Exception as error:
        print(f"ERROR: {error}")
        return 1

    results: List[CandidateResult] = []
    for candidate in candidates:
        # Preserve per-candidate outputs so failures can still be inspected if
        # no compatible implementation is found.
        candidate_result = run_candidate(script_dir, story_root, candidate, m, k, n)
        results.append(candidate_result)

    print_ranking(results)
    return 0 if any(result.succeeded for result in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
