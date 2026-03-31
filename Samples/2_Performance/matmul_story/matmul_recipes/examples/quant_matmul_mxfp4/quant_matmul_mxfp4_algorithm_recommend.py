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
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


MSPROF_OUTPUT_DIR_NAME = "msprof_recommend"
MSPROF_PROF_DIR_PREFIX = "PROF_"
MSPROF_OP_SUMMARY_GLOB = "op_summary_*.csv"
MSPROF_TASK_DURATION_COLUMN = "Task Duration(us)"
# Number of profiling runs executed for each candidate.
MSPROF_RUN_COUNT = 2


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


def cleanup_msprof_output_dir(msprof_output_dir: Path) -> None:
    # Recommendation only needs profiling artifacts transiently, so always
    # clean the output directory before returning control to the user.
    if msprof_output_dir.exists():
        shutil.rmtree(msprof_output_dir, ignore_errors=True)


def list_prof_directories(msprof_output_dir: Path) -> set[Path]:
    if not msprof_output_dir.exists():
        return set()

    return {
        entry.resolve()
        for entry in msprof_output_dir.iterdir()
        if entry.is_dir() and entry.name.startswith(MSPROF_PROF_DIR_PREFIX)
    }


def resolve_new_prof_directory(msprof_output_dir: Path, existing_prof_dirs: set[Path]) -> Path:
    current_prof_dirs = list_prof_directories(msprof_output_dir)
    new_prof_dirs = [entry for entry in current_prof_dirs if entry not in existing_prof_dirs]
    if not new_prof_dirs:
        raise FileNotFoundError(
            f"No new {MSPROF_PROF_DIR_PREFIX}* directory was generated under {msprof_output_dir}"
        )

    # `msprof` should generate exactly one fresh profile directory per run. If
    # multiple appear, prefer the newest one because it corresponds to the most
    # recent profiling session.
    return max(new_prof_dirs, key=lambda entry: entry.stat().st_mtime_ns)


def resolve_op_summary_csv(prof_dir: Path) -> Path:
    profiler_output_dir = prof_dir / "mindstudio_profiler_output"
    if not profiler_output_dir.is_dir():
        raise FileNotFoundError(f"mindstudio_profiler_output was not found in {prof_dir}")

    csv_files = sorted(
        profiler_output_dir.glob(MSPROF_OP_SUMMARY_GLOB),
        key=lambda entry: entry.stat().st_mtime_ns,
        reverse=True,
    )
    if not csv_files:
        raise FileNotFoundError(f"No {MSPROF_OP_SUMMARY_GLOB} file was found in {profiler_output_dir}")
    return csv_files[0]


def parse_kernel_time_us_from_csv(csv_path: Path) -> float:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        first_row = next(reader, None)

    if not header:
        raise ValueError(f"CSV header is missing in {csv_path}")
    if not first_row:
        raise ValueError(f"CSV data row is missing in {csv_path}")

    normalized_header = [column.strip() for column in header]
    try:
        duration_index = normalized_header.index(MSPROF_TASK_DURATION_COLUMN)
    except ValueError as error:
        raise ValueError(
            f"{MSPROF_TASK_DURATION_COLUMN} column was not found in {csv_path}"
        ) from error

    if duration_index >= len(first_row):
        raise ValueError(
            f"{MSPROF_TASK_DURATION_COLUMN} value is missing from the first data row in {csv_path}"
        )

    raw_value = first_row[duration_index].strip().replace(",", "")
    if not raw_value:
        raise ValueError(f"{MSPROF_TASK_DURATION_COLUMN} is empty in {csv_path}")

    try:
        return float(raw_value)
    except ValueError as error:
        raise ValueError(
            f"Failed to parse {MSPROF_TASK_DURATION_COLUMN} value '{raw_value}' from {csv_path}"
        ) from error


def run_candidate_with_msprof(script_dir: Path, executable_path: Path, m: int, k: int, n: int) -> tuple[float, str]:
    msprof_output_dir = script_dir / MSPROF_OUTPUT_DIR_NAME
    timed_kernel_time_us: Optional[float] = None
    outputs: List[str] = []

    for run_index in range(MSPROF_RUN_COUNT):
        existing_prof_dirs = list_prof_directories(msprof_output_dir)
        application = f"./{executable_path.name} {m} {k} {n}"
        result = run_command(
            ["msprof", f"--output=./{MSPROF_OUTPUT_DIR_NAME}", f"--application={application}"],
            script_dir,
        )
        output = result.stdout + result.stderr
        outputs.append(f"[msprof run {run_index + 1}]\n{output}".strip())
        if result.returncode != 0:
            raise RuntimeError("\n".join(outputs))

        try:
            prof_dir = resolve_new_prof_directory(msprof_output_dir, existing_prof_dirs)
            op_summary_csv = resolve_op_summary_csv(prof_dir)
            current_kernel_time_us = parse_kernel_time_us_from_csv(op_summary_csv)
        except Exception as error:
            outputs.append(f"[msprof parse error run {run_index + 1}]\n{error}")
            raise RuntimeError("\n".join(outputs)) from error

        # Use the final profiling run as the ranking sample.
        if run_index == MSPROF_RUN_COUNT - 1:
            timed_kernel_time_us = current_kernel_time_us

    if timed_kernel_time_us is None:
        outputs.append(f"Failed to collect the timed kernel duration for {executable_path.name}")
        raise RuntimeError("\n".join(outputs))

    return timed_kernel_time_us, "\n".join(outputs)


def run_candidate(script_dir: Path, candidate: Candidate, m: int, k: int, n: int) -> CandidateResult:
    # Each candidate executable is profiled against the same generated input so
    # the ranking compares kernel time under identical data and shape conditions.
    executable_path = resolve_executable(script_dir, candidate.executable_name)
    try:
        kernel_time_us, output = run_candidate_with_msprof(script_dir, executable_path, m, k, n)
        return_code = 0
    except Exception as error:
        kernel_time_us = None
        output = str(error)
        return_code = 1

    return CandidateResult(
        label=candidate.label,
        executable_path=executable_path,
        kernel_time_us=kernel_time_us,
        return_code=return_code,
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

    script_dir = Path(__file__).resolve().parent
    msprof_output_dir = script_dir / MSPROF_OUTPUT_DIR_NAME
    candidates = discover_candidates(script_dir)
    if not candidates:
        print(f"ERROR: No executable files were found in {script_dir}")
        return 1

    try:
        try:
            generate_input(script_dir, m, k, n)
        except Exception as error:
            print(f"ERROR: {error}")
            return 1

        results: List[CandidateResult] = []
        for candidate in candidates:
            # Preserve per-candidate outputs so failures can still be inspected if
            # no compatible implementation is found.
            candidate_result = run_candidate(script_dir, candidate, m, k, n)
            results.append(candidate_result)

        print_ranking(results)
        return 0 if any(result.succeeded for result in results) else 1
    finally:
        cleanup_msprof_output_dir(msprof_output_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
