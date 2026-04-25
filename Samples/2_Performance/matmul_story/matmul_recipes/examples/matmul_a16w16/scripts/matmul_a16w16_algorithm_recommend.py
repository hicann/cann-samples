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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


MSPROF_OUTPUT_DIR_NAME = "msprof_recommend"
MSPROF_PROF_DIR_PREFIX = "PROF_"
MSPROF_OP_SUMMARY_GLOB = "op_summary_*.csv"
PROFILE_METRIC_SPECS = (
    ("kernel_time_us", "kernel(us)", "Task Duration(us)"),
    ("mac_time_us", "mac(us)", "aic_mac_time(us)"),
    ("scalar_time_us", "scalar(us)", "aic_scalar_time(us)"),
    ("mte1_time_us", "mte1(us)", "aic_mte1_time(us)"),
    ("mte2_time_us", "mte2(us)", "aic_mte2_time(us)"),
    ("fixpipe_time_us", "fixpipe(us)", "aic_fixpipe_time(us)"),
    ("icache_miss_rate", "icache_miss(%)", "aic_icache_miss_rate"),
)


@dataclass(frozen=True)
class ProfileMetrics:
    """Performance fields extracted from one op_summary row."""

    kernel_time_us: float
    mac_time_us: float
    scalar_time_us: float
    mte1_time_us: float
    mte2_time_us: float
    fixpipe_time_us: float
    icache_miss_rate: float


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
    profile_metrics: Optional[ProfileMetrics]
    return_code: int
    output: str

    @property
    def succeeded(self) -> bool:
        return self.return_code == 0 and self.kernel_time_us is not None and self.profile_metrics is not None


def print_usage(program_name: str) -> None:
    print(f"Usage: {program_name} m k n [transA transB] [--print-target]")
    print("Args:")
    print("  m: row of matrix A")
    print("  k: col of matrix A")
    print("  n: col of matrix B")
    print("  transA: transdata of matrix A (optional, default: false)")
    print("  transB: transdata of matrix B (optional, default: true)")
    print("Options:")
    print("  --print-target: print only the recommended executable name")
    print(f"Example: {program_name} 100 50 200")
    print(f"Example: {program_name} 100 50 200 false true")


def parse_positive_uint64(arg: str, name: str) -> int:
    if not arg.isdigit():
        raise ValueError(f"{name} must be a positive integer")
    value = int(arg)
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    if value > 4294967295:
        raise ValueError(f"{name} must not exceed UINT32_MAX")
    return value


def parse_arguments(argv: List[str]) -> Tuple[int, int, int, bool, bool, bool]:
    if len(argv) >= 2 and argv[1] in ("-h", "--help"):
        print_usage(Path(argv[0]).name)
        raise SystemExit(0)

    print_target_only = False
    positional: List[str] = []
    for arg in argv[1:]:
        if arg == "--print-target":
            print_target_only = True
            continue
        if arg.startswith("-"):
            raise ValueError(f"Unknown option: {arg}")
        positional.append(arg)

    if len(positional) != 3 and len(positional) != 5:
        raise ValueError("Expected exactly 3 arguments: m k n, or 5 arguments: m k n transA transB")

    m = parse_positive_uint64(positional[0], "m")
    k = parse_positive_uint64(positional[1], "k")
    n = parse_positive_uint64(positional[2], "n")

    if len(positional) == 5:
        trans_a = positional[3].lower() == "true"
        trans_b = positional[4].lower() == "true"
    else:
        trans_a = False
        trans_b = True

    return m, k, n, trans_a, trans_b, print_target_only


def get_ranked_results(results: List[CandidateResult]) -> List[CandidateResult]:
    return sorted(
        [item for item in results if item.succeeded],
        key=lambda item: item.kernel_time_us if item.kernel_time_us is not None else float("inf"),
    )


def resolve_executable(script_dir: Path, executable_name: str) -> Path:
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


def read_command_log(log_file) -> str:
    log_file.seek(0)
    return log_file.read().strip()


def format_command_output(prefix: str, raw_output: str) -> str:
    if not raw_output:
        return prefix
    return f"{prefix}\n{raw_output}"


def resolve_gen_data_script(script_dir: Path) -> Path:
    script_path = script_dir / "gen_data.py"
    if script_path.exists():
        return script_path

    raise FileNotFoundError(f"gen_data.py was not found in {script_dir}")


def generate_input(script_dir: Path, m: int, k: int, n: int, trans_a: bool, trans_b: bool) -> None:
    script_path = resolve_gen_data_script(script_dir)
    args = [sys.executable, str(script_path), str(m), str(k), str(n)]
    if trans_a or trans_b:
        args.extend([str(trans_a).lower(), str(trans_b).lower()])
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log_file:
        result = subprocess.run(
            args,
            cwd=script_path.parent,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            output = read_command_log(log_file)
            raise RuntimeError(f"Failed to generate input data.\n{output}")


def cleanup_msprof_output_dir(msprof_output_dir: Path) -> None:
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


def resolve_latest_prof_directory(msprof_output_dir: Path) -> Path:
    prof_dirs = list_prof_directories(msprof_output_dir)
    if not prof_dirs:
        raise FileNotFoundError(
            f"No {MSPROF_PROF_DIR_PREFIX}* directory was generated under {msprof_output_dir}"
        )

    return max(prof_dirs, key=lambda entry: entry.stat().st_mtime_ns)


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


def parse_metric_value(raw_value: Optional[str], column_name: str, csv_path: Path) -> float:
    if raw_value is None:
        raise ValueError(f"{column_name} column was not found in {csv_path}")

    normalized_value = raw_value.strip().replace(",", "")
    if column_name == "aic_icache_miss_rate":
        normalized_value = normalized_value.rstrip("%")

    if not normalized_value:
        raise ValueError(f"{column_name} is empty in {csv_path}")

    try:
        return float(normalized_value)
    except ValueError as error:
        raise ValueError(f"Failed to parse {column_name} value '{raw_value}' from {csv_path}") from error


def parse_profile_metrics_from_csv(csv_path: Path) -> ProfileMetrics:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        header = reader.fieldnames
        first_row = next(reader, None)

    if not header:
        raise ValueError(f"CSV header is missing in {csv_path}")
    if not first_row:
        raise ValueError(f"CSV data row is missing in {csv_path}")

    metric_values = {
        field_name: parse_metric_value(first_row.get(column_name), column_name, csv_path)
        for field_name, _display_name, column_name in PROFILE_METRIC_SPECS
    }
    metric_values["icache_miss_rate"] *= 100.0
    return ProfileMetrics(**metric_values)


def resolve_candidate_msprof_output_dir(script_dir: Path, executable_path: Path) -> Path:
    return script_dir / MSPROF_OUTPUT_DIR_NAME / executable_path.stem


def run_candidate_with_msprof(
    script_dir: Path, executable_path: Path, m: int, k: int, n: int, trans_a: bool, trans_b: bool
) -> ProfileMetrics:
    msprof_output_dir = resolve_candidate_msprof_output_dir(script_dir, executable_path)
    cleanup_msprof_output_dir(msprof_output_dir)
    msprof_output_dir.parent.mkdir(parents=True, exist_ok=True)
    application = f"./{executable_path.name}"
    cmd_args = ["msprof", f"--output={msprof_output_dir}", application, str(m), str(k), str(n)]
    cmd_args.extend([str(trans_a).lower(), str(trans_b).lower()])
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log_file:
        result = subprocess.run(
            cmd_args,
            cwd=script_dir,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(format_command_output("[msprof]", read_command_log(log_file)))

        try:
            prof_dir = resolve_latest_prof_directory(msprof_output_dir)
            op_summary_csv = resolve_op_summary_csv(prof_dir)
            return parse_profile_metrics_from_csv(op_summary_csv)
        except Exception as error:
            command_output = format_command_output("[msprof]", read_command_log(log_file))
            raise RuntimeError(f"{command_output}\n[msprof parse error]\n{error}") from error


def run_candidate(
    script_dir: Path, candidate: Candidate, m: int, k: int, n: int, trans_a: bool, trans_b: bool
) -> CandidateResult:
    executable_path = resolve_executable(script_dir, candidate.executable_name)
    try:
        profile_metrics = run_candidate_with_msprof(script_dir, executable_path, m, k, n, trans_a, trans_b)
        kernel_time_us = profile_metrics.kernel_time_us
        output = ""
        return_code = 0
    except Exception as error:
        kernel_time_us = None
        profile_metrics = None
        output = str(error)
        return_code = 1

    return CandidateResult(
        label=candidate.label,
        executable_path=executable_path,
        kernel_time_us=kernel_time_us,
        profile_metrics=profile_metrics,
        return_code=return_code,
        output=output,
    )


def format_metric_cell(value: float) -> str:
    return f"{value:.3f}"


def build_ascii_table(headers: List[str], rows: List[List[str]], right_aligned_columns: set[int]) -> List[str]:
    widths = []
    for column_index, header in enumerate(headers):
        column_values = [row[column_index] for row in rows]
        widths.append(max(len(header), *(len(value) for value in column_values)))

    def format_row(row: List[str]) -> str:
        cells = []
        for column_index, value in enumerate(row):
            width = widths[column_index]
            if column_index in right_aligned_columns:
                cells.append(f" {value.rjust(width)} ")
            else:
                cells.append(f" {value.ljust(width)} ")
        return "|" + "|".join(cells) + "|"

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header_separator = "+" + "+".join("=" * (width + 2) for width in widths) + "+"
    lines = [border, format_row(headers), header_separator]
    for row in rows:
        lines.append(format_row(row))
    lines.append(border)
    return lines


def print_profile_table(results: List[CandidateResult]) -> None:
    headers = ["algorithm"] + [display_name for _field_name, display_name, _column_name in PROFILE_METRIC_SPECS]
    rows = []
    for result in results:
        if result.profile_metrics is None:
            raise ValueError(f"Profile metrics are missing for algorithm {result.label}")
        metric_row = [result.label]
        for field_name, _display_name, _column_name in PROFILE_METRIC_SPECS:
            metric_row.append(format_metric_cell(getattr(result.profile_metrics, field_name)))
        rows.append(metric_row)

    print("\n[Profile Breakdown]")
    for line in build_ascii_table(headers, rows, right_aligned_columns=set(range(1, len(headers)))):
        print(line)


def print_ranking(results: List[CandidateResult]) -> None:
    ranked_results = sorted(
        [item for item in results if item.succeeded],
        key=lambda item: item.kernel_time_us if item.kernel_time_us is not None else float("inf"),
    )

    print("\n[Recommended Algorithm Ranking]")

    if not ranked_results:
        print("  No compatible algorithm found for the current shape.")
        return

    for index, result in enumerate(ranked_results, start=1):
        print(f"  {index}. {result.label}")

    print_profile_table(ranked_results)
    print("Note: Only algorithms that support the current shape are listed.\n")


def main(argv: List[str]) -> int:
    try:
        m, k, n, trans_a, trans_b, print_target_only = parse_arguments(argv)
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
            generate_input(script_dir, m, k, n, trans_a, trans_b)
        except Exception as error:
            print(f"ERROR: {error}")
            return 1

        results: List[CandidateResult] = []
        for candidate in candidates:
            candidate_result = run_candidate(script_dir, candidate, m, k, n, trans_a, trans_b)
            results.append(candidate_result)

        ranked_results = get_ranked_results(results)
        if print_target_only:
            if not ranked_results:
                print("ERROR: No compatible algorithm found for the current shape.")
                return 1
            print(ranked_results[0].label)
            return 0

        print_ranking(results)
        return 0 if ranked_results else 1
    finally:
        cleanup_msprof_output_dir(msprof_output_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

