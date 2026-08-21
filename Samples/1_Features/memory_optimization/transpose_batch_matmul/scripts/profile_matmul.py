# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import csv
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


MSPROF_OUTPUT_DIR_NAME = "msprof_recommend"
MSPROF_PROF_DIR_PREFIX = "PROF_"
MSPROF_OP_SUMMARY_GLOB = "op_summary_*.csv"
EXECUTABLE_NAME = "transpose_batch_matmul"
# Keep the display order aligned with the recommendation table. The displayed
# MTE labels follow the sample's mte1/mte2 naming convention.
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
class MatmulShape:
    """Encapsulates the correlated matmul shape and transpose/split parameters."""

    m: int
    k: int
    n: int
    batch: int
    perm_x1: int
    perm_x2: int
    batch_split_factor: int


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


@dataclass
class ProfileResult:
    """Execution record for the single transpose_batch_matmul kernel."""

    label: str
    kernel_time_us: Optional[float]
    profile_metrics: Optional[ProfileMetrics]
    return_code: int
    output: str

    @property
    def succeeded(self) -> bool:
        return self.return_code == 0 and self.kernel_time_us is not None and self.profile_metrics is not None


def print_usage(program_name: str) -> None:
    print(f"Usage: {program_name} m k n batch [perm_x1] [perm_x2] [batch_split_factor]")
    print("Args:")
    print("  m: row of matrix A (M dimension)")
    print("  k: shared dimension of A and B")
    print("  n: col of matrix B (N dimension)")
    print("  batch: batch dimension")
    print("  perm_x1: 0=[0, 1, 2] (no transpose), 1=[1, 0, 2] (batch-M swap), default=1")
    print("  perm_x2: 0=[0, 1, 2] (no transpose), 1=[0, 2, 1] (B K-N swap), default=0")
    print("  batch_split_factor: 1=no split, >1=split batch, default=1")
    print(f"Example: {program_name} 32 512 128 16 1 0 1")
    print(f"Example: {program_name} 32 512 128 16 1 0 4")


def parse_positive_int(arg: str, name: str) -> int:
    if not arg.lstrip("-").isdigit():
        raise ValueError(f"{name} must be an integer")
    value = int(arg)
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def parse_arguments(argv: List[str]) -> MatmulShape:
    if len(argv) >= 2 and argv[1] in ("-h", "--help"):
        print_usage(Path(argv[0]).name)
        raise SystemExit(0)
    if len(argv) < 5:
        raise ValueError("Expected at least 4 arguments: m k n batch")

    m = parse_positive_int(argv[1], "m")
    k = parse_positive_int(argv[2], "k")
    n = parse_positive_int(argv[3], "n")
    batch = parse_positive_int(argv[4], "batch")
    perm_x1 = int(argv[5]) if len(argv) > 5 else 1
    perm_x2 = int(argv[6]) if len(argv) > 6 else 0
    batch_split_factor = int(argv[7]) if len(argv) > 7 else 1

    if perm_x1 not in (0, 1):
        raise ValueError("perm_x1 must be 0 or 1")
    if perm_x2 not in (0, 1):
        raise ValueError("perm_x2 must be 0 or 1")
    if batch_split_factor < 1 or batch % batch_split_factor != 0:
        raise ValueError("batch_split_factor must be >= 1 and divide batch")

    return MatmulShape(m, k, n, batch, perm_x1, perm_x2, batch_split_factor)


def read_command_log(log_file) -> str:
    log_file.seek(0)
    return log_file.read().strip()


def format_command_output(prefix: str, raw_output: str) -> str:
    if not raw_output:
        return prefix
    return f"{prefix}\n{raw_output}"


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


def resolve_latest_prof_directory(msprof_output_dir: Path) -> Path:
    prof_dirs = list_prof_directories(msprof_output_dir)
    if not prof_dirs:
        raise FileNotFoundError(
            f"No {MSPROF_PROF_DIR_PREFIX}* directory was generated under {msprof_output_dir}"
        )

    # Each candidate run uses its own clean msprof output directory. If
    # multiple profiling directories still appear, prefer the newest one.
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


def run_with_msprof(script_dir: Path, executable_path: Path, shape: MatmulShape) -> ProfileMetrics:
    msprof_output_dir = script_dir / MSPROF_OUTPUT_DIR_NAME / executable_path.stem
    cleanup_msprof_output_dir(msprof_output_dir)
    msprof_output_dir.parent.mkdir(parents=True, exist_ok=True)
    application = f"./{executable_path.name}"
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log_file:
        result = subprocess.run(
            ["msprof", f"--output={msprof_output_dir}", application,
             str(shape.m), str(shape.k), str(shape.n), str(shape.batch),
             str(shape.perm_x1), str(shape.perm_x2), str(shape.batch_split_factor)],
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


def run_profile(script_dir: Path, shape: MatmulShape) -> ProfileResult:
    # Each candidate executable is profiled against the same generated input so
    # the ranking compares kernel time under identical data and shape conditions.
    executable_path = script_dir / EXECUTABLE_NAME
    if not executable_path.exists():
        raise FileNotFoundError(f"Executable not found in {executable_path}")

    label = EXECUTABLE_NAME
    try:
        profile_metrics = run_with_msprof(script_dir, executable_path, shape)
        kernel_time_us = profile_metrics.kernel_time_us
        output = ""
        return_code = 0
    except Exception as error:
        kernel_time_us = None
        profile_metrics = None
        output = str(error)
        return_code = 1

    return ProfileResult(
        label=label,
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


def print_profile_table(result: ProfileResult, shape: MatmulShape) -> None:
    headers = ["shape"] + [display_name for _field_name, display_name, _column_name in PROFILE_METRIC_SPECS]
    shape_label = (
        f"m={shape.m},k={shape.k},n={shape.n},b={shape.batch},"
        f"px1={shape.perm_x1},px2={shape.perm_x2},bsf={shape.batch_split_factor}"
    )
    if result.profile_metrics is None:
        raise ValueError(f"Profile metrics are missing for {result.label}")
    metric_row = [shape_label]
    for field_name, _display_name, _column_name in PROFILE_METRIC_SPECS:
        metric_row.append(format_metric_cell(getattr(result.profile_metrics, field_name)))

    print("\n[Profile Breakdown]")
    for line in build_ascii_table(headers, [metric_row], right_aligned_columns=set(range(1, len(headers)))):
        print(line)


def main(argv: List[str]) -> int:
    try:
        shape = parse_arguments(argv)
    except ValueError as error:
        print(f"ERROR: {error}")
        print_usage(Path(argv[0]).name)
        return 1

    script_dir = Path(__file__).resolve().parent
    msprof_output_dir = script_dir / MSPROF_OUTPUT_DIR_NAME

    try:
        result = run_profile(script_dir, shape)
        if not result.succeeded:
            print(f"ERROR: Profiling failed for {result.label}")
            print(result.output)
            return 1
        print_profile_table(result, shape)
        print(f"\nKernel time: {result.kernel_time_us:.3f} us")
        return 0
    finally:
        cleanup_msprof_output_dir(msprof_output_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
