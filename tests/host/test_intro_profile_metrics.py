# Copyright (c) 2026 Ray
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host regressions using synthetic CSV files through the real parser; no msprof/NPU."""

import csv
import importlib.util
import math
import sys
from pathlib import Path

import pytest


COLUMNS = (
    "Task Duration(us)",
    "aic_mac_time(us)",
    "aic_scalar_time(us)",
    "aic_mte1_time(us)",
    "aic_mte2_time(us)",
    "aic_fixpipe_time(us)",
    "aic_icache_miss_rate",
)


@pytest.fixture(scope="module")
def profiler():
    source = (
        Path(__file__).resolve().parents[2]
        / "Samples/0_Introduction/matmul/scripts/profile_matmul.py"
    )
    spec = importlib.util.spec_from_file_location("intro_profile_metrics_test", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def write_csv(tmp_path, overrides=None, encoding="utf-8"):
    row = dict.fromkeys(COLUMNS, "1.25")
    row["aic_icache_miss_rate"] = "0.125"
    row.update(overrides or {})
    path = tmp_path / "op_summary_test.csv"
    with path.open("w", encoding=encoding, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    return path


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_valid_metrics_preserve_existing_values(profiler, tmp_path, encoding):
    path = write_csv(tmp_path, {"Task Duration(us)": "1,234.5"}, encoding)
    before = path.read_bytes()
    metrics = profiler.parse_profile_metrics_from_csv(path)
    assert metrics == profiler.ProfileMetrics(
        1234.5, 1.25, 1.25, 1.25, 1.25, 1.25, 12.5
    )
    assert path.read_bytes() == before


@pytest.mark.parametrize("column", COLUMNS)
@pytest.mark.parametrize("value", ["NaN", "inf", "-inf", "1e309", "-0.01"])
def test_invalid_metric_is_rejected(profiler, tmp_path, column, value):
    path = write_csv(tmp_path, {column: value})
    with pytest.raises(ValueError, match="finite, non-negative") as error:
        profiler.parse_profile_metrics_from_csv(path)
    assert column in str(error.value)
    assert str(path) in str(error.value)


@pytest.mark.parametrize("column", COLUMNS)
def test_zero_metrics_remain_valid(profiler, tmp_path, column):
    path = write_csv(tmp_path, {column: "0"})
    metrics = profiler.parse_profile_metrics_from_csv(path)
    assert all(math.isfinite(value) and value >= 0 for value in vars(metrics).values())


@pytest.mark.parametrize("duration", ["-0", "1e-300", "1e308"])
def test_finite_duration_boundaries(profiler, tmp_path, duration):
    path = write_csv(tmp_path, {"Task Duration(us)": duration})
    assert profiler.parse_profile_metrics_from_csv(path).kernel_time_us == float(
        duration
    )


def test_percentage_conversion_overflow_is_rejected(profiler, tmp_path):
    path = write_csv(tmp_path, {"aic_icache_miss_rate": "1e308"})
    with pytest.raises(ValueError, match="aic_icache_miss_rate"):
        profiler.parse_profile_metrics_from_csv(path)


@pytest.mark.parametrize("value", ["", "not-a-number"])
def test_existing_invalid_text_still_fails(profiler, tmp_path, value):
    path = write_csv(tmp_path, {"Task Duration(us)": value})
    with pytest.raises(ValueError):
        profiler.parse_profile_metrics_from_csv(path)
