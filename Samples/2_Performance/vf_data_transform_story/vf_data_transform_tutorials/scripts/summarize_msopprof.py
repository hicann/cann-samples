#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Print per-stage msopprof Task Duration values and their median."""

import argparse
import csv
import statistics
from pathlib import Path


def parse_duration(csv_path: Path) -> float:
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if len(rows) != 1 or "Task Duration(us)" not in rows[0]:
        raise ValueError(f"unexpected OpBasicInfo schema: {csv_path}")
    return float(rows[0]["Task Duration(us)"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", type=Path)
    args = parser.parse_args()

    found = False
    print("stage\truns_us\tmedian_us")
    for stage_dir in sorted(path for path in args.profile_dir.iterdir() if path.is_dir()):
        durations = [parse_duration(path) for path in sorted(stage_dir.glob("run*/OPPROF_*/OpBasicInfo.csv"))]
        if not durations:
            continue
        found = True
        runs = ",".join(f"{duration:.6f}" for duration in durations)
        print(f"{stage_dir.name}\t{runs}\t{statistics.median(durations):.6f}")
    if not found:
        raise FileNotFoundError(f"no OpBasicInfo.csv found below {args.profile_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
