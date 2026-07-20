#!/usr/bin/env python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""为共享机器上的 Python/BLAS 计算设置线程上限."""

import os


_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _available_cpu_threads() -> int:
    """返回当前进程 affinity 范围内的可用 CPU 线程数."""
    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except (AttributeError, OSError):
        return max(os.cpu_count() or 1, 1)


def _positive_int_env(name: str):
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def configure_python_threads() -> int:
    """设置线程池并返回实际线程数, 外部设置不得超过自动上限."""
    available_threads = _available_cpu_threads()
    auto_limit = min(max(available_threads // 2, 16), available_threads)

    requested = _positive_int_env("FA_VERIFY_THREADS")
    if os.environ.get("FA_VERIFY_THREADS", "").strip() and requested is None:
        raise ValueError("FA_VERIFY_THREADS 必须是正整数")

    actual = min(requested, auto_limit) if requested is not None else auto_limit
    for name in _THREAD_ENV_NAMES:
        existing = _positive_int_env(name)
        if existing is not None:
            actual = min(actual, existing)
    for name in _THREAD_ENV_NAMES:
        os.environ[name] = str(actual)
    return actual
