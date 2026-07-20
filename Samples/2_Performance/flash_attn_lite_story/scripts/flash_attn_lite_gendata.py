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

"""flash_attn_lite 输入数据生成器.

使用 numpy 生成 Q/K/V, 并以行主序 raw BF16 写入文件. BF16 转换优先使用
ml_dtypes, 否则使用 round-to-nearest-even 位运算.

用法:
    python3 flash_attn_lite_gendata.py <data_dir> <B> <S> <D>
产出:
    <data_dir>/q.bin, k.bin, v.bin, 物理 shape=(B,S,D), dtype=BF16.
    当前逻辑 shape 为 (B,1,S,D).
"""

import os
import sys
import time

from thread_limit import configure_python_threads

configure_python_threads()

import numpy as np

# ml_dtypes 和位运算路径均使用 round-to-nearest-even.
try:
    import ml_dtypes

    _HAS_ML_DTYPES = True
except Exception:  # pragma: no cover
    _HAS_ML_DTYPES = False


def fp32_to_bf16_bytes(x: np.ndarray) -> bytes:
    """FP32 ndarray -> 小端 BF16 字节流, 使用 round-to-nearest-even.

    位运算路径按下式舍入:
        rounded = (u32 + 0x7FFF + (lsb_of_truncated & 1)) >> 16
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if _HAS_ML_DTYPES:
        return x.astype(ml_dtypes.bfloat16).tobytes()
    u32 = x.view(np.uint32).copy()
    lsb = (u32 >> np.uint32(16)) & np.uint32(1)
    rounded = (u32 + np.uint32(0x7FFF) + lsb) >> np.uint32(16)
    return rounded.astype(np.uint16).tobytes()


def main() -> int:
    if len(sys.argv) != 5:
        print(f"用法: {sys.argv[0]} <data_dir> <B> <S> <D>", file=sys.stderr)
        return 1

    data_dir = sys.argv[1]
    b = int(sys.argv[2])
    s = int(sys.argv[3])
    d = int(sys.argv[4])

    if b <= 0 or s <= 0 or d <= 0:
        print(f"错误: B/S/D 必须为正整数，得到 B={b} S={s} D={d}", file=sys.stderr)
        return 1
    if s % 128 != 0:
        print(f"警告: S={s} 不是 128 的整数倍（规格要求 S 为 128 倍数）", file=sys.stderr)

    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(20260709)

    if _HAS_ML_DTYPES:
        bf16_tag = "ml_dtypes"
    else:
        bf16_tag = "位运算回退"
        print("gendata: 未装 ml_dtypes，bf16 走位运算回退", file=sys.stderr)

    t0 = time.perf_counter()
    qf = rng.standard_normal((b, s, d)).astype("<f4")
    kf = rng.standard_normal((b, s, d)).astype("<f4")
    vf = rng.standard_normal((b, s, d)).astype("<f4")

    # q.bin, k.bin, v.bin 均为 shape=(B,S,D) 的行主序 BF16.
    for name, x in (("q", qf), ("k", kf), ("v", vf)):
        with open(os.path.join(data_dir, f"{name}.bin"), "wb") as f:
            f.write(fp32_to_bf16_bytes(np.ascontiguousarray(x)))

    dt = time.perf_counter() - t0
    print(
        f"gendata: B={b} S={s} D={d} -> {{q,k,v}}.bin "
        f"(bf16/{bf16_tag}, {b*s*d*2} bytes each) {dt:.3f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
