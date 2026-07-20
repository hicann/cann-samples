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

"""flash_attn_lite Golden 计算与比对.

默认使用 torch CPU 计算, torch 不可用时回退到 numpy. BF16 读写优先使用
ml_dtypes, 否则使用 round-to-nearest-even 位运算.

处理流程:
  1. 读取 q.bin, k.bin, v.bin 和 npuout_o.bin, 并将 BF16 转为 FP32.
  2. 以 FP32 计算 softmax(scale * Q @ Kᵀ) @ V, 转为 BF16 后写入 golden_o.bin.
  3. 按 |npu-golden| <= atol + rtol * |golden| 逐元素比对.

环境变量:
  FA_VERIFY_BACKEND = auto|torch|numpy
  FA_VERIFY_THREADS = N
  FA_VERIFY_QUERY_BLOCK = N
  FA_VERIFY_FORCE_NUMPY = 1

用法:
    python3 flash_attn_lite_verify.py <data_dir> <B> <S> <D>
退出码:
    0 = 比对通过; 1 = 比对失败或执行出错.
"""

import os
import sys
import time

from thread_limit import configure_python_threads

_VERIFY_THREADS = configure_python_threads()

import numpy as np

# torch_npu autoload 失败可能抛出 RuntimeError, 因此捕获 Exception.
_HAS_TORCH = False
try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except Exception as _e:  # pragma: no cover - 环境相关
    _TORCH_IMPORT_ERR = repr(_e)
else:
    _TORCH_IMPORT_ERR = None


def _resolve_backend() -> str:
    """根据环境变量和 torch 可用性选择后端."""
    forced = os.environ.get("FA_VERIFY_FORCE_NUMPY", "").strip()
    backend_env = os.environ.get("FA_VERIFY_BACKEND", "auto").strip().lower()

    # 优先处理向后兼容的旧别名.
    if forced in ("1", "true", "yes", "on"):
        return "numpy"
    if backend_env == "numpy":
        return "numpy"
    if backend_env == "torch":
        if _HAS_TORCH:
            return "torch"
        # 强制 torch 但导入失败时, 警告并回退到 numpy.
        print(
            f"verify: 警告 FA_VERIFY_BACKEND=torch 但 torch 不可导入"
            f"（{_TORCH_IMPORT_ERR}），回退 numpy",
            file=sys.stderr,
        )
        return "numpy"
    return "torch" if _HAS_TORCH else "numpy"


# 模块加载时确定 Golden 后端.
_BACKEND = _resolve_backend()

if _BACKEND == "torch":
    torch.set_num_threads(_VERIFY_THREADS)
    torch.set_num_interop_threads(1)


# ml_dtypes 和位运算路径均使用 round-to-nearest-even.
try:
    import ml_dtypes

    _HAS_ML_DTYPES = True
except Exception:  # pragma: no cover
    _HAS_ML_DTYPES = False


def bf16_to_fp32(arr: np.ndarray) -> np.ndarray:
    """将 BF16 数组转为 FP32 数组."""
    if _HAS_ML_DTYPES and arr.dtype == ml_dtypes.bfloat16:
        return arr.astype(np.float32)
    # 位运算路径将 BF16 放入 FP32 的高 16 位.
    u16 = arr.view(np.uint16)
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


def fp32_to_bf16_bytes(x: np.ndarray) -> bytes:
    """FP32 ndarray -> 小端 BF16 字节流, 使用 round-to-nearest-even.

    位运算路径按下式舍入:
        rounded = (u32 + 0x7FFF + (lsb_of_truncated & 1)) >> 16
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if _HAS_ML_DTYPES:
        return x.astype(ml_dtypes.bfloat16).tobytes()
    u32 = x.view(np.uint32).copy()
    # 取保留部分的最低位, 用于 round-half-to-even tie-break.
    lsb = (u32 >> np.uint32(16)) & np.uint32(1)
    rounded = (u32 + np.uint32(0x7FFF) + lsb) >> np.uint32(16)
    return rounded.astype(np.uint16).tobytes()


# 采用 CANN BF16 混合容差, 并要求全部元素通过.
COMPARE_RTOL = 2.0 ** -6
COMPARE_ATOL = 2.0 ** -6
# 报告显示前 4 x 4 个元素.
PRINT_ROWS = 4
PRINT_COLS = 4


def read_bf16(path: str, shape, msg_ctx: str = "") -> np.ndarray:
    """读取行主序 raw BF16 文件, 并返回指定 shape 的 FP32 ndarray."""
    with open(path, "rb") as f:
        raw = f.read()
    expect = int(np.prod(shape)) * 2  # bf16 = 2 bytes
    if len(raw) != expect:
        raise ValueError(
            f"读取{msg_ctx} {path} 字节数不符：得到 {len(raw)}，"
            f"期望 {expect}（shape={shape}）"
        )
    if _HAS_ML_DTYPES:
        bf16 = np.frombuffer(raw, dtype=ml_dtypes.bfloat16).reshape(shape)
    else:
        bf16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    return bf16_to_fp32(bf16)


def compute_golden_torch(qf: np.ndarray, kf: np.ndarray, vf: np.ndarray,
                         scale: float) -> np.ndarray:
    """使用 torch FP32 计算 O = softmax(scale * Q @ Kᵀ) @ V."""
    # from_numpy 与 tensor 共享内存, 后续仅读取输入.
    q = torch.from_numpy(qf)
    k = torch.from_numpy(kf)
    v = torch.from_numpy(vf)

    query_block_env = os.environ.get("FA_VERIFY_QUERY_BLOCK", "").strip()
    if query_block_env:
        try:
            query_block = int(query_block_env)
        except ValueError:
            query_block = 0
        if query_block <= 0:
            raise ValueError("FA_VERIFY_QUERY_BLOCK 必须是正整数")
    else:
        query_block = q.shape[1]

    # 仅分块 Q 行, 每块仍使用完整 K/V, 避免一次性分配 B*S*S scores.
    o = torch.empty_like(q)
    kt = k.transpose(-2, -1)
    for row_begin in range(0, q.shape[1], query_block):
        row_end = min(row_begin + query_block, q.shape[1])
        scores = (q[:, row_begin:row_end] @ kt).mul_(scale)
        scores.sub_(scores.amax(dim=-1, keepdim=True))
        scores.exp_()
        scores.div_(scores.sum(dim=-1, keepdim=True))
        o[:, row_begin:row_end] = scores @ v
    return o.contiguous().numpy()


def compute_golden_numpy(qf: np.ndarray, kf: np.ndarray, vf: np.ndarray,
                         scale: float) -> np.ndarray:
    """使用 numpy FP32 计算 O = softmax(scale * Q @ Kᵀ) @ V."""
    s = scale * (qf @ kf.transpose(0, 2, 1))          # (B,S,S) FP32
    s = s - s.max(axis=2, keepdims=True)               # 减去 rowmax, 保持数值稳定.
    e = np.exp(s)                                      # (B,S,S)
    sm = e / e.sum(axis=2, keepdims=True)              # 归一化后的 P.
    o = sm @ vf                                        # (B,S,D) fp32
    return np.ascontiguousarray(o)


def compute_golden(qf: np.ndarray, kf: np.ndarray, vf: np.ndarray, scale: float):
    """使用选定的后端计算 Golden."""
    if _BACKEND == "torch":
        return compute_golden_torch(qf, kf, vf, scale)
    return compute_golden_numpy(qf, kf, vf, scale)


def compare(npu_fp32: np.ndarray, golden_fp32: np.ndarray):
    """逐元素执行 |npu-golden| <= atol + rtol * |golden| 比对."""
    tot = int(npu_fp32.size)
    abs_err = np.abs(npu_fp32 - golden_fp32)
    tol = COMPARE_ATOL + COMPARE_RTOL * np.abs(golden_fp32)
    ok = abs_err <= tol
    fail_count = int(np.count_nonzero(~ok))

    # denom 避免 |golden|=0, 与 C++ max(|golden|, float_min) 一致.
    denom_min = float(np.finfo(np.float32).tiny)  # ~1.18e-38, 等价于 C++ numeric_limits::min.
    denom = np.maximum(np.abs(golden_fp32), denom_min)
    rel_err = abs_err / denom

    max_abs_idx = int(np.argmax(abs_err.reshape(-1)))
    max_abs_err = float(abs_err.reshape(-1)[max_abs_idx])
    max_rel_err = float(np.max(rel_err))
    max_abs_pos = np.unravel_index(max_abs_idx, npu_fp32.shape)  # (b, i, d)

    lines = []
    lines.append(
        f"比对：总元素={tot} 失败={fail_count} "
        f"最大绝对误差={max_abs_err:.6e} @idx={max_abs_idx}"
        f"(b={max_abs_pos[0]}, row={max_abs_pos[1]}, col={max_abs_pos[2]}) "
        f"最大相对误差={max_rel_err:.6e}（rtol={COMPARE_RTOL} atol={COMPARE_ATOL}）"
    )

    # 显示 batch 0 的前 PRINT_ROWS x PRINT_COLS 个元素.
    B, S, D = npu_fp32.shape
    r = min(PRINT_ROWS, S)
    c = min(PRINT_COLS, D)
    lines.append(f"前 {r}x{c} O 元素（batch 0）：")
    for i in range(r):
        for j in range(c):
            a = float(npu_fp32[0, i, j])
            g = float(golden_fp32[0, i, j])
            tag = "OK" if ok[0, i, j] else "FAIL"
            lines.append(
                f"  [{i:3d},{j:3d}] npu={a:13.6e} golden={g:13.6e} "
                f"abs={abs(a - g):13.6e} {tag}"
            )

    passed = fail_count == 0
    return passed, lines


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

    # 记录实际 Golden 后端及 BF16 路径.
    wanted = os.environ.get("FA_VERIFY_BACKEND", "auto").strip().lower()
    forced_np = os.environ.get("FA_VERIFY_FORCE_NUMPY", "").strip() in (
        "1", "true", "yes", "on")
    passive_np = (
        _BACKEND == "numpy" and not _HAS_TORCH
        and wanted != "numpy" and not forced_np
    )
    if _BACKEND == "torch":
        print(f"verify: backend=torch({torch.get_num_threads()} threads)")
    else:
        print("verify: backend=numpy")
    if passive_np:
        print(
            f"verify: 警告 torch 不可用（{_TORCH_IMPORT_ERR}），用 numpy 回退",
            file=sys.stderr,
        )
    if not _HAS_ML_DTYPES:
        print("verify: bf16 走位运算回退（未装 ml_dtypes）", file=sys.stderr)

    shape = (b, s, d)
    try:
        t0 = time.perf_counter()
        qf = read_bf16(os.path.join(data_dir, "q.bin"), shape, "Q")
        kf = read_bf16(os.path.join(data_dir, "k.bin"), shape, "K")
        vf = read_bf16(os.path.join(data_dir, "v.bin"), shape, "V")
        npu_fp32 = read_bf16(os.path.join(data_dir, "npuout_o.bin"), shape, "NPU O")
        t_read = time.perf_counter() - t0
    except (OSError, ValueError) as e:
        print(f"读取输入失败：{e}", file=sys.stderr)
        return 1

    scale = 1.0 / float(d) ** 0.5
    t0 = time.perf_counter()
    o_golden_fp32 = compute_golden(qf, kf, vf, scale)
    t_golden = time.perf_counter() - t0

    # 以 round-to-nearest-even BF16 写入 golden_o.bin.
    golden_path = os.path.join(data_dir, "golden_o.bin")
    with open(golden_path, "wb") as f:
        f.write(fp32_to_bf16_bytes(o_golden_fp32))

    backend_tag = (
        f"torch×{torch.get_num_threads()}线程" if _BACKEND == "torch" else "numpy"
    )
    print(
        f"verify: 读入 {t_read:.3f}s，Golden(Q@Kᵀ→softmax→P@V, fp32/{backend_tag}) "
        f"{t_golden:.3f}s -> golden_o.bin ({b*s*d*2} bytes)"
    )

    # 重新读取落盘的 BF16 Golden, 确保比对精度口径一致.
    golden_fp32_disk = read_bf16(golden_path, shape, "Golden(disk)")
    passed, report_lines = compare(npu_fp32, golden_fp32_disk)
    print("\n".join(report_lines))

    if passed:
        print("比对成功 ✓（npuout 与 golden 在容差内）")
        return 0
    # 失败信息使用 stdout, 保持报告顺序稳定.
    print("比对失败 ✗（npuout 超出容差）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
