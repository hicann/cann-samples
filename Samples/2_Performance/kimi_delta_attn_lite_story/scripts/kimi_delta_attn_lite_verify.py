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

"""Kimi Delta Attention Lite Golden 计算与比对.

auto 模式使用 NumPy 处理小规格, 计算规模较大时优先使用 Torch CPU;
Torch 不可用时回退到 NumPy. BF16 读写优先使用 ml_dtypes, 否则使用
round-to-nearest-even 位运算.

处理流程:
  1. 读取已经落盘的 BF16 Q/K/V/beta 和 FP32 log_decay.
  2. 以 FP32 递推计算 O 与 final_state, 写出 Golden 文件.
  3. O 量化为 BF16 后比较, final_state 按 FP32 比较.

环境变量:
  KDA_VERIFY_BACKEND = auto|torch|numpy
    auto 在 B>=32 且 B*S>=262144 时优先使用 Torch, 其余使用 NumPy.
  KDA_PYTHON_THREADS = N

用法:
    python3 kimi_delta_attn_lite_verify.py <data_dir> <B> <S> <D>
退出码:
    0 = 比对通过; 1 = 比对失败或执行出错.
"""

import os
import sys
import time
from typing import Optional, Tuple

from kdalite_thread_limit import configure_python_threads

try:
    _PYTHON_THREADS = configure_python_threads()
except ValueError as error:
    print(f"错误: {error}", file=sys.stderr)
    sys.exit(1)

import numpy as np

torch = None
_TORCH_IMPORT_ATTEMPTED = False
_TORCH_IMPORT_ERROR = None
_TORCH_THREADS_CONFIGURED = False

try:
    import ml_dtypes

    _HAS_ML_DTYPES = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_ML_DTYPES = False


COMPARE_RTOL = 2.0**-6
COMPARE_ATOL = 2.0**-6
COMPARE_BLOCK_ELEMS = 1 << 20


_BACKEND_REQUEST = os.environ.get("KDA_VERIFY_BACKEND", "auto").strip().lower()
if _BACKEND_REQUEST not in ("auto", "torch", "numpy"):
    print("错误: KDA_VERIFY_BACKEND 必须是 auto、torch 或 numpy", file=sys.stderr)
    sys.exit(1)


def _load_torch() -> bool:
    """Import and configure Torch only when the selected shape can benefit."""
    global torch, _TORCH_IMPORT_ATTEMPTED, _TORCH_IMPORT_ERROR, _TORCH_THREADS_CONFIGURED
    if not _TORCH_IMPORT_ATTEMPTED:
        _TORCH_IMPORT_ATTEMPTED = True
        try:
            import torch as torch_module
        except Exception as error:  # pragma: no cover - environment dependent
            _TORCH_IMPORT_ERROR = repr(error)
        else:
            torch = torch_module
    if torch is None:
        return False
    if not _TORCH_THREADS_CONFIGURED:
        torch.set_num_threads(_PYTHON_THREADS)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _TORCH_THREADS_CONFIGURED = True
    return True


def _select_backend(batch_size: int, seq_len: int) -> str:
    if _BACKEND_REQUEST == "numpy":
        return "numpy"
    if _BACKEND_REQUEST == "auto" and (batch_size < 32 or batch_size * seq_len < 262144):
        return "numpy"
    if _load_torch():
        return "torch"
    return "numpy"


def bf16_to_fp32(value: np.ndarray) -> np.ndarray:
    """Convert ml_dtypes BF16 or raw uint16 BF16 to FP32."""
    if _HAS_ML_DTYPES and value.dtype == ml_dtypes.bfloat16:
        return value.astype(np.float32)
    bits = value.view(np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def fp32_to_bf16(value: np.ndarray) -> np.ndarray:
    """Convert FP32 to little-endian BF16 with round-to-nearest-even."""
    value = np.ascontiguousarray(value, dtype=np.float32)
    if _HAS_ML_DTYPES:
        return value.astype(ml_dtypes.bfloat16)
    bits = value.view(np.uint32).copy()
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = (bits + np.uint32(0x7FFF) + lsb) >> np.uint32(16)
    return rounded.astype("<u2")


def _check_file_bytes(path: str, shape, item_bytes: int, name: str) -> int:
    expected_elems = int(np.prod(shape))
    expected_bytes = expected_elems * item_bytes
    actual_bytes = os.path.getsize(path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"读取 {name} {path} 字节数不符：得到 {actual_bytes}，"
            f"期望 {expected_bytes}（shape={shape}）"
        )
    return expected_elems


def read_bf16(path: str, shape, name: str) -> np.ndarray:
    """Read row-major raw BF16 and return an FP32 ndarray."""
    expected_elems = _check_file_bytes(path, shape, 2, name)
    dtype = ml_dtypes.bfloat16 if _HAS_ML_DTYPES else np.dtype("<u2")
    value = np.fromfile(path, dtype=dtype, count=expected_elems).reshape(shape)
    return bf16_to_fp32(value)


def read_fp32(path: str, shape, name: str) -> np.ndarray:
    """Read row-major little-endian FP32."""
    expected_elems = _check_file_bytes(path, shape, 4, name)
    return np.fromfile(path, dtype="<f4", count=expected_elems).reshape(shape)


def require_finite(name: str, value: np.ndarray) -> None:
    """Check finite values in bounded-size blocks."""
    flat = value.reshape(-1)
    for begin in range(0, flat.size, COMPARE_BLOCK_ELEMS):
        block = flat[begin : begin + COMPARE_BLOCK_ELEMS]
        bad = np.flatnonzero(~np.isfinite(block))
        if bad.size:
            bad_flat = begin + int(bad[0])
            bad_pos = tuple(int(index) for index in np.unravel_index(bad_flat, value.shape))
            raise ValueError(f"{name} 含 NaN/Inf，首个位置={bad_pos}，值={value[bad_pos]}")


def require_range(name: str, value: np.ndarray, lower: Optional[float], upper: Optional[float]) -> None:
    """Check an inclusive range in bounded-size blocks."""
    flat = value.reshape(-1)
    for begin in range(0, flat.size, COMPARE_BLOCK_ELEMS):
        block = flat[begin : begin + COMPARE_BLOCK_ELEMS]
        invalid = np.zeros(block.shape, dtype=bool)
        if lower is not None:
            invalid |= block < lower
        if upper is not None:
            invalid |= block > upper
        bad = np.flatnonzero(invalid)
        if bad.size:
            bad_flat = begin + int(bad[0])
            bad_pos = tuple(int(index) for index in np.unravel_index(bad_flat, value.shape))
            if lower is None:
                requirement = f"必须 <= {upper}"
            elif upper is None:
                requirement = f"必须 >= {lower}"
            else:
                requirement = f"必须位于 [{lower},{upper}]"
            raise ValueError(f"{name} {requirement}，首个位置={bad_pos}，值={value[bad_pos]}")


def _validate_shapes(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    log_decay: np.ndarray,
    beta: np.ndarray,
    initial_state: Optional[np.ndarray],
) -> Tuple[int, int, int, int]:
    if q.shape != k.shape or q.shape != log_decay.shape:
        raise ValueError(f"Q/K/log_decay shape 必须一致，得到 {q.shape}/{k.shape}/{log_decay.shape}")
    if q.ndim != 3 or v.ndim != 3:
        raise ValueError("Q/K/V 必须是 [B,S,D] 三维数组")
    if q.shape[:2] != v.shape[:2] or beta.shape != q.shape[:2]:
        raise ValueError(f"V/beta shape 与 Q 不匹配：Q={q.shape} V={v.shape} beta={beta.shape}")

    batch_size, seq_len, key_dim = q.shape
    value_dim = v.shape[-1]
    if initial_state is not None and initial_state.shape != (batch_size, key_dim, value_dim):
        raise ValueError(
            f"initial_state shape 不符：得到 {initial_state.shape}，"
            f"期望 {(batch_size, key_dim, value_dim)}"
        )
    return batch_size, seq_len, key_dim, value_dim


def recurrent_kda_golden_numpy(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    log_decay: np.ndarray,
    beta: np.ndarray,
    initial_state: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Recurrent KDA with NumPy and reusable scratch buffers."""
    batch_size, seq_len, key_dim, value_dim = _validate_shapes(q, k, v, log_decay, beta, initial_state)
    if initial_state is None:
        state = np.zeros((batch_size, key_dim, value_dim), dtype=np.float32)
    else:
        state = np.array(initial_state, dtype=np.float32, copy=True)

    output = np.empty((batch_size, seq_len, value_dim), dtype=np.float32)
    decay = np.empty((batch_size, key_dim), dtype=np.float32)
    prediction = np.empty((batch_size, value_dim), dtype=np.float32)
    residual = np.empty((batch_size, value_dim), dtype=np.float32)
    state_update = np.empty_like(state)

    for token_idx in range(seq_len):
        np.exp(log_decay[:, token_idx, :], out=decay)
        state *= decay[:, :, None]
        np.einsum(
            "bi,bij->bj", k[:, token_idx, :], state, out=prediction, optimize=False
        )
        np.subtract(v[:, token_idx, :], prediction, out=residual)
        residual *= beta[:, token_idx, None]
        np.einsum(
            "bi,bj->bij", k[:, token_idx, :], residual, out=state_update, optimize=False
        )
        state += state_update
        np.einsum(
            "bi,bij->bj", q[:, token_idx, :], state, out=output[:, token_idx, :], optimize=False
        )
    return output, state


def recurrent_kda_golden_torch(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    log_decay: np.ndarray,
    beta: np.ndarray,
    initial_state: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Recurrent KDA with Torch CPU."""
    batch_size, seq_len, key_dim, value_dim = _validate_shapes(q, k, v, log_decay, beta, initial_state)
    q_tensor = torch.from_numpy(q)
    k_tensor = torch.from_numpy(k)
    v_tensor = torch.from_numpy(v)
    decay_tensor = torch.from_numpy(log_decay)
    beta_tensor = torch.from_numpy(beta)
    if initial_state is None:
        state = torch.zeros((batch_size, key_dim, value_dim), dtype=torch.float32)
    else:
        state = torch.from_numpy(np.array(initial_state, dtype=np.float32, copy=True))
    output = torch.empty((batch_size, seq_len, value_dim), dtype=torch.float32)

    with torch.inference_mode():
        for token_idx in range(seq_len):
            state.mul_(torch.exp(decay_tensor[:, token_idx, :]).unsqueeze(-1))
            prediction = torch.bmm(k_tensor[:, token_idx : token_idx + 1, :], state).squeeze(1)
            residual = beta_tensor[:, token_idx, None] * (v_tensor[:, token_idx, :] - prediction)
            state.add_(k_tensor[:, token_idx, :, None] * residual[:, None, :])
            output[:, token_idx, :] = torch.bmm(
                q_tensor[:, token_idx : token_idx + 1, :], state
            ).squeeze(1)
    return output.numpy(), state.numpy()


def recurrent_kda_golden(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    log_decay: np.ndarray,
    beta: np.ndarray,
    initial_state: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute token-by-token Recurrent KDA with the selected CPU backend."""
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    log_decay = np.asarray(log_decay, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    if initial_state is not None:
        initial_state = np.asarray(initial_state, dtype=np.float32)
    if _select_backend(q.shape[0], q.shape[1]) == "torch":
        return recurrent_kda_golden_torch(q, k, v, log_decay, beta, initial_state)
    return recurrent_kda_golden_numpy(q, k, v, log_decay, beta, initial_state)


def _format_position(flat_index: int, shape) -> str:
    position = tuple(int(index) for index in np.unravel_index(flat_index, shape))
    if len(position) == 3:
        return f"@idx={flat_index}(b={position[0]}, row={position[1]}, col={position[2]})"
    return f"@idx={flat_index}{position}"


def compare(name: str, actual: np.ndarray, expected: np.ndarray):
    """Compare in bounded-size blocks and return report lines."""
    if actual.shape != expected.shape:
        raise ValueError(f"{name} shape 不一致：NPU={actual.shape} Golden={expected.shape}")

    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    fail_count = 0
    first_fail = None
    max_abs_error = -1.0
    max_abs_index = 0
    max_rel_error = -1.0
    max_rel_index = 0
    tiny = float(np.finfo(np.float32).tiny)

    for begin in range(0, actual_flat.size, COMPARE_BLOCK_ELEMS):
        end = min(begin + COMPARE_BLOCK_ELEMS, actual_flat.size)
        actual_block = actual_flat[begin:end]
        expected_block = expected_flat[begin:end]

        for value_name, block in ((f"NPU {name}", actual_block), (f"Golden {name}", expected_block)):
            bad = np.flatnonzero(~np.isfinite(block))
            if bad.size:
                bad_flat = begin + int(bad[0])
                bad_pos = tuple(int(index) for index in np.unravel_index(bad_flat, actual.shape))
                raise ValueError(f"{value_name} 含 NaN/Inf，首个位置={bad_pos}，值={block[int(bad[0])]}")

        abs_error = np.abs(actual_block - expected_block)
        expected_abs = np.abs(expected_block)
        tolerance = COMPARE_ATOL + COMPARE_RTOL * expected_abs
        failed = abs_error > tolerance
        block_fail_count = int(np.count_nonzero(failed))
        fail_count += block_fail_count
        if block_fail_count and first_fail is None:
            local_first = int(np.flatnonzero(failed)[0])
            first_fail = (
                begin + local_first,
                float(actual_block[local_first]),
                float(expected_block[local_first]),
                float(abs_error[local_first]),
                float(tolerance[local_first]),
            )

        local_abs = int(np.argmax(abs_error))
        if float(abs_error[local_abs]) > max_abs_error:
            max_abs_error = float(abs_error[local_abs])
            max_abs_index = begin + local_abs

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            rel_error = abs_error / np.maximum(expected_abs, tiny)
        local_rel = int(np.argmax(rel_error))
        if float(rel_error[local_rel]) > max_rel_error:
            max_rel_error = float(rel_error[local_rel])
            max_rel_index = begin + local_rel

    lines = [
        f"比对 {name}：总元素={actual_flat.size} 失败={fail_count} "
        f"最大绝对误差={max_abs_error:.6e} {_format_position(max_abs_index, actual.shape)} "
        f"最大相对误差={max_rel_error:.6e} {_format_position(max_rel_index, actual.shape)} "
        f"（rtol={COMPARE_RTOL} atol={COMPARE_ATOL}）"
    ]
    if first_fail is not None:
        flat_index, actual_value, expected_value, abs_error, tolerance = first_fail
        lines.append(
            f"  首个失败：{_format_position(flat_index, actual.shape)} "
            f"npu={actual_value:.7e} golden={expected_value:.7e} "
            f"abs={abs_error:.7e} tolerance={tolerance:.7e}"
        )
    return fail_count == 0, lines


def main() -> int:
    if len(sys.argv) != 5:
        print(f"用法: {sys.argv[0]} <data_dir> <B> <S> <D>", file=sys.stderr)
        return 1
    try:
        batch_size = int(sys.argv[2])
        seq_len = int(sys.argv[3])
        dim = int(sys.argv[4])
    except ValueError:
        print("错误: B/S/D 必须是整数", file=sys.stderr)
        return 1
    if batch_size <= 0 or seq_len <= 0 or dim != 128:
        print(
            f"错误: KDALite 要求 B/S 为正整数且 D=128，"
            f"得到 B={batch_size} S={seq_len} D={dim}",
            file=sys.stderr,
        )
        return 1

    backend = _select_backend(batch_size, seq_len)
    if backend == "torch":
        print(f"verify: backend=torch({torch.get_num_threads()} threads)", flush=True)
    else:
        print(f"verify: backend=numpy(thread limit={_PYTHON_THREADS})", flush=True)
        if _TORCH_IMPORT_ATTEMPTED and torch is None:
            print(
                f"verify: 警告 torch 不可用（{_TORCH_IMPORT_ERROR}），回退 numpy",
                file=sys.stderr,
            )
    if not _HAS_ML_DTYPES:
        print("verify: bf16 走位运算回退（未装 ml_dtypes）", file=sys.stderr)

    data_dir = sys.argv[1]
    sequence_shape = (batch_size, seq_len, dim)
    state_shape = (batch_size, dim, dim)
    try:
        start = time.perf_counter()
        q = read_bf16(os.path.join(data_dir, "q.bin"), sequence_shape, "Q")
        k = read_bf16(os.path.join(data_dir, "k.bin"), sequence_shape, "K")
        v = read_bf16(os.path.join(data_dir, "v.bin"), sequence_shape, "V")
        log_decay = read_fp32(os.path.join(data_dir, "log_decay.bin"), sequence_shape, "log_decay")
        beta = read_bf16(os.path.join(data_dir, "beta.bin"), (batch_size, seq_len), "beta")
        read_seconds = time.perf_counter() - start
    except (OSError, ValueError, MemoryError) as error:
        print(f"读取输入失败：{error}", file=sys.stderr)
        return 1

    try:
        for name, value in (("Q", q), ("K", k), ("V", v), ("log_decay", log_decay), ("beta", beta)):
            require_finite(name, value)
        require_range("log_decay", log_decay, None, 0.0)
        require_range("beta", beta, 0.0, 1.0)
    except ValueError as error:
        print(f"输入校验失败：{error}", file=sys.stderr)
        return 1

    try:
        start = time.perf_counter()
        golden_o_fp32, golden_state = recurrent_kda_golden(q, k, v, log_decay, beta)
        golden_seconds = time.perf_counter() - start
        require_finite("Golden O(FP32)", golden_o_fp32)
        require_finite("Golden final_state", golden_state)
        del q, k, v, log_decay, beta

        golden_o_bf16 = fp32_to_bf16(golden_o_fp32)
        del golden_o_fp32
        golden_o_bf16.tofile(os.path.join(data_dir, "golden_o.bin"))
        golden_o = bf16_to_fp32(golden_o_bf16)
        del golden_o_bf16
        np.ascontiguousarray(golden_state, dtype="<f4").tofile(
            os.path.join(data_dir, "golden_final_state.bin")
        )
    except (OSError, ValueError, RuntimeError, MemoryError) as error:
        print(f"Golden 计算失败：{error}", file=sys.stderr)
        return 1

    try:
        start = time.perf_counter()
        npu_o = read_bf16(os.path.join(data_dir, "npuout_o.bin"), sequence_shape, "NPU O")
        npu_state = read_fp32(
            os.path.join(data_dir, "npuout_final_state.bin"), state_shape, "NPU final_state"
        )
        read_seconds += time.perf_counter() - start
    except (OSError, ValueError, MemoryError) as error:
        print(f"读取 NPU 输出失败：{error}", file=sys.stderr)
        return 1

    backend_tag = (
        f"torch×{torch.get_num_threads()}线程"
        if backend == "torch"
        else f"numpy，线程上限={_PYTHON_THREADS}"
    )
    print(
        f"verify: 读入 {read_seconds:.3f}s，Golden(recurrent KDA, fp32/{backend_tag}) "
        f"{golden_seconds:.3f}s -> {{golden_o,golden_final_state}}.bin"
    )

    try:
        o_passed, o_lines = compare("O(BF16)", npu_o, golden_o)
        del npu_o, golden_o
        state_passed, state_lines = compare("final_state(FP32)", npu_state, golden_state)
    except ValueError as error:
        print(f"比对失败：{error}", file=sys.stderr)
        return 1

    print("\n".join(o_lines + state_lines))
    if o_passed and state_passed:
        print("比对成功 ✓（O 与 final_state 均在容差内）")
        return 0
    print("比对失败 ✗（O 或 final_state 超出容差）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
