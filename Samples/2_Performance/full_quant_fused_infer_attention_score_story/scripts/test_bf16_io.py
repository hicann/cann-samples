#!/usr/bin/python3
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

"""Host regressions for the FIA sample's BF16 binary contract."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_modules(monkeypatch):
    # NPU generation is not exercised; these are unused import-time dependencies.
    monkeypatch.setitem(sys.modules, "en_dtypes", types.ModuleType("en_dtypes"))
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    modules = []
    for name in ("gen_data", "verify_result"):
        path = Path(__file__).parent / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"fia_host_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def test_reader_decodes_bf16_bits_instead_of_integer_values(tmp_path, sample_modules):
    _, verifier = sample_modules
    bits = np.array([0x3F80, 0xC000, 0x8000, 0x7F80, 0x7FC0], dtype=np.uint16)
    path = tmp_path / "values.bin"
    bits.tofile(path)
    actual = verifier.load_bf16_bin(path).float()
    expected = torch.tensor([1.0, -2.0, -0.0, float("inf"), float("nan")])
    torch.testing.assert_close(actual, expected, equal_nan=True, rtol=0, atol=0)
    assert torch.signbit(actual[2])


def test_writer_uses_bf16_storage_for_float16_golden(tmp_path, sample_modules):
    generator, verifier = sample_modules
    values = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float16)
    path = tmp_path / "golden.bin"
    generator.save_bf16_bin(values, path)
    assert (
        path.read_bytes()
        == np.array([0x3F80, 0xC000, 0x3F00], dtype=np.uint16).tobytes()
    )
    torch.testing.assert_close(
        verifier.load_bf16_bin(path).float(), values.float(), rtol=0, atol=0
    )


def test_verifier_rejects_large_numeric_error_with_adjacent_bf16_bits(
    tmp_path, sample_modules, monkeypatch
):
    _, verifier = sample_modules
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    np.array([0x4480], dtype=np.uint16).tofile(output / "golden_out.bin")
    np.array([0x4481], dtype=np.uint16).tofile(output / "npu_out.bin")
    assert verifier.verify_result() is False


def test_verifier_accepts_equal_signed_zeros(tmp_path, sample_modules, monkeypatch):
    _, verifier = sample_modules
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    np.array([0], dtype=np.uint16).tofile(output / "golden_out.bin")
    np.array([0x8000], dtype=np.uint16).tofile(output / "npu_out.bin")
    assert verifier.verify_result() is True
