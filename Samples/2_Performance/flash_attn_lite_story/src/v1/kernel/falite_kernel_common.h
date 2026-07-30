/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../flash_attn_lite_common.h"
#include "_shared_k.h"

namespace FALite {

// Fixpipe L0C→UB (v1 新增)
template <typename dT, typename sT>
__aicore__ inline void FixpipeToVecUB(
    const AscendC::LocalTensor<dT>& d, const AscendC::LocalTensor<sT>& s, uint32_t m, uint32_t n,
    uint8_t dualDstCtl = 1)
{
    using namespace AscendC;
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> p;
    constexpr uint32_t FA = 8, FMA = 2;
    p.nSize = CeilAlign<uint32_t>(n, FA);
    p.mSize = CeilAlign<uint32_t>(m, FMA);
    p.srcStride = CeilAlign<uint32_t>(p.mSize, BLOCK_CUBE);
    p.dstStride = dualDstCtl == 2 ? p.nSize / 2 : CeilAlign<uint32_t>(p.nSize, BLOCK_CUBE);
    p.dualDstCtl = dualDstCtl;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    Fixpipe<dT, sT, PFA_CFG_UB>(d, s, p);
}

constexpr uint16_t FLAG_S_READY = 0;
constexpr uint16_t FLAG_O_READY = 1;
constexpr uint16_t FLAG_DONE = 2;
constexpr uint16_t FLAG_P_READY = 4;

} // namespace FALite
