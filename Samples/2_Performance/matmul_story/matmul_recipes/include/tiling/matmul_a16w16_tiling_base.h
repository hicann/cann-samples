/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_a16w16_tiling_base.h
 * \brief Base tiling class for A16W16 matmul.
 */

#pragma once

#include <cstdint>
#include <memory>
#include "tiling/matmul_a16w16_tiling_data.h"
#include "tiling/matmul_a16w16_tiling_common.h"
#include "utils/matmul_a16w16_constant.h"
#include "platform/platform_ascendc.h"

class MatmulA16W16TilingBase {
public:
    MatmulA16W16TilingBase() = default;
    virtual ~MatmulA16W16TilingBase() = default;

    virtual void GetTilingData(uint64_t m, uint64_t n, uint64_t k, bool transA, bool transB, MatmulA16W16TilingData& tilingData)
    {
        InitCompileInfo();
        InitShapeArgs(m, n, k, transA, transB);
        DoOpTiling(tilingData);
        PrintTilingData(tilingData);
    };

    // Calculate the size of the intermediate space fo GM
    virtual size_t GetWorkSpace()
    {
        return 0;
    };

protected:
    MatmulA16W16PlatformInfo platformInfo_;
    MatmulA16W16Args args_;
    MatmulA16W16RunInfo runInfo_;

    virtual const char* TilingName() const = 0;
    virtual void DoOpTiling(MatmulA16W16TilingData& tilingData) = 0;

private:
    void InitCompileInfo()
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
        platformInfo_.aicNum = ascendcPlatform->GetCoreNumAic();
        platformInfo_.aivNum = ascendcPlatform->GetCoreNumAiv();
        platformInfo_.socVersion = ascendcPlatform->GetSocVersion();
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo_.ubSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo_.l1Size);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo_.l0aSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo_.l0bSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo_.l0cSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo_.l2Size);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::BT, platformInfo_.btSize);
    }

    void InitShapeArgs(uint64_t m, uint64_t n, uint64_t k, bool transA, bool transB, bool hasBias = false)
    {
        args_.m = m;
        args_.n = n;
        args_.k = k;
        args_.isATrans = transA;
        args_.isBTrans = transB;
        args_.hasBias = hasBias;
    }

    void PrintTilingData(const MatmulA16W16TilingData& tilingData) const
    {
        printf("[Matmul Strategy]\n");
        printf("  strategy           : %s\n", TilingName());
        printf("[Matmul Tiling Data]\n");
        printf("  usedCoreNum        : %u\n", tilingData.usedCoreNum);
        printf("  m                  : %u\n", tilingData.m);
        printf("  n                  : %u\n", tilingData.n);
        printf("  k                  : %u\n", tilingData.k);
        printf("  mL1                : %u\n", tilingData.mL1);
        printf("  nL1                : %u\n", tilingData.nL1);
        printf("  kL1                : %u\n", tilingData.kL1);
        printf("  baseM              : %u\n", tilingData.baseM);
        printf("  baseN              : %u\n", tilingData.baseN);
        printf("  baseK              : %u\n", tilingData.baseK);
        printf("  skSingleCoreK      : %u\n", tilingData.skSingleCoreK);
        printf("  mTailCnt           : %u\n", tilingData.mTailCnt);
        printf("  nTailCnt           : %u\n", tilingData.nTailCnt);
        printf("  mBaseTailSplitCnt  : %u\n", tilingData.mBaseTailSplitCnt);
        printf("  nBaseTailSplitCnt  : %u\n", tilingData.nBaseTailSplitCnt);
        printf("  mTailMain          : %u\n", tilingData.mTailMain);
        printf("  nTailMain          : %u\n", tilingData.nTailMain);
        printf("  l1BufferNum        : %u\n", tilingData.l1BufferNum);
        printf("  l0cDB              : %u\n", tilingData.l0cDB);
    }
};

