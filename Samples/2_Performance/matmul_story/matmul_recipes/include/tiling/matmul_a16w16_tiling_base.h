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

#ifndef MATMUL_A16W16_TILING_BASE_H
#define MATMUL_A16W16_TILING_BASE_H

#include <cstdint>
#include <memory>
#include "tiling/matmul_a16w16_tiling_data.h"
#include "utils/matmul_a16w16_constant.h"
#include "platform/platform_ascendc.h"

struct MatmulA16W16PlatformInfo {
    uint32_t aicNum{0};
    uint32_t aivNum{0};
    uint64_t ubSize{0};
    uint64_t l1Size{0};
    uint64_t l0aSize{0};
    uint64_t l0bSize{0};
    uint64_t l0cSize{0};
    uint64_t l2Size{0};
    uint64_t btSize{0};
    platform_ascendc::SocVersion socVersion{0};
};

struct MatmulA16W16Args {
    uint64_t m{0};
    uint64_t n{0};
    uint64_t k{0};
    bool hasBias{false};
    bool isATrans{false};
    bool isBTrans{false};
};

struct MatMulV3TailInfo {
    uint64_t mCnt = 1UL;
    uint64_t nCnt = 1UL;
    uint64_t kCnt = 1UL;
    uint64_t mTailMain = 0UL;
    uint64_t nTailMain = 0UL;
};

struct MatmulA16W16RunInfo {
    uint64_t baseM{1};
    uint64_t baseN{1};
    uint64_t baseK{1};
    uint64_t singleCoreM{1};
    uint64_t singleCoreN{1};
    uint64_t singleCoreK{1};
    uint32_t mBaseTailSplitCnt{1};
    uint32_t nBaseTailSplitCnt{1};
    uint64_t usedCoreNum{1};
    uint64_t depthA1{1};
    uint64_t depthB1{1};
    uint64_t stepKa{1};
    uint64_t stepKb{1};
    uint64_t stepM{1};
    uint64_t stepN{1};
    uint64_t iterateOrder{0};
    uint64_t dbL0c{0};
    uint64_t l1BufferNum{2};
    MatMulV3TailInfo tailInfo;
    double defaultBalance{0.0};
    uint64_t redundantData{0};
};

class MatmulA16W16TilingBase {
public:
    MatmulA16W16TilingBase() = default;
    virtual ~MatmulA16W16TilingBase() = default;

    virtual void GetTilingData(uint64_t m, uint64_t n, uint64_t k, MatmulA16W16TilingData& tilingData)
    {
        InitCompileInfo();
        InitShapeArgs(m, n, k);
        DoOpTiling(tilingData);
        PrintTilingData(tilingData);
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

    void InitShapeArgs(uint64_t m, uint64_t n, uint64_t k, bool hasBias = false)
    {
        args_.m = m;
        args_.n = n;
        args_.k = k;
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

#endif
