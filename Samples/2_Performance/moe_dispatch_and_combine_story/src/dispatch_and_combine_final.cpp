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
 * \file quant_matmul_mxfp4.cpp
 * \brief
 */
#include <iostream>
#include <cstdlib>
#include <memory>

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "kernel_operator.h"

#include "shmem.h"
#include "moe_distribute_dispatch.h"
#include "moe_distribute_combine.h"

inline int32_t TestSetAttr(int32_t myPe, int32_t nPes, uint64_t localMemSize, const char *ipPort, aclshmemx_uniqueid_t flagUid,
                       aclshmemx_init_attr_t *attributes)
{
    size_t ip_len = 0;
    if (ipPort != nullptr) {
        ip_len = std::min(strlen(ipPort), static_cast<size_t>(64-1);
        std::copy_n(ipPort, ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            return 1;
        }
    }
    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = myPe;
    attributes->n_pes = nPes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = localMemSize;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, 
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void *>(&flagUid);
    return 0;
}

/** 
    描述当前sample 实现与transformer仓下实现，参数做如下调整：
    GM_ADDR shmemSpace,             shmem 申请的单卡内存空间
    GM_ADDR expandX,                token输入
    GM_ADDR expertIds,              专家ID
    GM_ADDR assistInfoForCombine,   不需要
    GM_ADDR epSendCount,            不需要
    GM_ADDR scales,                 不需要
    GM_ADDR tpSendCount,            不需要
    GM_ADDR xActiveMask,            不需要
    GM_ADDR activationScale,        不需要
    GM_ADDR weightScale,            不需要
    GM_ADDR groupList,              不需要
    GM_ADDR expandScales,           不需要
    GM_ADDR sharedExpertX,          不需要
    GM_ADDR elasticInfo,            不需要
    GM_ADDR oriX,                   不需要
    GM_ADDR constExpertAlpha1,      不需要
    GM_ADDR constExpertAlpha2,      不需要
    GM_ADDR constExpertV,           不需要
    GM_ADDR XOut,                   输出
    GM_ADDR workspaceGM,            需要考虑下？
    GM_ADDR tilingGM                需要提前申请下
*/
__global__ __aicore__ void MoeDistributeDispatchKernel(
    GM_ADDR shmemSpace, GM_ADDR x, GM_ADDR expertIds, GM_ADDR expandXOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // 待补齐kernel实现
}



/**
    描述当前sample 实现与transformer仓下实现，参数做如下调整
    GM_ADDR shmemSpace,         shmem 申请的单卡内存空间，新增
    GM_ADDR x,                  token输入
    GM_ADDR expertIds,          专家ID
    GM_ADDR scales,             不需要
    GM_ADDR xActiveMask,        不需要
    GM_ADDR expertScales,       专家权重
    GM_ADDR elasticInfo,        不需要
    GM_ADDR expandXOut,         输出
    GM_ADDR dynamicScalesOut,   不需要
    GM_ADDR assistInfoOut,      不需要
    GM_ADDR expertTokenNumsOut, 不需要
    GM_ADDR epSendCountsOut,    不需要
    GM_ADDR tpSendCountsOut,    不需要
    GM_ADDR expandScalesOut,    不需要
    GM_ADDR workspaceGM,
    GM_ADDR tilingGM
*/
__global__ __aicore__ void MoeDistributeCombineKernel(
    GM_ADDR shmemSpace, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInForCombine, GM_ADDR epSendCount, GM_ADDR expertScales, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_INLING_DATA_WIHT_STRUCT(MoeDistributeCombineShmemTilingData, tilingData, tilingGM);
    TPipe pipe;
    MoeDistributeCombineShmem<float, float, int32_t> op();
    op.Init(shmemSpace, expandX, expertIds, assistInfoForCombine, epSendCount, expertScales, XOut, workspaceGM, &pipe, &tilingData);
    op.Process();

}

void SetDispatchTilingData(MoeDistributeDispatchTilingData& dispatchTilingData)
{
    // 待补齐tilingData数据填写
}

void SetCombineTilingData(MoeDistributeCombineShmemTilingData& combineTilingData)
{
    auto ascendPlatform = platform_ascendc::platformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    combineTilingData.epWorldSize = 2;
    combineTilingData.epRankId = 0;
    combineTilingData.moeExpertNum = 8;
    combineTilingData.moeExpertPerRankNum = 4;
    combineTilingData.globalBs = 16;
    combineTilingData.bs = 8;
    combineTilingData.k = 8;
    combineTilingData.h = 7168;
    combineTilingData.aivNum = aivNum;
    combineTilingData.totalUbSize = ubSize;
    combineTilingData.totalWinSize = 100 * 1024 * 1024;  //100MB
}

int main(int argc, char* argv[])
{
    int status = ACLSHMEM_SUCCESS;
    int rankNum = atoi(argv[1]);
    int rankId = atoi(argv[2]);
    std::string ipport = argv[3];

    // Acl && Shmem init
    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = atoi(argv[4]) + rankId;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    aclshmemx_uniqueid_t flagUid;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    TestSetAttr(rankId, rankNum, localMemSize, ipport.c_str(), flagUid, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    int32_t aclshmem_size = (504 * 1024 * 1024) * sizeof(__fp16);
    void *symmPtr = aclshmem_malloc(aclshmem_size);
    uint8_t *symmetricPtr = (uint8_t *) symmPtr;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    MoeDistributeDispatchTilingData dispatchTilingData;
    MoeDistributeCombineShmemTilingData combineTilingData;

    // 待补齐相应参数生成和传递
    for (int i = 0; i < 1; ++i) {
        MoeDistributeDispatchKernel<<<BLOCK_NUM, nullptr, stream>>>(symmetricPtr, dispatchTilingData);
        MoeDistributeCombineKernel<<<BLOCK_NUM, nullptr, stream>>>(symmetricPtr, combineTilingData);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    aclshmem_free(symmPtr);
    status = aclrtDestroyStream(stream);
    status = aclshmem_finalize();
    status = aclrtResetDevice(deviceId);
    status = aclFinalize();
    if (status) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] demo run success in relative_pe_id " << rankId << std::endl;
    return 0;
}
