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
 * \file bw_common.h
 * \brief 带宽测试样例（bw_rcw / bw_read / bw_rw ...）共用的宏与常量。
 */

#ifndef BW_COMMON_H
#define BW_COMMON_H

#include <cstdint>
#include <iostream>
#include "acl/acl.h"

#define CHECK_ACL(call)                                                                              \
    do {                                                                                             \
        aclError err = (call);                                                                       \
        if (err != ACL_SUCCESS) {                                                                    \
            std::cerr << "ACL error: " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return 1;                                                                                \
        }                                                                                            \
    } while (0)

constexpr int64_t DATA_BYTES = 56LL * 64 * 1024 * 1024;
// clang-format off

constexpr int32_t DEFAULT_BUFFER_NUM_LIST[] = {
    2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4,
    6, 6, 6, 6, 6,
    8, 8, 8, 8, 8
};

constexpr int64_t DEFAULT_QUE_BYTES_LIST[] = {
    1024, 2 * 1024, 4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024, 64 * 1024,
    1024, 2 * 1024, 4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024,
    1024, 2 * 1024, 4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024,
    1024, 2 * 1024, 4 * 1024, 8 * 1024, 16 * 1024,
    1024, 2 * 1024, 4 * 1024, 8 * 1024, 16 * 1024,
};
// clang-format on

#endif // BW_COMMON_H
