/**
 * 单 VF 拆分版的共享头：统一 case 规格 + ACL 错误检查宏 + host 侧输入生成/比对（elemwise / reduce 两个用例的 16 个 .asc 共用）。
 *
 * - 统一规格 [D0,D1]=[78,250]，N=D0*D1=19500。按最受限算子 elemwise（x/y/z 三块 UB buffer，约
 *   234KB<256KB UB）反推尺寸；reduce 类只需一块输入 buffer。D1>VL 且 %8≠0、D0%4=2，保留分块/
 *   尾块/展开尾行路径。改规格即改此处常量。
 * - 输入生成（GenInput）统一放这里：固定种子、16 个 VF 共用同一份随机输入。各算子的「标杆 golden」
 *   因实现而异（max/sum、ar/ra），留在各自 .asc 的 main 里就地计算。
 * - 算子专属算法常量（AXPY 的 A、展开因子 UNROLL、Max 归约初值 NEG_INF）不属于「规格」，各自留在对应文件。
 */

#ifndef VF_COMMON_H
#define VF_COMMON_H

#include <cstdint>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include "acl/acl.h"

// ACL 调用错误检查（在 main 中使用，失败返回 1）
#define CHECK_ACL(call)                                                        \
    do {                                                                       \
        aclError err = (call);                                                 \
        if (err != ACL_SUCCESS) {                                              \
            std::cerr << "ACL error " << err << " at " << __LINE__ << "\n";    \
            return 1;                                                          \
        }                                                                      \
    } while (0)

// ===== 统一 case 规格 =====
static constexpr uint32_t VL_B32 = 256 / sizeof(float);    // 64 lane
static constexpr uint32_t UB_ALIGN = 32 / sizeof(float);   // 8：UB 行按 32B 对齐（reduce 类用）
static constexpr uint32_t D0 = 78;                         // 统一形状首维（%4=2）
static constexpr uint32_t D1 = 250;                        // 统一形状尾维（>VL 且 %8≠0）
static constexpr uint32_t N = D0 * D1;                     // 19500：elemwise 按一维向量处理
static constexpr int VF_REPEAT = 5;                        // 每个 VF 在 kernel 内连续跑的次数（profiling 取稳态）

namespace vf {

// 确定性随机输入（固定种子，16 个 VF 共用）；范围 [-1,1) 使 sum 良态
inline std::vector<float> GenInput(uint32_t n)
{
    std::vector<float> x(n);
    std::mt19937 gen(0u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (uint32_t i = 0; i < n; i++) x[i] = dist(gen);
    return x;
}

// 绝对容差比对（max / elemwise：精确）
inline bool VerifyAbs(const std::vector<float>& got, const std::vector<float>& ref, float tol)
{
    for (size_t k = 0; k < ref.size(); k++)
        if (std::fabs(got[k] - ref[k]) > tol) return false;
    return true;
}

// 相对容差比对（sum：浮点重排有低位差异）
inline bool VerifyRel(const std::vector<float>& got, const std::vector<float>& ref, double rtol)
{
    for (size_t k = 0; k < ref.size(); k++)
        if (std::fabs((double)got[k] - (double)ref[k]) > rtol * (std::fabs((double)ref[k]) + 1.0))
            return false;
    return true;
}

}  // namespace vf

#endif  // VF_COMMON_H
