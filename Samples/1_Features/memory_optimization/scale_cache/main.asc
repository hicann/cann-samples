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
 * \file main.cpp
 * \brief Main implementation file for Ascend matrix multiplication kernel
 *        This file contains the kernel implementation and host-side setup code
 */

#include <iostream>
#include <vector>
#include <filesystem>
#include <sys/stat.h>
#include <fstream>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

#include "acl/acl.h"
#include "kernel_basic_intf.h"
#include "tiling/platform/platform_ascendc.h"
#include "include/tensor.h"

#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

using AscendC::Te::AttrInfo;
using AscendC::Te::C0_SIZE;
using AscendC::Te::GetCacheModeFromTensor;
using AscendC::Te::GetEleFromLayout;
using AscendC::Te::IsScaleANDFormat;
using AscendC::Te::IsScaleBNDFormat;
using AscendC::Te::IsZZFormat;
using AscendC::Te::MX_SCALE_K0;

namespace tool {
// Memory and buffer configuration constants
constexpr static uint64_t DOUBLE_BUFFER_COUNT = 2;                       // Double buffering for ping-pong operation
constexpr static int64_t L0A_SIZE = 64 * 1024;                           // L0A buffer size (64KB)
constexpr static int64_t TOTAL_L0C_SIZE = 256 * 1024;                    // Total L0C buffer size (256KB)
constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT; // Half L0A for ping-pong

constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr static uint64_t MXFP_GROUP_SIZE = 32UL;

// Synchronization flag values
constexpr static uint16_t ZERO_FLAG = 0;  // First flag value
constexpr static uint16_t FIRST_FLAG = 1; // Second flag value
constexpr uint16_t SCALE_BUFFER_FLAG_0 = 4;
constexpr uint16_t SCALE_BUFFER_FLAG_1 = 5;
constexpr uint16_t MTE1_MTE2_EVENT_ID_NUM = 6;

constexpr uint32_t FINAL_ACCUMULATION = 3;
constexpr uint32_t NON_FINAL_ACCUMULATION = 2;

// Helper function declarations for host-side operations
template <typename T>
void FillRandomData(std::vector<T>& data, T min, T max); // Fill vector with random data
float Bf16ToFloat(uint16_t h);
uint16_t FloatToBf16(float f);
__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b);
__aicore__ inline uint64_t CeilAlign(uint64_t a, uint64_t b); // Ceiling division
uint64_t CeilDivHost(uint64_t a, uint64_t b);
inline bool ReadFile(const std::string& filePath, size_t& fileSize, void* buffer, size_t bufferSize);
inline bool WriteFile(const std::string& filePath, const void* buffer, size_t size);
} // namespace tool

namespace Tile {
struct CopyScaleGM2L1Atom {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src)
    {
        if constexpr (IsZZFormat<T>::value) {
            if constexpr (IsScaleANDFormat<U>::value) {
                CopyScaleADn2nz<Tp, traits, T, U>(dst, src);
            } else {
                // The ND variant is only used by layouts that already expose
                // the expanded 64-element MXFP scale width on the GM side.
                CopyScaleANd2nz<Tp, traits, T, U>(dst, src);
            }
        } else {
            if constexpr (IsScaleBNDFormat<U>::value) {
                CopyScaleBNd2nz<Tp, traits, T, U>(dst, src);
            } else {
                CopyScaleBDn2nz<Tp, traits, T, U>(dst, src);
            }
        }
    }

    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void CopyScaleADn2nz(const T& dst, const U& src)
    {
        if ASCEND_IS_AIV {
            return;
        }
        using type = typename U::elementType;
        static_assert(AscendC::Std::is_same_v<type, __gm__ fp8_e8m0_t>, "The data type is not supported.");

        // scaleA in DN format stores one logical MX scale value for every
        // 32-K group, while the destination expects an NZ-style packed layout.
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout(); // shape: 1,m,1,k/32, stride: 0,m,0,1

        uint16_t nValue =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout) / MX_SCALE_K0;
        uint32_t dValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint64_t srcDValue =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout) / MX_SCALE_K0;
        uint16_t dstNzC0Stride =
            GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout) / C0_SIZE<T>;
        CopyGmToCbufDn2nz<Tp, traits, T, U>(dst, src, nValue, dValue, srcDValue, dstNzC0Stride);
    }

    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void CopyScaleANd2nz(const T& dst, const U& src)
    {
        if ASCEND_IS_AIV {
            return;
        }
        using type = typename U::elementType;
        static_assert(AscendC::Std::is_same_v<type, __gm__ fp8_e8m0_t>, "The data type is not supported.");

        // The ND flavor already exposes the expanded 64-K divisor width, so
        // only the source stride interpretation changes during repacking.
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout(); // shape: 1,m,2,k/64, stride: 0,2,1,2*m

        uint16_t nValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t dValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);

        uint64_t srcDValue =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / MX_SCALE_K0;
        uint16_t dstNzC0Stride =
            GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout) / C0_SIZE<T>;
        CopyGmToCbufNd2nz<Tp, traits, T, U>(dst, src, nValue, dValue, srcDValue, dstNzC0Stride);
    }

    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void CopyScaleBDn2nz(const T& dst, const U& src)
    {
        if ASCEND_IS_AIV {
            return;
        }
        using type = typename U::elementType;
        static_assert(AscendC::Std::is_same_v<type, __gm__ fp8_e8m0_t>, "The data type is not supported.");

        // scaleB flips the major axis relative to scaleA, so D/N extraction
        // mirrors the A-side helper even though the copy primitive is shared.
        auto dstLayout = dst.Layout(); // shape: 2,k/64,16,n/16, stride: 1,32,2,(k/32)*16
        auto srcLayout = src.Layout(); // shape: 1,k/32,1,n stride: 0,1,0,k/32

        uint16_t dValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t nValue =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout) / MX_SCALE_K0;
        uint64_t srcDValue =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / MX_SCALE_K0;
        uint16_t dstNzC0Stride =
            GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout) / C0_SIZE<T>;
        CopyGmToCbufDn2nz<Tp, traits, T, U>(dst, src, nValue, dValue, srcDValue, dstNzC0Stride);
    }

    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void CopyScaleBNd2nz(const T& dst, const U& src)
    {
        if ASCEND_IS_AIV {
            return;
        }
        using type = typename U::elementType;
        static_assert(AscendC::Std::is_same_v<type, __gm__ fp8_e8m0_t>, "The data type is not supported.");

        // This is the B-side counterpart of `CopyScaleANd2nz`, with row/column
        // semantics swapped to match the filter-major scale layout.
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout(); // shape: 2,k/64,1,n, stride: 1,2*n,0,2

        uint16_t nValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t dValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

        uint64_t srcDValue =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout) / MX_SCALE_K0;
        uint16_t dstNzC0Stride =
            GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout) / C0_SIZE<T>;
        CopyGmToCbufNd2nz<Tp, traits, T, U>(dst, src, nValue, dValue, srcDValue, dstNzC0Stride);
    }

    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void CopyGmToCbufDn2nz(
        const T& dst, const U& src, uint16_t nValue, uint32_t dValue, uint64_t srcDValue, uint16_t dstNzC0Stride)
    {
        uint16_t dnNum = 1;
        uint64_t srcDnMatrixStride = 0;
        uint16_t dstNzNStride = 1;
        uint32_t dstNzMatrixStride = 0;

        uint64_t loop1SrcStride = srcDValue * sizeof(half);
        uint64_t loop4SrcStride = srcDnMatrixStride * sizeof(half);

        uint16_t loop2DstStride = dstNzNStride;  // loop2_dst_stride = dst_nz_n_stride
        uint16_t loop3DstStride = dstNzC0Stride; // loop3_dst_stride = dst_nz_c0_Stride
        // loop4_dst_stride: dst_nz_matrix_stride * size_of_dst_type / C0_SIZE<T>
        uint16_t loop4DstStride = static_cast<uint16_t>(dstNzMatrixStride * sizeof(half) / C0_SIZE<T>);

        uint8_t cacheMode = GetCacheModeFromTensor(src);
        // The hardware DN2NZ DMA expects a packed register describing the
        // destination NZ strides. The helper derives those fields once here.
        uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
        mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
        mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
        mte2NzPara |= static_cast<uint64_t>(dnNum);                        // MTE2_NZ_PARA[15:0]
        set_mte2_nz_para(mte2NzPara); // CCE: store parameters for DN2NZ DMA instructions
        copy_gm_to_cbuf_multi_dn2nz(
            (__cbuf__ half*)dst.Data().Get(), (__gm__ half*)src.Data().Get(), 0, loop1SrcStride, cacheMode, nValue,
            dValue, loop4SrcStride, false);
    }

    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void CopyGmToCbufNd2nz(
        const T& dst, const U& src, uint16_t nValue, uint32_t dValue, uint64_t srcDValue, uint16_t dstNzC0Stride)
    {
        uint16_t ndNum = 1;
        uint64_t srcNdMatrixStride = 0;

        uint16_t dstNzNStride = 1;
        uint32_t dstNzMatrixStride = 0;

        uint64_t loop1SrcStride = srcDValue * sizeof(half);
        uint64_t loop4SrcStride = srcNdMatrixStride * sizeof(half);

        uint16_t loop2DstStride = dstNzNStride;  // loop2_dst_stride = dst_nz_n_stride
        uint16_t loop3DstStride = dstNzC0Stride; // loop3_dst_stride = dst_nz_c0_Stride
        // loop4_dst_stride: dst_nz_matrix_stride * size_of_dst_type / C0_SIZE<T>
        uint16_t loop4DstStride = static_cast<uint16_t>(dstNzMatrixStride * sizeof(half) / C0_SIZE<T>);

        uint8_t cacheMode = GetCacheModeFromTensor(src);
        // The ND2NZ path uses the same register layout as DN2NZ; only the
        // source indexing logic differs in how ND tiles are walked in GM.
        uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
        mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
        mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
        mte2NzPara |= static_cast<uint64_t>(ndNum);                        // MTE2_NZ_PARA[15:0]
        set_mte2_nz_para(mte2NzPara); // CCE: store parameters for ND2NZ DMA instructions
        copy_gm_to_cbuf_multi_nd2nz(
            (__cbuf__ half*)dst.Data().Get(), (__gm__ half*)src.Data().Get(), 0, loop1SrcStride, cacheMode, nValue,
            dValue, loop4SrcStride, false);
    }
};
struct CopyL12L0MxScaleA3510Atom {
    template <typename Tp, const Tp& traits, typename T, typename U, class Coord>
    __aicore__ inline static void Copy(const T& dst, const U& src, const Coord& coord)
    {
        if ASCEND_IS_AIV {
            return;
        }
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
        static_assert(
            AscendC::Std::is_one_of_v<
                AscendC::Std::tuple<dstType, srcType>, AscendC::Std::tuple<__ca__ fp8_e8m0_t, __cbuf__ fp8_e8m0_t>>,
            "The data type is not supported.");
        // `coord` is expressed in the original M/K element space; the helper
        // converts it to the packed MX scale coordinates expected by the L0A
        // scale layout and issues one hardware MX load.
        // (m1, k/64, m0, 2)
        // shape ((m0, m1), (2, k/64))
        // stride ((2, k/64*m0*2), (1, m0*2))
        // Zz -> Zz
        uint16_t mStartPosition = tool::CeilDiv(AscendC::Std::get<0>(coord), AscendC::BLOCK_CUBE);
        uint16_t kStartPosition = tool::CeilDiv(AscendC::Std::get<1>(coord), tool::MXFP_DIVISOR_SIZE);
        auto mStep = AscendC::Std::get<1>(AscendC::Std::get<0>(dst.Layout().Shape()));
        auto kStep = AscendC::Std::get<1>(AscendC::Std::get<1>(dst.Layout().Shape()));
        auto srcStride = AscendC::Std::get<1>(AscendC::Std::get<0>(src.Layout().Stride())) >> 5;
        auto dstStride = kStep;
        // The intrinsic takes a 16-byte unit address, hence the right shift.
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst.Data().Get())) >> 4;
        load_cbuf_to_ca_mx(
            mxDstAddr, static_cast<__cbuf__ void*>(src.Data().Get()), mStartPosition, kStartPosition, mStep, kStep,
            srcStride, dstStride);
    }
};

struct CopyL12L0MxScaleB3510Atom {
    template <typename Tp, const Tp& traits, typename T, typename U, class Coord>
    __aicore__ inline static void Copy(const T& dst, const U& src, const Coord& coord)
    {
        if ASCEND_IS_AIV {
            return;
        }
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
        // `coord` is expressed in the original K/N element space; the helper
        // converts it to the packed MX scale coordinates expected by the L0B
        // scale layout and issues one hardware MX load.
        // (n1, k/64, n0, 2)
        // shape ((2, k/64), (n0, n1))
        // stride ((2, k/64*n0*2), (1, n0*2))
        // Nn -> Nn
        uint16_t nStartPosition = tool::CeilDiv(AscendC::Std::get<1>(coord), AscendC::BLOCK_CUBE);
        uint16_t kStartPosition = tool::CeilDiv(AscendC::Std::get<0>(coord), tool::MXFP_DIVISOR_SIZE);
        auto nStep = AscendC::Std::get<1>(AscendC::Std::get<1>(dst.Layout().Shape()));
        auto kStep = AscendC::Std::get<1>(AscendC::Std::get<0>(dst.Layout().Shape()));
        auto srcStride = AscendC::Std::get<1>(AscendC::Std::get<1>(src.Layout().Stride())) >> 5;
        auto dstStride = kStep;
        // The intrinsic takes a 16-byte unit address, hence the right shift.
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst.Data().Get())) >> 4;
        load_cbuf_to_cb_mx(
            mxDstAddr, static_cast<__cbuf__ void*>(src.Data().Get()), nStartPosition, kStartPosition, nStep, kStep,
            srcStride, dstStride);
    }
};
struct MmadMx {
    template <typename Tp, const Tp& traits, typename T, typename U, typename S>
    __aicore__ inline static void Mad(
        const T& dst, const U& fm, const S& filter, uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlagCtrl,
        bool btBuffCtrl, bool initCMatrixCtrl)
    {
        if ASCEND_IS_AIV {
            return;
        }
        // Forward the generic TE MMAD request to the MX-specific hardware
        // intrinsic used by quantized MXFP4 matmul.
        mad_mx(
            dst.Data().Get(), fm.Data().Get(), filter.Data().Get(), m, k, n, unitFlagCtrl, true, btBuffCtrl,
            initCMatrixCtrl);
    }
};

} // namespace Tile
namespace AscendC::Te {
template <typename Opration, typename TraitStruct>
struct MmadTraits<Opration, TraitStruct> {
    using TraitType = typename TraitStruct::TraitType;
    static constexpr const TraitType defaultTrait = TraitStruct::value;

    template <const TraitType& trait = defaultTrait, typename... Args>
    __aicore__ inline void MmadUnpack(const Args&... args) const
    {
        // Store the scalar MMAD parameters in the trait object once, then
        // append them automatically to every unpacked operator invocation.
        Opration::template Mad<TraitType, trait, Args...>(args..., m, k, n, unitFlagCtrl, btBuffCtrl, initCMatrixCtrl);
    }

    uint16_t m = 0;
    uint16_t k = 0;
    uint16_t n = 0;
    uint8_t unitFlagCtrl = 0;
    bool btBuffCtrl = false;
    bool initCMatrixCtrl = false;
};

template <>
struct AscendC::Te::CopyTraits<::Tile::CopyScaleGM2L1Atom>
    : public CopyTraits<
          ::Tile::CopyScaleGM2L1Atom, AscendC::Te::LoadDataTraitDefault, ::Tile::CopyScaleGM2L1Atom,
          AscendC::Te::LoadDataTraitDefault> {};
template <>
struct AscendC::Te::CopyTraits<::Tile::CopyL12L0MxScaleA3510Atom>
    : public CopyTraits<
          ::Tile::CopyL12L0MxScaleA3510Atom, LoadDataTraitDefault, ::Tile::CopyL12L0MxScaleA3510Atom,
          LoadDataTraitDefault> {};

template <>
struct AscendC::Te::CopyTraits<::Tile::CopyL12L0MxScaleB3510Atom>
    : public CopyTraits<
          ::Tile::CopyL12L0MxScaleB3510Atom, LoadDataTraitDefault, ::Tile::CopyL12L0MxScaleB3510Atom,
          LoadDataTraitDefault> {};

template <>
struct MmadTraits<::Tile::MmadMx>
    : public MmadTraits<::Tile::MmadMx, MmadTraitDefault, ::Tile::MmadMx, MmadTraitDefault> {};

// Layout definitions for matrices A and B (NZ format by default, can be transposed to ZN)
static constexpr bool transA = false;
static constexpr bool transB = true;
using MakeLayoutAL1 = AscendC::Std::conditional_t<
    transA, AscendC::Te::ZnLayoutFormat<fp4x2_e2m1_t>, AscendC::Te::NzLayoutFormat<fp4x2_e2m1_t>>;
using MakeLayoutBL1 = AscendC::Std::conditional_t<
    transB, AscendC::Te::ZnLayoutFormat<fp4x2_e2m1_t>, AscendC::Te::NzLayoutFormat<fp4x2_e2m1_t>>;

using MakeLayoutA = AscendC::Te::NDLayoutFormat<fp4x2_e2m1_t>;
using MakeLayoutB = AscendC::Te::DNLayoutFormat<fp4x2_e2m1_t>;
using MakeLayoutScaleA = AscendC::Te::ScaleANDLayoutFormat<fp8_e8m0_t>;
using MakeLayoutScaleB = AscendC::Te::ScaleBDNLayoutFormat<fp8_e8m0_t>;
} // namespace AscendC::Te

namespace matmul {
/**
 * @brief Matrix multiplication kernel for Ascend AI processor
 *
 * This kernel implements C = A * B using optimized memory hierarchy:
 * - Double buffering between GM -> L1 and L1 -> L0
 * - Tiled computation to fit in on-chip memory
 * - Multi-core parallelization
 *
 * @tparam T Data type (float in this implementation)
 * @param aGm Global memory pointer to matrix A (size m*k)
 * @param bGm Global memory pointer to matrix B (size k*n)
 * @param cGm Global memory pointer to output matrix C (size m*n)
 * @param m Rows of A and C
 * @param k Columns of A, rows of B
 * @param n Columns of B and C
 */
template <typename T>
__global__ __aicore__ void MatmulKernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR scaleAGm, GM_ADDR scaleBGm, GM_ADDR cGm, uint32_t m, uint32_t k, uint32_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    __gm__ fp4x2_e2m1_t* aGmAddr = reinterpret_cast<__gm__ fp4x2_e2m1_t*>(aGm);
    __gm__ fp4x2_e2m1_t* bGmAddr = reinterpret_cast<__gm__ fp4x2_e2m1_t*>(bGm);
    __gm__ T* cGmAddr = reinterpret_cast<__gm__ T*>(cGm);
    __gm__ fp8_e8m0_t* scaleAGmAddr = reinterpret_cast<__gm__ fp8_e8m0_t*>(scaleAGm);
    __gm__ fp8_e8m0_t* scaleBGmAddr = reinterpret_cast<__gm__ fp8_e8m0_t*>(scaleBGm);

    // Initialize tiling parameters for memory hierarchy
    uint64_t baseM = 128;
    uint64_t baseN = 128;
    uint64_t baseK = 256;
    uint64_t kL1 = 1024;
    uint64_t scaleKL1 = 8192;
    uint64_t scaleKL1Ratio = scaleKL1 / kL1;
    uint64_t mTileNum = tool::CeilDiv(m, baseM);
    uint64_t nTileNum = tool::CeilDiv(n, baseN);
    uint64_t tileNum = mTileNum * nTileNum;
    uint64_t kL1TileNum = tool::CeilDiv(k, kL1);
    uint64_t tailKL1 = k - (kL1TileNum - 1) * kL1;
    uint64_t tailBaseM = m - (mTileNum - 1) * baseM;
    uint64_t tailBaseN = n - (nTileNum - 1) * baseN;
    uint64_t l0cOffset = 0; 

    uint64_t curBlockIdx = AscendC::GetBlockIdx();
    uint64_t blockNum = AscendC::GetBlockNum();

    uint64_t l0PingPong = 0;
    uint64_t l1PingPong = 0;
    uint64_t scaleLoopCnt = 0;
    uint64_t l1BufferAOffset[2] = {0UL};
    uint64_t l1BufferBOffset[2] = {0UL};
    uint64_t l1BufferScaleAOffset[2] = {0UL};
    uint64_t l1BufferScaleBOffset[2] = {0UL};

    auto layoutA = AscendC::Te::MakeLayoutA{}(m, k);
    auto layoutB = AscendC::Te::MakeLayoutB{}(k, n);
    auto layoutScaleA =
        AscendC::Te::MakeLayoutScaleA{}(m, tool::CeilDiv(k, tool::MXFP_DIVISOR_SIZE) * tool::MXFP_MULTI_BASE_SIZE);
    auto layoutScaleB =
        AscendC::Te::MakeLayoutScaleB{}(tool::CeilDiv(k, tool::MXFP_DIVISOR_SIZE) * tool::MXFP_MULTI_BASE_SIZE, n);
    auto layoutC = AscendC::Te::MakeNDLayout<T>(m, n);

    auto tensorAgm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(aGmAddr), layoutA);
    auto tensorBgm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(bGmAddr), layoutB);
    auto ScaleAgm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(scaleAGmAddr), layoutScaleA);
    auto ScaleBgm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(scaleBGmAddr), layoutScaleB);
    auto tensorCgm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(cGmAddr), layoutC);

    // Initialize hardware event flags for synchronization
    for (uint8_t i = 0; i < tool::MTE1_MTE2_EVENT_ID_NUM; ++i) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
    }
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(tool::ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(tool::FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(tool::ZERO_FLAG);

    for (uint64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
        uint64_t mTileIdx = tileIdx / nTileNum;
        uint64_t nTileIdx = tileIdx % nTileNum;
        int64_t curM = mTileIdx == (mTileNum - 1) ? tailBaseM : baseM;
        int64_t curN = nTileIdx == (nTileNum - 1) ? tailBaseN : baseN;

        auto tensorAGmBlock = tensorAgm(AscendC::Te::MakeCoord(mTileIdx * baseM, 0L), AscendC::Te::MakeShape(curM, k));
        auto tensorBGmBlock = tensorBgm(AscendC::Te::MakeCoord(0L, nTileIdx * baseN), AscendC::Te::MakeShape(k, curN));
        // Define the source tile for Scale-A in global memory (GM)
        // Coordinates: (mTileIdx * baseM, 0) - starting from the current M tile offset, K dimension starts at 0
        // Shape: (curM, total_K_blocks) - each block represents a group of MXFP elements
        auto ScaleAgmBlock = ScaleAgm(
            AscendC::Te::MakeCoord(mTileIdx * baseM, 0L),
            AscendC::Te::MakeShape(curM, tool::CeilDiv(k, tool::MXFP_DIVISOR_SIZE) * tool::MXFP_MULTI_BASE_SIZE));
        auto ScaleBgmBlock = ScaleBgm(
            AscendC::Te::MakeCoord(0L, nTileIdx * baseN),
            AscendC::Te::MakeShape(tool::CeilDiv(k, tool::MXFP_DIVISOR_SIZE) * tool::MXFP_MULTI_BASE_SIZE, curN));
        auto tensorCGmBlock =
            tensorCgm(AscendC::Te::MakeCoord(mTileIdx * baseM, nTileIdx * baseN), AscendC::Te::MakeShape(curM, curN));

        auto layoutL0C = AscendC::Te::MakeL0CLayout(curM, curN);
        auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeL0CmemPtr<float>(l0cOffset), layoutL0C);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(tool::ZERO_FLAG);
        for (uint64_t iter0 = 0; iter0 < kL1TileNum; ++iter0) {
            uint64_t l1BufId = l1PingPong & 1;
            uint64_t scaleL1BufId = scaleLoopCnt & 1;

            uint64_t AOffsetL1 = (baseM * kL1) >> 1;
            uint64_t BOffsetL1 = (baseN * kL1) >> 1;
            // Calculate L1 offset for Scale-A based on the base M dimension
            // The offset accounts for the scaled K dimension grouped into MXFP blocks,
            // multiplied by the base size of each MXFP block and the data type size.
            uint64_t scaleAL1Offset = baseM * tool::CeilDiv(scaleKL1, tool::MXFP_DIVISOR_SIZE) *
                                      tool::MXFP_MULTI_BASE_SIZE * sizeof(fp8_e8m0_t);
            // Calculate L1 offset for Scale-B based on the base N dimension
            // Similar to Scale-A, but indexed by N dimension instead of M
            uint64_t scaleBL1Offset = baseN * tool::CeilDiv(scaleKL1, tool::MXFP_DIVISOR_SIZE) *
                                      tool::MXFP_MULTI_BASE_SIZE * sizeof(fp8_e8m0_t);
            l1BufferAOffset[l1BufId] = l1BufId * AOffsetL1;
            l1BufferBOffset[l1BufId] = tool::DOUBLE_BUFFER_COUNT * AOffsetL1 + l1BufId * BOffsetL1;

            l1BufferScaleAOffset[scaleL1BufId] =
                tool::DOUBLE_BUFFER_COUNT * (AOffsetL1 + BOffsetL1) + scaleL1BufId * scaleAL1Offset;
            l1BufferScaleBOffset[scaleL1BufId] =
                tool::DOUBLE_BUFFER_COUNT * (AOffsetL1 + BOffsetL1 + scaleAL1Offset) + scaleL1BufId * scaleBL1Offset;

            // Conditional execution based on the iteration count and scaling factor ratio
            if (iter0 % scaleKL1Ratio == 0) {
                // Calculate the offset in the K dimension for the scaling factor
                uint64_t scaleKL1Offset = iter0 * kL1;

                // Wait for the MTE1/MTE2 hardware event corresponding to the current L1 buffer
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(tool::SCALE_BUFFER_FLAG_0 + scaleL1BufId);

                // Create a copy operation atom for moving scale factors from GM to L1
                auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(::Tile::CopyScaleGM2L1Atom{});

                // Determine the actual number of K elements to copy for this iteration, clamping to remaining elements
                uint64_t curScaleKL1 = scaleKL1;
                if (scaleKL1Offset + curScaleKL1 > k) {
                    curScaleKL1 = k - scaleKL1Offset;
                }

                // --- Handle Scale-A (LHS) ---
                // Define the layout for scale-A in L1 (ZZ layout)
                auto layoutScaleAL1 =
                    AscendC::Te::MakeZzLayout<fp8_e8m0_t>(curM, tool::CeilDiv(scaleKL1, tool::MXFP_GROUP_SIZE));
                // Create a tensor for scale-A in L1 buffer
                auto tensorScaleAL1Buf = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(l1BufferScaleAOffset[scaleL1BufId]), layoutScaleAL1);
                // Define the source tile for scale-A in global memory (GM)
                auto tensorScaleAGmTile = ScaleAgmBlock(
                    AscendC::Te::MakeCoord(0, scaleKL1Offset / tool::MXFP_GROUP_SIZE),
                    AscendC::Te::MakeShape(
                        curM, tool::CeilDiv(curScaleKL1, tool::MXFP_DIVISOR_SIZE) * tool::MXFP_MULTI_BASE_SIZE));
                // Execute the copy: GM -> L1 for scale-A
                AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleAL1Buf, tensorScaleAGmTile);

                // --- Handle Scale-B (RHS) ---
                // Define the layout for scale-B in L1 (NN layout)
                auto layoutScaleBL1 =
                    AscendC::Te::MakeNnLayout<fp8_e8m0_t>(tool::CeilDiv(scaleKL1, tool::MXFP_GROUP_SIZE), curN);
                // Create a tensor for scale-B in L1 buffer
                auto tensorScaleBL1Buf = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(l1BufferScaleBOffset[scaleL1BufId]), layoutScaleBL1);
                // Define the source tile for scale-B in global memory (GM)
                auto tensorScaleBGmTile = ScaleBgmBlock(
                    AscendC::Te::MakeCoord(scaleKL1Offset / tool::MXFP_GROUP_SIZE, 0),
                    AscendC::Te::MakeShape(
                        tool::CeilDiv(curScaleKL1, tool::MXFP_DIVISOR_SIZE) * tool::MXFP_MULTI_BASE_SIZE, curN));
                // Execute the copy: GM -> L1 for scale-B
                AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleBL1Buf, tensorScaleBGmTile);
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);

            auto curGmBKL1 = (iter0 + 1 == kL1TileNum) ? (k - iter0 * kL1) : kL1;
            auto curGmAKL1 = curGmBKL1;
            uint64_t curPadKL1 = tool::CeilAlign(curGmBKL1, tool::MXFP_DIVISOR_SIZE);

            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            auto layoutAL1 = AscendC::Te::MakeLayoutAL1{}(curM, curGmAKL1);
            auto tensorAL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<fp4x2_e2m1_t>(l1BufferAOffset[l1BufId]), layoutAL1);

            auto tensorAGmTile =
                tensorAGmBlock(AscendC::Te::MakeCoord(0, iter0 * kL1), AscendC::Te::MakeShape(curM, curGmAKL1));
            AscendC::Te::Copy(copyGM2L1, tensorAL1, tensorAGmTile);

            auto layoutBL1 = AscendC::Te::MakeLayoutBL1{}(curGmBKL1, curN);
            auto tensorBL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<fp4x2_e2m1_t>(l1BufferBOffset[l1BufId]), layoutBL1);

            auto tensorBGmTile =
                tensorBGmBlock(AscendC::Te::MakeCoord(iter0 * kL1, 0), AscendC::Te::MakeShape(curGmBKL1, curN));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, tensorBGmTile);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            uint64_t scaleKL1IterOffset = (iter0 % scaleKL1Ratio) * kL1;

            auto layoutScaleAL1ForL0 = AscendC::Te::MakeZzLayout<fp8_e8m0_t>(
                curM, tool::CeilDiv(scaleKL1, tool::MXFP_DIVISOR_SIZE) * tool::MXFP_MULTI_BASE_SIZE);
            auto tensorBlockScaleAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(l1BufferScaleAOffset[scaleL1BufId]), layoutScaleAL1ForL0);

            auto layoutScaleBL1ForL0 = AscendC::Te::MakeNnLayout<fp8_e8m0_t>(
                tool::CeilDiv(scaleKL1, tool::MXFP_DIVISOR_SIZE) * tool::MXFP_MULTI_BASE_SIZE, curN);
            auto tensorBlockScaleBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(l1BufferScaleBOffset[scaleL1BufId]), layoutScaleBL1ForL0);

            uint64_t kL0IterNum = tool::CeilDiv(curGmBKL1, baseK);

            for (uint16_t iter1 = 0; iter1 < kL0IterNum; ++iter1) {
                uint64_t kL0Offset = iter1 * baseK;
                uint64_t l0BufId = l0PingPong & 1;                
                uint64_t l0Offset = tool::HALF_L0_SIZE * l0BufId; 
                uint64_t curKL0 = (kL0Offset + baseK > curPadKL1) ? (curPadKL1 - kL0Offset) : baseK;
                uint64_t curScaleKL0 = tool::CeilDiv(curKL0, tool::MXFP_DIVISOR_SIZE) * tool::MXFP_MULTI_BASE_SIZE;

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);

                auto copyL12L0 = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0{});
                auto layoutAL0 = AscendC::Te::MakeNzLayout<fp4x2_e2m1_t>(curM, curKL0);
                auto tensorAL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<fp4x2_e2m1_t>(l0Offset), layoutAL0);
                auto tensorAL1Tile =
                    tensorAL1(AscendC::Te::MakeCoord(0, iter1 * baseK), AscendC::Te::MakeShape(curM, curKL0));
                AscendC::Te::Copy(copyL12L0, tensorAL0, tensorAL1Tile);

                auto layoutScaleAL0 = AscendC::Te::MakeZzLayout<fp8_e8m0_t>(curM, curScaleKL0);
                auto tensorScaleAL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<fp8_e8m0_t>(l0Offset), layoutScaleAL0);
                auto CopyL12L0MxScaleA = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleA3510Atom{});
                AscendC::Te::Copy(
                    CopyL12L0MxScaleA, tensorScaleAL0, tensorBlockScaleAL1,
                    AscendC::Te::MakeCoord(0, scaleKL1IterOffset + kL0Offset));

                auto layoutBL0 = AscendC::Te::MakeZnLayout<fp4x2_e2m1_t>(curKL0, curN);
                auto tensorBL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr<fp4x2_e2m1_t>(l0Offset), layoutBL0);
                auto tensorBL1Tile =
                    tensorBL1(AscendC::Te::MakeCoord(iter1 * baseK, 0), AscendC::Te::MakeShape(curKL0, curN));
                AscendC::Te::Copy(copyL12L0, tensorBL0, tensorBL1Tile);

                auto layoutScaleBL0 = AscendC::Te::MakeNnLayout<fp8_e8m0_t>(curScaleKL0, curN);
                auto tensorScaleBL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr<fp8_e8m0_t>(l0Offset), layoutScaleBL0);
                auto CopyL12L0MxScaleB = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleB3510Atom{});
                AscendC::Te::Copy(
                    CopyL12L0MxScaleB, tensorScaleBL0, tensorBlockScaleBL1,
                    AscendC::Te::MakeCoord(scaleKL1IterOffset + kL0Offset, 0));

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);

                uint8_t mmadUnitFlag = (iter0 + 1 == kL1TileNum && iter1 + 1 == kL0IterNum) ?
                                           tool::FINAL_ACCUMULATION :
                                           tool::NON_FINAL_ACCUMULATION;
                bool mmadCmatrixInitVal = (iter0 == 0 && iter1 == 0);
                AscendC::Te::Mad(
                    AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<::Tile::MmadMx>>{}.with(
                        static_cast<uint16_t>(curM),
                        static_cast<uint16_t>(tool::CeilAlign(curKL0, tool::MXFP_DIVISOR_SIZE)),
                        static_cast<uint16_t>(curN), mmadUnitFlag, false, mmadCmatrixInitVal),
                    tensorL0C, tensorAL0, tensorBL0);

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
                l0PingPong++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            l1PingPong++;

            if ((iter0 + 1) % scaleKL1Ratio == 0 || iter0 == kL1TileNum - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(tool::SCALE_BUFFER_FLAG_0 + scaleL1BufId);
                scaleLoopCnt++;
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(tool::ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(tool::ZERO_FLAG);

        auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        AscendC::Te::Copy(CopyL0C2GM, tensorCGmBlock, tensorL0C, AscendC::Te::FixpipeParams{tool::FINAL_ACCUMULATION});

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(tool::ZERO_FLAG);
    }

    // Final synchronization waits
    for (uint8_t i = 0; i < tool::MTE1_MTE2_EVENT_ID_NUM; ++i) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
    }
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(tool::ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(tool::FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(tool::ZERO_FLAG);
}

} // namespace matmul

// Utility macro for condition checking with error message
#define CHECK_COND(cond, message, return_expr)              \
    do {                                                    \
        if (!(cond)) {                                      \
            std::cerr << "ERROR: " << message << std::endl; \
            return_expr;                                    \
        }                                                   \
    } while (0)

// Print command-line usage help
void printUsage(const std::string& programName)
{
    std::cerr << "Usage: " << programName << " m k n" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A" << std::endl;
    std::cerr << "  k: col of matrix A" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "Example: " << programName << " 100 50 200" << std::endl;
}

// Brief parses and validates command-line arguments
void parseArguments(int argc, char* argv[], int& m, int& k, int& n)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        printUsage(argv[0]);
        exit(1);
    }
    if (argc < 4) {
        throw std::invalid_argument("ERROR: Lacks Arguments");
    }
    try {
        m = std::stoi(argv[1]);
        k = std::stoi(argv[2]);
        n = std::stoi(argv[3]);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument("ERROR: m k n must be Integer");
    }

    if (m <= 0 || k <= 0 || n <= 0) {
        throw std::invalid_argument("ERROR: m k n must be positive");
    }
}

/**
 * @brief Main function - host-side setup and execution
 *
 * This function:
 * 1. Parses command line arguments
 * 2. Initializes Ascend Computing Language (ACL) resources
 * 3. Allocates and initializes host/device memory
 * 4. Launches the kernel
 * 5. Verifies results against CPU reference
 * 6. Cleans up resources
 */
int main(int argc, char* argv[])
{
    using namespace tool;

    int m, k, n;
    try {
        parseArguments(argc, argv, m, k, n);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Initialize ACL resources
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclInit(nullptr);
    CHECK_COND(ret == ACL_SUCCESS, "aclInit failed.", return 1);
    ret = aclrtSetDevice(deviceId);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtSetDevice failed.", return 1);
    ret = aclrtCreateStream(&stream);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtCreateStream failed.", return 1);

    // Allocate host memory and fill with random data
    std::vector<uint8_t> hostA((m * k + 1) >> 1, 0);
    std::vector<uint8_t> hostB((k * n + 1) >> 1, 0);
    std::vector<uint8_t> hostScaleA(m * CeilDivHost(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, 0);
    std::vector<uint8_t> hostScaleB(n * CeilDivHost(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, 0);
    std::vector<half> hostOutput(m * n, 0);
    auto sizeA = static_cast<size_t>(1) * hostA.size() * sizeof(uint8_t);
    auto sizeB = static_cast<size_t>(1) * hostB.size() * sizeof(uint8_t);
    auto sizeScaleA = static_cast<size_t>(1) * hostScaleA.size() * sizeof(uint8_t);
    auto sizeScaleB = static_cast<size_t>(1) * hostScaleB.size() * sizeof(uint8_t);
    auto sizeOutput = static_cast<size_t>(1) * hostOutput.size() * sizeof(half);

    std::string cmd = "python3 gen_data.py " + std::to_string(m) + " " + std::to_string(k) + " " + std::to_string(n);
    system(cmd.c_str());

    std::string baseDir = std::filesystem::current_path();
    std::string inputDir = baseDir + "/input";
    std::string outputDir = baseDir + "/output";
    ReadFile(inputDir + "/input_a.bin", sizeA, hostA.data(), sizeA);
    ReadFile(inputDir + "/input_b.bin", sizeB, hostB.data(), sizeB);
    ReadFile(inputDir + "/input_scaleA.bin", sizeScaleA, hostScaleA.data(), sizeScaleA);
    ReadFile(inputDir + "/input_scaleB.bin", sizeScaleB, hostScaleB.data(), sizeScaleB);

    // Allocate device memory
    GM_ADDR deviceA = nullptr;
    GM_ADDR deviceB = nullptr;
    GM_ADDR deviceScaleA = nullptr;
    GM_ADDR deviceScaleB = nullptr;
    GM_ADDR deviceOutput = nullptr;
    ret = aclrtMalloc((void**)&deviceA, sizeA, ACL_MEM_MALLOC_HUGE_ONLY);
    std::unique_ptr<void, aclError (*)(void*)> DeviceAAddr(deviceA, aclrtFree);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceA failed.", return 1);
    ret = aclrtMalloc((void**)&deviceB, sizeB, ACL_MEM_MALLOC_HUGE_ONLY);
    std::unique_ptr<void, aclError (*)(void*)> DeviceBAddr(deviceB, aclrtFree);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceB failed.", return 1);
    ret = aclrtMalloc((void**)&deviceScaleA, sizeScaleA, ACL_MEM_MALLOC_HUGE_ONLY);
    std::unique_ptr<void, aclError (*)(void*)> DeviceScaleAAddr(deviceScaleA, aclrtFree);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceScaleA failed.", return 1);
    ret = aclrtMalloc((void**)&deviceScaleB, sizeScaleB, ACL_MEM_MALLOC_HUGE_ONLY);
    std::unique_ptr<void, aclError (*)(void*)> DeviceScaleBAddr(deviceScaleB, aclrtFree);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceScaleB failed.", return 1);
    ret = aclrtMalloc((void**)&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_ONLY);
    std::unique_ptr<void, aclError (*)(void*)> DeviceOutputAddr(deviceOutput, aclrtFree);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceOutput failed.", return 1);

    ret = aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceA failed.", return 1);
    ret = aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceB failed.", return 1);
    ret = aclrtMemcpy(deviceScaleA, sizeScaleA, hostScaleA.data(), sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceScaleA failed.", return 1);
    ret = aclrtMemcpy(deviceScaleB, sizeScaleB, hostScaleB.data(), sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceScaleB failed.", return 1);

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    CHECK_COND(ascendcPlatform != nullptr, "get ascendcPlatform failed.", return 1);
    uint32_t numBlocks = ascendcPlatform->GetCoreNumAic();

    // Launch kernel on all available AI cores
    matmul::MatmulKernel<bfloat16_t>
        <<<numBlocks, nullptr, stream>>>(deviceA, deviceB, deviceScaleA, deviceScaleB, deviceOutput, m, k, n);

    ret = aclrtSynchronizeStream(stream);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed.", return 1);

    ret = aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceOutput failed.", return 1);

    WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), sizeOutput);

    cmd = "python3 verify_result.py " + std::to_string(m) + " " + std::to_string(n);
    if (std::system(cmd.c_str()) != 0) {
        return 1;
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

namespace tool {
/**
 * @brief Ceiling division for integer arithmetic
 */
__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t CeilAlign(uint64_t a, uint64_t b)
{
    return CeilDiv(a, b) * b;
}

/**
 * @brief Ceiling division for integer arithmetic in host
 */
uint64_t CeilDivHost(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

/**
 * @brief Convert a 16-bit brain floating-point (bfloat16) value to a 32-bit float
 * @param h 16-bit bfloat16 value stored in uint16_t format
 * @return The converted 32-bit floating-point value
 */
float Bf16ToFloat(uint16_t h)
{
    uint32_t sign = (h & 0x8000U) ? 0x80000000U : 0x00000000U;
    uint32_t exponent = (h >> 7) & 0x00FFU;
    uint32_t mantissa = h & 0x007FU;
    uint32_t f_bits = sign | (exponent << 23) | (mantissa << (23 - 7));
    return *reinterpret_cast<float*>(&f_bits);
}

/**
 * @brief Convert a 32-bit float to a 16-bit brain floating-point (bfloat16) value
 * @param f 32-bit floating-point value to convert
 * @return The converted 16-bit bfloat16 value stored in uint16_t format (truncated rounding)
 */
uint16_t FloatToBf16(float f)
{
    uint32_t f_bits;
    std::memcpy(&f_bits, &f, sizeof(f_bits));

    // Extract the high 16 bits (simple truncation)
    return static_cast<uint16_t>(f_bits >> 16);
}

inline bool ReadFile(const std::string& filePath, size_t& fileSize, void* buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file");
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf* buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char*>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

/**
 * @brief Write data to file
 * @param [in] filePath: file path
 * @param [in] buffer: data to write to file
 * @param [in] size: size to write
 * @return write result
 */
inline bool WriteFile(const std::string& filePath, const void* buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    size_t writeSize = write(fd, buffer, size);
    (void)close(fd);
    if (writeSize != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}

} // namespace tool