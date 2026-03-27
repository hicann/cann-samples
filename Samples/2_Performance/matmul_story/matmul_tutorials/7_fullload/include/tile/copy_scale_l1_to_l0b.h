#ifndef MATMUL_TILE_DATAMOVE_COPY_L1_TO_L0B_H
#define MATMUL_TILE_DATAMOVE_COPY_L1_TO_L0B_H
#include "impl/atom/cube_datamove/copy_l12l0.h"
#include "kernel_utils/common_utils.h"

namespace Tile {
struct CopyL12L0MxScaleB3510 {
    template <typename Tp, const Tp& traits, typename T, typename U, class Coord>
    __aicore__ inline static void Copy(const T& dst, const U& src, const Coord& coord)
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
        uint16_t nStartPosition = CeilDiv(AscendC::Std::get<1>(coord), AscendC::BLOCK_CUBE);
        uint16_t kStartPosition = CeilDiv(AscendC::Std::get<0>(coord), MXFP_DIVISOR_SIZE);
        auto nStep = AscendC::Std::get<1>(AscendC::Std::get<1>(dst.Layout().Shape()));
        auto kStep = AscendC::Std::get<1>(AscendC::Std::get<0>(dst.Layout().Shape()));
        auto srcStride = AscendC::Std::get<1>(AscendC::Std::get<1>(src.Layout().Stride())) >> 5;
        auto dstStride = kStep;
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst.Data().Get())) >> 4;
        load_cbuf_to_cb_mx(
            mxDstAddr, static_cast<__cbuf__ void*>(src.Data().Get()), nStartPosition, kStartPosition, nStep, kStep,
            srcStride, dstStride);
    }
};

} // namespace Tile

template <>
struct AscendC::Te::CopyTraits<::Tile::CopyL12L0MxScaleB3510>
    : public CopyTraits<
        ::Tile::CopyL12L0MxScaleB3510, LoadDataTraitDefault, ::Tile::CopyL12L0MxScaleB3510,
        LoadDataTraitDefault> {};

namespace AscendC::Te {
constexpr LoadDataTrait LOAD_DATA_B_TRAIT{true};

struct LoadData2BTrait {
    using TraitType = LoadDataTrait;
    static constexpr const TraitType value = LOAD_DATA_B_TRAIT;
};
}

#endif
