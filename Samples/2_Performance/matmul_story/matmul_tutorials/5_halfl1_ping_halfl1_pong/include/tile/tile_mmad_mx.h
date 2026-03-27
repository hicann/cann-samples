#ifndef MATMUL_TILE_MMAD_MX_H
#define MATMUL_TILE_MMAD_MX_H
#include "impl/atom/cube_compute/mmad.h"
namespace Tile {

struct MmadMx {
    template <typename Tp, const Tp& traits, typename T, typename U, typename S>
    __aicore__ inline static void Mad(
        const T& dst, const U& fm, const S& filter, uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlagCtrl,
        bool btBuffCtrl, bool initCMatrixCtrl)
    {
        mad_mx(
            dst.Data().Get(), fm.Data().Get(), filter.Data().Get(), m, k, n, unitFlagCtrl, true, btBuffCtrl,
            initCMatrixCtrl);
    }
};

} // namespace Tile

namespace AscendC {
namespace Te {
template <typename Opration, typename TraitStruct>
struct MmadTraits<Opration, TraitStruct> {
    using TraitType = typename TraitStruct::TraitType;
    static constexpr const TraitType defaultTrait = TraitStruct::value;

    template <const TraitType& trait = defaultTrait, typename... Args>
    __aicore__ inline void MmadUnpack(const Args&... args) const
    {
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
struct MmadTraits<::Tile::MmadMx>
    : public MmadTraits<::Tile::MmadMx, MmadTraitDefault, ::Tile::MmadMx, MmadTraitDefault> {};

} // namespace Te
} // namespace AscendC
#endif
