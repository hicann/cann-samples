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
* \file mad_atom_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_DETAIL_ATOM_MAD_ATOM_IMPL_H
#define IMPL_TENSOR_API_DETAIL_ATOM_MAD_ATOM_IMPL_H

#include "impl/atom/cube_compute/cube_compute_impl.h"

namespace AscendC {
namespace Te {

template <typename... Args>
struct MmadAtom;

template <typename MadOperation>
struct MmadAtom<MadOperation> : MmadAtom<MmadTraits<MadOperation>> {};

template <typename MadOperation, typename... Args>
struct MmadAtom<MmadTraits<MadOperation, Args...>> : MmadTraits<MadOperation, Args...>
{
    using MadTraitType = MmadTraits<MadOperation, Args...>;
    using TraitType = typename MadTraitType::TraitType;
    static constexpr const TraitType defaultTrait = MadTraitType::defaultTrait;

    template <const TraitType& traits = defaultTrait, typename... Params>
    __aicore__ inline void Call(const Params& ...params) const {
        MadTraitType::template MmadUnpack<traits>(params...);
    }

    template <typename... TraitsArgs>
    __aicore__ inline auto with(TraitsArgs&&... args) const {
        auto traits = MadTraitType::with(static_cast<TraitsArgs&&>(args)...);
        return MmadAtom<decltype(traits)>{traits};
    }
};

}
}

#endif // IMPL_TENSOR_API_DETAIL_ATOM_MAD_ATOM_IMPL_H