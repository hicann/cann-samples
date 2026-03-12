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
 * \file fia_enum.h
 * \brief
 */

#ifndef FIA_ENUM_H_
#define FIA_ENUM_H_

#define ASCENDC_TPL_5_BW 5
#define ASCENDC_TPL_10_BW 10
#define ASCENDC_TPL_3_BW 3

#define PARSE_PARAMS_NoQuant(inOutLayoutType, config, ...) \
    constexpr LayOutTypeEnum inputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]); \
    constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]); \
    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1); \
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2); \
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d); \
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv);

enum class inferFaLayOutTypeEnum { None = 0, LAYOUT_BSH = 1, LAYOUT_SBH = 2, LAYOUT_BNSD = 3, LAYOUT_TND = 4, LAYOUT_NTD_TND = 5};
enum class inferFIALayoutTypeEnum { LAYOUT_BSH = 0, LAYOUT_BNSD = 1, LAYOUT_TND = 2};
enum class inferDTemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned48 = 48,
    Aligned64 = 64,
    Aligned80 = 80,
    Aligned96 = 96,
    Aligned128 = 128,
    Aligned160 = 160,
    Aligned192 = 192,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned576 = 576,
    NotAligned,
};

enum class inferS1TemplateType {
    Aligned16 = 16,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    NotAligned,
};

enum class inferS2TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned1024 = 1024,
    NotAligned,
};

enum class inferImplModeEnum {
    AA_HIGH_PRECISION = 0,
    AA_HIGH_PERFORMANCE = 1,
    AA_INVALID_LINE_HIGH_PRECISION = 2
};

#define ImplMode_AA_HIGH_PRECISION 0
#define ImplMode_AA_HIGH_PERFORMANCE 1
#define ImplMode_AA_INVALID_LINE_HIGH_PRECISION 2

static constexpr  inferFaLayOutTypeEnum InOutLayoutTypeValue[3][2] ={
    {inferFaLayOutTypeEnum::LAYOUT_BNSD, inferFaLayOutTypeEnum::LAYOUT_BNSD},    
    {inferFaLayOutTypeEnum::LAYOUT_BSH, inferFaLayOutTypeEnum::LAYOUT_BSH},
    {inferFaLayOutTypeEnum::LAYOUT_TND, inferFaLayOutTypeEnum::LAYOUT_TND},
};



#define InOutLayoutType_BNSD_BNSD 0
#define InOutLayoutType_BSH_BSH 1 //BSND由于排布与BSH一样，因此两个会编译成相同取值
#define InOutLayoutType_TND_TND 2

struct ConfigParams {
    inferS1TemplateType s1;
    inferS2TemplateType s2;
    inferDTemplateType d;
    inferDTemplateType dv;
};

// Config
static constexpr ConfigParams ConfigValue[] ={
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //0
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128, inferDTemplateType::Aligned128}, //1
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //2
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned128, inferDTemplateType::Aligned128}, //3
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned192, inferDTemplateType::Aligned128}, //4
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned256, inferDTemplateType::Aligned128}, //5
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned256, inferDTemplateType::Aligned256}, //6
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned512, inferDTemplateType::Aligned512}, //7
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //8
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned576, inferDTemplateType::Aligned512}, //9
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned64, inferDTemplateType::Aligned256, inferDTemplateType::Aligned256}, //10
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned64, inferDTemplateType::Aligned512, inferDTemplateType::Aligned512}, //11
   {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned1024, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //12
   {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned128, inferDTemplateType::Aligned128}, //13
   {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256, inferDTemplateType::Aligned256}, //14
   {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned512, inferDTemplateType::Aligned512}, //15
   {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //16
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128, inferDTemplateType::Aligned128}, //17
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256, inferDTemplateType::Aligned256}, //18
};

#define Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64 0
#define Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128 1
#define Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64 2
#define Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128 3
#define Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128 4
#define Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128 5
#define Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256 6
#define Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512 7
#define Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64 8
#define Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512 9
#define Config_S1Aligned64_S2Aligned64_DAligned256_DVAligned256 10
#define Config_S1Aligned64_S2Aligned64_DAligned512_DVAligned512 11
#define Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64 12
#define Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128 13
#define Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256 14
#define Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512 15
#define Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64 16
#define Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128 17
#define Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256 18


#endif
