/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/operator.h"
#include "register/register.h"
#include <nlohmann/json.hpp>

using namespace ge;
using json = nlohmann::json;

namespace domi {
namespace {
const int kTypeFloat = 1;
}
Status ParseOnnxParamsLeakyRelu(const ge::Operator& op_src, ge::Operator& op_dest) {
  // trans op_src to op_dest
  // if op_src get required attr failed, need to return Failed
  // if op_src get optional attr failed, need to return Failed or set a default value
  float negative_slope = 0.01f;
  string negative_slope_str;
  AscendString attrs_string;
  if (ge::GRAPH_SUCCESS == op_src.GetAttr("attribute", attrs_string)) {
    json attrs = json::parse(attrs_string.GetString());
    for (json attr : attrs["attribute"]) {
      if (attr["name"] == "alpha" && attr["type"] == kTypeFloat) {
        negative_slope_str = attr["f"];  // float type in json has accuracy loss, so we use string type to store it
        negative_slope = atof(negative_slope_str.c_str());
      }
    }
  }

  op_dest.SetAttr("negative_slope", negative_slope);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("LeakyRelu")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::LeakyRelu"),
                   ge::AscendString("ai.onnx::9::LeakyRelu"),
                   ge::AscendString("ai.onnx::10::LeakyRelu"),
                   ge::AscendString("ai.onnx::11::LeakyRelu"),
                   ge::AscendString("ai.onnx::12::LeakyRelu"),
                   ge::AscendString("ai.onnx::13::LeakyRelu")})
    .ParseParamsByOperatorFn(ParseOnnxParamsLeakyRelu)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
