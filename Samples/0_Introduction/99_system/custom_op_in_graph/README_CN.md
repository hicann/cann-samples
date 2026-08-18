# 自定义算子框架插件

## 概述

本样例从完整自定义算子工程中独立出来的**框架插件（Framework Plugin）**模块，包含 ONNX 和 TensorFlow 两个框架的自定义算子适配插件代码。这些插件用于告诉 CANN 编译器如何识别和映射第三方框架模型中的自定义算子。

## 目录结构

```
custom_op_in_graph/
├── CMakeLists.txt           // 构建入口
├── build.sh                 // 便捷编译脚本
├── README_CN.md
├── image/                     // 模型转换效果对比图
│   ├── addn_original.svg      // AddN 原始模型图
│   └── addn_subgraph.svg      // AddN 拆分为子图效果
├── onnx_plugin              // ONNX 框架算子适配插件
│   ├── CMakeLists.txt
│   ├── add_plugin.cc        // Add 算子映射
│   ├── addn_plugin.cc       // AddN 算子映射
│   ├── leaky_relu_plugin.cc // LeakyRelu 算子映射
└── tf_plugin                // TensorFlow 框架算子适配插件
    ├── CMakeLists.txt
    ├── add_block_cust_plugin.cc                      // AddBlockCust 算子注册
    ├── add_dsl_plugin.cc                             // AddDsl 算子注册
    ├── decode_bbox_v2_scope_fusion_plugin.cc         // DecodeBboxV2 融合算子适配
    ├── lstm_tik_plugin.cc                            // LSTMTik 算子注册
    ├── reshape_cust_plugin.cc                        // ReshapeCust 算子注册
    ├── scatter_nd_add_plugin.cc                      // ScatterNdAdd 算子注册
    └── unique_cust_plugin.cc                         // UniqueCust 算子注册
```

## 样例介绍

### ONNX 框架插件（onnx_plugin）

将第三方框架中的算子映射为 CANN 算子。本目录包含以下样例：

- **add_plugin.cc**：将 ONNX Add 算子直接映射为 CANN Add 算子（一对一映射），通过 `AutoMappingByOpFn` 自动完成输入输出映射。
- **addn_plugin.cc**：将 ONNX AddN 算子映射为多个 CANN Add 算子组成的子图（一对多映射 "PartitionedCall"）。通过 `ParseOpToGraphFn` 构建子图，将 AddN(x, y, z) 拆解为 Add(Add(x, y), z)。
- **leaky_relu_plugin.cc**：将 ONNX LeakyRelu 算子（兼容 ai.onnx::8 ~ 13 多个 opset 版本）映射为 CANN LeakyRelu 算子。通过 `ParseParamsByOperatorFn` 从 ONNX 属性中解析 `alpha` 参数。

### TensorFlow 框架插件（tf_plugin）

将 TensorFlow 自定义算子注册到 CANN 框架。本目录包含以下样例：

- **add_block_cust_plugin.cc**：注册 AddBlockCust 算子，指定为 AI_CPU 实现。
- **add_dsl_plugin.cc**：注册 AddDsl 算子，指定为 TVM 实现。
- **decode_bbox_v2_scope_fusion_plugin.cc**：DecodeBboxV2 融合算子的适配插件，通过 `FusionParseParamsFn` 从 Scope 内的小算子中提取缩放参数并设置到融合算子。
- **lstm_tik_plugin.cc**：注册 LSTMTik 算子，指定为 TVM 实现。
- **reshape_cust_plugin.cc**：注册 ReshapeCust 算子，指定为 AI_CPU 实现。
- **scatter_nd_add_plugin.cc**：注册 ScatterNdAdd 算子，指定为 TVM 实现。
- **unique_cust_plugin.cc**：注册 UniqueCust 算子，指定为 AI_CPU 实现。

## 环境要求

环境要求与主仓一致，详见主仓 [README.md](../../../../README.md) 中的"环境部署"章节。本样例额外需要：

- onnx >= 1.12.0（用于生成验证模型）

## 编译

### 方式一：通过 build.sh 便捷编译（推荐）

在 `custom_op_in_graph` 目录下执行：

```bash
chmod +x build.sh
./build.sh
```

`build.sh` 会自动调用主仓统一构建系统（Bisheng 编译器）完成配置和编译，产物输出到当前目录的 `build_out/makepkg/` 下。若重新编译，先执行 `./build.sh clean` 清理产物。

### 方式二：通过主仓统一构建

从项目根目录启动构建，参考项目 [README.md](../../../../README.md)：

```bash
# 1. 配置项目（NPU_ARCH 为必填参数，本模块为 Host 侧代码，不依赖具体架构）
cmake -S . -B build -DNPU_ARCH=dav-3510

# 2. 编译插件
cmake --build build --target cust_onnx_parsers cust_tf_parsers
```

编译后产物直接输出到 `custom_op_in_graph/build_out/makepkg/` 目录下。

### 编译产物

```
custom_op_in_graph/build_out/makepkg/
├── set_env.bash                           // 环境变量脚本
└── packages/vendors/customize/
    └── framework/
        ├── onnx/libcust_onnx_parsers.so
        └── tensorflow/libcust_tf_parsers.so
```

## 部署

编译完成后，`custom_op_in_graph/build_out/makepkg/` 目录结构与算子包一致，包含所有框架的插件库：

    custom_op_in_graph/build_out/makepkg/
    ├── set_env.bash                           // 环境变量脚本
    └── packages/vendors/customize/
        └── framework/
            ├── onnx/libcust_onnx_parsers.so
            └── tensorflow/libcust_tf_parsers.so

部署方式如下：
 
1.  指定目录安装（推荐用于验证）：

    执行 `source build_out/makepkg/set_env.bash`，将编译输出路径追加到 `ASCEND_CUSTOM_OPP_PATH` 环境变量。ONNX/TF的描述转换成GEOP的时会遍历该路径下的 `framework/` 子目录，自动发现所有框架的插件库。在当前终端生效后，可直接进行模型转换验证。

    多个厂商的算子包共存时，按照 `ASCEND_CUSTOM_OPP_PATH` 中从左到右的顺序搜索，后 source 的路径优先级更高。

2.  默认安装：将 `packages/vendors/customize/` 目录整体拷贝到 CANN OPP 算子库路径 `<CANN>/opp/vendors/` 下,注意必须从 `customize` 目录层级拷贝。编辑 `<CANN>/opp/vendors/config.ini`，将 `customize` 写入 `load_priority` 值头部，以逗号分隔其他厂商，例如 `load_priority=customize,other_vendor`。

## 将算子映射为子图（一对多映射）验证

用户可使用ATC模型转换工具对算子映射为子图的效果进行验证。下面给出验证方法：

1.  构造包含AddN算子的onnx模型。

    生成模型的方法为：

    1.  假设用户工作路径为  _<work\_dir\>_，在工作路径下创建python脚本gen\_addn.py， 脚本内容参考：


        ```
        import os
        import numpy as np
        import onnx

        def gen_onnx():
            X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [5])
            Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [5])
            Z = onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [5])
            output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [5])

            node0 = onnx.helper.make_node("AddN", inputs=["X", "Y", "Z"], outputs=["output"])

            inputs = [X, Y, Z]
            outputs = [output]

            graph_def = onnx.helper.make_graph(
                [node0],
                "addn_model",
                inputs,
                outputs
            )

            model_def = onnx.helper.make_model(graph_def)
            model_def.opset_import[0].version = 11
            onnx.save(model_def, "addn_model.onnx")
            print(model_def)
        
        if __name__ == "__main__":
            gen_onnx()
        ```

    2.  执行脚本，生成的onnx模型文件"addn_model.onnx"位于  _<work\_dir\>_目录下。

        **python3 gen\_addn.py**

2.  通过ATC模型转换功能验证算子映射子图效果。
    1.  设置环境变量。

        完成CANN软件基础环境变量配置后，还需要额外配置如下环境变量。
        
        ```
        export DUMP_GE_GRAPH=2     # 控制dump图的内容多少
        export DUMP_GRAPH_LEVEL=2  # 控制dump图的个数
        ```
        
    2. 进行模型转换。

       **atc --model=./addn_model.onnx --framework=5 --output=./addn --input_format=NCHW --soc\_version=$\{soc\_version\}**

       其中，soc\_version：昇腾AI处理器的型号，请根据实际情况替换。可从ATC安装路径下的"<arch\>-linux/data/platform\_config"目录下查看支持的昇腾AI处理器的类型，对应"\*.ini"文件的名字即为{soc\_version}。
       模型转换完成后会在执行atc命令的当前目录下生成一系列按"ge_onnx\*.pbtxt"命名方式命名的文件。这些文件是基于ONNX的开源模型描述结构，可以使用Netron等可视化软件打开。

    3. 结果验证。

       ge\_onnx\_00000000\_graph\_0\_PreRunBegin.pbtxt是ge获取到的经过parse处理的整张下沉图。使用Netron等可视化软件打开原始模型和 ge\_onnx\_00000000\_graph\_0\_PreRunBegin.pbtxt可以看到算子映射子图的实际效果。

        转换前（原始 AddN 模型）：

        ![AddN原始模型](./image/addn_original.svg)

        转换后（AddN 被拆分为 Add + Add 子图）：

        ![AddN子图拆分效果](./image/addn_subgraph.svg)
