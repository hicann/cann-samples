# Vector Add C API

## 描述

本样例展示了如何在昇腾AI处理器的VectorCore硬件单元上使用Ascend C API（C语言风格接口）实现向量加法操作。

## 关键特性

- C API编程：使用asc_simd.h提供的C语言风格API，无需C++模板
- 参数可配：支持自定义向量长度进行测试
- 精度对比：提供标准的CPU实现作为精度基准

## 支持架构

NPU ARCH dav-2201

## 参数说明

- totalLength: 向量长度

算子Kernel目前支持FLOAT32

## 编译运行

1. 编译样例

从项目根目录启动构建，参考项目[README.md](../../../README.md)

指定vector_add_c_api的编译命令：
```shell
cmake --build build --target vector_add_c_api
```

2. 运行样例

切换到可执行目录文件的所在目录`build/Samples/0_Introduction/vector_add_c_api/`, 使用可执行文件直接执行算子用例。
```shell
cd ./build/Samples/0_Introduction/vector_add_c_api/
./vector_add_c_api
```
打印如下执行结果，证明样例执行成功。
```shell
Vector add completed successfully!
```
如果存在精度问题，则会打印错误数据，并显示如下结果。
```shell
Vector add failed!
```
