# Vector Add

## 描述

本样例展示了如何在昇腾AI处理器的VectorCore硬件单元上使用AscendC编程语言实现向量加法操作。

## 关键特性

- 流水并行：具备DoubleBuffer能力开启流水并行
- 参数可配：支持自定义向量长度进行测试
- 精度对比：提供标准的CPU实现作为精度基准

## 支持架构

NPU ARCH 3510

## 参数说明

- totalLength: 向量长度

算子Kernel支持Dtype模板参数，目前支持FLOAT32

## 编译运行

1. 编译样例

从项目根目录启动构建，参考项目[README.md](../../../README.md)

指定vector_add的编译命令：
```shell
cmake --build build --target vector_add
```

2. 运行样例

切换到可执行目录文件的所在目录`build/Samples/0_Introduction/vector_add/`, 使用可执行文件直接执行算子用例。
```shell
cd ./build/Samples/0_Introduction/vector_add/
./vector_add
```
打印如下执行结果，证明样例执行成功。
```shell
Vector add completed successfully!
```
如果存在精度问题，则会打印错误数据，并显示如下结果。
```shell
Vector add failed!
```
