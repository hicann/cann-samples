# Vector Function Add (RegBase)

## 描述

本样例演示了如何在昇腾AI处理器的 Vector Core 硬件单元上使用 **RegBase**（Register Based）编程模型实现向量加法 `z = x + y`，是 RegBase 编程模型的 **Hello World** 入门样例。

它与 [vector_add](../vector_add)（TQue / MemBase 编程模型）功能相同，可直接对照学习两种编程模型在写法上的差异。配套概念文档见 [vector_function_getting_started](../vector_function_getting_started)。

## 关键特性

- RegBase / Vector Function（VF）：计算在 `__simd_vf__` 函数中完成，中间结果留在 Vector 寄存器里，不往返 UB
- Mask 尾块自适应：`UpdateMask` 自动处理凑不满一个寄存器宽度的尾块，无需单独处理
- 流水并行：block / tile 两级切分，可运行在多个 Vector Core 上
- 精度对比：提供标准的 CPU 实现作为精度基准

## 编程模型

VF 是 RegBase 模型最核心的编程载体，用 `__simd_vf__` 标记。函数体内用 `AscendC::Reg::*` API 操作寄存器：

```cpp
__simd_vf__ inline void VectorFunctionAdd(
    __ubuf__ float* xAddr, __ubuf__ float* yAddr, __ubuf__ float* zAddr, uint32_t total, uint16_t loopNum)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> xReg, yReg, zReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remain = total;
    for (uint16_t i = 0; i < loopNum; ++i) {
        mask = AscendC::Reg::UpdateMask<float>(remain);
        AscendC::Reg::LoadAlign(xReg, xAddr + i * vectorLength);
        AscendC::Reg::LoadAlign(yReg, yAddr + i * vectorLength);
        AscendC::Reg::Add(zReg, xReg, yReg, mask);
        AscendC::Reg::StoreAlign(zAddr + i * vectorLength, zReg, mask);
    }
}
```

- 一条 `Add` 指令同时作用于一个寄存器宽度的一整批元素（SIMD），不是单个标量
- 最后一批元素不满一个寄存器宽度时，mask 只让前 `remain` 个元素生效，其余 lane 被屏蔽
- Kernel 通过 `asc_vf_call` 调用 VF，VF 的 Load / 运算 / Store 全部由开发者控制

## 支持架构

NPU ARCH dav-3510 (Ascend 950)

RegBase / Vector Function 是 Ascend 950 特性，`dav-2201`（Ascend 910B/C）配置下本样例会自动跳过。

## 参数说明

- totalLength: 向量长度

算子 Kernel 支持 Dtype 模板参数，目前支持 FLOAT32。测试长度 409603 不是寄存器宽度
（`VECTOR_REG_WIDTH / sizeof(float)`）的整数倍，用于演示 Mask 对尾块的自适应处理。

## 编译运行

1. 编译样例

从项目根目录启动构建，参考项目[README.md](../../../README.md)

指定vector_function_add的编译命令：
```shell
cmake --build build --target vector_function_add
```

2. 运行样例

切换到可执行目录文件的所在目录`build/Samples/0_Introduction/vector_function_add/`, 使用可执行文件直接执行算子用例。
```shell
cd ./build/Samples/0_Introduction/vector_function_add/
./vector_function_add
```
打印如下执行结果，证明样例执行成功。
```shell
Vector function add completed successfully!
```
如果存在精度问题，则会打印错误索引，并显示如下结果。
```shell
Vector function add failed at index N!
```

## 与 vector_add 的对比

| | [vector_add](../vector_add) | vector_function_add |
| --- | --- | --- |
| 编程模型 | MemBase（TQue / TPipe） | RegBase（Vector Function） |
| 计算方式 | `AscendC::Add(zLocal, xLocal, yLocal, n)` 操作 UB 上的 LocalTensor | `AscendC::Reg::Add` 操作 Vector 寄存器，中间结果不碰 UB |
| 数据流 | 每步运算都在 UB 上读写 | Load 进寄存器 -> 寄存器间运算 -> Store 回 UB |
| 支持架构 | dav-2201, dav-3510 | dav-3510 |

## References

- [vector_function_getting_started](../vector_function_getting_started): RegBase / VF 编程模型入门概念文档
- [gelu_eltwise_regbase_story](../../2_Performance/gelu_eltwise_regbase_story): RegBase 性能优化实践
