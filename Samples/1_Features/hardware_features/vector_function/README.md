# Vector Function

## 🚀 快速开始

### 步骤1：环境检查
```bash
# 检查Ascend环境
echo $ASCEND_HOME_PATH
# 预期有路径输出

# 检查CMake版本
cmake --version | head -1
# CMake版本 >= 3.16

# 检查编译器
which bisheng
# 预期返回bisheng的绝对路径
```

### 步骤2：编译运行示例

#### 2.1 从项目根目录构建（推荐）
```bash
# 1. 配置项目（首次构建需要）
cmake -S . -B build

# 2. 编译
cmake --build build --parallel

# 3. 安装到build_out目录
cmake --install build --prefix ./build_out

# 6. 运行示例
./build_out/1_Features/hardware_features/vector_function/gelu_without_vf
./build_out/1_Features/hardware_features/vector_function/gelu_with_vf
```

**预期输出**：
```
GeLU completed successfully!
```

> 💡 **提示**：如果遇到环境配置问题，请确保：
> 1. `ASCEND_HOME_PATH`环境变量已正确设置
> 2. Bisheng编译器已安装并可用
> 3. CMake版本为3.16或更高

### 步骤3：观察性能差异

#### 3.1 使用msprof进行性能分析
msprof是Ascend工具链中的性能分析工具，可以测量Kernel的执行时间：

```bash
# 分析无VF融合版本性能
msprof --application='./gelu_without_vf'

# 分析VF优化版本性能
msprof --application='./gelu_with_vf'
```

#### 3.2 性能对比数据
运行上述命令后，您将看到类似以下的性能数据：

| 性能指标 | 传统版本 | VF优化版本 | 加速比 |
|---------|---------|-----------|--------|
| **Task Duration** | 69.2μs | 25.3μs | **2.74x** |
| **AIV Time** | 67.8μs | 24.1μs | 2.81x |
| **AIV Vec Time** | 67.4μs | 23.7μs | 2.84x |
| **AIV Vec Ratio** | 97.9% | 94.9% | - |

**绝对性能提升**：VF优化版本比传统版本快约**2.8倍**

---

## 📚 什么是Vector Function？

### 核心定义
**Vector Function（向量函数）** 是Ascend NPU引入的编程概念，通过显式控制向量寄存器实现极致计算性能。

### 关键特征
1. **"标量调用、向量执行"**：
   - **标量调用**：由Main Scalar（主标量单元）发起VF调用，处理程序控制流
   - **向量执行**：VF内部的向量计算由专用VF计算单元并行执行

2. **数据驻留**：
   - 中间结果直接在向量寄存器间传递，无需写回Unified Buffer (UB)
   - 消除冗余的数据搬运，提高计算密度

3. **运行时灵活性**：
   - 可根据运行时参数动态调整处理的数据量
   - 支持硬件循环和掩码处理尾部数据

### 与普通函数的本质区别
| 特性 | 普通函数 | Vector Function |
|------|----------|-----------------|
| 执行单元 | Main Scalar逐个执行指令 | VF计算单元接管内部计算 |
| 数据流 | UB-to-UB，中间结果写回UB | 寄存器到寄存器，数据驻留在寄存器中 |
| 优化级别 | 指令级优化 | 计算融合、寄存器重用 |
| 硬件要求 | 所有Ascend平台 | Ascend 950PR/950DT |

### 硬件平台支持
- **支持VF的平台**：Ascend 950PR、Ascend 950DT
- **传统平台**：Atlas A2/A3（不支持VF特性，使用MemoryBased编程模型）

---

## ⚙️ 为什么需要VF？访存墙

### Memory Based编程模型
在Memory Based编程模型中（对应Atlas A2/A3芯片），计算遵循**Load-Compute-Store三阶段**：

![programming_model_spmd](./images/image-1.png)

**性能问题**：
1. **冗余数据搬运**：每个计算步骤都需要完整的`加载-计算-存储`循环
2. **计算单元闲置**：计算单元频繁等待数据搬运完成
3. **UB带宽压力**：大量中间结果占用UB带宽

### GeLU计算示例分析
传统GeLU实现需要8步计算：
```
1. x² = x * x          # 结果写回UB
2. x³ = x² * x         # 结果写回UB
3. t = x * factor      # 结果写回UB
4. sum = x³ + t        # 结果写回UB
5. scaled = sum * k    # 结果写回UB
6. exp = exp(scaled)   # 结果写回UB
7. exp_plus1 = exp + 1 # 结果写回UB
8. y = x / exp_plus1   # 最终结果
```

**问题**：7次中间结果写回UB，产生大量存储-加载开销

### VF的突破性优化
Vector Function通过**计算融合**打破访存墙：

1. **寄存器驻留**：中间结果保留在寄存器中
2. **指令融合**：多步计算合并为单个VF指令块
3. **硬件并行**：利用乱序执行和指令双发

**效果**：将8步计算融合为单个连续执行流，消除中间数据搬运。

![programming_model_vf](./images/image-2.png)
> 🎯 **关键洞察**：VF不是"更快地做同样的事"，而是"用不同的方式做更少的事"（减少数据搬运）。


## 💻 实战示例：GeLU优化

### GeLU计算公式
```
GeLU(x) ≈ x * σ(1.702x)
近似公式：x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715x³)))
```

### 传统实现分析
```cpp
// gelu_without_vf.cpp 关键代码
__aicore__ void gelu_compute(...) {
    AscendC::PipeBarrier<PIPE_V>();          // 同步流水线
    AscendC::Mul(xCube, xLocal, xLocal, n);  // x²，结果写回UB
    AscendC::PipeBarrier<PIPE_V>();          // 等待写回完成
    AscendC::Mul(xCube, xCube, xLocal, n);   // x³，结果写回UB
    AscendC::Muls(tLocal, xLocal, factor, n);// t = x * factor
    AscendC::PipeBarrier<PIPE_V>();          // 等待写回完成
    // ... 总共8个PipeBarrier
}
```

**问题诊断**：
- 8步计算需要7次中间结果写回UB
- 每个`PipeBarrier`强制流水线排空，计算单元闲置
- 大量时间消耗在数据搬运而非计算

### VF优化实现
```cpp
// gelu_with_vf.cpp 关键代码
__simd_vf__ inline void gelu_vf(__ubuf__ float *xAddr, __ubuf__ float *yAddr, uint32_t n, uint32_t loopNum)
{
    const float NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715;
    const float TANH_APPROX_FACTOR = 1 / 0.044715;
    constexpr static uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::MicroAPI::MaskReg pMask;
    // 寄存器声明：使用MicroAPI::RegTensor定义向量寄存器中的张量
    AscendC::MicroAPI::RegTensor<float> xReg, yReg, cubeReg, tReg;
    uint32_t count;
    count = static_cast<uint32_t>(n);
    
    for (uint16_t i = 0; i < loopNum; ++i) {
        pMask = AscendC::MicroAPI::UpdateMask<float>(count);
        // 数据加载到寄存器（一次）
        // LoadDist::DIST_NORM: 连续对齐搬入模式，从UB加载数据到寄存器
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
            xReg, (__ubuf__ float *)xAddr + i * vectorLength);
        // 计算融合（中间结果驻留寄存器）
        // 所有中间结果在寄存器间传递，无需写回UB
        AscendC::MicroAPI::Mul(cubeReg, xReg, xReg, pMask);                       // x² → cubeReg
        AscendC::MicroAPI::Mul(cubeReg, cubeReg, xReg, pMask);                    // x³ → cubeReg（寄存器重用）
        AscendC::MicroAPI::Muls(tReg, xReg, TANH_APPROX_FACTOR, pMask);           // t = x * factor
        AscendC::MicroAPI::Add(cubeReg, cubeReg, tReg, pMask);                    // x³ + t → cubeReg
        // ... 后续计算（指数、加法、除法）全部在寄存器中进行
        AscendC::MicroAPI::Muls(cubeReg, cubeReg, NEG_SQRT_EIGHT_OVER_PI, pMask);
        AscendC::MicroAPI::Exp(cubeReg, cubeReg, pMask);
        AscendC::MicroAPI::Adds(cubeReg, cubeReg, 1.0f, pMask);
        AscendC::MicroAPI::Div(yReg, xReg, cubeReg, pMask);
        // 最终结果写回UB（一次）
        // StoreDist::DIST_NORM_B32: 连续对齐搬出模式，B32表示32Byte对齐
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ float *)yAddr + i * vectorLength, yReg, pMask);
    }
}

__aicore__ inline void gelu_compute(...)
{
    constexpr static uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t loopNum = (n + vectorLength - 1) / vectorLength;
    __ubuf__ float *xAddr = (__ubuf__ float *)xLocal.GetPhyAddr();
    __ubuf__ float *yAddr = (__ubuf__ float *)yLocal.GetPhyAddr();
    gelu_vf(xAddr, yAddr, static_cast<uint32_t>(n), loopNum);
}
```

**优化亮点**：
1. 消除中间结果UB写回（7次→0次）
2. 移除PipeBarrier（8个→0个）
3. 寄存器重用（xReg多次使用）
4. 硬件自动管理指令依赖

## 支持架构

NPU ARCH 3510
