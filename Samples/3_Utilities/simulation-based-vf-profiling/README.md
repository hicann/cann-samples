# Simulation based VF Profiling

基于 `cannsim` 的 VF 性能分析手段。


## 一、为什么用 CANNSim 观测 VF 性能

VectorFunction（VF）编写完成后，上板 profiling 只能看到端到端耗时，难以精确判断瓶颈究竟来自计算指令、load/store、标量控制，还是某类 single-issue 指令。若想单独观察 VF 的指令级行为，`cannsim` 是更适合作为第一步的分析工具。

`cannsim` 能够提供 cycle 级仿真结果，帮助开发者快速定位 VF 的性能瓶颈。

本样例通过一个最小工程，展示如何利用 CANNSim 仿真准确获取 VF 的执行 Trace。


## 二、为获取准确的 VF 数据所做的工作

核心思路：尽可能剥离 VF 之外的干扰因素，使 Trace 中保留的开销尽量收敛到 VF 指令本身。

### 2.1 使用纯 VF kernel 隔离数据搬运

`pure_vf_kernel` 仅在 UB 上定义本地数组，通过 `asc_vf_call` 发起 VF 调用，全程不涉及 GM 与 UB 之间的 DataCopy。

这样设计的原因在于：GM↔UB 的数据搬运与 VF 内部的 UB↔Reg 访存复用同一套 UB 读写端口，二者会争用端口带宽、相互阻塞。若在测量路径中保留搬运指令，VF 的实测开销会被搬运竞争所污染。剔除搬运后，Trace 上保留的主要是 VF 指令，从而避免将搬运开销错误地计入 VF 的性能判断。

### 2.2 用 Warmup 消除冷启动开销

`main` 中先启动 `warmup_kernel`，再启动待测的 `pure_vf_kernel`：

```cpp
warmup_kernel<<<1, nullptr, stream>>>();
pure_vf_kernel<<<1, nullptr, stream>>>();
```

首个 Kernel 启动时，其指令段与数据段尚未建立地址映射，首次访问会触发 TLB Miss，这部分开销会以噪声形式混入 Trace。`warmup_kernel` 内部仅执行 `AscendC::Nop<8>()`，用于预先完成一次页表建立、填充 TLB；待 `pure_vf_kernel` 执行时 TLB 已就绪，采集到的 cycle 数据便能更真实地反映 VF 指令自身的执行开销。

### 2.3 每个 VF 重复调用 3 次

`pure_vf_kernel` 中每个 VF 均重复调用 3 次。首次调用时，VF 对应的指令段可能尚未驻留 ICache，会因 ICache Miss 而引入额外延迟，干扰性能评估；后两次调用时，指令已驻留 ICache，可在排除 ICache Miss 的稳态下取得更干净的测量值。

### 2.4 关闭 VF 自动融合

样例的 `CMakeLists.txt` 为 ASC 编译追加了如下选项：

```cmake
"$<$<COMPILE_LANGUAGE:ASC>:--cce-simd-vf-fusion=false>"
```

编译器会依据 VF 代码内容，将可融合的 VF 自动合并。其后果是：源码中写了 3 段 VF，Trace 中看到的却可能是融合后的单个 VF，导致测量对象与源码意图不一致。关闭自动融合后，Trace 中呈现的即为源码原样的 VF，做到所见即所得，保证测量粒度与代码结构一一对应。

## 三、环境检查

本样例只支持 Ascend 950 对应架构：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
```

如果使用其他 `NPU_ARCH`，CMake 会跳过该样例。

运行前请确认：

- 已安装 CANN Toolkit，并执行过 `${install_path}/ascend-toolkit/set_env.sh`
- 可使用 `cannsim`
- CMake 版本满足仓库根 README 的要求

### 编译与运行

#### 使用 CMake 编译

在仓库根目录执行：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --target simulation_based_vf_profiling
```

用 `cannsim` 运行并生成报告：

```bash
cannsim record ./build/Samples/3_Utilities/simulation-based-vf-profiling/simulation_based_vf_profiling -s Ascend950 --gen-report
```

#### 使用 bisheng 独立编译

如果只想编译并运行当前样例，也可以直接调用 `bisheng`：

```bash
cd Samples/3_Utilities/simulation-based-vf-profiling

bisheng main.asc -o simulation_based_vf_profiling --npu-arch=dav-3510 -O3 --cce-simd-vf-fusion=false
```

用 `cannsim` 运行并生成报告：

```bash
cannsim record ./simulation_based_vf_profiling -s Ascend950 --gen-report
```

运行成功后，终端会看到：

```text
Kernel launched successfully!
```

同时会生成形如下面的产物目录：

```text
cannsim_<时间戳>_simulation_based_vf_profiling/
```
