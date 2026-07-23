# 以 Scatter 算子为例：SIMT 不规则写与写冲突处理指南

本样例面向 Ascend 950 系列上的 SIMT 编程教学，围绕 Scatter 算子展示两类核心问题：

- **不规则写**：每个 update 根据 `indices[i]` 写到离散的 GM 地址，地址不连续，难以用传统 SIMD/MTE 连续搬运高效覆盖。
- **写冲突**：多个 update 指向同一个输出地址时，不能让多个 SIMT 线程同时无序覆盖同一位置，否则结果不可重复。要求使用最后写入值为准的冲突处理语义。

样例采用递进式 story 结构，每个 step 都是独立可执行目标，方便用户对照代码和 msprof 结果。

## 支持范围

| 项目 | 说明 |
|:---|:---|
| **硬件 / 架构** | Ascend 950PR / 950DT，`NPU_ARCH=dav-3510` |
| **特性线** | SIMT |
| **算子** | Scatter，覆盖语义 |
| **数据类型** | `int32` |
| **输入 / 输出 shape** | `base[4096, 8]`，`updates[N, 8]`，`indices[N]`，`y[4096, 8]` |
| **冲突语义** | 重复 index 采用 last-writer-wins，即原始 update 顺序中最后一次写生效 |

已知限制：

- 本样例只覆盖覆盖语义 Scatter，不覆盖 ScatterAdd / ScatterMax 等规约语义。
- 本样例的冲突处理依赖“按目标地址分组后再写”的模式；真实业务中若输入未分组，可在前处理阶段用 Sort / ArgSort / owner map 等方式生成同等结构。
- 性能数据与硬件、CANN 版本、shape、index 分布强相关，请以本地 msprof 为准。

## 目录说明

```text
simt_scatter_story/
├── CMakeLists.txt
├── README.md
├── include/
│   └── sample_common.h          # ACL 初始化、数据生成、bin 读取、golden 校验、RunSample 入口
├── scripts/
│   └── gen_data.py              # 生成 unique / conflict 两套输入和 golden
└── src/
    ├── 0_direct_unique.asc      # Step 0：唯一 index，SIMT 直接离散写
    ├── 1_grouped_conflict.asc   # Step 1：重复 index，分组后单写者处理冲突
    └── 2_grouped_conflict_2d.asc# Step 2：二维 SIMT 线程布局，并行写 row 内元素
```

## Scatter 语义

本样例计算：

```text
y = base
for i in range(updateRows):
    y[indices[i], :] = updates[i, :]
```

当 `indices` 不重复时，每个 update 写不同输出行，可以直接并行。

当 `indices` 重复时，多个 update 会写同一行。若多个 SIMT 线程直接执行 `y[indices[i]] = updates[i]`，线程调度顺序会影响最终结果。为获得确定结果，本样例定义 last-writer-wins：

```text
y[dst, :] = updates[last_i, :]
last_i = max(i) where indices[i] == dst
```

## Step 0：唯一 index 的 SIMT 直接写

源文件：`src/0_direct_unique.asc`

目标：先理解 SIMT 为什么适合 Scatter 的不规则写。

核心代码：

```cpp
for (int32_t row = rowTid + coreId * rowThreadNum; row < updateRows; row += coreNum * rowThreadNum) {
    int32_t dst = indices[row];
    int64_t srcOffset = static_cast<int64_t>(row) * innerDim;
    int64_t dstOffset = static_cast<int64_t>(dst) * innerDim;
    for (int32_t col = colTid; col < innerDim; col += colThreadNum) {
        y[dstOffset + col] = updates[srcOffset + col];
    }
}
```

要点：

- SIMT 线程可直接访问 GM，`indices[row]` 决定每个线程的离散写地址。
- 当 `indices` 唯一时，没有两个线程写同一目标行，直接写是正确的。
- 这里用二维线程布局：x 维负责 update row，y 维负责 row 内 `innerDim` 元素。

运行：

```bash
cmake --build build --target simt_scatter_0_direct_unique
./build/Samples/2_Performance/simt_scatter_story/simt_scatter_0_direct_unique
```

预期输出：

```text
[0_direct_unique] step 0 PASSED
```

## Step 1：重复 index 的单写者处理

源文件：`src/1_grouped_conflict.asc`

目标：解决重复 index 导致的写冲突。

如果直接写：

```cpp
y[indices[row], col] = updates[row, col];
```

当两个线程的 `indices[row]` 相同，就会出现多写者覆盖同一 GM 地址。覆盖语义下这不是可交换规约，不能靠线程调度获得稳定结果。

本样例采用“先分组，后单写者”的处理模式。`gen_data.py` 会把冲突输入按 `(dst, 原始位置)` 排序，因此同一个 `dst` 的 update 连续排列，且组内仍保持原始顺序。kernel 中只让每个目标地址所在分组的最后一行写出：

```cpp
int32_t dst = indices[row];
bool isLastInGroup = (row == updateRows - 1) || (dst != indices[row + 1]);
if (!isLastInGroup) {
    continue;
}
```

这样每个目标行最终只被一个 SIMT 线程写，结果就等价于 last-writer-wins。

运行：

```bash
cmake --build build --target simt_scatter_1_grouped_conflict
./build/Samples/2_Performance/simt_scatter_story/simt_scatter_1_grouped_conflict
```

预期输出：

```text
[1_grouped_conflict] step 1 PASSED
```

## Step 2：二维 SIMT 布局处理 row 内并行

源文件：`src/2_grouped_conflict_2d.asc`

目标：在 Step 1 的冲突处理基础上，让 row 内元素也由 SIMT y 维线程并行写出。

Step 1 中，一个线程负责一个目标 row 的全部 `innerDim` 元素：

```cpp
for (int32_t col = 0; col < innerDim; ++col) {
    y[dstOffset + col] = updates[srcOffset + col];
}
```

Step 2 改为二维线程：

```cpp
for (int32_t col = colTid; col < innerDim; col += colThreadNum) {
    y[dstOffset + col] = updates[srcOffset + col];
}
```

要点：

- x 维线程负责不同 update row。
- y 维线程负责同一个 row 内不同列。
- `isLastInGroup` 仍然在 row 维判断，确保一个目标 row 只有组尾 update 写出。

运行：

```bash
cmake --build build --target simt_scatter_2_grouped_conflict_2d
./build/Samples/2_Performance/simt_scatter_story/simt_scatter_2_grouped_conflict_2d
```

预期输出：

```text
[2_grouped_conflict_2d] step 2 PASSED
```

## 实验运行

以下命令在 `cann-samples` 仓库根目录执行。

### 环境准备

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
# 或：source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Python 侧至少需要 `numpy`：

```bash
pip install numpy
```

### 构建

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --target simt_scatter_story
```

也可以只构建单个 step：

```bash
cmake --build build --target simt_scatter_0_direct_unique
cmake --build build --target simt_scatter_1_grouped_conflict
cmake --build build --target simt_scatter_2_grouped_conflict_2d
```

### 运行

```bash
./build/Samples/2_Performance/simt_scatter_story/simt_scatter_0_direct_unique
./build/Samples/2_Performance/simt_scatter_story/simt_scatter_1_grouped_conflict
./build/Samples/2_Performance/simt_scatter_story/simt_scatter_2_grouped_conflict_2d
```

每个可执行文件启动后会自动调用 `gen_data.py --output <exe_dir>` 生成输入和 golden，再执行 kernel 并校验输出。

## 什么时候使用这种模式

适合：

- `indices` 离散，SIMD 连续搬运难以高效覆盖。
- update 粒度较小，直接 GM 访问和多线程调度可以隐藏部分访存延迟。
- 覆盖语义下存在重复 index，需要确定性结果。

不适合：

- `innerDim` 很大且每个目标连续搬运占主导，此时 MTE 搬运加 SIMD 计算可能更合适。
- 重复 index 需要加和、最大值等规约语义，应优先考虑原子操作或分块归约，而不是 last-writer-wins。
- 输入完全未分组且冲突率极高，前处理成本可能成为主瓶颈，需要结合业务分布评估。

## 总结

| Step | 场景 | 关键点 | 源文件 |
|:---:|:---|:---|:---|
| 0 | 唯一 index | SIMT 直接离散写 GM | `src/0_direct_unique.asc` |
| 1 | 重复 index | 目标地址分组，只让组尾写 | `src/1_grouped_conflict.asc` |
| 2 | 重复 index + row 内并行 | 二维 SIMT 线程布局 | `src/2_grouped_conflict_2d.asc` |

处理 SIMT Scatter 的推荐思路是：**先确认写语义 -> 再判断 index 是否唯一 -> 对重复 index 先收敛为单写者或规约者 -> 最后再做 SIMT 线程布局优化**。
