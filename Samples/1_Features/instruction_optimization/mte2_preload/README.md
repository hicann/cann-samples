# MTE2预加载特性介绍
## 1. 原理介绍
### 1.1 背景
&ensp;&ensp;在实现启用双缓冲（Double Buffer）的矩阵乘法时，每条 MTE2_PONG 指令必须等待与其配对的 MTE2_PING 指令之后的所有指令全部发射完毕后，才能获得发射机会。因此，若这两条指令之间插入的指令数量过多，超出了芯片预设的指令队列深度，即使不存在同步依赖，MTE2_PONG 指令仍会因队列阻塞而无法发射。

<div align="center">
  <img src="./images/image-1.png" alt="背景流水图" style="width: 80%; height: auto;">
</div>

&ensp;&ensp;限制 KL1 长度虽能减少指令队列中的指令数量，但在某些 shape 场景下，KL1 已无法进一步缩减，否则会导致性能损失。此时，需借助 MTE2 预加载来避免指令阻塞。

### 1.2 原理
&ensp;&ensp;通过提前将两组数据搬运到 L1 缓存中，使 MTE2 指令得以提前发射，从而避免因指令堵塞而导致的流水线断流。

**计算流水图如下**：

&ensp;&ensp;优先发送 PONG 对应的 MTE2，且下一轮的 PING 无需等待。随后再执行上一轮 MTE2（即已发送 PONG 的那一轮）所对应的 MTE1 与 MMAD 指令，从而实现解耦。
<div align="center">
  <img src="./images/image-2.png" alt="原理图" style="width: 80%; height: auto;">
</div>

### 1.3 预期效果

* **消除指令队列阻塞**：MTE2_PONG 不再被动等待 PING 之后的所有指令发射完毕，而是可以提前发射并预取数据，避免因指令队列深度不足导致的发射停顿。
* **提升流水线连续性**：在 KL1 无法进一步缩减的场景下，仍能维持计算单元持续工作，减少流水线断流。

## 2. 实践：使用MTE2预加载特性优化计算流水

### 2.1 代码
下面演示 M 方向 MTE2 预加载的实现：
首先，在第一轮搬运两份 A 矩阵到 L1 缓存：

```C++
// 第一段：处理第一个分片（tileIdx / blockNum == 0 且是第一轮迭代 iter0 == 0）
if (tileIdx / blockNum == 0 && iter0 == 0) {
    // 等待该 L1 缓冲区上之前的 MTE1（L1 -> GM）和 MTE2（GM -> L1）操作完成
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
    
    // 为 L1 缓冲区 A 创建张量，并从全局内存中拷贝第一个分片的 A 数据
    auto tensorAL1First =
        AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<T>(l1BufferAOffset[l1BufId]), layoutAL1);
    auto tensorAGmTileFirst =
        tensorAGmBlock(AscendC::Te::MakeCoord(0, iter0 * kL1), AscendC::Te::MakeShape(curM, curGmAKL1));
    AscendC::Te::Copy(copyGM2L1, tensorAL1First, tensorAGmTileFirst);

    // 为 L1 缓冲区 B 创建张量，并从全局内存中拷贝第一个分片的 B 数据
    auto tensorBL1First =
        AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<T>(l1BufferBOffset[l1BufId]), layoutBL1);
    auto tensorBGmTileFirst =
        tensorBGmBlock(AscendC::Te::MakeCoord(iter0 * kL1, 0), AscendC::Te::MakeShape(curGmBKL1, curN));
    AscendC::Te::Copy(copyGM2L1, tensorBL1First, tensorBGmTileFirst);

    // 设置标志位，同步 MTE2 -> MTE1，表示拷贝完成
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
}
```

接下来，在非首轮迭代或首轮迭代的后续分片时，利用双缓冲机制预取下一分片数据到另一块 L1 缓冲区，实现数据加载与计算的重叠:

```C++
// 第二块：双缓冲预取下一个tile
if (iter0 + 1 < kL1TileNum) {     
    // 等待目标缓冲区空闲
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1 - l1BufId);
    
    // 重新计算下一块的数据布局
    layoutAL1 = AscendC::Te::MakeLayoutAL1<T>{}(curM, NextCurGmAKL1);
    layoutBL1 = AscendC::Te::MakeLayoutBL1<T>{}(NextCurGmBKL1, curN);
    
    // 预取A矩阵的下一块到备用L1缓冲区
    auto tensorAL1Sec =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, T>(l1BufferAOffset[1 - l1BufId]), layoutAL1);
    auto tensorAGmTileSec = tensorAGmBlock.Slice(
        AscendC::Te::MakeCoord(0, (iter0 + 1) * kL1), AscendC::Te::MakeShape(curM, NextCurGmAKL1));
    AscendC::Te::Copy(copyGM2L1, tensorAL1Sec, tensorAGmTileSec);

    // 预取B矩阵的下一块到备用L1缓冲区
    auto tensorBL1Sec =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, T>(l1BufferBOffset[1 - l1BufId]), layoutBL1);
    auto tensorBGmTileSec = tensorBGmBlock.Slice(
        AscendC::Te::MakeCoord((iter0 + 1) * kL1, 0), AscendC::Te::MakeShape(NextCurGmBKL1, curN));
    AscendC::Te::Copy(copyGM2L1, tensorBL1Sec, tensorBGmTileSec);

    // 标记数据已准备好
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1 - l1BufId);
}
```

最后处理轮次边界：当完成当前K维度的最后一个分片时，提前预取下一轮区域的第一个K分片，实现跨轮次的双缓冲无缝衔接。
```C++
// 第三块：完成当前轮次最后一个tile时，预取下一轮的第一个tile（实现轮间无缝衔接）
// 触发条件：
//   1. iter0 + 1 == kL1TileNum：当前K维度的tile是最后一轮（没有下一个tile了）
//   2. tileIdx + blockNum < tileNum：还有未处理的tile块（不是全局最后一个块）
if (iter0 + 1 == kL1TileNum && tileIdx + blockNum < tileNum) {
    // 等待备用缓冲区上的异步操作完成，确保可以安全写入下一轮的数据
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1 - l1BufId);
    
    // 计算下一轮要处理的tile索引（将一维索引转换为二维(M,N)坐标）
    // blockNum：每个tile包含的K维子块数量，用于跨轮次索引跳转
    uint64_t nextMTileIdx = (tileIdx + blockNum) / nTileNum;
    uint64_t nextNTileIdx = (tileIdx + blockNum) % nTileNum;

    // 处理M/N维度的边界情况：最后一行的tile可能高度或宽度不完整（tail）
    int64_t nextRoundM = nextMTileIdx == (mTileNum - 1) ? tailBaseM : baseM;
    int64_t nextRoundN = nextNTileIdx == (nTileNum - 1) ? tailBaseN : baseN;

    // 处理K维度的边界：当前轮次的K维度tile数量不一时，下一轮的第一个K分片大小需要重新计算
    // 注意：这里kL1TileNum表示当前轮次有多少个K分片，当只有一个分片时直接取剩余长度
    int64_t nextRoundAKL1 = kL1TileNum == 1 ? (k - (kL1TileNum - 1) * kL1) : kL1;
    int64_t nextRoundBKL1 = nextRoundAKL1;  // A和B在K维度大小必须一致

    // 根据新的tile尺寸重新计算L1缓冲区布局（适应可能的边界形状变化）
    layoutAL1 = AscendC::Te::MakeLayoutAL1<T>{}(nextRoundM, nextRoundAKL1);
    layoutBL1 = AscendC::Te::MakeLayoutBL1<T>{}(nextRoundBKL1, nextRoundN);

    // 从全局张量中切出下一轮要处理的完整块（包含所有K维度数据）
    auto nextTensorAGmBlock = tensorAgm.Slice(
        AscendC::Te::MakeCoord(nextMTileIdx * baseM, 0L), 
        AscendC::Te::MakeShape(nextRoundM, k));
    auto nextTensorBGmBlock = tensorBgm.Slice(
        AscendC::Te::MakeCoord(0L, nextNTileIdx * baseN), 
        AscendC::Te::MakeShape(k, nextRoundN));

    // 预取下一轮的A矩阵第一个K分片（从新GM块中取起始位置）
    auto tensorAL1Sec =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, T>(l1BufferAOffset[1 - l1BufId]), layoutAL1);
    auto tensorAGmTileSec = nextTensorAGmBlock.Slice(
        AscendC::Te::MakeCoord(0, 0),  // 从新块的开头开始取
        AscendC::Te::MakeShape(nextRoundM, nextRoundAKL1));
    AscendC::Te::Copy(copyGM2L1, tensorAL1Sec, tensorAGmTileSec);

    // 预取下一轮的B矩阵第一个K分片
    auto tensorBL1Sec =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, T>(l1BufferBOffset[1 - l1BufId]), layoutBL1);
    auto tensorBGmTileSec = nextTensorBGmBlock.Slice(
        AscendC::Te::MakeCoord(0, 0),  // 从新块的开头开始取
        AscendC::Te::MakeShape(nextRoundBKL1, nextRoundN));
    AscendC::Te::Copy(copyGM2L1, tensorBL1Sec, tensorBGmTileSec);

    // 标记数据准备就绪，下一轮计算开始时可以直接使用
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1 - l1BufId);
}
```

**关键改动点**：

* **初始预加载双份数据**: 在首轮（iter0 == 0）时，一次性搬运两份 A 矩阵数据到 L1 缓存，使得 MTE2_PONG 提前发射。
* **缓冲区切换**：在首轮块内，每次预取前更新curL1BufId = 1 - l1BufId，切换到另一块缓冲区；同时更新全局内存偏移量（curOffsetL1）和分片尺寸（curGmAKL1, curGmBKL1）。
* **轮次边界处理**：当完成当前区域的最后一个 K 分片（iter0 + 1 == kL1TileNum）且还有下一个区域待处理时，提前预取下一轮区域的第一个 K 分片到备用缓冲区，实现轮间无缝切换，避免计算单元空闲。

### 2.2 修改注意点

* **只预取所需分片**：条件判断确保只在还有剩余分片时（iter0 + 1 < kL1TileNum）才发起预取，避免无效搬运。
* **首轮特殊处理边界**：首轮（iter0 == 0）预加载两份数据后，需确保首次 MMAD 计算使用的是第一份数据，而非等待第二份搬运完成。
* **末轮搬运与计算的收尾**：在最后一轮迭代时，需注意不再触发下一轮预取（避免越界访问），同时确保最后一轮的计算仍能正确访问已搬运的数据块，不要提前释放或覆盖。
* **受影响部分**：代码复杂度增加，流水管理需注意错开操作，同时指令规模扩大，导致耗时增加。

## 3. 性能结果对比
### 3.1 case前后性能
&ensp;&ensp;以基础 MatMul 算子开启 DB 为例，在相同输入规模（M=1024, K=2048, N=4096）下，将 basek 设为 32 以增加指令深度并进行性能测试，通过 Profiling 工具采集硬件流水线的执行状态。

<div align="center">
  <img src="./images/image-3.png" alt="计算流水图1" style="width: 80%; height: auto;">
</div>

&ensp;&ensp;从流水对比图可以看出，开启 MTE2 预加载后，Pong 的 MTE2 指令并未因超过预设队列深度而延后启动，整体 MTE2 处理过程向前平移，计算更加连续。

## 4. 结论

适用场景：
* **指令队列深度受限的高密度调度场景**：当 PING 与 PONG 指令之间插入的指令数量超出芯片预设队列深度时，传统双缓冲会出现 MTE2_PONG 发射阻塞，预加载机制可有效解除该依赖。
* **KL1 无法进一步缩减的性能敏感场景**：在某些矩阵 shape 下，继续缩减 KL1 会导致计算性能下降。此时 MTE2 预加载能够在保持 KL1 不变的前提下，通过提前发射MTE_PONG指令来避免流水线断流。

&ensp;&ensp;MTE2 预加载通过提前搬运矩阵到L1，消除了 MTE2_PONG 的发射阻塞，提升了流水线连续性与吞吐效率。

## 5. 编译 执行

1. 编译样例

从项目根目录启动构建，参考项目[README.md](../../../README.md)

在仓库根目录下完成编译和安装后，进入当前样例目录：
```shell
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --parallel
cmake --install build --prefix ./build_out
cd ./build_out/1_Features/instruction_optimization/mte2_preload/
```

如需单独编译当前样例，可使用以下指令：
```shell
cmake --build build --target mte2_preload
cp ./Samples/1_Features/instruction_optimization/mte2_preload/scripts/* ./build/Samples/1_Features/instruction_optimization/mte2_preload/
cd ./build/Samples/1_Features/instruction_optimization/mte2_preload/
```

2. 运行样例

使用可执行文件直接执行算子用例，需要指定矩阵乘维度，并随机生成输入数据。
```shell
./mte2_preload 1024 2048 4096
```
运行成功后，终端将打印如下类似信息：
```txt
Data generated successfully!

[verify] shape(1024, 4096), elements=4194304 - summary (large matrix, full tensors omitted)
  abs_err: max=2.560000e+02, mean=6.103516e-03, rmse=1.250000e+00
  rel_err: max=6.410256e-03
  count(|abs_err| > 0.001): 100 / 4194304
  cpu golden (top-left 4x4):
tensor([[41728., 42240., 41216., 41216.],
        [40960., 41728., 40704., 40960.],
        [40960., 41472., 40704., 40448.],
        [40704., 41246., 40192., 39680.]], dtype=torch.bfloat16)
  npu out (top-left 4x4):
tensor([[41728., 42240., 41216., 41216.],
        [40960., 41728., 40704., 40960.],
        [40960., 41472., 40704., 40448.],
        [40704., 41246., 40192., 39680.]], dtype=torch.bfloat16)
max abs diff: 256.0
point error count(>0.1): 0/4194304
ratio error count(>0.001): 100/4194304, error ratio: 0.000024
[PASS] NPU results are consistent with CPU.
```
如果存在精度问题，则会打印错误数据，并显示如下结果。
```txt
[ERROR] NPU results differ from CPU.
```

3. 测试性能
运行性能测试脚本，指定矩阵乘法的维度后执行。
```shell
python3 profile_matmul.py 1024 2048 4096
```
打印如下执行结果，证明样例性能测试成功。
```shell
[Profile Breakdowm]
+--------------+------------+---------+------------+----------+----------+-------------+----------------+
| candidate    | kernel(us) | mac(us) | scalar(us) | mte1(us) | mte2(us) | fixpipe(us) | icache_miss(%) |
+==============+============+=========+============+==========+==========+=============+================+
| mte2_preload |     53.328 |  40.791 |      3.166 |   13.191 |   38.916 |       2.110 |          0.300 |
+--------------+------------+---------+------------+----------+----------+-------------+----------------+
```
与相同规模下的基础 MatMul 算子开启 double-buffer对比：
```shell
[Profile Breakdowm]
+-----------+------------+---------+------------+----------+----------+-------------+----------------+
| candidate | kernel(us) | mac(us) | scalar(us) | mte1(us) | mte2(us) | fixpipe(us) | icache_miss(%) |
+===========+============+=========+============+==========+==========+=============+================+
| n_buffer  |     66.000 |  40.810 |      2.558 |   10.659 |   37.595 |       1.980 |          1.200 |
+-----------+------------+---------+------------+----------+----------+-------------+----------------+
```
可以看到，整体kernel运算时间缩短，性能有所提升。

## 6. 支持架构

NPU ARCH 3510