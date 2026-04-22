# SIMT特性介绍

## 1. 概述

### 1.1 什么是SIMT
&ensp;&ensp;SIMT(Single Instruction Multiple Thread, 单指令多线程)是一种专为大规模并行计算设计的编程模型。其核心思想是将大量线程组织为逻辑上的线程束(Warp), 每个Warp包含固定数量的线程(如32个)，并由硬件统一调度执行。在SIMT架构中，所有线程在Warp内共享同一指令流，但每个线程拥有独立的寄存器、程序计数器和状态，使得程序员能够以标量风格编写并行代码，而无需显示处理SIMD(单指令多数据)的向量化细节。
硬件层面，SIMT单元在每个指令周期内选择准备就绪的Warp，并向其活跃线程分发指令。线程间的执行分歧(如条件分支)通过硬件隐式处理，即当线程路径不同时，硬件会暂停分析线程的执行，仅运行符合条件的线程，并通过流水调度和隐藏技术(如多Warp并行执行)来掩盖延迟，从而显著提升吞吐量。这种机制允许程序员专注于数据并行逻辑，而无需手动管理线程同步或分支优化，极大简化了并行程序的开发复杂度。

### 1.2 与传统SIMD编程模型的对比

|特性        |SIMD(单指令多数据)|SIMT(单指令多线程)|
|------------|-----------------|-----------------|
|**执行单位**|基于向量寄存器和向量指令并行执行|以线程为单位并行执行|
|**内存管理**|用户显示管理|Dcache(SIMT专用内存空间)分配大小后，硬件自动管理|
|**时序管理**|用户显示管理|编译器和硬件配合完成，用户不感知|
|**分支处理**|if和else分支都会执行|部分场景配合编译器可以仅执行一个分支|

举例说明SIMD和SIMT在数据处理上的区别:

如下图所示,`a=[a1, a2, a3, a4]`, `b=[b1, b2, b3, b4]`,实现`c = a + b = [a1 + b1, a2 + b2, a3 + b3, a4 + b4]`。

![add_demo](./images/image-1.png)

如果**采用SIMD**, 那么只需要一条指令即可完成.

```assembly
VADD.f32 V2, V0, V1, P1
```
SIMD使用一个独立线程，该线程能够同时进行多个数据计算，由于ALU(Arithmetic Logic Unit, 算术逻辑单元)的宽度限制， 其数据对类型、格式、大小要求严格对齐。

![add_demo_simd](./images/image-2.png)

如果**采用SIMT**, 那么会起4个线程(P寄存器控制warp内4个线程生效)。

```assembly
FADD R2, R0, R1 wait:0b0000000 stall:1
```

![add_demo_simt](./images/image-3.png)

**SIMT的好处**是无需开发者费力把数据凑成合适的矢量长度，允许一条指令对多数据分开寻址，而SIMD在数据处理时，对数据在内存中的排布有严格要求(连续、对齐等)，因此SIMT在使用上比SIMD更加灵活.


## 2. 编程模型

### 2.1 Vector Core硬件架构抽象
&ensp;&ensp;理解硬件是写好高性能代码的前提， 在深入阐述SIMT的编程范式之前， 有必要首先理解其赖以运行的硬件基础架构。在Ascend芯片上SIMT既可以单独执行，又可以与SIMD混合编程，通过统一的计算资源和内存层级，实现向量级并行与线程级并行的高效协同。

![vectore_arch](./images/image-6.png)

VectorCore中主要包括: 存储单元、搬运单元和计算单元; 其中计算单元包括标量计算单元。
- 存储单元
   - Global Memory: VectorCore能够访问的外部存储，即为Device上的HBM/GDDR等内存;
   - Local Memory: VectorCore内部的片上高速缓存, 即位Unified Buffer(UB), 是向量指令直接访问的存储空间；
   - SIMT DCache: SIMT专有的Data Cache空间，用于SIMT执行过程中各种数据的缓存区，本质是UB上划出来的一块空间，其大小可由程序员根据需要设置。
- 搬运单元: 负责处理DMA指令(如DataCopy)， 用于在Global Memory和Local Memory之间搬运数据；
- 计算单元(ALU): 执行向量计算的单元，底层计算资源由SIMT和SIMD共享，因此同一时刻，一个AIV(Vector Core)核只能执行SIMT或者SIMD任务，不同的任务在时序上会串行;


### 2.2 编程范式
#### 2.2.1 线程层次
SIMT编程模型的线程层次结构分为三层:
* 线程(Thread): 单个线程块中的数据结构，可以使用`blockDim.x`/`blockDim.y`/`blockDim.z`接口获取每个维度的线程数，3个维度的乘积必须小于等于函数定义时launchbound指定的值。
* 块(Block): 一个线程块最大支持2048线程(硬件当前上限)。需要注意，与GPU不同，当前SIMT VF一次在一个AIV上只能运行一个Block。
* 执行块数(NumBlock): 执行的块数量，类似GPU的grid含义，但是只支持单维。

每个线程块被切分成多个Warp(执行相同指令的线程集合)，每个Warp包含32个线程, 一个Block的多个Warp被依次调度到AI Core中的同一个AIV执行。

![thread_model](./images/image-4.png)

#### 2.2.2 内存模型
* 寄存器: 每个线程拥有独立的寄存器和栈，用于存储局部变量。每个线程可用寄存器数量与线程块中线程数有关(每个AIV总寄存器数固定，启动的线程越多，每个线程可用的寄存器越少)。
* UB(Unified Buffer)： 类似GPU的shared memory，每个block内所有线程共享的本地内存，该内存区域由block内所有线程共同访问。
* Global Memory: 所有线程均可直接访问的全局内存.

![mem_model](./images/image-5.png)

UB(Unified Buffer)内存空间总大小为256KB(Ascend 950PR, 不同型号芯片有差异)， 按功能划分为四个主要区域，从低地址向高地址依次为静态内存、动态内存、预留内存、Data Cache。
1、静态内存: 从地址的起始地址分配一段指定大小的内存空间，其大小在编译时确定，不可动态修改。**该方式待后续支持**
```
//静态内存通过数组分配，例如:
__ubuf__ char staticBuf[1024];
```

2、动态内存: 位于静态内存之后,通过`<<<>>>`中参数dynUBufSize指定的动态内存大小空间，可以通过以下方式申请使用：
* 通过TPipe的相关接口申请
* 通过LocalMemAllocator的Alloc接口申请
* 使用动态数组分配(后续版本支持)
由于上述三种方法申请动态内存时均从静态内存结束位置之后开始分配，如果同时使用可能导致地址空间重叠，从而发生未定义行为，因此只能选择其中一种方法进行申请。
3、预留空间：编译器和AscendC预留空间，大小固定为8KB。
4、Data Cache: SIMT专有的缓存空间，必须大于或者等于32KB,最大不超过128KB。

**静态内存、动态内存的动态数组分配方式目前正在开发中，将在后续版本支持，请关注后续版本。**

* Data Cache = UB总大小(256KB) - 静态内存 - 动态内存 - 预留空间
* 若Data Cache小于32KB会校验报错
* 在SIMD与SIMT混合编程的场景下，算子内部不能使用全部的Unified Buffer空间，除了预留8KB保留空间外，还需要为SIMT预留32KB的Data Cache空间

#### 2.2.3 SIMT VF函数定义
```
__simt_vf__ __aicore__ __launch_bounds__(MAX_THREADNUM) void simt_vector_function(__gm__ float* input, ...)
```
SIMT VF函数使用__simt_vf__、__aicore__修饰符表示是一个simt调用的device函数，使用__launch_bounds__指定该函数可启动的最大线程数量

#### 2.2.4 函数调用
在`host`侧使用`<<<...>>>`方式启动核函数，核函数内部使用`asc_vf_call`调用SIMT子函数，通过参数配置，启动指定数量的线程，执行SIMT函数。

* 核函数：使用`__global__ __aicore__`标识，是Device侧的入口函数，在Host侧通过`<<<...>>>`语法进行调用。
* __aicore__函数: 使用`__aicore__`标识的函数在Device侧执行，核函数内部可以调用`__aicore__`函数。

**核函数**调用方式如下:
```
kernel_name<<<numBlocks, dynUBufSize, stream>>>(args...)
```
* numBlocks:设置核函数启用的核数。
* dynUBufSize：用于指定动态内存的总大小。
* stream：类型为aclrtStream, 用于维护异步操作的执行顺序，确保在device上按照程序中的代码调用顺序执行。

**SIMT VF**调用方式如下:
```
asc_vf_call<funcptr>(dim3 threadNums, ...Args)
```
* funcptr: 待调用的simt函数
* dim3 threadNums: 启动的三维线程
* ...Args: funcptr使用的参数

**💡 备注**:
* 1、开发者需要保证核函数内使用的动态内存大小不超过`dynUBufSize`,超出会越界访问预留空间或者Data Cache，引发未定义行为。
* 2、`asc_vf_call`启动SIMT_VF子任务时，子任务函数不能是类的成员函数，推荐使用普通函数或者类静态函数，且函数入口必须使用`__simt_vf__ `修饰符。
* 3、`asc_vf_call`启动SIMT_VF子任务时，传递的参数当前只支持裸指针，常见的基本数据类型，当前不支持传递结构体、数组等。

## 3. 实践：使用SIMT实现Gather算子
Gather是根据索引从张量中收集元素的一个算子，常用于特征提取、嵌入查找、动态切片等场景，广泛应用于图像、语音等多种领域的模型。

本示例实现的功能类似于tensorflow的gather、pytorch的index_select算子，功能为给定一个输入x、indcies在指定的dim轴上进行gather，可参考如下公式
$$
out[i][j][k] = input[index[i]][j][k]   if dim == 0  \\
out[i][j][k] = input[i][index[j]][k]   if dim == 1  \\
out[i][j][k] = input[i][j][index[k]]   if dim == 2  \\
$$

考虑到要扩展支持任意维度的输入，为了简化kernel的计算过程，根据该算子语义信息，可对输入、输出、索引进行合轴处理:
* indices整体可以合为一根轴[indices_size]
* 输入x可以合并为gather轴之前的部分、gather轴、gather轴之后的部分，总计三根轴[outer, gather_dim_size, inner]
* 输入y可以合并为gather轴之前的部分、gather轴、gather轴之后的部分，总计三根轴[outer, indices_size, inner]

合轴之后kernel仅用支持三根轴的gather，就可以支持任意维度的输入。

### 3.1 代码
* 在host侧完成合轴，计算算子使用的核数numBlock
```
size_t outerDimSize = segmentProduct(inputShape, 0, gatherDim);
size_t gatherDimSize = inputShape[gatherDim];
size_t indicesDimSize = segmentProduct(indicesShape, 0, indicesShape.size());
size_t innerDimSize = segmentProduct(inputShape, gatherDim + 1, outputShape.size());

uint64_t numBlock = std::min((static_cast<uint64_t>(outerDimSize) + threadNum - 1) / threadNum, maxCoreNum);
```

* gather的simt kernel实现
```
template <uint32_t MAX_THREADNUM, typename DATA_TYPE, typename INDICES_TYPE, typename INDEX_SIZE_TYPE>
inline __simt_vf__ __aicore__ __launch_bounds__(MAX_THREADNUM) void gather_function(__gm__ DATA_TYPE *x, __gm__ INDICES_TYPE *indices, __gm__ DATA_TYPE *y, 
    INDEX_SIZE_TYPE gatherDimSize, INDEX_SIZE_TYPE indicesDimSize, INDEX_SIZE_TYPE innerDimSize, INDEX_SIZE_TYPE outNum) {
    for (INDEX_SIZE_TYPE idx = threadIdx.x + blockIdx.x * blockDim.x; idx < outNum; 
        idx += block_num * blockDim.x) {
        INDEX_SIZE_TYPE outerI = idx / (gatherDimSize * innerDimSize);
        INDEX_SIZE_TYPE tmpI = idx - outerI * (gatherDimSize * innerDimSize);
        INDEX_SIZE_TYPE gatherI = tmpI / innerDimSize;
        INDEX_SIZE_TYPE innerI = tmpI - gatherI * innerDimSize;
        INDICES_TYPE indicesValue = indices[gatherI];
        INDEX_SIZE_TYPE indicesValueI = static_cast<INDEX_SIZE_TYPE>(indicesValue);
        INDEX_SIZE_TYPE xIndex = outerI * gatherDimSize * innerDimSize + indicesValueI * innerDimSize + innerI;
        // indices overflow
        bool indexOutOfBound = indicesValue < 0 || indicesValue >= gatherDimSize;
        y[idx] = indexOutOfBound ? 0 : x[xIndex];
    }
}
```
注：__launch_bounds__是个可选参数，用于控制这个simt函数可启动的最大线程数。其影响每个线程可获取的寄存器数量，开的线程越多，每个线程可使用的寄存器越少。该参数若不设置，默认为1024。
* gather的simt函数调用
```
template <typename DATA_TYPE, typename INDICES_TYPE>
__global__ __aicore__ __vector__ void gather(__gm__ DATA_TYPE *x, __gm__ INDICES_TYPE *indices, __gm__ DATA_TYPE *y, 
    size_t outerDimSize, size_t gatherDimSize, size_t innerDimSize, size_t indicesDimSize) {
    constexpr uint64_t INT32_MAX_SIZE = std::numeric_limits<int32_t>::max();
    if ((outerDimSize * gatherDimSize * innerDimSize < INT32_MAX_SIZE) && (outerDimSize * indicesDimSize * innerDimSize < INT32_MAX_SIZE)) {
        asc_vf_call<gather_function<2048, DATA_TYPE, INDICES_TYPE, int32_t>>(dim3(2048), x, indices, y, gatherDimSize, indicesDimSize, innerDimSize, indicesDimSize * innerDimSize);
    } else {
        asc_vf_call<gather_function<2048, DATA_TYPE, INDICES_TYPE, int64_t>>(dim3(2048), x, indices, y, gatherDimSize, indicesDimSize, innerDimSize, indicesDimSize * innerDimSize);
    }
}
```

## 4. 结论
SIMT支持线程**直接访问Global Memory**, 允许一条指令对多数据分开寻址，无需开发者费力将数据拼凑成合适的矢量，并通过硬件自动的warp调度来完成计算和访存的流水掩盖，在使用上比SIMD更加灵活，非常适合处理离散访存等场景。

## 5. 支持架构

NPU ARCH 3510