# HiFloat8介绍

## 简要描述

HiFloat8（以下简称HiF8）是一种适用于深度学习的创新性8位浮点数据格式。HiF8采用梯度精度设计：在常规值编码模式下，提供7个3位尾数位指数值、8个2位尾数位指数值以及16个1位尾数位指数值。对于非标准值编码，它通过额外增加7个2的幂次将动态范围从31扩展至38位二进制数（需注意FP16覆盖40位二进制数）。同时，HiF8编码了所有特殊值，但正零和负零仅由单一比特模式表示。得益于精度与动态范围的更佳平衡，HiF8可同时应用于AI训练的前向与后向传播。

## 设计实现

详细介绍参考以下论文

[Ascend HiFloat8 Format for Deep Learning](https://arxiv.org/abs/2409.16626)


## 支持架构

NPU ARCH 3510

## 算子实践
```c++
// 在算子开发中不感知 HIF8 类型转换的具体计算逻辑。根据实际需求，将对应输入/输出的数据类型设置为 hifloat8_t 即可。
// 支持创建数据类型为 hifloat8_t 的 GM 和 UB
AscendC::GlobalTensor<hifloat8_t> yGm;
yGm.SetGlobalBuffer((__gm__ hifloat8_t *)y, TOTAL_LENGTH);

AscendC::LocalTensor<hifloat8_t> yLocal = outQueueY.AllocTensor<hifloat8_t>();
AscendC::LocalTensor<float> tmpLocal = tmpCalc.Get<float>();

// 直接使用 AscendC::Cast API 进行类型转换，无需额外操作
AscendC::Cast<hifloat8_t, float>(yLocal, tmpLocal, AscendC::RoundMode::CAST_ROUND, TOTAL_LENGTH);

// DataCopy时按照每个数据 1 Byte 计算搬运量，正常搬出即可
outQueueY.EnQue<hifloat8_t>(yLocal);
AscendC::LocalTensor<hifloat8_t> yOutput = outQueueY.DeQue<hifloat8_t>();
AscendC::DataCopy(yGm, yOutput, TOTAL_LENGTH);
```


# 算子样例
## Quantize算子
- 算子功能：  
  Quantize算子实现将数据量化为Hifloat8类型的功能。

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Quantize</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">1 * 2048</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">scale</td><td align="center">1 * 2048</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">offset</td><td align="center">1 * 2048</td><td align="center">float32</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">1 * 2048</td><td align="center">hifloat8</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">quantize_custom</td></tr>
  </table>
- 计算公式：  
  ```
  y = (x / scale + offset).to(hifloat8)
  ```

- 样例执行
  ```bash
  # 根据 ${git_clone_path}/README.md 编译Samples仓的所有执行用例
  cd build_out/1_Features/hardware_features/hif8 # 进入hif8的build结果目录
  python3 scripts/gen_data.py   # 生成测试输入数据
  ./quantize_hif8_demo  # 执行编译生成的可执行程序，执行样例
  python3 scripts/verify_result.py output/output_y.bin output/golden_y.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  如果看到以下执行结果，说明精度对比成功。
  ```bash
  test pass
  ```
