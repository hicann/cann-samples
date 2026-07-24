<h1 align="center">CANN-SAMPLES</h1>

<p align="center">
  昇腾 CANN 算子领域的实战样例仓库<br>
  提供高性能实现示例与体系化调优知识库，从入门概念到极致性能，覆盖 MatMul、MoE、Attention 等核心算子的完整优化链路
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Perf%20Stories-9-blueviolet.svg" alt="Perf Stories">
  <img src="https://img.shields.io/badge/CANN-9.0%2B-green.svg" alt="CANN">
  <a href="https://zread.ai/hicann/cann-samples"><img src="https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff" alt="Zread"></a>
</p>

<p align="center">
  <a href="#概述">📖 概述</a> ·
  <a href="#最新动态">🔥 最新动态</a> ·
  <a href="#环境部署">🛠️ 环境部署</a> ·
  <a href="#快速入门">🚀 快速入门</a> ·
  <a href="#样例列表">📦 样例列表</a> ·
  <a href="https://gitcode.com/cann/cann-samples/issues">💬 社区讨论</a>
</p>

---

## 🚀概述

`cann-samples` 是 [CANN](https://hiascend.com/software/cann)（Compute Architecture for Neural Networks）算子领域的实战样例仓库，提供高性能实现示例与体系化调优知识库。

本仓已集成代码仓库智能体，点击 [![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/hicann/cann-samples) 徽章，进入其专属页面，开启在线智能代码学习与知识问答体验！

### 🗺️ 用户导航

| 你是... | 推荐入口 | 预计耗时 |
|---------|---------|:-------:|
| 👋 NPU 算子初学者 | [入门样例](Samples/0_Introduction/README.md) → `vector_add` / `matmul` 建立基本概念 | 30 min |
| 🏗️ 自有环境部署 | [环境部署](#环境部署) → [快速入门](#快速入门) 编译运行第一个样例 | 20 min |
| 🚀 追求极致性能 | [样例列表](#样例列表) → 按算子类型查找调优实践 | 按需 |
| ✨ 贡献代码 | [所属 SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-basic) | 15 min |

### 为什么使用 cann-samples

| 维度 | 说明 |
|------|------|
| ⚡ 体系化调优知识库 | 从入门概念到极致性能，覆盖 MatMul、MoE、Attention 等核心算子的完整优化链路 |
| 🔬 端到端可复现 | 每个 story 提供 baseline→优化分步教程与可运行 recipe，配合 cannsim trace 量化性能差异 |
| 🧠 硬件特性深挖 | 直击 ScalarBound、流水线排布、寄存器 Spill、SIMT / SIMD VF 编程等底层优化点 |
| 🧩 多数据类型覆盖 | 支持 BF16 / FP16 / HiFloat8 / MXFP4 / MXFP8 等多种精度与量化方案 |
| 🎯 多代际硬件适配 | 覆盖 Ascend 950（`dav-3510`）与 Ascend 910B/C（`dav-2201`）平台 |

---

## 🔥最新动态

- [2026/05] cann-samples在matmul_story新增[matmul_a16w16](Samples/2_Performance/matmul_story/matmul_recipes/examples/matmul_a16w16/README.md) streamK等高性能example，并在既有quant_matmul_mxfp4、quant_matmul_mxfp8 MX量化矩阵乘中支持Weight NZ。
- [2026/05] cann-samples在grouped_matmul_story新增[weight_quant_grouped_matmul_mxfp8fp4](Samples/2_Performance/grouped_matmul_story/grouped_matmul_recipes/examples/weight_quant_grouped_matmul_mxfp8fp4/README.md) MXA8W4量化example，并在既有quant_grouped_matmul_mxfp4、quant_grouped_matmul_mxfp8分组矩阵乘中支持Weight NZ。

---

## 📝环境部署

当前仓库已验证通过的社区版 CANN Toolkit 如下：

| CANN 版本 | 时间戳 | 验证结果 | 下载链接 |
| --- | --- | --- | --- |
| `9.1.0` | `20260513000324948` | ✅ PASS | [aarch64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260513000324948/Ascend-cann-toolkit_9.1.0_linux-aarch64.run) / [x86_64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260513000324948/Ascend-cann-toolkit_9.1.0_linux-x86_64.run) |
| `9.1.0` | `20260508171052185` | ✅ PASS | [aarch64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260508171052185/Ascend-cann-toolkit_9.1.0_linux-aarch64.run) / [x86_64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260508171052185/Ascend-cann-toolkit_9.1.0_linux-x86_64.run) |
| `9.0.0` | `20260422000325096` | ✅ PASS | [aarch64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260422000325096/Ascend-cann-toolkit_9.0.0_linux-aarch64.run) / [x86_64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260422000325096/Ascend-cann-toolkit_9.0.0_linux-x86_64.run) |

请根据实际 CPU 架构，从上述链接目录中自行选择对应的 `.run` 安装包。

### 兼容性声明

cann-samples中矩阵乘类example更新，引入ops-tensor子模块。涉及Tensor API的样例需使用上表中已验证通过的 Toolkit 版本构建，并按下文[ops-tensor子模块与Toolkit约束](#ops-tensor子模块与toolkit约束)初始化子模块、配置环境变量。

toolkit 安装包文件名格式如下：

- `Ascend-cann-toolkit_${cann_version}_linux-aarch64.run`
- `Ascend-cann-toolkit_${cann_version}_linux-x86_64.run`

1. **安装社区版 CANN Toolkit**

    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
    ```
    - `${cann_version}`：表示 toolkit 安装包版本号，需满足上文的最低版本要求。
    - `${arch}`：表示 CPU 架构，如 `aarch64`、`x86_64`。
    - `${install_path}`：表示指定安装路径，默认安装在 `/usr/local/Ascend` 目录。

2. **配置环境变量**

   安装完成后，请先执行：

    ```bash
    source ${install_path}/ascend-toolkit/set_env.sh
    ```

   请将 `${install_path}` 替换为 toolkit 的实际安装目录，例如 `/usr/local/Ascend` 或 `${HOME}/Ascend`。

3. **前置依赖**

   编译用到的依赖如下，请确保已安装并且满足版本要求：

   - cmake >= 3.16.0
   - python >= 3.8.0
   - zip
   - git
   - python三方库依赖：通过`pip3 install -r requirements.txt`安装

4. **ops-tensor子模块与Toolkit约束**

   - **仓库与路径**：子模块路径为`third_party/ops-tensor`，对应上游仓库[ops-tensor](https://gitcode.com/cann/ops-tensor)（默认跟踪`master`，见仓库根目录`.gitmodules`）。
   - **获取源码**：克隆本仓库时建议执行 `git clone --recurse-submodules <仓库 URL>`；若已克隆未带子模块，在仓库根目录执行：
     ```bash
     git submodule update --init --recursive third_party/ops-tensor
     ```
     若未提前初始化子模块，CMake在构建依赖`cann_samples::tensor_api`的目标时也会尝试执行上述子模块更新命令。
   - **Toolkit要求**：Tensor API相关样例会使用`third_party/ops-tensor/include/tensor_api`以及Toolkit中的Ascend C头文件，因此必须安装完整的CANN Toolkit并先执行`source ${install_path}/ascend-toolkit/set_env.sh`。当前请使用上表中已验证通过的版本构建；Toolkit版本过旧、仅安装Run包或环境变量未生效时，可能出现头文件缺失、符号未定义或编译选项报错。
   - **NPU架构**：`matmul_story`、`grouped_matmul_story`额外要求`NPU_ARCH=dav-3510`（Ascend 950）；使用`dav-2201`全量配置工程时，这两项样例会被跳过，属预期行为。

## ⚡️快速入门

1. 配置项目

   `NPU_ARCH` 为必填参数，用于指定目标 NPU 架构。当前支持的取值如下：

   | NPU 平台 | NPU_ARCH |
   | --- | --- |
   | Ascend950 | `dav-3510` |
   | Ascend910B/C | `dav-2201` |

   以 Ascend950 为例，使用以下命令初始化构建配置，CMake 会自动创建 `build` 目录：
   ```sh
   cmake -S . -B build -DNPU_ARCH=dav-3510
   ```
   在 Ascend910B/C 平台构建时，请使用 `-DNPU_ARCH=dav-2201`。不支持当前架构的样例会在配置阶段跳过，因此 `target help` 和后续构建只包含当前架构生效的样例。

2. 查看可用 Target（可选）

   编译前可先查看当前项目中支持单独构建的目标列表：
   ```sh
   cmake --build build --target help
   ```

3. 编译与安装

   - 选项 A：编译指定 Target（部分构建）

     将 `<target_name>` 替换为上一步查到的目标名称：
     ```sh
     cmake --build build --target <target_name>
     ```

   - 选项 B：编译所有 Target（推荐，全量构建）

     支持多线程加速构建：
     ```sh
     cmake --build build --parallel
     ```

     安装编译产物，将生成的二进制文件整理到 `build_out` 目录：
     ```sh
     cmake --install build --prefix ./build_out
     ```

4. 运行验证

   - 选项A: 运行指定的Target(以vector_add为例)

     上一步将`<target_name>` 替换为`vector_add`编译成功后，编译输出二进制文件在`./build/Samples/0_Introduction/vector_add/`目录下，即编译产物在第一步构建的`build`文件夹下与样例目录对应的位置，执行如下命令运行：
     ```sh
     ./build/Samples/0_Introduction/vector_add/vector_add
     ```
     可以得到结果如下：
     ```
     Vector add completed successfully!
     ```

   - 选项B: 运行全量编译并安装后的matmul用例

     完成第三步的安装后，所有编译生成文件都在`build_out`文件夹下，`matmul`用例的可运行文件在`./build_out/0_Introduction/matmul`目录下，执行如下命令运行：
     ```
     cd ./build_out/0_Introduction/matmul/
     ./matmul 100 50 200
     ```
      运行成功后，终端将打印如下类似信息：
      ```txt
      Data generated successfully!

     [verify] shape(100, 200), elements=20000 - summary (large matrix, full tensors omitted)
     abs_err: max=0.000000e+00, mean=0.000000e-00, rmse=0.000000e+00
     rel_err: max=0.000000e+00
     count(|abs_err| > 0.001): 0 / 20000
     cpu golden (top-left 4x4):
     tensor([[1144., 1088., 1012., 1040.],
           [1104., 1072., 1004., 972.],
           [1056., 968., 888., 984.],
           [1012., 932., 876., 912.]], dtype=torch.bfloat16)
     npu out (top-left 4x4):
     tensor([[1144., 1088., 1012., 1040.],
           [1104., 1072., 1004., 972.],
           [1056., 968., 888., 984.],
           [1012., 932., 876., 912.]], dtype=torch.bfloat16)
     max abs diff: 0.0
     point error count(>0.1): 0/20000
     ratio error count(>0.001): 0/20000, error ratio: 0.000000
      [PASS] NPU results are consistent with CPU.
      ```
      开发者可自行尝试运行`build_out`下的其它用例。

---

## 📂目录结构
```
├── Samples                                  # 样例目录
│   ├── 0_Introduction                       # 入门样例
│   ├── 1_Features                           # 功能特性样例
│   │   ├── memory_optimization              # 访存优化方法
│   │   ├── instruction_optimization         # 指令优化方法
│   │   ├── system_optimization              # 系统优化方法
│   │   └── hardware_features                # 芯片特性样例
│   ├── 2_Performance                        # 性能调优样例
│   │   ├── matmul_story                     # 矩阵乘性能优化实践
│   │   ├── grouped_matmul_story             # 分组矩阵乘性能优化实践
│   │   └── ...                              # 其它性能调优样例
│   └── CMakeLists.txt
├── third_party                              # 外部依赖（Git 子模块）
│   ├── ops-tensor                          # ops-tensor：Ascend C Tensor API 头文件
│   ├── shmem                                # 共享内存相关组件
│   └── ...                                  # 其它第三方依赖
├── cmake                                    # 工程编译配置
├── .clang-format                            # 代码格式配置
├── CMakeLists.txt                           # 根 CMake 配置
├── LICENSE                                  # 许可证
├── SECURITY.md                              # 安全声明
└── README.md                                # 项目说明文档
```



## 💬相关信息

- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-basic)

## 🤝联系我们

本项目的功能与文档会持续更新。

- **问题反馈**：通过 GitCode [Issues](https://gitcode.com/cann/cann-samples/issues) 提交问题
- **社区互动**：通过 GitCode [Discussions](https://gitcode.com/cann/cann-samples/discussions) 参与交流

---

<p align="center">
  <sub>Made with ❤️ by the CANN Team · <a href="https://gitcode.com/cann">More CANN Projects</a></sub>
</p>
