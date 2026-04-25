# cann-samples

## 🔥Latest News
- [2026/03] ops-samples更名为cann-samples。
- [2026/02] ops-samples项目上线，提供算子领域高性能实战演进样例与体系化调优知识库。

## 🚀概述

`cann-samples` 是 [CANN](https://hiascend.com/software/cann)（Compute Architecture for Neural Networks）算子领域的实战样例仓库，提供高性能实现示例与体系化调优知识库。

本仓已集成代码仓库智能体，点击 [![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/hicann/cann-samples) 徽章，进入其专属页面，开启在线智能代码学习与知识问答体验！

## 📝环境部署

当前仓库已验证通过的社区版 CANN Toolkit 如下：

| CANN 版本 | 时间戳 | 验证结果 | 下载链接 |
| --- | --- | --- | --- |
| `9.0.0` | `20260325000325538` | ✅ PASS | [aarch64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260325000325538/aarch64/) / [x86_64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260325000325538/x86_64/) |

请根据实际 CPU 架构，从上述链接目录中自行选择对应的 `.run` 安装包。

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

## ⚡️快速入门

1. 配置项目

   使用以下命令初始化构建配置，CMake 会自动创建 `build` 目录：
   ```sh
   cmake -S . -B build
   ```

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
     ./build_out/0_Introduction/matmul/matmul 100 50 200
     ```
     可以得到结果如下：
     ```
     matmul run successfully!
     ```
     开发者可自行尝试运行`build_out`下的其它用例。

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
│   └── CMakeLists.txt
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
