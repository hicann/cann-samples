# cann-samples

## 🔥Latest News
- [2026/03] ops-samples更名为cann-samples。
- [2026/02] ops-samples项目上线，提供算子领域高性能实战演进样例与体系化调优知识库。

## 🚀概述

`cann-samples` 是 [CANN](https://hiascend.com/software/cann)（Compute Architecture for Neural Networks）算子领域的实战样例仓库，提供高性能实现示例与体系化调优知识库。

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
