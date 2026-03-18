# cann-samples

## 🔥Latest News
- [2026/03] ops-samples更名为cann-samples。
- [2026/02] ops-samples项目上线，提供算子领域高性能实战演进样例与体系化调优知识库。

## 🚀概述

cann-samples是[CANN](https://hiascend.com/software/cann)（Compute Architecture for Neural Networks）算子库中提供算子领域高性能实战演进样例与体系化调优知识库。

## 📝环境部署

**请获取最新版本的CANN toolkit包以获得完整的体验。**

单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)，根据实际环境架构，获取```Ascend-cann-toolkit_${cann_version}_linux-${arch}.run```。

1. **安装社区CANN toolkit包**

    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
    ```
    - \$\{cann\_version\}：表示CANN包版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
    - \$\{install\_path\}：表示指定安装路径，默认安装在`/usr/local/Ascend`目录。

## ⚡️快速入门

1. 配置项目

   使用以下命令初始化构建配置，CMake会自动创建`build`文件夹
   ```sh
   cmake -S . -B build
   ```

2. 查看可用的Target(可选)

   在编译前，可查看当前项目中所有支持单独编译的目标列表
   ```sh
   cmake --build build --target help
   ```

3. 执行编译安装

   - 选项A: 编译指定的Target(部分构建)

     将`<target_name>` 替换为上一步查到的名称：
     ```sh
     cmake --build build --target <target_name>
     ```

   - 选项B: 编译所有Target(推荐，全量构建)

     支持多线程加速：
     ```sh
     cmake --build build --parallel
     ```

     安装编译产物：

     执行安装命令，将编译生成的二进制文件整理到`build_out`文件夹下：
     ```sh
     cmake --install build --prefix ./build_out
     ```

## 📂目录结构

```
├── Samples                        # 样例目录
│   ├── 0_Introduction            # 入门介绍样例
│   ├── 1_Features                # 功能特性样例
│   │   ├── memory_optimization   # 访存优化方法
│   │   ├── instruction_optimization # 指令优化方法
│   │   ├── system_optimization   # 系统优化方法
│   │   └── hardware_features     # 芯片特性样例
│   ├── 2_Performance             # 性能调优样例
│   └── CMakeLists.txt
├── cmake                         # 项目工程编译目录
├── .clang-format                 # 代码格式配置
├── CMakeLists.txt                # 项目根编译配置
├── LICENSE                       # 许可证
├── SECURITY.md                   # 安全声明
└── README.md                     # 项目说明文档
```

当前 `1_Features` 下已有样例统一归类在 `hardware_features`，其余分类目录预留用于后续扩展。

## 💬相关信息

- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-basic)

## 🤝联系我们

本项目功能和文档正在持续更新和完善中，欢迎您关注最新版本。

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/cann-samples/issues)提交问题
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/cann-samples/discussions)参与交流
