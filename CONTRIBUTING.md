# 贡献指南

本文档说明 `cann-samples` 的贡献范围、目录约束、本地验证要求和 Pull Request 提交流程。

首次贡献前，建议先阅读以下章节：

1. [仓库定位与贡献范围](#仓库定位与贡献范围)
2. [Sample 目录结构规范](#sample-目录结构规范)
3. [README 和教程必须包含的内容](#readme-和教程必须包含的内容)
4. [构建、验证与提交前检查](#构建验证与提交前检查)

## 目录

- [仓库定位与贡献范围](#仓库定位与贡献范围)
- [开始之前](#开始之前)
- [社区协作与沟通](#社区协作与沟通)
- [仓库结构](#仓库结构)
- [Sample 目录结构规范](#sample-目录结构规范)
- [README 和教程必须包含的内容](#readme-和教程必须包含的内容)
- [贡献流程](#贡献流程)
- [编码与文档规范](#编码与文档规范)
- [构建、验证与提交前检查](#构建验证与提交前检查)
- [Pull Request 要求与评审关注点](#pull-request-要求与评审关注点)
- [Issue 反馈](#issue-反馈)
- [安全与许可证](#安全与许可证)

---

## 仓库定位与贡献范围

接受以下类型的贡献：

- 新增可独立构建、可验证结果的样例。
- 对现有样例进行问题修复、性能优化或可读性改进。
- 完善 README、tutorial、图示、验证脚本和构建说明。
- 补充通用脚本、公共组件或工程化改进，但必须服务于样例使用场景。

以下内容不应提交：

- 只有结论、没有可复现代码或验证过程的“经验总结”。
- 与样例无关的大规模基础设施改造。
- 仅适用于个人环境的绝对路径、本地脚本、私有依赖或临时调试代码。
- 构建产物、压缩包、日志、核心转储、下载文件等生成物，例如 `build/`、`build_out/`、`*.zip`。

提交前请确认改动符合以下目录定位：

- `0_Introduction`：介绍基础知识，建立基本概念，补全从入门到精通过程中的知识空缺。
- `1_Features`：介绍关键特性，解耦大模型底层算子能力，包括公共优化技巧和关键芯片特性。
- `2_Performance`：面向典型性能问题，展示专题化优化方法、演进过程和设计取舍。

跨多个主题的改动应拆分为独立 PR。不要在同一个 PR 中混合提交新样例、重构和大规模文档调整。

## 开始之前

### 环境要求

本仓库要求以下环境：

- 已安装社区版 CANN Toolkit，并正确配置 `ASCEND_HOME_PATH`
- CANN Toolkit 版本要求与 [README.md](README.md) 保持一致。请优先使用 README 中说明的最新版本社区包。
- `CMake >= 3.16`
- `Python >= 3.10`
- `GCC >= 11.3.0`
- `clang-format`
- `requirements.txt` 中声明的 Python 依赖
- CMake 可发现的 `ASC` 工具链包

### 兼容范围声明

每个样例不需要覆盖全部硬件和场景，但每个贡献都必须声明以下内容：

- 已验证的硬件型号或架构
- 已验证的 CANN 版本
- 支持的数据类型、shape 约束和已知限制
- 未覆盖但可能被用户误用的场景

以上信息必须同时出现在样例 `README.md` 和 PR 描述中。

### Fork 与远程配置

在 GitCode 上 Fork [cann/cann-samples](https://gitcode.com/cann/cann-samples)，然后配置远程：

```bash
git clone https://gitcode.com/<your-username>/cann-samples.git
cd cann-samples
git remote add upstream https://gitcode.com/cann/cann-samples.git
```

提交前同步最新主线：

```bash
git fetch upstream
git checkout master
git rebase upstream/master
```

## 社区协作与沟通

提交 Issue、PR 或参与讨论时，请聚焦问题本身。

- 优先提供事实、复现步骤、边界条件和数据。
- 对评审意见有异议时，请给出代码、日志、数据或文档依据。
- 若社区已有统一行为规范或讨论规范，请以该规范为准。

## 仓库结构

```text
cann-samples/
├── Samples/
│   ├── 0_Introduction/     # 基础知识与入门样例
│   ├── 1_Features/         # 关键特性与能力演示样例
│   └── 2_Performance/      # 性能专题与优化演进样例
├── cmake/                  # 工具链与公共 CMake 配置
├── .ci/                    # CI 相关脚本
├── .gitcode/               # Issue / PR 模板
├── CMakeLists.txt          # 根工程构建入口
├── README.md               # 仓库总览
└── CONTRIBUTING.md         # 贡献指南
```

## Sample 目录结构规范

新增样例应使用模板 A、B 或 C 之一。

- 历史样例可以在既有结构内演进，但新增目录和新增专题应优先对齐本节模板。
- 同一层级不要混用无语义命名。
- 新增样例必须同时接入对应父级 `CMakeLists.txt`。
- 非标准目录结构只应用于收益明确的场景，并在 PR 中说明原因及其与模板 A/B/C 的对应关系。

模板选择规则：

- 同时提供“可直接复用的最佳实践变体”和“按步骤展开的教程路径”时，优先使用模板 C。
- 样例围绕单一主题按优化阶段展开，且需要在 `src/` 中平铺维护多个阶段实现时，使用模板 B。
- 当样例代码可以直接平铺在样例根目录下，不需要稳定拆分 `src/`、`include/`、`common/` 等职责目录时，使用模板 A；即使根目录下存在多个用于演示或对比的 `.cpp` 文件，也不必因此升级为模板 B。

### 模板 A：简单 Sample

适用于源文件少、逻辑集中的样例，例如 `vector_add` 一类最小可运行示例。

```text
<sample_name>/
├── CMakeLists.txt                # 必须，构建配置
├── <demo_a>.cpp                  # 必须，演示源码
├── <demo_b>.cpp                  # 可选，并列对比的演示源码
├── <helper>.h                    # 可选，平铺放置的辅助头文件
├── README.md                     # 必须，样例说明文档
├── images/                       # 可选，README 引用的图片资源
└── scripts/                      # 可选，数据生成和结果验证脚本
    ├── gen_data.py
    └── verify_result.py
```

适用条件：

- 样例源码可以直接平铺在根目录下，不需要再拆出 `src/`、`include/`、`common/` 等稳定职责目录。
- 允许存在多个并列的 demo `.cpp` 文件，用于展示不同写法、不同参数或不同实现之间的对比。
- 可以有少量平铺放置的辅助 `.cpp` / `.h` 文件，但不应继续演化出独立模块层级。
- 验证逻辑简单，可通过脚本或内嵌校验完成。
- 样例目录名、目标名、README 标题应保持可映射，不应出现三套互不对应的命名。

### 模板 B：复杂 Sample

适用于围绕单一主题按优化阶段展开、需要集中展示基线实现与后续优化阶段的样例。

```text
<sample_name>/
├── CMakeLists.txt                # 必须，构建配置
├── README.md                     # 必须，样例说明文档
├── src/                          # 必须，源码目录
│   ├── 0_naive.cpp               # 必须，基线实现
│   ├── 1_<stage>.cpp             # 可选，第一阶段优化实现
│   └── 2_<stage>.cpp             # 可选，后续阶段优化实现
├── include/                      # 可选，业务代码对应的头文件
├── scripts/                      # 可选，数据生成和结果验证脚本
├── common/                       # 可选，工具类或公共辅助代码
├── images/                       # 可选，README/docs 引用的图片资源
└── docs/                         # 可选，README 之外的补充文档
```

目录约束：

- `src/` 采用按阶段编号的平铺命名，文件名格式为 `<index>_<stage>.cpp`，例如 `0_naive.cpp`、`1_multi_core.cpp`、`2_double_buffer.cpp`。编号应连续，并与 README 的讲解顺序一致。
- `include/`、`common/`、`scripts/`、`images/`、`docs/` 是该类内容的标准目录名。没有对应内容时可以省略；存在对应内容时应使用这些目录名。
- `common/` 仅用于存放具有明确复用价值的工具类或公共辅助代码，不用于承载单个阶段私有实现。
- `docs/` 仅承载 README 无法容纳的补充内容，不重复维护 README 已覆盖的信息。
- 存在共享头文件、工具函数、验证脚本或验证数据时，应在 README 中说明其入口、适用范围和依赖关系，避免形成隐式依赖。

### 模板 C：专题型 Story Sample

适用于同一算子主题下，同时提供最佳实践与演进教程的专题样例。新建专题应直接采用下述结构；历史专题样例若暂时无法完全迁移，也必须满足本节列出的职责划分、命名规则和共享资源约束。

```text
<sample_name>_story/
├── CMakeLists.txt                    # 必须，顶层构建配置
├── README.md                         # 必须，专题总览
│
├── <sample_name>_recipes/            # 必须，最佳实践代码
│   ├── CMakeLists.txt                # 顶层，add_subdirectory 各变体
│   ├── README.md                     # 变体总览
│   ├── include/                      # recipes 内共享的业务头文件
│   ├── common/                       # recipes 内共享的公共辅助代码
│   └── <variant>/                    # 每个变体一个子目录
│       ├── CMakeLists.txt            # 当前变体的构建配置
│       ├── <variant>.cpp             # 当前变体的实现代码
│       ├── images/                   # 当前变体 README 引用的图片资源
│       ├── scripts/                  # 当前变体的数据生成和结果验证脚本
│       │   ├── gen_data.py
│       │   └── verify_result.py
│       ├── README.md                 # 当前变体的使用说明
│       └── tutorial.md               # 当前变体的设计说明或实现解读
│
└── <sample_name>_tutorials/          # 必须，演进教程代码
    ├── CMakeLists.txt                # 顶层，add_subdirectory 各 step
    ├── README.md                     # 教程总述
    ├── include/                      # tutorials 内共享的业务头文件
    ├── common/                       # tutorials 内共享的公共辅助代码
    ├── scripts/                      # 教程公共的数据生成和结果验证脚本
    │   ├── gen_data.py
    │   └── verify_result.py
    ├── images/                       # 教程总述或阶段文档引用的图片资源
    ├── docs/                         # 补充教程说明、图示或阶段说明
    ├── 0_naive/
    │   ├── CMakeLists.txt            # 当前阶段的构建配置
    │   ├── naive.cpp                 # 基线实现
    │   └── include/                  # 当前阶段的业务头文件
    ├── 1_multi_core/
    │   ├── CMakeLists.txt            # 当前阶段的构建配置
    │   ├── multi_core.cpp            # 第一阶段优化实现
    │   └── include/                  # 当前阶段的业务头文件
    └── 2_double_buffer/
        ├── CMakeLists.txt            # 当前阶段的构建配置
        ├── double_buffer.cpp         # 后续阶段优化实现
        └── include/                  # 当前阶段的业务头文件
```

适用场景：

- 同一主题下存在多个实现变体。
- 需要同时满足“直接复用代码”和“理解优化过程”两类读者。
- 需要给出逐步优化步骤、性能数据和设计取舍。

两条路径的职责必须清晰区分：

| 维度 | `*_recipes/` | `*_tutorials/` |
|------|------|------|
| 目标读者 | 有经验开发者 | 入门或中级开发者 |
| 主要目的 | 直接复用或二次开发 | 理解优化过程和原理 |
| 内容组织 | 各变体一个目录 | 按编号 step 组织 |
| 文档侧重点 | 怎么用、适合什么场景 | 为什么这样做、每一步改了什么 |

结构约束：

- 顶层必须包含专题总览 `README.md`、顶层 `CMakeLists.txt`、`*_recipes/` 和 `*_tutorials/`。
- `*_recipes/` 只存放可直接复用或对比的实现变体；`*_tutorials/` 只存放按步骤展开的教学代码。
- `include/`、`common/`、`scripts/`、`images/`、`docs/` 是专题内对应内容的标准目录名；存在对应内容时应使用这些目录名。
- `*_recipes/` 与 `*_tutorials/` 相互独立，不共享代码、脚本、图片或验证数据。
- 共享 `common/` 只允许位于 `*_recipes/` 或 `*_tutorials/` 各自目录内，不得上提到专题顶层。
- 变体或阶段可以引用所在目录层级内的公共代码。
- 不允许不同 sample 目录之间相互复用代码、脚本或验证数据。

### 验证方式

以下两种验证方式都可以使用，但结果必须可复现。

**方式一：Python 验证脚本**

适用于生成输入、调用框架计算标杆结果或进行批量验证的场景。

```bash
python3 scripts/gen_data.py
./<executable>
python3 scripts/verify_result.py
```

要求：

- 明确输入规模、数据类型和随机种子。
- 明确容差参数，例如 `atol`、`rtol`。
- 输出清晰的 `PASS` / `FAIL` 结论。
- 验证失败时必须返回非零退出码，并打印可定位问题的关键信息。
- 标杆输入、临时输出、性能测试原始数据应通过脚本生成，不直接提交大文件或本地导出结果。

**方式二：C++ 内嵌验证**

适用于简单、无额外依赖且标杆结果计算容易表达的样例。

要求：

- 不要把大量测试数据硬编码进源码。
- 失败时输出足够的定位信息，例如索引、期望值、实际值。

### 新增样例接入清单

- [ ] 补齐本目录 `CMakeLists.txt`
- [ ] 补齐所属分组父目录 `add_subdirectory(<sample_name>)`
- [ ] 如引入专题子层级，补齐各层 `CMakeLists.txt`
- [ ] 确保 `cmake --build build --target help` 可见对应目标
- [ ] 确保 README 中的目标名、可执行名、运行命令与实际构建结果一致

## README 和教程必须包含的内容

每个样例的 `README.md` 至少包含以下内容：

- **功能简介**：1 到 3 句话说明样例目标。
- **支持范围**：硬件架构、数据类型、shape 约束、输入限制。
- **目录说明**：关键文件的作用。
- **构建与运行**：完整命令。
- **验证方式**：如何获得标杆结果或判定依据、如何检查结果。
- **预期输出**：关键日志或结果示例。
- **性能说明**：性能样例需说明主要优化点与收益。

### Story / Tutorial 要求

Story / Tutorial 文档除 README 最小内容外，还应包含以下章节：

1. **引言**
   - 说明待优化算法的背景、计算公式或计算流图。
   - 说明该算子或算法的重要性，以及典型应用场景或代表性网络。
   - 说明优化目标，例如吞吐、时延或接近硬件理论峰值的目标。
   - 说明测试所用硬件环境及关键规格。

2. **硬件架构基础**
   - 只介绍与当前优化直接相关的硬件概念。
   - 必要时引用 `0_Introduction` 或 `1_Features` 中的相关内容。
   - 给出用于后续对比的理论上限，例如 Roofline Model。

3. **基准实现**
   - 指向基线代码文件，例如 `0_naive.cpp`。
   - 说明实现原理及其功能正确性。
   - 给出运行结果、性能测试结果和 profiling 数据。
   - 基于理论分析或 profiling 数据说明主要瓶颈。
   - 提供必要图示，说明数据流、访存路径或瓶颈位置。

4. **各优化阶段**
   - 每个优化阶段单独成节，标题与阶段文件名保持对应，例如“优化阶段一：Multi Core”对应 `1_multi_core.cpp`。
   - 每个阶段均应说明修改内容、设计动机、性能结果和瓶颈变化。
   - 每个阶段均应提供必要图示，用于说明数据流、并行策略、访存变化或关键实现差异。

5. **总结与最终结果**
   - 提供从基线实现到最终版本的性能对比图表。
   - 提供与加速库版本或参考实现的性能对比。
   - 说明最终结果达到理论上限的比例。
   - 说明后续可继续优化的方向。

涉及性能结论的 Story / Tutorial 文档，必须写清以下上下文：

- 硬件型号及关键规格
- CANN 版本
- 输入规模
- 数据类型
- 测量方法

## 贡献流程

### 1. 创建分支

基于最新 `master` 创建独立分支：

```bash
git checkout -b <branch-name>
```

建议使用以下分支命名方式：

| 类型 | 格式 | 示例 |
|------|------|------|
| 新样例 | `sample/<sample_name>` | `sample/softmax` |
| 问题修复 | `fix/<description>` | `fix/matmul-shape-check` |
| 文档修改 | `docs/<description>` | `docs/contributing-guide` |
| 性能优化 | `perf/<description>` | `perf/matmul-tile-opt` |

### 2. 实现改动

实现改动时请遵循以下原则：

- 一个样例目录应尽量自洽，避免对其他样例产生隐式依赖。
- 优先保持示例代码直观可读，再考虑过度抽象。
- 公共逻辑只有在多个样例明确复用时才抽取。
- 如果引入限制条件，请在代码和文档里都写清楚。
- 如果修改已有样例的入口、目标名、目录结构或脚本参数，必须在 README 和 PR 中写明兼容性影响；若存在破坏性变更，必须提供迁移说明。

### 3. 本地构建与验证

根目录基础构建方式：

```bash
cmake -S . -B build
cmake --build build --parallel
```

如果 `cmake -S . -B build` 失败，请先检查：

- `ASC` 工具链是否已被 CMake 正确发现
- `ASCEND_HOME_PATH` 是否指向有效安装目录
- 样例是否遗漏父级 `add_subdirectory(...)` 

可选：查看可编译目标

```bash
cmake --build build --target help
```

可选：安装构建产物

```bash
cmake --install build --prefix ./build_out
```

如果 PR 只改动单个样例，请至少验证：

- 根工程可以成功配置。
- 受影响目标可以成功编译。
- 样例结果验证通过。
- 文档中的命令可以按描述执行。
- 目标已被根工程正式接纳；仅存在目录但未被父级 `CMakeLists.txt` 通过 `add_subdirectory(...)` 接线的内容，不属于仓库正式可构建样例，不应在 README 或 PR 中声明“已支持”。

### 4. 提交代码

建议使用 Conventional Commit 风格：

```text
<type>(<scope>): <subject>
```

常用类型：

- `feat`：新增样例或新增能力
- `fix`：修复功能问题
- `perf`：性能优化
- `docs`：文档更新
- `refactor`：重构但不改变行为
- `test`：测试或验证逻辑调整
- `build`：构建系统或依赖调整

示例：

```text
feat(sample): add softmax example with verification scripts
docs(contributing): clarify review checklist and PR expectations
fix(matmul): reject unsupported shape combinations
```

### 5. 推送并发起 PR

```bash
git push origin <branch-name>
```

然后在 GitCode 上向 `cann/cann-samples` 的 `master` 分支提交 Pull Request。

### 6. 触发并通过 CI 门禁

提交 PR 后，需在 PR 评论区输入 `compile` 触发 CI 门禁。

要求：

- 根据 CI 检测结果修复构建、验证、格式或文档问题。
- 修复后重新推送分支，并再次触发门禁，直至 CI 通过。
- 在 PR 描述或评论中同步关键修复内容和当前状态。

### 7. 响应 Committer 检视

CI 通过后，等待 Committer 检视。

要求：

- 根据 Committer 检视意见继续修改代码、文档或构建配置。
- 修改完成后重新推送分支，并在 PR 中同步更新说明。
- 修改完成后 `@` 指派的 Committer，通知其继续检视。

### 8. Maintainer 最终审核与合入

Committer 检视通过后，PR 会标注 `/lgtm` 标签。

要求：

- 等待 Maintainer 在 1 天内完成最终审核。
- Maintainer 确认无问题后，会标注 `/approve` 标签并合入 PR。

## 编码与文档规范

### 代码风格

项目使用根目录 [`.clang-format`](.clang-format) 统一格式。提交前请格式化变更文件：

```bash
clang-format -i --style=file <source_files>
```

主要规则如下：

| 项目 | 要求 |
|------|------|
| 基础风格 | `BasedOnStyle: Google` |
| 缩进 | 4 空格，禁止 Tab |
| 行宽 | 120 |
| 指针对齐 | 左对齐，例如 `int* ptr` |
| 大括号 | 函数定义换行 |
| include 顺序 | 不强制自动重排 |

### 命名与可读性

- 目录名、样例名使用小写加下划线。
- 变量、函数、类型命名保持与所在目录既有风格一致。
- 避免使用无含义缩写，尤其在教程类样例中。
- 注释应解释“为什么这样做”，不要机械复述代码。

### 许可证声明

新增源文件、头文件、CMake 文件请补齐许可证头。格式可参考现有文件。

### 文档规范

- 所有命令示例默认以仓库根目录为起点，若不是，请显式说明。
- 文档中的路径、目标名、脚本名必须与仓库实际内容一致。
- 示例输出不要伪造；如果有简化，需说明是示意输出。
- 若复用了公共资源目录，例如共享标杆数据、公共辅助代码或共享脚本，README 必须写明其位置和用途。
- 如果新增脚本或流程会稳定产生输出文件，应同步更新 `.gitignore` 或在文档中明确约定输出目录。

## 构建、验证与提交前检查

提交前请至少检查以下内容：

- [ ] 改动范围与仓库定位一致，没有顺手夹带无关修改。
- [ ] 代码已经按 `.clang-format` 格式化。
- [ ] 根工程可成功 `cmake -S . -B build`。
- [ ] 受影响目标能够成功编译。
- [ ] 新增样例已正确接入父级 `CMakeLists.txt`，根工程能够发现目标。
- [ ] 样例结果验证通过，且验证过程可复现。
- [ ] README / tutorial / 图片 / 脚本已经与代码同步更新。
- [ ] 明确写出支持范围、限制条件和已知前提。
- [ ] 没有提交 `build/`、`build_out/`、压缩包、日志、下载包、临时标杆输出等生成物。
- [ ] 新增脚本或流程产生的稳定输出已通过 `.gitignore` 或目录约定隔离。
- [ ] 提交信息符合约定，PR 描述能说明改动原因和测试结果。

如果改动涉及性能结论，请额外检查：

- [ ] 性能数据包含测试环境与输入条件。
- [ ] 优化收益可复现，不是偶然结果。
- [ ] 同时说明收益和代价，例如可读性、适用范围或资源占用变化。

## Pull Request 要求与评审关注点

PR 模板位于 [`.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`](.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md)。

PR 至少应包含以下信息：

- 影响范围：改动了哪个样例、哪个目录、哪些目标。
- 改动动机：是新增能力、修复问题，还是补全文档。
- 验证命令：直接给出你本地执行过的命令。
- 结果摘要：构建结果、验证结果、性能结果。
- 限制说明：未覆盖的硬件、shape、数据类型或后续工作。

代码类 PR 还必须补充：

- 已验证的平台、CANN 版本和关键依赖版本
- 目录结构是否完全符合模板；如有例外，给出映射关系和原因
- 如涉及破坏性变更，给出迁移说明

建议补充以下信息：

- 关键设计取舍或替代方案说明
- 典型日志、截图或性能图表
- 未纳入本次 PR 的后续工作说明

评审通常重点关注以下内容：

- 改动是否与仓库定位一致。
- 样例是否完整，能否独立构建和验证。
- 文档是否足够让其他开发者复现。
- 代码是否清晰，是否保持样例的教学价值。
- 性能结论是否有数据支撑，且上下文完整。

以下问题通常需要在评审前补齐：

- README 只写背景，不写运行方法。
- 只给构建命令，不给验证命令。
- 性能图表没有测试条件。
- 新增脚本但没有说明依赖和入口。
- PR 描述只写“优化代码”或“修复问题”，缺少上下文。

## Issue 反馈

仓库已提供以下模板：

- Bug： [`.gitcode/ISSUE_TEMPLATE/bug-report.yml`](.gitcode/ISSUE_TEMPLATE/bug-report.yml)
- 需求： [`.gitcode/ISSUE_TEMPLATE/feature-request.yml`](.gitcode/ISSUE_TEMPLATE/feature-request.yml)
- 文档： [`.gitcode/ISSUE_TEMPLATE/documentation.yml`](.gitcode/ISSUE_TEMPLATE/documentation.yml)
- 咨询： [`.gitcode/ISSUE_TEMPLATE/question.yml`](.gitcode/ISSUE_TEMPLATE/question.yml)

提交 Issue 前，请先搜索已有的 Issue、PR 和 Discussion；提问前请先阅读根 README 和对应样例文档。

提交 Issue 时，至少提供以下信息：

- 使用的硬件型号和架构
- CANN 版本
- 操作系统与关键依赖版本
- 复现步骤
- 期望结果与实际结果
- 错误日志、截图或最小复现样例

如果问题只在特定输入下出现，请写清输入规模、数据类型和约束条件。

## 安全与许可证

- 请勿提交账号、口令、Token、许可证文件或任何敏感配置。
- 请勿在文档中泄露内网地址、私有镜像源或本地路径。
- 发现安全问题时，请优先参考 [`SECURITY.md`](SECURITY.md) 的说明处理。
- 本仓库基于 [CANN Open Software License Agreement Version 2.0](LICENSE) 开源，贡献内容需与仓库许可证兼容。
