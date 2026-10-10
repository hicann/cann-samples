# 子模块准备阶段的 Host 集成测试

从仓库根目录执行，需要 Python、Git、CMake 和 CMake 默认构建工具：

```bash
python3 -m unittest discover -s cmake/tests -p test_submodule_update.py -v
```

测试在含空格的临时路径创建本地 Git 子模块，使用实际 shmem/tensor_api
CMake 依赖目标和 `--parallel 2`。Git 代理记录调用区间，并在真实子模块更新前
延迟 0.2 秒以暴露并发；版本查询和所有 Git 操作仍交由系统 Git 执行。
覆盖初始化、重复构建后的锁释放、`.git` 为文件的 linked worktree，以及 Git
失败的非零返回和锁释放。SHMEM fixture 仅提供 Host 安装目标。

没有 Git/CMake 时显式跳过；该测试不验证 CANN Toolkit、ASC 编译、SHMEM
真实代码构建或 NPU 运算。更新锁只协调这两个依赖目标使用的共同入口，外部
手动 Git 命令仍须避免与构建同时修改子模块。锁在 Git common directory 内，
不产生源代码目录中的未跟踪文件；仅包围子模块更新，不覆盖后续 SHMEM 编译。
