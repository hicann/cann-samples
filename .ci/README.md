# CI 使用说明

## 定位

本文件只说明 `.ci/` 下执行入口的用途和调用方式。
测试清单字段、补充规则、清单维护方式统一见 [`tests/README.md`](../tests/README.md)。

## 入口

- `build.sh`：构建、安装、打包 `build_out`
- `run_ci_functional.sh`：执行功能测试 CI
- `run_ci_functional.py`：manifest 驱动执行器
- 清单文件：[`tests/ci_functional_test.yaml`](../tests/ci_functional_test.yaml)

## 前提

```bash
pip install -r requirements.txt
pip install -r .ci/requirements.txt
```

要求当前机器可正常使用 `cmake`、`python3`、CANN 工具链和设备运行时。

## 命令

仅构建：

```bash
bash .ci/build.sh
```

执行完整功能测试：

```bash
bash .ci/run_ci_functional.sh
```

只跑一个 sample：

```bash
bash .ci/run_ci_functional.sh --sample vector_add
```

在支持 `dav-2201` 的环境中，可通过环境变量选择构建架构（默认 `dav-3510`）：

```bash
NPU_ARCH=dav-2201 bash .ci/run_ci_functional.sh --sample vector_add
```

覆盖清单路径或产物目录：

```bash
MANIFEST_PATH=tests/ci_functional_test.yaml \
ARTIFACTS_DIR=test_artifacts/ci_functional \
bash .ci/run_ci_functional.sh
```

已手动完成 build/install 后，直接执行 runner：

```bash
python3 .ci/run_ci_functional.py --manifest tests/ci_functional_test.yaml
```

## 输出

- 默认目录：`test_artifacts/ci_functional/`
- 输出内容：`summary.json`、每一步 `stdout`、每一步 `stderr`、每一步退出码
