---
name: pixi-usage
description: >
  询问 pixi 用法时使用。
  适用于环境管理、依赖安装、任务运行、shell 交互、清理环境等。
---

本技能记录 SPONGE 项目中 pixi 的使用方式和注意事项。当前使用的 pixi 版本为 0.65.0。

## pixi 是什么

pixi 是基于 conda 生态的开发工作流工具，用于管理依赖、环境和任务。SPONGE 项目通过 `pixi.toml` 定义所有环境和任务，取代手动 conda/mamba 操作。

## 安装 pixi

### Linux / macOS

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

SPONGE 镜像（国内推荐）：

```bash
curl -fsSL https://conda.spongemm.cn/pixi/install.sh | bash
```

安装脚本会将 pixi 放到 `~/.pixi/bin/` 并自动添加到 shell 的 PATH 中。

### Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://pixi.sh/install.ps1 | iex"
```

SPONGE 镜像（国内推荐）：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://conda.spongemm.cn/pixi/install.ps1 | iex"
```

### 更新 pixi

```bash
pixi self-update              # 更新到最新版本
pixi self-update --version 0.65.0   # 更新到指定版本
```

## 核心概念

### 环境（environment）

环境由 feature 组合而成，在 `pixi.toml` 的 `[environments]` 中定义。每个环境有独立的依赖集和任务集。环境安装在 `.pixi/envs/<环境名>/` 下。

大多数 pixi 命令通过 `-e <环境名>` 指定环境，不指定时使用 `default` 环境。

### Feature

Feature 是可组合的依赖和任务单元，环境由一个或多个 feature 组成。例如 `dev-cuda13 = ["dev", "cuda", "cuda13"]` 组合了开发工具、CUDA 通用依赖和 CUDA 13 特有依赖。

### 任务（task）

任务在 `pixi.toml` 中定义，通过 `pixi run` 执行。任务可以是简单字符串命令，也可以带参数模板。

## 常用命令

### 安装环境

```bash
pixi install -e dev-cuda13      # 安装指定环境的依赖
pixi install -a                 # 安装所有环境
```

`pixi run` 会自动触发 install，但显式执行可以提前发现依赖问题。

### 运行任务

```bash
pixi run -e dev-cuda13 configure    # 在指定环境中运行任务
pixi run -e dev-cuda13 compile      # 编译
pixi run -e dev-cuda13 perf-amber   # 运行 benchmark
```

### 任务参数传递

pixi task 支持通过 `{{ arg_name }}` 模板定义参数。以 SPONGE 项目中的 task 为例：

```toml
# pixi.toml 中的定义
configure = { cmd = "cmake ... -DPARALLEL='{{ parallel }}'", args = [{ arg = "parallel", default = "none" }] }
compile = { cmd = "cmake --build ... --parallel {{ jobs }}", args = [{ arg = "jobs", default = "4" }] }
```

调用时，参数按位置直接跟在 task 名后面传值：

```bash
pixi run -e dev-cpu configure avx2      # avx2 替换 {{ parallel }}
pixi run -e dev-cuda13 compile 8        # 8 替换 {{ jobs }}
pixi run -e dev-cuda13 compile          # 不传则使用默认值 4
```

如果 task 定义了多个参数，按定义顺序依次传入。

以下写法都是**错误的**，不要使用：

```bash
pixi run -e dev-cpu configure --parallel avx2    # 错误：被当作额外参数，报错
pixi run -e dev-cpu configure parallel=avx2      # 错误：被当作字面字符串传给 cmake
pixi run -e dev-cpu configure -- avx2            # 错误：-- 后面被当作额外参数
```

注意：以上规则仅适用于定义了 `args` 的 task。如果 task 没有定义 `args`（即简单字符串命令），后面附加的内容会直接拼接到命令末尾：

```toml
# 没有定义 args 的 task
perf-amber = "python -X utf8 -m pytest benchmarks/performance/amber/tests -s --tb=short"
```

```bash
pixi run -e dev-cuda13 perf-amber -k test_rdf
# 实际执行：python -X utf8 -m pytest benchmarks/performance/amber/tests -s --tb=short -k test_rdf
```

### 进入 shell

```bash
pixi shell -e dev-cuda13        # 进入环境的交互式 shell
```

进入后所有环境变量（`PATH`、`CONDA_PREFIX` 等）已设置好，可以直接调用环境内的工具。用 `exit` 退出。

### 在环境中运行任意命令

```bash
pixi run -e dev-cuda13 python script.py       # 运行非 task 的命令
pixi run -e dev-cuda13 pip install -e ./plugins/prips
pixi run -e dev-cuda13 which SPONGE
```

`pixi run` 后面如果不是已定义的 task 名，会当作普通 shell 命令在环境中执行。

### 添加依赖

```bash
pixi add numpy                              # 添加 conda 依赖到 default feature
pixi add numpy -f dev                       # 添加到 dev feature
pixi add numpy -p linux-64 -f dev           # 指定平台
pixi add --pypi xponge -f dev               # 添加 PyPI 依赖
```

添加后会自动更新 lockfile 和 `pixi.toml`。

### 查看信息

```bash
pixi info                          # 显示系统和工作区信息
pixi list -e dev-cuda13            # 列出环境中的包
pixi task list -e dev-cuda13       # 列出可用任务
```

## 环境清理与重建

### 清理环境

```bash
pixi clean -e dev-cuda13           # 清理指定环境
pixi clean                         # 清理所有环境
pixi clean cache                   # 清理全局缓存
```

### 重装环境

```bash
pixi reinstall -e dev-cuda13              # 重装整个环境
pixi reinstall -e dev-cuda13 numpy        # 只重装指定包
```

### 干净环境运行

```bash
pixi run --clean-env -e dev-cuda13 configure    # 隔离系统环境变量运行任务
```

`--clean-env` 会忽略当前 shell 的环境变量，只使用 pixi 环境自身的变量。适用于排查系统环境污染问题。

注意：如果任务本身依赖 pixi 环境外部的工具链（如 HIP/ROCm、MSVC），使用 `--clean-env` 会导致找不到这些工具。

## lockfile 控制

```bash
pixi run --frozen -e dev-cuda13 compile     # 使用 lockfile 原样安装，不更新
pixi run --locked -e dev-cuda13 compile     # 检查 lockfile 是否最新，不一致则报错
```

- `--frozen`：适用于 CI 或确定不想更新依赖时
- `--locked`：适用于确保 lockfile 和 manifest 一致

## pixi.toml 中的 task 定义

### 简单字符串

```toml
format-check = "GITHOOK_MODIFY=0 ... python -X utf8 scripts/pre-commit"
```

### 带参数模板

```toml
compile = { cmd = "cmake --build ... --parallel {{ jobs }}", args = [{ arg = "jobs", default = "4" }] }
```

### 按平台/feature 定义

task 可以定义在不同的 section 下，按平台和 feature 区分：

```toml
[feature.cuda.target.linux-64.tasks]     # CUDA + Linux x86_64
[feature.cpu.target.osx-arm64.tasks]     # CPU + macOS ARM
[feature.dev.tasks]                      # dev feature（跨平台）
```

同名 task 在不同 section 下定义时，实际使用的版本取决于当前环境包含哪些 feature 和当前平台。

## 目录结构

```
.pixi/
  envs/
    dev-cuda13/          # 环境安装目录
    dev-cpu/
    ...
pixi.toml                # 项目配置
pixi.lock                # lockfile
```

`pixi.lock` 应提交到版本控制，确保团队和 CI 使用一致的依赖版本。

## 常见问题

### pixi run 找不到命令

确认命令在指定环境中可用。用 `pixi run -e <env> which <cmd>` 检查。如果是非 dev 环境，可能缺少 python、pytest 等工具。

### 依赖求解失败

检查 `pixi.toml` 中的版本约束是否存在冲突。用 `pixi add <pkg> -f <feature> --no-install` 可以只更新 lockfile 不安装，方便调试。

### lockfile 过期

`pixi.toml` 修改后 lockfile 会自动在下次 run/install 时更新。如果只想更新 lockfile 不安装环境，用 `pixi lock`。
