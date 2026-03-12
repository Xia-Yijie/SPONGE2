---
name: SPONGE-Development
description: 当用户需要开发本项目（SPONGE）时使用该技能
---

本技能适配`SPONGE`版本号：`2.0.0-alpha`

# pixi

`SPONGE`使用`pixi`管理项目。

`pixi` 是一个快速、可复现、跨平台的项目级包与环境管理工具，能统一管理来自 Conda 与 PyPI 的依赖。

`pixi --version`查看是否安装以及版本。`pixi.toml`是项目的`pixi`配置文件。`SPONGE`推荐使用大于0.65.0版本的`pixi`。

## `pixi`的安装

```shell
# Linux / MacOS
curl -fsSL https://pixi.sh/install.sh | sh
# For users in China (if GitHub connection issues)
curl -fsSL https://conda.spongemm.cn/pixi/install.sh | sh
# Windows powershell
irm -useb https://pixi.sh/install.ps1 | iex
# For users in China (if GitHub connection issues)
irm -useb https://conda.spongemm.cn/pixi/install.ps1 | iex
```

阅读`pixi.toml`了解`SPONGE`配置。如果`pixi`的使用方法存在疑问使用`pixi --help`以及`pixi (命令) --help`了解。

## `SPONGE`的安装与分发

```shell
# 安装依赖
pixi install -e (环境名)
# 存在 conda / pypi 中缺少的依赖时，需要手动额外安装
## 例如对于Windows，需要`pixi run install-msvc`安装`MSVC Build Tools`
## 例如对于AMD显卡、海光DCU，需要自行安装对应的HIP套件
# 进行编译配置
pixi run -e (环境名) configure
# 对于CPU系环境（cpu、cpu-mpi、dev-cpu、dev-cpu-mpi），configure还额外接受指令集参数
## 例如对于AMD64的AVX512指令集，可以使用pixi run configure AVX512
## 例如对于ARM64的SVE指令集，可以使用pixi run configure SVE
## `pixi run configure`默认为无编译器优化版本
## `pixi run configure cpu`可以自动检测本机最优的CPU指令集
# 执行编译
pixi run -e (环境名) compile (编译并行度，默认为4)
# 测试SPONGE
pixi run -e (环境名) SPONGE -v
# 打包SPONGE
pixi run -e (环境名) package
```
