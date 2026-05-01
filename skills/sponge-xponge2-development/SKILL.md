---
name: sponge-xponge2-development
description: >
  询问 SPONGE Xponge/xponge2 C++ 或 Python wheel 开发时使用。
  适用于 assignment、GAFF typing、mol2 IO、topology loader、
  plugins/xponge2、Amber/Gromacs/CHARMM/OPLS 后续支持。
---

本技能适配 SPONGE 版本号：2.0.0-beta.1

本技能用于修改 `SPONGE/xponge/`、`plugins/xponge2/`，或为 xponge2 设计未来可复用的 C++ 能力。

## Core Position

xponge2 是在 SPONGE monorepo 内构建的 Python wheel，不要求 `plugins/xponge2` 目录脱离 SPONGE repo 单独源码构建。`plugins/xponge2` 是 Python 打包层；所有 C++ 实现和 Limited Python C API 绑定都属于 `SPONGE/xponge`，由主程序 CMake target 构建。

核心原则：

- `SPONGE/xponge` 拥有实际实现，包括 assignment、typing、参数加载、topology writer、以及 Python C API interface。
- `plugins/xponge2` 不编译任何 C++ 文件，也不维护自己的 CMake；它只包装已经由 SPONGE CMake 生成并复制进来的 `_core.abi3.so`、Python API、数据文件、`.pyi` 和 `py.typed`。
- `xponge_core` 是 SPONGE 本体 CMake target，当前短期只包含 `assign/`、`model.*`、`forcefield/` 和 `pyinterface.cpp`，不把 `SPONGE/xponge/xponge.cpp` 或 `load/` adapter 拉进 Python extension。
- 以后若需要在 xponge 中编译 CPU 版 SPONGE，可以继续从主程序 CMake target 组织扩展，而不是在 plugin 目录复制源码。

## Boundaries

- 构建边界：`plugins/xponge2` 不拥有 C++ build，也不引入 `py-build-cmake`。使用 `pixi run -e dev-cpu build-xponge` 调主 CMake 构建 `xponge_core`，并复制生成的 `_core.abi3.so` 到 `plugins/xponge2/xponge2/`。使用 `pack-xponge` 先构建 native extension，再用 setuptools 打 wheel。
- 代码边界：优先复用 `SPONGE/xponge`、`SPONGE/common.{h,cpp}`、`SPONGE/utils/*` 和已有 parser/IR。不要为了 plugin 独立性重复实现已经稳定的主程序逻辑。
- 运行边界：wheel 可以来自 SPONGE 本体构建产物，但普通 API 不应依赖外部 `SPONGE` 可执行文件、当前工作目录约定或即时运行主程序，除非该功能明确声明需要这些环境。
- ABI 边界：`std::filesystem` 不应重新进入 xponge2 关键路径；普通 C++ parser、主程序公共工具、CPU SPONGE 相关代码可以按 target 边界复用。CUDA/HIP/MPI 依赖只在明确需要对应功能时引入。

## Layout Guidance

- `SPONGE/xponge/assign/` 应保持接近独立 core：标准库、本目录内部依赖，以及 `SPONGE/utils/control/{string,file}.hpp` 等纯 header-only stdlib 工具；不能依赖 Python、`CONTROLLER`、GPU/MPI 或 SPONGE 主程序运行状态。
- `SPONGE/xponge/pyinterface.cpp` 是 Limited Python C API 包装和异常转换入口，可以包含 `Python.h`，但不要把 Python API 泄漏到 `assign/`、`model.*` 或 force-field core headers。
- `plugins/xponge2/src/` 不应再存放 C++ 源文件；如果看到新的 C++ 被加到 plugin 目录，应移动到 `SPONGE/xponge` 并纳入 `xponge_core` target。
- `SPONGE/xponge/load/` 可以作为 SPONGE adapter 使用 `CONTROLLER`。如果未来 xponge2 也要 load Gromacs/Amber/native，优先把可复用 parser/IR 放进不依赖 `CONTROLLER` 的 core 层，再由 SPONGE adapter 和 Python wrapper 分别调用。
- force-field 特异逻辑优先放到清楚的命名空间或目录中，例如 GAFF/Amber、CHARMM、OPLS 分开；公共 mol2、ring、bond-order、SMART-like matching 能力保留在 shared assign/core 层。

## Build Flow

开发或打包 xponge2 时优先使用：

```bash
pixi run -e dev-cpu build-xponge
pixi run -e dev-cpu pack-xponge
```

`build-xponge` 应：

1. 通过主仓库 CMake 配置 `TARGETS=xponge_core`。
2. 构建 `cmake/targets/xponge_core.cmake`。
3. 从构建目录复制 `_core.abi3.so` 到 `plugins/xponge2/xponge2/_core.abi3.so`。

`pack-xponge` 应先执行同样的 native build/copy，再构建 wheel。不要重新添加 `plugins/xponge2/CMakeLists.txt`、plugin-local C++ build step 或 `py-build-cmake` 配置。

## Review Checklist

修改 xponge2 相关代码时检查：

1. Python wheel 是否仍能在 monorepo 内构建。
2. C++ 文件是否都在 `SPONGE/xponge`，且已纳入 `xponge_core` 或明确属于主程序 adapter。
3. `plugins/xponge2` 是否仍只包含 Python 包装、数据、typing stubs、打包配置和复制进来的 native artifact。
4. 新 parser 或 typing 逻辑是否放在可被 Python wrapper 和 SPONGE adapter 共同复用的位置。
5. 新依赖是否合理地通过 SPONGE 本体 CMake target 引入，而不是在 plugin CMake 里手写源码或编译选项。
6. `SPONGE/xponge/assign` 是否仍保持低依赖。
7. C++ ABI 风险是否被控制，尤其避免重新引入 `std::filesystem` 到 xponge2 关键路径。
8. API 是否尽量保持原 Python 版 Xponge 包的接口语义。
