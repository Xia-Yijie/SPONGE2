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

xponge2 是在 SPONGE monorepo 内构建的 Python wheel，不要求 `plugins/xponge2` 目录脱离 SPONGE repo 单独源码构建。优先复用 SPONGE 主程序已有代码，但要控制 wheel 的运行时依赖和 ABI 风险。

## Boundaries

- 构建边界：`plugins/xponge2` 是 Python wheel 入口，构建上下文是整个 SPONGE repo。不要为了独立 sdist 复制 `SPONGE/xponge` 源码；只有在明确要支持离仓库源码构建时才改变这个边界。
- 代码边界：优先复用 `SPONGE/xponge`、`SPONGE/common.{h,cpp}` 和已有 parser/IR。不要为了“未来可能析出”重复实现已经稳定的主程序逻辑。
- 运行边界：wheel 安装后不应依赖 SPONGE 可执行文件、mdin/mdout、全局 `CONTROLLER` 状态、当前工作目录约定、GPU/MPI runtime，除非该功能明确声明需要这些环境。
- ABI 边界：谨慎把 CUDA/HIP/MPI、插件加载、全局平台宏、`std::filesystem` 或其他容易受编译器/运行时影响的依赖带进 Python extension。普通 C++ parser、纯数据结构和稳定 header 工具可以复用。

## Layout Guidance

- `SPONGE/xponge/assign/` 应保持接近独立 core：标准库 + 本目录内部依赖，不能依赖 Python、`CONTROLLER`、GPU/MPI 或 SPONGE 主程序运行状态。
- `plugins/xponge2/src/` 只做 Python C API 包装和异常转换，不把 `Python.h` 泄漏进 `SPONGE/xponge` core。
- `SPONGE/xponge/load/` 可以作为 SPONGE adapter 使用 `CONTROLLER`。如果未来 xponge2 也要 load Gromacs/Amber/native，优先把可复用 parser/IR 放进不依赖 `CONTROLLER` 的 core 层，再由 SPONGE adapter 和 Python wrapper 分别调用。
- force-field 特异逻辑优先放到清楚的命名空间或目录中，例如 GAFF/Amber、CHARMM、OPLS 分开；公共 mol2、ring、bond-order、SMART-like matching 能力保留在 shared assign/core 层。

## Review Checklist

修改 xponge2 相关代码时检查：

1. Python wheel 是否仍能在 monorepo 内构建。
2. 新依赖是否会把 wheel 运行时绑定到 SPONGE executable、GPU/MPI 或全局 controller。
3. 新 parser 或 typing 逻辑是否放在可被 Python wrapper 和 SPONGE adapter 共同复用的位置。
4. `SPONGE/xponge/assign` 是否仍保持低依赖。
5. C++ ABI 风险是否被控制，尤其避免重新引入 `std::filesystem` 到 xponge2 关键路径。
6. API 是否尽量保持原 Python 版 Xponge 包的接口语义。
