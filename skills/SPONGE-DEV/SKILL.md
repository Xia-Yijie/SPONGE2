---
name: SPONGE-DEV
description: >
  询问 SPONGE 开发时 问题 时 使用
  - 环境 安装
  - 程序 打包
---

本技能适配 SPONGE 版本号：2.0.0-alpha

SPONGE2 推荐使用 pixi 进行不同版本和依赖控制，因此其调用方式可能为`pixi run -e 环境名 SPONGE`（项目级） 或 `SPONGE` （全局）

`(pixi run -e 环境名) SPONGE -v` 可以确认版本。

不同的 pixi 环境请参考 reference/pixi_environments.md

## 环境安装

建议使用 pixi 进行开发环境安装，以方便后期打包分发

```shell
pixi install -e 环境名
pixi run --clean-env -e 环境名 configure
pixi run --clean-env -e 环境名 compile
```

