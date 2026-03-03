---
name: SPONGE-USAGE
description: >
  询问 SPONGE 运行时 问题 时 使用
  - 程序 安装
  - 版本 查询
  - 启动 程序
  - 参数 设置
  - 解决 错误
  - 反馈 问题
---

本技能适配 SPONGE 版本号：2.0.0-alpha

SPONGE2 推荐使用 pixi 进行不同版本和依赖控制，因此其调用方式可能为`pixi run -e 环境名 SPONGE`（项目级） 或 `SPONGE` （全局）

`(pixi run -e 环境名) SPONGE -v` 可以确认版本。

不同的 pixi 环境请参考 reference/pixi_environments.md

## 安装

### 源码安装

在项目源码中，可以使用 pixi 进行安装。

```shell
# Windows系统下，因为微软协议限制，需要额外手动安装MSVC。
# pixi run install-msvc
pixi install -e 环境名
pixi run --clean-env -e 环境名 configure
pixi run --clean-env -e 环境名 compile
pixi run --clean-env -e 环境名 SPONGE -v
```
