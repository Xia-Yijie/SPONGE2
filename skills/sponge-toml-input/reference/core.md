# 核心模拟参数

## 模拟标识

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `md_name` | string | `"Default SPONGE MD Task Name"` | 模拟任务名称 |

## 模拟模式

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | string | **必填** | 模拟模式 |

`mode` 可选值：

| 值 | 说明 |
|----|------|
| `"nve"` | 微正则系综（恒能量） |
| `"nvt"` | 正则系综（恒温度），需要 thermostat |
| `"npt"` | 等温等压系综，需要 thermostat + barostat |
| `"minimization"` / `"min"` | 能量最小化 |
| `"rerun"` | 轨迹重分析 |

## 时间与步数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `step_limit` | int | `1000` | 总模拟步数 |
| `dt` | float | `0.001` | 时间步长（ps） |
| `frame_limit` | int | - | rerun 模式下的帧数限制 |

## 力场截断

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cutoff` | float | `8.0` | 非键相互作用截断距离（A） |
| `skin` | float | `2.0` | 邻居表额外皮层距离（A） |

## 温度与压力目标

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_temperature` | float | `300.0` | 目标温度（K），NVT/NPT 必填 |
| `target_pressure` | float | `1.0` | 目标压力（bar），NPT 必填 |

温度和压力支持 schedule（分步/线性变化），详见 [thermostat.md](thermostat.md) 和 [barostat.md](barostat.md)。

## 工作目录

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `workspace` | string | mdin 所在目录 | 工作目录路径，可为绝对或相对路径 |
