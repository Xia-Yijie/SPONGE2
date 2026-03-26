# 增强采样参数

## 集体变量（CV）

| 参数 | 类型 | 说明 |
|------|------|------|
| `cv_in_file` | string | 集体变量定义文件路径 |

也可通过 `[CV]` section 内联定义：

| 参数 | 类型 | 说明 |
|------|------|------|
| `CV_type` | string | CV 类型 |
| `CV_period` | int | CV 更新频率 |
| `CV_minimal` / `CV_maximum` | float | CV 取值范围 |

## Metadynamics

通过 `[META]` 或 `[meta]` section 配置：

| 参数 | 类型 | 说明 |
|------|------|------|
| `sink` | string | Sink metadynamics 模式 |

## SITS

SITS（Self-guided Integrated Tempering Sampling）参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `SITS_mode` | string | 运行模式 |
| `SITS_atom_numbers` | int | 参与 SITS 的原子数 |
| `SITS_k_numbers` | int | k 空间点数 |
| `SITS_T_low` / `SITS_T_high` | float | 温度范围（K） |
| `SITS_record_interval` | int | 记录间隔 |
| `SITS_update_interval` | int | 更新间隔 |
| `SITS_nk_fix` | int | 固定 k 空间点数 |
| `SITS_nk_in_file` | string | k 空间输入文件 |
| `SITS_pe_a` / `SITS_pe_b` | float | 势能参数 |
| `SITS_fb_interval` | int | 反馈间隔 |

`SITS_mode` 可选值：

| 值 | 说明 |
|----|------|
| `"observation"` | 观测阶段 |
| `"iteration"` | 迭代阶段 |
| `"production"` | 生产阶段 |
| `"empirical"` | 经验模式 |
| `"amd"` | AMD 模式 |
| `"gamd"` | GaMD 模式 |
