# 恒压器参数

## 恒压器选择

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `barostat` | string | **NPT 必填** | 恒压器算法 |

`barostat` 可选值：

| 值 | 说明 |
|----|------|
| `"andersen_barostat"` | Andersen 恒压器 |
| `"berendsen_barostat"` | Berendsen 恒压器 |
| `"bussi_barostat"` | Bussi 恒压器 |
| `"monte_carlo_barostat"` | Monte Carlo 恒压器 |

## 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_pressure` | float | `1.0` | 目标压力（bar） |
| `barostat_tau` | float | - | 耦合时间常数（ps） |
| `barostat_update_interval` | int | - | 恒压器更新间隔（步） |
| `barostat_compressibility` | float | - | 等温压缩率（bar⁻¹） |

## 各向异性控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `barostat_isotropy` / `isotropy` | string | `"isotropic"` | 压力耦合类型 |

`barostat_isotropy` 可选值：

| 值 | 说明 |
|----|------|
| `"isotropic"` | 各向同性，三轴等比缩放 |
| `"semiisotropic"` | 半各向同性 |
| `"semianisotropic"` | 半各向异性 |
| `"anisotropic"` | 各向异性，三轴独立缩放 |

## 盒子形变控制

| 参数 | 类型 | 说明 |
|------|------|------|
| `barostat_g11` / `g21` / `g22` / `g31` / `g32` / `g33` | float | 盒子形变矩阵元素 |
| `barostat_x_constant` / `y_constant` / `z_constant` | float | 固定特定轴的盒子尺寸 |

## 表面张力

| 参数 | 类型 | 说明 |
|------|------|------|
| `surface_tensor` | string | 表面张力控制模式 |
| `surface_number` | int | 表面数量 |
| `surface_tension` | float | 表面张力值 |

## 压力 schedule

与温度 schedule 类似，支持按计划变化目标压力：

| 参数 | 类型 | 说明 |
|------|------|------|
| `target_pressure_schedule_mode` | string | `"step"` 或 `"linear"` |
| `target_pressure_schedule_steps` | array | 内联 schedule |
| `target_pressure_schedule_file` | string | 外部 schedule 文件路径 |
