# 恒温器参数

## 恒温器选择

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `thermostat` | string | **NVT/NPT 必填** | 恒温器算法 |

`thermostat` 可选值：

| 值 | 说明 |
|----|------|
| `"middle_langevin"` / `"langevin"` | Middle Langevin 动力学（推荐） |
| `"andersen"` | Andersen 恒温器 |
| `"berendsen_thermostat"` | Berendsen 恒温器 |
| `"bussi_thermostat"` | Bussi (velocity rescaling) 恒温器 |
| `"nose_hoover_chain"` | Nose-Hoover 链恒温器 |

## 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_temperature` | float | `300.0` | 目标温度（K） |
| `thermostat_tau` / `tau` | float | - | 耦合时间常数（ps） |
| `thermostat_seed` / `seed` | int | - | 随机种子（随机性恒温器需要） |
| `velocity_max` | float | - | 最大允许速度 |

## 温度 schedule

支持模拟过程中按计划变化目标温度：

| 参数 | 类型 | 说明 |
|------|------|------|
| `target_temperature_schedule_mode` | string | `"step"`（阶跃）或 `"linear"`（线性插值） |
| `target_temperature_schedule_steps` | array | 内联 schedule，格式为 `[{step = N, value = T}, ...]` |
| `target_temperature_schedule_file` | string | 外部 schedule 文件路径（TOML 格式） |

示例：

```toml
target_temperature = 300.0
target_temperature_schedule_mode = "linear"
target_temperature_schedule_steps = [
    { step = 0, value = 100.0 },
    { step = 10000, value = 300.0 },
    { step = 50000, value = 300.0 },
]
```
