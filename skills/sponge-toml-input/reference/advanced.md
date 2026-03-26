# 高级参数

## 设备与并行

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `device` | string | 自动检测 | GPU 设备 ID |
| `device_optimized_block` | int | - | GPU block size 优化 |

## 墙约束

### 硬墙

| 参数 | 类型 | 说明 |
|------|------|------|
| `hard_wall_x_low` / `hard_wall_x_high` | float | X 方向硬墙位置（A） |
| `hard_wall_y_low` / `hard_wall_y_high` | float | Y 方向硬墙位置（A） |
| `hard_wall_z_low` / `hard_wall_z_high` | float | Z 方向硬墙位置（A） |

### 软墙

| 参数 | 类型 | 说明 |
|------|------|------|
| `soft_walls_in_file` | string | 软墙势能参数文件路径 |

## ReaxFF 反应力场

通过 `[REAXFF]` section 配置：

```toml
[REAXFF]
in_file = "ffield.reax.cho"
type_in_file = "atom_types.txt"
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `in_file` | string | ReaxFF 参数文件 |
| `type_in_file` | string | 原子类型映射文件 |

## 自定义力

| 参数 | 类型 | 说明 |
|------|------|------|
| `custom_force` | section | 自定义力配置 |
| `pairwise_force` | section | 自定义成对力配置 |

## 插件

通过 `[plugin]` section 配置外部插件（如 PRIPS）。

## 控制参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `command_only` | bool | `false` | 仅使用命令行参数，跳过 mdin 文件 |
| `dont_check_input` | bool | `false` | 跳过未使用参数的检查 |
| `end_pause` | bool | `false` | 模拟结束后暂停 |
