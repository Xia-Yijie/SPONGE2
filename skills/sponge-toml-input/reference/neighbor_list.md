# 邻居表参数

邻居表参数通过 `[neighbor_list]` TOML section 设置：

```toml
[neighbor_list]
max_neighbor_numbers = 1200
skin_permit = 0.5
```

## 参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `refresh_interval` | int | 自动确定 | 邻居表重建间隔（步） |
| `skin_permit` | float | `0.5` | 皮层距离允许增长量，用于触发重建 |
| `max_neighbor_numbers` | int | `1200` | 单原子最大邻居数 |
| `max_atom_in_grid_numbers` | int | `150` | 单网格最大原子数 |
| `max_ghost_in_grid_numbers` | int | `150` | 单网格最大幽灵原子数 |
| `max_padding` | int | - | 数组最大填充量 |
| `min_padding` | int | - | 数组最小填充量 |
| `check_overflow_interval` | int | - | 内存溢出检查间隔 |
| `throw_error_when_overflow` | bool | `false` | 溢出时是否报错（否则自动扩容） |

邻居表的重建策略默认基于 `skin` 距离自动判断，`refresh_interval = 0` 表示自动模式。
