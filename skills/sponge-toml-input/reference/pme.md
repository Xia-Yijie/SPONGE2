# PME 静电参数

PME（Particle Mesh Ewald）参数通过 `[PME]` TOML section 设置：

```toml
[PME]
grid_spacing = 1.0
```

## 参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fftx` / `ffty` / `fftz` | int | 自动确定 | FFT 网格维度 |
| `grid_spacing` | float | `1.0` | 网格间距（A），用于自动确定 FFT 维度 |
| `update_interval` | int | `1` | 倒空间求和更新间隔（步） |
| `Direct_Tolerance` | float | - | 直接空间截断容差 |
| `calculate_excluded_part` | bool | - | 是否计算排除对相互作用 |
| `calculate_reciprocal_part` | bool | - | 是否计算倒空间部分 |

如果不指定 `fftx/ffty/fftz`，SPONGE 会根据 `grid_spacing` 和盒子大小自动计算。
