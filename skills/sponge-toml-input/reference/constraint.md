# 约束算法参数

## 约束模式

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `constrain_mode` | string | 无（不启用约束） | 约束算法 |

`constrain_mode` 可选值：

| 值 | 说明 |
|----|------|
| `"SETTLE"` | SETTLE 算法，专用于刚性水分子（三角形约束） |
| `"SHAKE"` | SHAKE 算法，通用键长约束 |

## SETTLE 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `settle_disable` | bool | `false` | 禁用 SETTLE |

## SHAKE 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `SHAKE_step_length` | float | - | SHAKE 迭代收敛参数 |
