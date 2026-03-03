# Restrain 模块输入参数

下面列出 `restrain` 模块在 `mdin` 中支持的输入命令。所有键名仅支持下划线形式。

## 必需
- `restrain_atom_id`
  - 类型：文件路径
  - 内容：逐行原子索引（int），用于指定 restrain 的原子列表。

## 可选
- `restrain_refcoord_scaling`
  - 类型：字符串
  - 取值：`no` / `all` / `com_ug` / `com_res` / `com_mol`
  - 说明：参考坐标随盒子缩放的方式。
  - 注意：如果选择 Monte-Carlo 控压，至少要选择`com_ug` / `com_res` / `com_mol`。

- `restrain_calc_virial`
  - 类型：bool（`true`/`false` 或 `0/1`）
  - 默认：`true`
  - 说明：仅当为 `true` 时 restrain 才会计算并累加 virial。

- `restrain_coordinate_in_file`
  - 类型：文件路径
  - 内容：第一行原子数，之后每行三个浮点数（x y z）。
  - 说明：作为 restrain 参考坐标；优先级高于 `amber_rst7`。

- `restrain_amber_rst7`
  - 类型：文件路径
  - 说明：从 Amber rst7 读取 restrain 参考坐标（当未提供 `coordinate_in_file` 时使用）。

- `restrain_single_weight`
  - 类型：float
  - 说明：单一各向同性 restrain 力常数。设置后忽略 `weight_in_file`。

- `restrain_weight_in_file`
  - 类型：文件路径
  - 内容：逐行三个浮点数（wx wy wz），数量与 `restrain_atom_id` 一致。
  - 说明：各向异性权重；仅在未提供 `single_weight` 时使用。

## 示例
```text
restrain_atom_id = atom_id.dat
restrain_refcoord_scaling = com_mol
restrain_calc_virial = true
restrain_coordinate_in_file = refcrd.dat
restrain_single_weight = 20.0
```
