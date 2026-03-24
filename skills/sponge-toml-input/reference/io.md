# 输入输出参数

## 输入文件

### 通用前缀

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `default_in_file_prefix` | string | - | 输入文件名前缀，自动匹配 `<prefix>_coordinate.txt` 等 |
| `default_out_file_prefix` | string | - | 输出文件名前缀 |

设置 `default_in_file_prefix = "WAT"` 后，SPONGE 会自动查找：
- `WAT_coordinate.txt` — 坐标
- `WAT_mass.txt` — 质量
- `WAT_charge.txt` — 电荷
- `WAT_LJ.txt` — LJ 参数
- `WAT_bond.txt` — 键
- `WAT_exclude.txt` — 排除列表
- 等等

### 单独指定输入文件

| 参数 | 类型 | 说明 |
|------|------|------|
| `coordinate_in_file` | string | 坐标文件路径 |
| `velocity_in_file` | string | 速度文件路径 |
| `mass_in_file` | string | 质量文件路径 |
| `charge_in_file` | string | 电荷文件路径 |
| `atom_in_file` | string | 原子信息文件 |
| `atom_type_in_file` | string | 原子类型文件 |
| `edge_in_file` | string | 键/边信息文件 |
| `angle_in_file` | string | 角度信息文件 |
| `dihedral_in_file` | string | 二面角信息文件 |
| `atom_numbers` | int | 原子总数（可从文件自动获取） |

### 外部格式导入

| 参数 | 类型 | 说明 |
|------|------|------|
| `amber_parm7` | string | AMBER parm7 力场/拓扑文件 |
| `amber_rst7` | string | AMBER rst7 坐标/速度文件 |
| `gromacs_gro` | string | GROMACS .gro 坐标文件 |
| `gromacs_top` | string | GROMACS .top 拓扑文件 |

## 输出文件

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mdout` | string | `"mdout.txt"` | 标准输出文件（能量、温度等） |
| `mdinfo` | string | `"mdinfo.txt"` | 模拟信息/日志文件 |
| `crd` | string | - | 坐标轨迹文件（二进制） |
| `vel` | string | - | 速度轨迹文件（二进制） |
| `frc` | string | - | 力轨迹文件（二进制） |
| `box` | string | - | 盒子信息轨迹文件 |
| `rst` | string | - | 重启文件 |

## 输出频率控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `write_information_interval` | int | `1000` | mdinfo/mdout 写入间隔（步） |
| `write_mdout_interval` | int | `1000` | mdout 写入间隔 |
| `write_trajectory_interval` | int | 同 `write_information_interval` | 轨迹写入间隔 |
| `write_restart_file_interval` | int | `step_limit` | 重启文件写入间隔 |
| `max_restart_export_count` | int | - | 保留的重启文件最大数量 |
| `buffer_frame` | int | `10` | 文件缓冲帧数（影响 I/O 性能） |

## 输出内容控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `print_zeroth_frame` | int | `0` | 是否输出第 0 步的帧（1 = 是） |
| `print_pressure` | int | `0` | 是否输出压力 |
| `print_detail` | int | `0` | 详细输出级别 |
