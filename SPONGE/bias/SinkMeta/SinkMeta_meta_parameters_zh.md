# SinkMeta/meta 输入参数（中文）

本文档来源于 `META::Initial()`（`SPONGE/SPONGE/bias/SinkMeta/Meta.cpp`）以及 `Meta.h` 中的默认值。

## 参数（模块名：`meta`）

| 参数名 | 类型 / 维度 | 是否必须 | 默认值（如有） | 含义 / 说明 |
| --- | --- | --- | --- | --- |
| `CV` | 字符串列表 | 是 | 无 | Meta 使用的 CV 模块名列表；决定维度。缺失则不会初始化。 |
| `dip` | float | 否 | `0.0` | submarine/sink 额外下陷项（以 `kB*T` 参与偏置位移）。 |
| `welltemp_factor` | float | 否 | `1e9` | Well-tempered 偏置因子；> 1 时启用 well-tempered。 |
| `Ndim` | int | 否 | `CV` 个数 | 显式指定维度；必须与 `CV` 列表长度一致。 |
| `subhill` | 标志位 | 否 | `false` | 启用 sub-hill（Gaussian）行为。只判断是否存在，不读数值。 |
| `kde` | int | 否 | `0` | 非零启用 KDE，并设置 `subhill=true`；sigma 缩放为 `1.414/sigma`。 |
| `mask` | int | 否 | `0` | 启用 mask 模式（n 维区域 exit 标记）。 |
| `maxforce` | float | 否 | `0.1` | 边界力阈值（exit 标记）；仅在 `mask` 存在时读取。 |
| `sink` | int | 否 | `0` | 非零启用负 hill（sink/submarine）行为。 |
| `sumhill_freq` | int | 否 | `0` | `sumhill` 历史频率（影响 Rbias/RCT）。 |
| `catheter` | 标志位 | 否 | `false` | 只判断存在：强制 `use_scatter=true`、`usegrid=false`、`do_negative=true`，并把 `catheter` 设为 3（硬编码）。 |
| `convmeta` | int | 否 | `0` | ConvolutionMeta 开关；同时设置 `do_negative=true`。 |
| `grw` | int | 否 | `0` | GRW 开关；同时设置 `do_negative=true`。 |
| `CV_period` | float 数组（ndim） | 是 | META 无默认 | 每个 CV 的周期长度；在 `META::Initial()` 中总是读取。 |
| `CV_sigma` | float 数组（ndim） | 是 | META 无默认 | 每个 CV 的高斯宽度（必须 > 0）。内部会转为 `1/sigma`。 |
| `cutoff` | float 数组（ndim） | 否 | `3 * CV_sigma` | 邻域截断（查表/边界墙用）；存在则启用 `do_cutoff`。 |
| `potential_in_file` | string | 否 | 无 | 从文件读取势能；若设置则调用 `Read_Potential()`（跳过网格/散点设置）。 |
| `scatter_in_file` | string | 否 | 无 | 从文件读取散点势能；设置后 `use_scatter=true`、`usegrid=false` 并读取势能。 |
| `scatter` | int | 否 | `0` | 散点数；> 0 时使用散点而非网格。 |
| `CV_minimal` | float 数组（ndim） | 条件必需 | META 无默认 | 网格下界；当未使用 `potential_in_file`/`scatter_in_file` 时需要。 |
| `CV_maximum` | float 数组（ndim） | 条件必需 | META 无默认 | 网格上界；必须大于 `CV_minimal`。 |
| `CV_grid` | int 数组（ndim） | 条件必需 | META 无默认 | 网格点数；必须 > 1。 |
| `height` | float | 否 | `1.0` | 初始 hill 高度（`height_0`）。 |
| `wall_height` | float | 否 | 无 | 启用边界墙并设置 `border_potential_height`。 |
| `potential_out_file` | string | 否 | `Meta_Potential.txt` | 势能输出文件名。 |
| `potential_update_interval` | int | 否 | `write_information_interval` 或 `1000` | 势能写出步长；<= 0 时强制为 1000。 |

## 其他模块 / 全局使用的参数

| 参数名 | 作用域 | 类型 / 维度 | 是否必须 | 默认值（如有） | 含义 / 说明 |
| --- | --- | --- | --- | --- | --- |
| `CV_point` | 每个 CV 模块 | float 数组（scatter_size） | 条件必需 | META 无默认 | 每个 CV 的散点坐标；仅在 `scatter > 0` 时读取。 |
| `write_information_interval` | Controller | int | 否 | `1000` | 全局信息写出间隔；作为 `potential_update_interval` 默认值。 |

## 备注

- 若同时设置 `potential_in_file` 与 `scatter_in_file`，代码采用 `if/else if`，`potential_in_file` 优先。
- `CV_sigma` 在内部会取倒数；`kde` 模式使用 `1.414/sigma`，否则为 `1.0/sigma`。
- 代码中的默认文件名：`read_potential_file_name` 和 `write_potential_file_name` 初始均为 `Meta_Potential.txt`。
