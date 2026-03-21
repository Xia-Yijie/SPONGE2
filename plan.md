# 计划：将 quantum_chemistry.cpp 拆分为多个编译单元

## 背景

当前 quantum_chemistry 全部代码（约 20,000 行 hpp）通过一个 `quantum_chemistry.cpp` 编译为单个目标文件。在低配机器上编译时内存和 CPU 压力极大。需要拆分为多个 `.cpp` 分别编译，降低峰值资源占用。

## 当前结构

```
quantum_chemistry.cpp (855 行)
  ├─ #include basis/* (14 个基组文件, 11,074 行数据)
  ├─ #include integrals/eri.hpp, one_e.hpp
  ├─ #include scf/matrix.hpp
  ├─ 6 个成员函数 (Parsing_Arguments, Initial_Molecule 等)
  └─ 文件末尾 #include:
     ├─ dft/* (ao, grid, vxc, xc, lebedev: 3,794 行)
     ├─ integrals/cart2sph.hpp
     └─ scf/scf.hpp → 包含 7 个 SCF 文件 (3,734 行)
```

全部 ~20,238 行编译为一个目标文件。

## 拆分方案（4 个编译单元）

### 1. `quantum_chemistry_init.cpp`
**内容：** 初始化相关函数
- `Parsing_Arguments`、`Initial_Molecule`、`Initial_Integral_Tasks`、`Initial`、`Memory_Allocate`、`Step_Print`
- 包含：`quantum_chemistry.h`、所有 `basis/*.hpp`、`structure/*.h`

**拆分理由：** 基组数据文件非常大（11K 行），但只在初始化时使用，不需要和 SCF/DFT 代码一起编译。

### 2. `quantum_chemistry_scf.cpp`
**内容：** SCF 循环及所有 SCF 相关函数
- 包含 `scf/scf.hpp`（→ pre_scf、build_fock、diag_density、apply_diis、mix_converge、accumulate_energy、workspace）
- 函数：`Solve_SCF` 及其调用的所有 SCF 成员函数

**拆分理由：** SCF 链路函数紧密耦合，需要在一个编译单元中，但与基组初始化和 DFT 无关。

### 3. `quantum_chemistry_dft.cpp`
**内容：** DFT 相关函数
- 包含 `dft/*.hpp`（ao、grid、vxc、xc、lebedev）
- 函数：`Update_DFT_Grid` 及 DFT 相关成员函数

**拆分理由：** `lebedev.hpp` 单独就有 2,234 行常量数据，加上 XC 泛函代码共 3,794 行。

### 4. `quantum_chemistry_integrals.cpp`
**内容：** 积分和基组变换函数
- 包含 `integrals/one_e.hpp`、`eri.hpp`、`cart2sph.hpp`
- 函数：`Compute_OneE_Integrals`、`Cart2Sph_OneE_Integrals`、`Build_Cart2Sph_Matrix`

**拆分理由：** ERI 代码有复杂的内联函数，与 SCF 循环逻辑分离。

## 每个新 .cpp 文件的模式

```cpp
#include "quantum_chemistry.h"
// 只包含该编译单元需要的 hpp 文件
#include "scf/scf.hpp"  // 或 dft/*.hpp 或 basis/*.hpp

// 实现 QUANTUM_CHEMISTRY:: 成员函数子集
void QUANTUM_CHEMISTRY::Solve_SCF(...) { ... }
```

## 需要修改的文件

| 文件 | 操作 |
|------|------|
| `quantum_chemistry.cpp` | 拆分为 4 个 .cpp 文件 |
| `cmake/targets/SPONGE.cmake` | 将新的 .cpp 加入 `SPONGE_SOURCES` |
| `quantum_chemistry.h` | 可能需要补充前置声明 |

## 注意事项

1. **hpp 中的 static 函数**：kernel 函数（`__global__`）和 static inline 辅助函数在 hpp 中定义，被 include 到对应的 .cpp 即可，不会重复定义。
2. **宏定义**：`ONEE_MD_BASE`、`ONEE_MD_IDX` 等宏在 `quantum_chemistry.cpp` 顶部定义，需要移到对应的 .cpp 或单独的头文件中。
3. **编译顺序无关**：各 .cpp 独立编译，通过链接合并，顺序无关。
4. **GPU 兼容**：所有 .cpp 在 GPU 构建时用 nvcc 编译（CMake 已处理 `.cpp.o` → CUDA object）。

## 验证步骤

1. `pixi run -e dev-cpu compile` — CPU 编译通过
2. `pixi run -e dev-cuda13 compile` — GPU 编译通过
3. `pixi run -e dev-cpu comp-pyscf` — 30 passed
4. `pixi run -e dev-cuda13 comp-pyscf` — 30 passed
5. benzene/def2-qzvp 2 轮 CPU 测试确认正确性
