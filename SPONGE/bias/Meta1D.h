/*
    Notice: Meta1D 已被弃用，目前MetaD使用 SinkMeta 实现
*/

#pragma once
#include "../collective_variable/collective_variable.h"

struct META1D
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int last_modify_date = 20260216;

    // 作用的CV
    COLLECTIVE_VARIABLE_PROTOTYPE* cv;

    // 一维Meta的边界和细度记录
    float cv_min = 0;
    float cv_max = 0;
    float dcv = 0.01;
    int grid_numbers = 0;
    float border_potential_height = 100000.;  // cv边界近似无限高势垒
    float cv_period = 0;

    // 存储背景势场的离散列表
    float* potential_list = NULL;

    // 多次运行SPONGE相关的记录存储操作
    char read_potential_file_name[256];
    char write_potential_file_name[256];
    void Read_Potential(CONTROLLER* controller);
    void Write_Potential();
    // 高斯峰的高度和宽度（目前暂不支持动态变化）
    float height = 1.;
    float sigma = 1.;

    float welltemp_factor =
        1000000000.;  // Biasfactor无限大时，即为普通的Meta无时间衰减，因此预留一个大值。

    // 计算Meta1D的力
    // 为与其他force函数有相似性，这里面输入的是体系的总原子数目（即使meta只对部分原子起作用）
    // frc则是系统核心迭代的力，并且是累加性的。
    // cv则是相匹配的cv模块算出来的cv值（GPU上）
    // cv_grad也是cv模块cv值对体系所有原子坐标的梯度（不参与cv计算的原子的对应梯度为0）
    void Meta_Force_With_Energy_And_Virial(int atom_numbers, VECTOR* frc,
                                           int need_potential,
                                           int need_pressure,
                                           float* d_potential,
                                           LTMatrix3* d_virial);

    void Initial(CONTROLLER* controller,
                 COLLECTIVE_VARIABLE_CONTROLLER* cv_controller,
                 char* module_name = NULL);
    void AddPotential(int steps);
    void Do_Metadynamics(int atom_numbers, VECTOR* crd, LTMatrix3 cell,
                         LTMatrix3 rcell, int step, int need_potential,
                         int need_pressure, VECTOR* frc, float* d_potential,
                         LTMatrix3* d_virial);
    float Potential(float x);
    float DPotential(float x);
    void Step_Print(CONTROLLER* controller);
    int potential_update_interval;
};
