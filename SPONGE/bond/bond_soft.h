#pragma once

#include "../common.h"
#include "../control.h"

struct BOND_SOFT
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    int soft_bond_numbers = 0;
    int* h_atom_a = NULL;
    int* d_atom_a = NULL;
    int* h_atom_b = NULL;
    int* d_atom_b = NULL;
    int* h_ABmask = NULL;
    int* d_ABmask = NULL;
    float* d_k = NULL;
    float* h_k = NULL;
    float* d_r0 = NULL;
    float* h_r0 = NULL;
    float lambda;
    float alpha;

    float* h_soft_bond_ene = NULL;
    float* d_soft_bond_ene = NULL;
    float* d_sigma_of_soft_bond_ene = NULL;
    float* h_sigma_of_soft_bond_ene = NULL;

    float* h_soft_bond_dH_dlambda = NULL;
    float* d_soft_bond_dH_dlambda = NULL;
    float* h_sigma_of_dH_dlambda = NULL;
    float* d_sigma_of_dH_dlambda = NULL;

    void Initial(CONTROLLER* controller, CONECT* connectivity,
                 PAIR_DISTANCE* con_dis, const char* module_name = NULL);

    void Clear();

    void Memory_Allocate();

    void Parameter_Host_To_Device();

    void Soft_Bond_Force_With_Atom_Energy_And_Virial(
        const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
        VECTOR* frc, int need_atom_energy, float* atom_energy, int need_virial,
        LTMatrix3* atom_virial, int need_dH_dlambda);

    float Get_Partial_H_Partial_Lambda(CONTROLLER* controller);

    /*
        以下用于区域分解
    */
    int* d_atom_a_local = NULL;
    int* d_atom_b_local = NULL;
    int* d_ABmask_local = NULL;
    float* d_k_local = NULL;
    float* d_r0_local = NULL;

    // 局部信息
    int local_atom_numbers = 0;
    int num_bond_local = 0;  // 进程内bond数
    int* d_num_bond_local = NULL;
    // 局部函数：allocated模块，查询当前进程domain内需要计算的bond序号
    void Get_Local(int* atom_local, int local_atom_numbers, int ghost_numbers,
                   char* atom_local_label,
                   int* atom_local_id);  // 为domain分配bond信息
    void Step_Print(CONTROLLER* controller);
};
