#ifndef REAXFF_BOND_ORDER_H
#define REAXFF_BOND_ORDER_H

#include "../../common.h"
#include "../../control.h"

struct REAXFF_BOND_ORDER
{
    int is_initialized = 0;

    int atom_numbers = 0;
    int atom_type_numbers = 0;

    // General parameters from ffield
    float gp_boc1 = 0.0f;
    float gp_boc2 = 0.0f;
    float gp_bo_cut = 0.001f;

    // Per-atom-type boc parameters (for computing per-pair p_boc3/4/5)
    float* h_b_o_131 = NULL;  // p_boc4 per atom type
    float* h_b_o_132 = NULL;  // p_boc3 per atom type
    float* h_b_o_133 = NULL;  // p_boc5 per atom type

    // Parameters from ffield
    float* h_ro_sigma = NULL;
    float* h_ro_pi = NULL;
    float* h_ro_pi2 = NULL;
    float* h_bo_1 = NULL;
    float* h_bo_2 = NULL;
    float* h_bo_3 = NULL;
    float* h_bo_4 = NULL;
    float* h_bo_5 = NULL;
    float* h_bo_6 = NULL;
    // Pair-specific bond-order radii (r_s, r_p, r_pp). Initialized to averages
    // and overridden by off-diagonal entries if present in ffield.
    float* h_r_s = NULL;
    float* h_r_p = NULL;
    float* h_r_pp = NULL;

    float* d_ro_sigma = NULL;
    float* d_ro_pi = NULL;
    float* d_ro_pi2 = NULL;
    float* d_bo_1 = NULL;
    float* d_bo_2 = NULL;
    float* d_bo_3 = NULL;
    float* d_bo_4 = NULL;
    float* d_bo_5 = NULL;
    float* d_bo_6 = NULL;
    float* d_r_s = NULL;
    float* d_r_p = NULL;
    float* d_r_pp = NULL;

    int* h_atom_type = NULL;
    int* d_atom_type = NULL;

    // 键级修正参数
    float* h_valency = NULL;      // 价态参数
    float* h_valency_val = NULL;  // 价态参数
    float* h_ovc = NULL;          // 过配位修正开关
    float* h_v13cor = NULL;       // 1-3键修正开关
    float* h_p_boc1 = NULL;       // 过配位修正参数1
    float* h_p_boc2 = NULL;       // 过配位修正参数2
    float* h_p_boc3 = NULL;       // 1-3键修正参数3
    float* h_p_boc4 = NULL;       // 1-3键修正参数4
    float* h_p_boc5 = NULL;       // 1-3键修正参数5

    float* d_valency = NULL;
    float* d_valency_val = NULL;
    float* d_ovc = NULL;
    float* d_v13cor = NULL;
    float* d_p_boc1 = NULL;
    float* d_p_boc2 = NULL;
    float* d_p_boc3 = NULL;
    float* d_p_boc4 = NULL;
    float* d_p_boc5 = NULL;

    // Temporary device arrays
    float* d_bo_sigma = NULL;
    float* d_bo_pi = NULL;
    float* d_bo_pi2 = NULL;
    float* d_total_bo = NULL;
    float* d_total_bond_order = NULL;  // 每个原子的总键级
    float* d_total_corrected_bond_order = NULL;
    float* d_bo_s = NULL;              // 未修正的sigma键级
    float* d_bo_p = NULL;              // 未修正的pi键级
    float* d_bo_p2 = NULL;             // 未修正的pi-pi键级
    float* d_total_bo_raw = NULL;      // 未修正的总键级
    float* d_corrected_bo = NULL;      // 修正后的键级
    float* d_corrected_bo_s = NULL;    // 修正后的sigma键级
    float* d_corrected_bo_pi = NULL;   // 修正后的pi键级
    float* d_corrected_bo_pi2 = NULL;  // 修正后的pi-pi键级
    int* d_pair_i = NULL;              // 存储原子对的i索引
    int* d_pair_j = NULL;              // 存储原子对的j索引
    float* d_pair_distances = NULL;    // 存储原子对的距离
    int* d_num_pairs_ptr = NULL;       // 指向设备上原子对数量的指针
    int h_num_pairs = 0;               // 主机上的原子对数量

    // Derivatives of total energy with respect to bond orders (accumulated from
    // all modules)
    float* d_dE_dBO_s = NULL;
    float* d_dE_dBO_pi = NULL;
    float* d_dE_dBO_pi2 = NULL;

    // Derivatives of corrected bond orders w.r.t r, Delta_i, Delta_j
    float* d_dbo_s_dr = NULL;
    float* d_dbo_pi_dr = NULL;
    float* d_dbo_pi2_dr = NULL;
    float* d_dbo_s_dDelta_i = NULL;
    float* d_dbo_pi_dDelta_i = NULL;
    float* d_dbo_pi2_dDelta_i = NULL;
    float* d_dbo_s_dDelta_j = NULL;
    float* d_dbo_pi_dDelta_j = NULL;
    float* d_dbo_pi2_dDelta_j = NULL;
    float* d_dbo_raw_total_dr = NULL;
    float* d_CdDelta_prime = NULL;

    void Initial(CONTROLLER* controller, int atom_numbers,
                 const char* parameter_in_file, const char* type_in_file,
                 const float cutoff, float* cutoff_full);
    void Calculate_Bond_Order(int atom_numbers, const VECTOR* d_crd,
                              const LTMatrix3 cell, const LTMatrix3 rcell,
                              const ATOM_GROUP* fnl_d_nl, float cutoff);
    void Calculate_Corrected_Bond_Order(int atom_numbers, const VECTOR* d_crd,
                                        const LTMatrix3 cell,
                                        const LTMatrix3 rcell,
                                        const ATOM_GROUP* fnl_d_nl,
                                        float cutoff);
    void Calculate_Forces(int atom_numbers, const VECTOR* d_crd, VECTOR* d_frc,
                          const LTMatrix3 cell, const LTMatrix3 rcell,
                          float cutoff, float* d_CdDelta, int need_virial,
                          LTMatrix3* atom_virial);
    void Clear_Derivatives(int atom_numbers, float* d_CdDelta);

    // GPU kernel declarations
    void Calculate_Uncorrected_Bond_Orders_GPU(
        int atom_numbers, const VECTOR* d_crd, const LTMatrix3 cell,
        const LTMatrix3 rcell, float cutoff, int* d_pair_i, int* d_pair_j,
        float* d_distances, int* d_num_pairs_ptr);
    void Calculate_Corrected_Bond_Orders_GPU(
        int atom_numbers, const VECTOR* d_crd, const LTMatrix3 cell,
        const LTMatrix3 rcell, float cutoff, int num_pairs, int* d_pair_i,
        int* d_pair_j, float* d_distances);
};

#endif
