#ifndef QC_STRUCTURE_CART2SPH_H
#define QC_STRUCTURE_CART2SPH_H

#include "../../common.h"

// 笛卡尔变球谐的缓冲变量
struct QC_CARTESIAN_TO_SPHERICAL
{
    // 笛卡尔缓冲变量
    float* d_S_cart = NULL;
    float* d_T_cart = NULL;
    float* d_V_cart = NULL;
    float* d_ERI_cart = NULL;
    float* d_cart2sph_mat = NULL;
    float* d_cart2sph_1e_tmp = NULL;

    float* d_cart2sph_eri_t1 = NULL;
    float* d_cart2sph_eri_t2 = NULL;
    float* d_cart2sph_eri_t3 = NULL;
};

#endif
