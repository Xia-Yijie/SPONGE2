#pragma once

#include "../scf/matrix.hpp"

// 用于将归一化笛卡尔基映射到归一化实球谐基
// l=2（d 轨道）
// 行：笛卡尔顺序
// 列：球谐顺序
const float CART2SPH_MAT_D[6][5] = {
    {0.00000000f, 0.00000000f, -0.31539157f, 0.00000000f, 0.54627422f},
    {1.09254843f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 1.09254843f, 0.00000000f},
    {0.00000000f, 0.00000000f, -0.31539157f, 0.00000000f, -0.54627422f},
    {0.00000000f, 1.09254843f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.63078313f, 0.00000000f, 0.00000000f},
};

// l=3（f 轨道）
// 行：笛卡尔顺序
// 列：球谐顺序
const float CART2SPH_MAT_F[10][7] = {
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, -0.45704580f,
     0.00000000f, 0.59004359f},
    {1.77013077f, 0.00000000f, -0.45704580f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, -1.11952900f, 0.00000000f,
     1.44530572f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, -0.45704580f,
     0.00000000f, -1.77013077f},
    {0.00000000f, 2.89061144f, 0.00000000f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 1.82818320f,
     0.00000000f, 0.00000000f},
    {-0.59004359f, 0.00000000f, -0.45704580f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, -1.11952900f, 0.00000000f,
     -1.44530572f, 0.00000000f},
    {0.00000000f, 0.00000000f, 1.82818320f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.74635267f, 0.00000000f,
     0.00000000f, 0.00000000f},
};

// l=4（g 轨道）
// 行：笛卡尔顺序
// 列：球谐顺序
const float CART2SPH_MAT_G[15][9] = {
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.31735664f,
     0.00000000f, -0.47308735f, 0.00000000f, 0.62583574f},
    {2.50334294f, 0.00000000f, -0.94617470f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
     -2.00713963f, 0.00000000f, 1.77013077f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.63471328f,
     0.00000000f, 0.00000000f, 0.00000000f, -3.75501441f},
    {0.00000000f, 5.31039231f, 0.00000000f, -2.00713963f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, -2.53885313f,
     0.00000000f, 2.83852409f, 0.00000000f, 0.00000000f},
    {-2.50334294f, 0.00000000f, -0.94617470f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
     -2.00713963f, 0.00000000f, -5.31039231f, 0.00000000f},
    {0.00000000f, 0.00000000f, 5.67704817f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
     2.67618617f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.31735664f,
     0.00000000f, 0.47308735f, 0.00000000f, 0.62583574f},
    {0.00000000f, -1.77013077f, 0.00000000f, -2.00713963f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, -2.53885313f,
     0.00000000f, -2.83852409f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 2.67618617f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.84628438f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
};

static __global__ void QC_Cart2Sph_MatMul_UT_RowRow_Kernel(
    const int m, const int n, const int kdim, const float* U_row_k_m,
    const float* B_row_k_n, float* C_row_m_n)
{
    SIMPLE_DEVICE_FOR(idx, m * n)
    {
        int i = idx / n;
        int j = idx - i * n;
        double sum = 0.0;
        for (int k = 0; k < kdim; k++)
        {
            sum += (double)U_row_k_m[k * m + i] * (double)B_row_k_n[k * n + j];
        }
        C_row_m_n[i * n + j] = (float)sum;
    }
}

static __global__ void QC_ERI_Cart2Sph_Stage1_Kernel(int n_out, int nc, int ns,
                                                     const float* U_row_nc_ns,
                                                     const float* eri_c,
                                                     float* t1)
{
    SIMPLE_DEVICE_FOR(idx, n_out)
    {
        int tmp = idx;
        const int d = (int)(tmp % nc);
        tmp /= nc;
        const int c = (int)(tmp % nc);
        tmp /= nc;
        const int b = (int)(tmp % nc);
        tmp /= nc;
        const int p = (int)tmp;
        double sum = 0.0;
        for (int a = 0; a < nc; a++)
        {
            const int idc = (((int)a * nc + b) * nc + c) * nc + d;
            sum += (double)U_row_nc_ns[a * ns + p] * (double)eri_c[idc];
        }
        t1[idx] = (float)sum;
    }
}

static __global__ void QC_ERI_Cart2Sph_Stage2_Kernel(int n_out, int nc, int ns,
                                                     const float* U_row_nc_ns,
                                                     const float* t1, float* t2)
{
    SIMPLE_DEVICE_FOR(idx, n_out)
    {
        int tmp = idx;
        const int d = (int)(tmp % nc);
        tmp /= nc;
        const int c = (int)(tmp % nc);
        tmp /= nc;
        const int q = (int)(tmp % ns);
        tmp /= ns;
        const int p = (int)tmp;
        double sum = 0.0;
        for (int b = 0; b < nc; b++)
        {
            const int id1 = (((int)p * nc + b) * nc + c) * nc + d;
            sum += (double)U_row_nc_ns[b * ns + q] * (double)t1[id1];
        }
        t2[idx] = (float)sum;
    }
}

static __global__ void QC_ERI_Cart2Sph_Stage3_Kernel(int n_out, int nc, int ns,
                                                     const float* U_row_nc_ns,
                                                     const float* t2, float* t3)
{
    SIMPLE_DEVICE_FOR(idx, n_out)
    {
        int tmp = idx;
        const int d = (int)(tmp % nc);
        tmp /= nc;
        const int r = (int)(tmp % ns);
        tmp /= ns;
        const int q = (int)(tmp % ns);
        tmp /= ns;
        const int p = (int)tmp;
        double sum = 0.0;
        for (int c = 0; c < nc; c++)
        {
            const int id2 = (((int)p * ns + q) * nc + c) * nc + d;
            sum += (double)U_row_nc_ns[c * ns + r] * (double)t2[id2];
        }
        t3[idx] = (float)sum;
    }
}

static __global__ void QC_ERI_Cart2Sph_Stage4_Kernel(int n_out, int nc, int ns,
                                                     const float* U_row_nc_ns,
                                                     const float* t3,
                                                     float* eri_s)
{
    SIMPLE_DEVICE_FOR(idx, n_out)
    {
        int tmp = idx;
        const int s = (int)(tmp % ns);
        tmp /= ns;
        const int r = (int)(tmp % ns);
        tmp /= ns;
        const int q = (int)(tmp % ns);
        tmp /= ns;
        const int p = (int)tmp;
        double sum = 0.0;
        for (int d = 0; d < nc; d++)
        {
            const int id3 = (((int)p * ns + q) * ns + r) * nc + d;
            sum += (double)U_row_nc_ns[d * ns + s] * (double)t3[id3];
        }
        eri_s[idx] = (float)sum;
    }
}

void QUANTUM_CHEMISTRY::Build_Cart2Sph_Matrix()
{
    int nao_c = mol.nao_cart;
    int nao_s = mol.nao_sph;
    std::vector<float> cart2sph_mat(nao_c * nao_s, 0.0f);

    int offset_c = 0;
    int offset_s = 0;

    for (int k = 0; k < mol.h_l_list.size(); k++)
    {
        int l = mol.h_l_list[k];
        int dim_c = (l + 1) * (l + 2) / 2;
        int dim_s = 2 * l + 1;

        switch (l)
        {
            case 0:  // s 轨道
                cart2sph_mat[offset_c * nao_s + offset_s] = 1.0f;
                break;
            case 1:  // p 轨道
                // 球谐 p 顺序：p_y, p_z, p_x（m=-1,0,1）
                // 笛卡尔 p 顺序：x, y, z
                // 映射关系：球谐 0/1/2 -> 笛卡尔 1/2/0
                cart2sph_mat[(offset_c + 1) * nao_s + (offset_s + 0)] = 1.0f;
                cart2sph_mat[(offset_c + 2) * nao_s + (offset_s + 1)] = 1.0f;
                cart2sph_mat[(offset_c + 0) * nao_s + (offset_s + 2)] = 1.0f;
                break;
            case 2:  // d 轨道
                for (int i = 0; i < 6; i++)
                {
                    for (int j = 0; j < 5; j++)
                    {
                        cart2sph_mat[(offset_c + i) * nao_s + (offset_s + j)] =
                            CART2SPH_MAT_D[i][j];
                    }
                }
                break;
            case 3:  // f 轨道
                for (int i = 0; i < 10; i++)
                {
                    for (int j = 0; j < 7; j++)
                    {
                        cart2sph_mat[(offset_c + i) * nao_s + (offset_s + j)] =
                            CART2SPH_MAT_F[i][j];
                    }
                }
                break;
            case 4:  // g 轨道
                for (int i = 0; i < 15; i++)
                {
                    for (int j = 0; j < 9; j++)
                    {
                        cart2sph_mat[(offset_c + i) * nao_s + (offset_s + j)] =
                            CART2SPH_MAT_G[i][j];
                    }
                }
                break;
            default:
                printf("Error: l=%d not supported in Cart2Sph transform\n", l);
                exit(1);
        }
        offset_c += dim_c;
        offset_s += dim_s;
    }
    if (cart2sph_mat.empty())
    {
        cart2sph.d_cart2sph_mat = nullptr;
    }
    else
    {
        Device_Malloc_Safely((void**)&cart2sph.d_cart2sph_mat,
                             sizeof(float) * cart2sph_mat.size());
        deviceMemcpy(cart2sph.d_cart2sph_mat, cart2sph_mat.data(),
                     sizeof(float) * cart2sph_mat.size(),
                     deviceMemcpyHostToDevice);
    }
}

void QUANTUM_CHEMISTRY::Cart2Sph_OneE_Integrals()
{
    if (!mol.is_spherical) return;
    const int nao_c = mol.nao_cart;
    const int nao_s = mol.nao_sph;
    const int threads = 256;
    const int total = nao_s * nao_s;
    auto cart2sph_1e = [&](float* d_src, float* d_dst)
    {
        QC_MatMul_RowRow_Blas(blas_handle, nao_c, nao_s, nao_c, d_src,
                              cart2sph.d_cart2sph_mat,
                              cart2sph.d_cart2sph_1e_tmp);
        Launch_Device_Kernel(QC_Cart2Sph_MatMul_UT_RowRow_Kernel,
                             (total + threads - 1) / threads, threads, 0, 0,
                             nao_s, nao_s, nao_c, cart2sph.d_cart2sph_mat,
                             cart2sph.d_cart2sph_1e_tmp, d_dst);
    };
    cart2sph_1e(cart2sph.d_S_cart, scf_ws.d_S);
    cart2sph_1e(cart2sph.d_T_cart, scf_ws.d_T);
    cart2sph_1e(cart2sph.d_V_cart, scf_ws.d_V);
}

void QUANTUM_CHEMISTRY::Cart2Sph_ERI()
{
    if (!mol.is_spherical) return;

    // (ab|cd) -> (pb|cd) -> (pq|cd) -> (pqr|d) -> (pq|rs)
    const int nc = mol.nao_cart;
    const int ns = mol.nao_sph;
    const int n_t1 = ns * nc * nc * nc;
    const int n_t2 = ns * ns * nc * nc;
    const int n_t3 = ns * ns * ns * nc;
    const int n_s = ns * ns * ns * ns;

    const int threads = 256;
    Launch_Device_Kernel(QC_ERI_Cart2Sph_Stage1_Kernel,
                         (n_t1 + threads - 1) / threads, threads, 0, 0, n_t1,
                         nc, ns, cart2sph.d_cart2sph_mat, cart2sph.d_ERI_cart,
                         cart2sph.d_cart2sph_eri_t1);
    Launch_Device_Kernel(
        QC_ERI_Cart2Sph_Stage2_Kernel, (n_t2 + threads - 1) / threads, threads,
        0, 0, n_t2, nc, ns, cart2sph.d_cart2sph_mat, cart2sph.d_cart2sph_eri_t1,
        cart2sph.d_cart2sph_eri_t2);
    Launch_Device_Kernel(
        QC_ERI_Cart2Sph_Stage3_Kernel, (n_t3 + threads - 1) / threads, threads,
        0, 0, n_t3, nc, ns, cart2sph.d_cart2sph_mat, cart2sph.d_cart2sph_eri_t2,
        cart2sph.d_cart2sph_eri_t3);
    Launch_Device_Kernel(QC_ERI_Cart2Sph_Stage4_Kernel,
                         (n_s + threads - 1) / threads, threads, 0, 0, n_s, nc,
                         ns, cart2sph.d_cart2sph_mat,
                         cart2sph.d_cart2sph_eri_t3, scf_ws.d_ERI);
}
