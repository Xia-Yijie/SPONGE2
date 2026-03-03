#pragma once

// ============================ 坐标同步 ===========================
// 从 MD 坐标更新 QC 的原子环境与壳层中心（含周期边界修正）
// ================================================================
static __global__ void QC_Update_Env_From_Crd_Kernel(
    const int natm, const int* atom_local, const VECTOR* crd, const int* atm,
    float* env, const float to_bohr, const VECTOR box_length)
{
    SIMPLE_DEVICE_FOR(i, natm)
    {
        const int md_idx = atom_local[i];
        const VECTOR r = crd[md_idx];
        const int ptr_coord = atm[i * 6 + 1];
        const VECTOR prev(env[ptr_coord + 0] / to_bohr,
                          env[ptr_coord + 1] / to_bohr,
                          env[ptr_coord + 2] / to_bohr);
        const VECTOR dr = Get_Periodic_Displacement(r, prev, box_length);
        env[ptr_coord + 0] = (prev.x + dr.x) * to_bohr;
        env[ptr_coord + 1] = (prev.y + dr.y) * to_bohr;
        env[ptr_coord + 2] = (prev.z + dr.z) * to_bohr;
    }
}

static __global__ void QC_Update_Centers_From_Env_Kernel(const int nbas,
                                                         const int* bas,
                                                         const int* atm,
                                                         const float* env,
                                                         VECTOR* centers)
{
    SIMPLE_DEVICE_FOR(ish, nbas)
    {
        const int iatm = bas[ish * 8 + 0];
        const int ptr_coord = atm[iatm * 6 + 1];
        centers[ish] = {env[ptr_coord + 0], env[ptr_coord + 1],
                        env[ptr_coord + 2]};
    }
}

void QUANTUM_CHEMISTRY::Update_Coordinates_From_MD(const VECTOR* crd,
                                                   const VECTOR box_length)
{
    const int threads = 256;
    Launch_Device_Kernel(QC_Update_Env_From_Crd_Kernel,
                         (mol.natm + threads - 1) / threads, threads, 0, 0,
                         mol.natm, d_atom_local, crd, mol.d_atm, mol.d_env,
                         CONSTANT_ANGSTROM_TO_BOHR, box_length);
    Launch_Device_Kernel(QC_Update_Centers_From_Env_Kernel,
                         (mol.nbas + threads - 1) / threads, threads, 0, 0,
                         mol.nbas, mol.d_bas, mol.d_atm, mol.d_env,
                         mol.d_centers);
}

// ========================== SCF 状态重置 =========================
// 清零本轮 SCF 的收敛标志、能量缓存与密度矩阵，并重置 DIIS 历史
// ================================================================
void QUANTUM_CHEMISTRY::Reset_SCF_State()
{
    const int nao2 = mol.nao2;

    scf_ws.diis_hist_count = scf_ws.diis_hist_head = 0;
    scf_ws.diis_hist_count_b = scf_ws.diis_hist_head_b = 0;

    deviceMemset(scf_ws.d_scf_energy, 0, sizeof(double));
    deviceMemset(scf_ws.d_converged, 0, sizeof(int));
    deviceMemset(scf_ws.d_P, 0, sizeof(float) * nao2);
    if (scf_ws.unrestricted)
        deviceMemset(scf_ws.d_P_b, 0, sizeof(float) * nao2);
}

// =========================== 单电子积分 ===========================
// 计算 S/T/V 单电子积分，并在球谐基下执行笛卡尔到球谐变换
// ================================================================
void QUANTUM_CHEMISTRY::Compute_OneE_Integrals()
{
    const int nao_c = mol.nao_cart;
    float* p_S = mol.is_spherical ? cart2sph.d_S_cart : scf_ws.d_S;
    float* p_T = mol.is_spherical ? cart2sph.d_T_cart : scf_ws.d_T;
    float* p_V = mol.is_spherical ? cart2sph.d_V_cart : scf_ws.d_V;

    deviceMemset(p_S, 0, sizeof(float) * nao_c * nao_c);
    deviceMemset(p_T, 0, sizeof(float) * nao_c * nao_c);
    deviceMemset(p_V, 0, sizeof(float) * nao_c * nao_c);

    const int chunk_size = ONE_E_BATCH_SIZE;
    for (int i = 0; i < task_ctx.n_1e_tasks; i += chunk_size)
    {
        int current_chunk = std::min(chunk_size, task_ctx.n_1e_tasks - i);
        QC_ONE_E_TASK* task_ptr = task_ctx.d_1e_tasks + i;
        Launch_Device_Kernel(
            OneE_Kernel, (current_chunk + 63) / 64, 64, 0, 0, current_chunk,
            task_ptr, mol.d_centers, mol.d_l_list, mol.d_exps, mol.d_coeffs,
            mol.d_shell_offsets, mol.d_shell_sizes, mol.d_ao_offsets, mol.d_atm,
            mol.d_env, mol.natm, p_S, p_T, p_V, nao_c);
    }
    Cart2Sph_OneE_Integrals();
}

// ============================ 核排斥能 ===========================
// 累加核间库仑排斥能，结果写入设备侧 d_nuc_energy_dev
// ================================================================
static __global__ void QC_Accumulate_Nuclear_Repulsion_Kernel(
    const int natm, const int* z_nuc, const int* atm, const float* env,
    double* e_nuc, const VECTOR box_length)
{
    SIMPLE_DEVICE_FOR(i, natm)
    {
        const int ptr_i = atm[i * 6 + 1];
        const double zi = (double)z_nuc[i];
        const VECTOR ri(env[ptr_i + 0], env[ptr_i + 1], env[ptr_i + 2]);
        double local = 0.0;
        for (int j = i + 1; j < natm; j++)
        {
            const int ptr_j = atm[j * 6 + 1];
            const double zj = (double)z_nuc[j];
            const VECTOR rj(env[ptr_j + 0], env[ptr_j + 1], env[ptr_j + 2]);
            const VECTOR dr = Get_Periodic_Displacement(ri, rj, box_length);
            const double r = sqrt((double)dr.x * dr.x + (double)dr.y * dr.y +
                                  (double)dr.z * dr.z);
            local += zi * zj / fmax(r, 1e-12);
        }
        atomicAdd(e_nuc, local);
    }
}

void QUANTUM_CHEMISTRY::Compute_Nuclear_Repulsion(const VECTOR box_length)
{
    deviceMemset(scf_ws.d_nuc_energy_dev, 0, sizeof(double));
    const int threads = 256;
    Launch_Device_Kernel(QC_Accumulate_Nuclear_Repulsion_Kernel,
                         (mol.natm + threads - 1) / threads, threads, 0, 0,
                         mol.natm, mol.d_Z, mol.d_atm, mol.d_env,
                         scf_ws.d_nuc_energy_dev, box_length);
}

// =========================== 积分预处理 ===========================
// 归一化单电子积分并构建 Hcore，同时计算、变换和归一化 ERI
// ================================================================
static __global__ void QC_Build_Norms_From_S_Kernel(const int nao,
                                                    const float* S,
                                                    float* norms)
{
    SIMPLE_DEVICE_FOR(i, nao)
    {
        float sii = S[i * nao + i];
        norms[i] = 1.0f / sqrtf(fmaxf(sii, 1e-20f));
    }
}

static __global__ void QC_Scale_OneE_And_Build_Hcore_Kernel(const int nao,
                                                            const float* norms,
                                                            float* S, float* T,
                                                            float* V,
                                                            float* H_core)
{
    const int total = nao * nao;
    SIMPLE_DEVICE_FOR(idx, total)
    {
        int i = idx / nao;
        int j = idx - i * nao;
        float scale = norms[i] * norms[j];
        S[idx] *= scale;
        T[idx] *= scale;
        V[idx] *= scale;
        H_core[idx] = T[idx] + V[idx];
    }
}

static __global__ void QC_Normalize_ERI_Kernel(const int eri4, const int nao,
                                               const float* norms, float* ERI)
{
    SIMPLE_DEVICE_FOR(idx, eri4)
    {
        int t = idx;
        int l = t % nao;
        t /= nao;
        int k = t % nao;
        t /= nao;
        int j = t % nao;
        t /= nao;
        ERI[idx] *= norms[t] * norms[j] * norms[k] * norms[l];
    }
}

void QUANTUM_CHEMISTRY::Prepare_Integrals()
{
    const int nao_c = mol.nao_cart;
    const int nao = mol.nao;
    const int nao2 = mol.nao2;
    const int eri4 = nao2 * nao2;
    const int eri4_calc = nao_c * nao_c * nao_c * nao_c;

    // 单电子积分归一化合并至 Hcore
    const int threads = 256;
    Launch_Device_Kernel(QC_Build_Norms_From_S_Kernel,
                         (nao + threads - 1) / threads, threads, 0, 0, nao,
                         scf_ws.d_S, scf_ws.d_norms);
    Launch_Device_Kernel(QC_Scale_OneE_And_Build_Hcore_Kernel,
                         (nao2 + threads - 1) / threads, threads, 0, 0, nao,
                         scf_ws.d_norms, scf_ws.d_S, scf_ws.d_T, scf_ws.d_V,
                         scf_ws.d_H_core);

    // 计算双电子积分 ERI
    float* p_ERI = mol.is_spherical ? cart2sph.d_ERI_cart : scf_ws.d_ERI;
    deviceMemset(p_ERI, 0, sizeof(float) * eri4_calc);
    const int chunk_size = ERI_BATCH_SIZE;
    const int eri_threads = 64;
    for (int i = 0; i < task_ctx.n_eri_tasks; i += chunk_size)
    {
        int current_chunk = std::min(chunk_size, task_ctx.n_eri_tasks - i);
        QC_ERI_TASK* task_ptr = task_ctx.d_eri_tasks + i;
        Launch_Device_Kernel(
            ERI_Kernel, (current_chunk + eri_threads - 1) / eri_threads,
            eri_threads, 0, 0, current_chunk, task_ptr, mol.d_atm, mol.d_bas,
            mol.d_env, mol.d_ao_loc, p_ERI, d_hr_pool, nao_c,
            task_ctx.eri_hr_base, task_ctx.eri_hr_size,
            task_ctx.eri_shell_buf_size, task_ctx.eri_prim_screen_tol);
    }

    // 球谐变换 + 归一化 ERI
    if (mol.is_spherical) Cart2Sph_ERI();
    Launch_Device_Kernel(QC_Normalize_ERI_Kernel,
                         (eri4 + threads - 1) / threads, threads, 0, 0, eri4,
                         nao, scf_ws.d_norms, scf_ws.d_ERI);
}

// ========================= 重叠正交化矩阵 =========================
// 对重叠矩阵 S 做本征分解，并构建正交化变换矩阵 X
// ================================================================
static __global__ void QC_Build_X_From_EigCol_Kernel(const int nao,
                                                     const float* U_col,
                                                     const float* W,
                                                     float* X_row)
{
#ifdef USE_GPU
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nao || j >= nao) return;
#else
#pragma omp parallel for collapse(2)
    for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++)
#endif
    {
        double sum = 0.0;
        for (int k = 0; k < nao; k++)
        {
            float wk = fmaxf(W[k], 1e-10f);
            float uik = U_col[i + k * nao];
            float ujk = U_col[j + k * nao];
            sum += (double)uik * (double)ujk / sqrt((double)wk);
        }
        X_row[i * nao + j] = (float)sum;
    }
}

void QUANTUM_CHEMISTRY::Build_Overlap_X()
{
    const int nao = mol.nao;
    const int nao2 = mol.nao2;

    deviceMemcpy(scf_ws.d_Work, scf_ws.d_S, sizeof(float) * nao2,
                 deviceMemcpyDeviceToDevice);
    QC_Diagonalize(solver_handle, mol.nao, scf_ws.d_Work, scf_ws.d_W,
                   scf_ws.d_solver_work, scf_ws.lwork, scf_ws.d_solver_iwork,
                   scf_ws.liwork, scf_ws.d_info);

    const dim3 block2d(16, 16);
    const dim3 grid2d((nao + block2d.x - 1) / block2d.x,
                      (nao + block2d.y - 1) / block2d.y);
    Launch_Device_Kernel(QC_Build_X_From_EigCol_Kernel, grid2d, block2d, 0, 0,
                         nao, scf_ws.d_Work, scf_ws.d_W, scf_ws.d_X);
}
