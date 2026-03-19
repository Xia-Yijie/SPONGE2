#pragma once

#ifndef USE_GPU
static inline int QC_Count_Active_Partners_By_Bound(
    const std::vector<int>& sorted_pair_ids, const float* shell_pair_bounds,
    const float threshold)
{
    int low = 0;
    int high = (int)sorted_pair_ids.size();
    while (low < high)
    {
        const int mid = low + (high - low) / 2;
        if (shell_pair_bounds[sorted_pair_ids[mid]] >= threshold)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

static inline int QC_Count_Active_Partners_By_Activity(
    const std::vector<int>& sorted_pair_ids, const float* pair_activity,
    const float threshold)
{
    int low = 0;
    int high = (int)sorted_pair_ids.size();
    while (low < high)
    {
        const int mid = low + (high - low) / 2;
        if (pair_activity[sorted_pair_ids[mid]] >= threshold)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

static inline float QC_Exact_Quartet_Screen_CPU(
    const QC_INTEGRAL_TASKS& task_ctx, const int pair_ij, const int pair_kl,
    const float* shell_pair_bounds, const float* pair_density_coul,
    const float* pair_density_exx_a, const float* pair_density_exx_b,
    const float exx_scale_a, const float exx_scale_b)
{
    const QC_ONE_E_TASK& ij = task_ctx.h_shell_pairs[pair_ij];
    const QC_ONE_E_TASK& kl = task_ctx.h_shell_pairs[pair_kl];
    const int ik_pair = QC_Shell_Pair_Index(ij.x, kl.x);
    const int il_pair = QC_Shell_Pair_Index(ij.x, kl.y);
    const int jk_pair = QC_Shell_Pair_Index(ij.y, kl.x);
    const int jl_pair = QC_Shell_Pair_Index(ij.y, kl.y);

    const float shell_bound =
        shell_pair_bounds[pair_ij] * shell_pair_bounds[pair_kl];
    const float coul_screen =
        shell_bound *
        fmaxf(pair_density_coul[pair_ij], pair_density_coul[pair_kl]);
    const float exx_screen_a =
        exx_scale_a == 0.0f
            ? 0.0f
            : shell_bound * exx_scale_a *
                  QC_Max4(pair_density_exx_a[ik_pair], pair_density_exx_a[il_pair],
                          pair_density_exx_a[jk_pair], pair_density_exx_a[jl_pair]);
    float exx_screen_b = 0.0f;
    if (pair_density_exx_b != NULL && exx_scale_b != 0.0f)
    {
        exx_screen_b =
            shell_bound * exx_scale_b *
            QC_Max4(pair_density_exx_b[ik_pair], pair_density_exx_b[il_pair],
                    pair_density_exx_b[jk_pair], pair_density_exx_b[jl_pair]);
    }
    return fmaxf(coul_screen, fmaxf(exx_screen_a, exx_screen_b));
}

struct QC_Shell_Pair_Meta_CPU
{
    int sh[2];
    int l[2];
    int np[2];
    int p_exp[2];
    int p_cof[2];
    int dims_cart[2];
    int dims_sph[2];
    int dims_eff[2];
    int off_cart[2];
    int off_eff[2];
    float R[2][3];
    float pair_dist2;
    int comp_x[2][MAX_CART_SHELL];
    int comp_y[2][MAX_CART_SHELL];
    int comp_z[2][MAX_CART_SHELL];
};

struct QC_Bra_Prim_Cache_CPU
{
    float P[3];
    float inv_p;
    float n_ab;
    float E_bra[3][5][5][9];
};

static inline void QC_Init_Shell_Pair_Meta_CPU(
    const QC_ONE_E_TASK& pair, const int* atm, const int* bas,
    const float* env, const int* ao_offsets_cart, const int* ao_offsets_sph,
    const int is_spherical, QC_Shell_Pair_Meta_CPU& meta)
{
    meta.sh[0] = pair.x;
    meta.sh[1] = pair.y;
    for (int s = 0; s < 2; s++)
    {
        const int sh = meta.sh[s];
        meta.l[s] = bas[sh * 8 + 1];
        meta.np[s] = bas[sh * 8 + 2];
        meta.p_exp[s] = bas[sh * 8 + 5];
        meta.p_cof[s] = bas[sh * 8 + 6];
        meta.dims_cart[s] = (meta.l[s] + 1) * (meta.l[s] + 2) / 2;
        meta.dims_sph[s] = 2 * meta.l[s] + 1;
        meta.dims_eff[s] = QC_Shell_Dim(meta.l[s], is_spherical);
        meta.off_cart[s] = ao_offsets_cart[sh];
        meta.off_eff[s] =
            is_spherical ? ao_offsets_sph[sh] : meta.off_cart[s];
        const int ptr_R = atm[bas[sh * 8 + 0] * 6 + 1];
        meta.R[s][0] = env[ptr_R + 0];
        meta.R[s][1] = env[ptr_R + 1];
        meta.R[s][2] = env[ptr_R + 2];
        for (int c = 0; c < meta.dims_cart[s]; c++)
        {
            QC_Get_Lxyz_Host(meta.l[s], c, meta.comp_x[s][c], meta.comp_y[s][c],
                             meta.comp_z[s][c]);
        }
    }
    meta.pair_dist2 =
        (meta.R[0][0] - meta.R[1][0]) * (meta.R[0][0] - meta.R[1][0]) +
        (meta.R[0][1] - meta.R[1][1]) * (meta.R[0][1] - meta.R[1][1]) +
        (meta.R[0][2] - meta.R[1][2]) * (meta.R[0][2] - meta.R[1][2]);
}

static inline void QC_Build_Bra_Prim_Cache_CPU(
    const QC_Shell_Pair_Meta_CPU& bra, const float* env,
    const float prim_screen_tol, std::vector<QC_Bra_Prim_Cache_CPU>& prims)
{
    prims.clear();
    prims.reserve((size_t)bra.np[0] * (size_t)bra.np[1]);
    for (int ip = 0; ip < bra.np[0]; ip++)
    {
        for (int jp = 0; jp < bra.np[1]; jp++)
        {
            const float ai = env[bra.p_exp[0] + ip];
            const float aj = env[bra.p_exp[1] + jp];
            const float p = ai + aj;
            const float inv_p = 1.0f / p;
            const float kab = expf(-(ai * aj * inv_p) * bra.pair_dist2);
            const float n_ab =
                env[bra.p_cof[0] + ip] * env[bra.p_cof[1] + jp] * kab;
            if (fabsf(n_ab) < prim_screen_tol) continue;

            QC_Bra_Prim_Cache_CPU prim = {};
            prim.P[0] = (ai * bra.R[0][0] + aj * bra.R[1][0]) * inv_p;
            prim.P[1] = (ai * bra.R[0][1] + aj * bra.R[1][1]) * inv_p;
            prim.P[2] = (ai * bra.R[0][2] + aj * bra.R[1][2]) * inv_p;
            prim.inv_p = inv_p;
            prim.n_ab = n_ab;
            for (int d = 0; d < 3; d++)
            {
                compute_md_coeffs(prim.E_bra[d], bra.l[0], bra.l[1],
                                  prim.P[d] - bra.R[0][d],
                                  prim.P[d] - bra.R[1][d], 0.5f * inv_p);
            }
            prims.push_back(prim);
        }
    }
}

static inline bool QC_Compute_Shell_Quartet_ERI_Buffer_CPU_BraCached(
    const QC_Shell_Pair_Meta_CPU& bra, const QC_Shell_Pair_Meta_CPU& ket,
    const float* env, const float* norms, const int is_spherical,
    const float* cart2sph_mat, const int nao_sph,
    const std::vector<QC_Bra_Prim_Cache_CPU>& bra_prims, float* HR,
    float* shell_eri, float* shell_tmp, int hr_base, int shell_buf_size,
    float prim_screen_tol, int* dims_eff, int* off_eff)
{
    const int dims_cart[4] = {bra.dims_cart[0], bra.dims_cart[1],
                              ket.dims_cart[0], ket.dims_cart[1]};
    const int dims_sph[4] = {bra.dims_sph[0], bra.dims_sph[1], ket.dims_sph[0],
                             ket.dims_sph[1]};
    const int off_cart[4] = {bra.off_cart[0], bra.off_cart[1], ket.off_cart[0],
                             ket.off_cart[1]};
    const int l[4] = {bra.l[0], bra.l[1], ket.l[0], ket.l[1]};
    dims_eff[0] = bra.dims_eff[0];
    dims_eff[1] = bra.dims_eff[1];
    dims_eff[2] = ket.dims_eff[0];
    dims_eff[3] = ket.dims_eff[1];
    off_eff[0] = bra.off_eff[0];
    off_eff[1] = bra.off_eff[1];
    off_eff[2] = ket.off_eff[0];
    off_eff[3] = ket.off_eff[1];

    const int shell_size =
        dims_cart[0] * dims_cart[1] * dims_cart[2] * dims_cart[3];
    if (shell_size > shell_buf_size) return false;
    for (int i = 0; i < shell_size; i++) shell_eri[i] = 0.0f;
    if (bra_prims.empty()) return true;

    float E_ket[3][5][5][9];
    for (const QC_Bra_Prim_Cache_CPU& prim : bra_prims)
    {
        const float p = 1.0f / prim.inv_p;
        for (int kp = 0; kp < ket.np[0]; kp++)
        {
            for (int lp = 0; lp < ket.np[1]; lp++)
            {
                const float ak = env[ket.p_exp[0] + kp];
                const float al = env[ket.p_exp[1] + lp];
                const float q = ak + al;
                const float inv_q = 1.0f / q;
                const float kcd = expf(-(ak * al * inv_q) * ket.pair_dist2);
                const float pref = 2.0f * PI_25 / (p * q * sqrtf(p + q));
                const float n_abcd = prim.n_ab * env[ket.p_cof[0] + kp] *
                                     env[ket.p_cof[1] + lp] * kcd * pref;
                if (fabsf(n_abcd) < prim_screen_tol) continue;

                float Q[3] = {(ak * ket.R[0][0] + al * ket.R[1][0]) * inv_q,
                              (ak * ket.R[0][1] + al * ket.R[1][1]) * inv_q,
                              (ak * ket.R[0][2] + al * ket.R[1][2]) * inv_q};
                const float alpha = p * q / (p + q);
                float PQ_val[3] = {prim.P[0] - Q[0], prim.P[1] - Q[1],
                                   prim.P[2] - Q[2]};
                const int L_sum = l[0] + l[1] + l[2] + l[3];
                float F_vals[17];
                compute_boys_stable(
                    F_vals,
                    alpha * (PQ_val[0] * PQ_val[0] + PQ_val[1] * PQ_val[1] +
                             PQ_val[2] * PQ_val[2]),
                    L_sum);
                compute_hr_tensor(HR, F_vals, alpha, PQ_val, L_sum, hr_base);

                for (int d = 0; d < 3; d++)
                {
                    compute_md_coeffs(E_ket[d], ket.l[0], ket.l[1],
                                      Q[d] - ket.R[0][d], Q[d] - ket.R[1][d],
                                      0.5f * inv_q);
                }

                for (int i = 0; i < bra.dims_cart[0]; i++)
                {
                    const int ix = bra.comp_x[0][i];
                    const int iy = bra.comp_y[0][i];
                    const int iz = bra.comp_z[0][i];
                    for (int j = 0; j < bra.dims_cart[1]; j++)
                    {
                        const int jx = bra.comp_x[1][j];
                        const int jy = bra.comp_y[1][j];
                        const int jz = bra.comp_z[1][j];
                        for (int k = 0; k < ket.dims_cart[0]; k++)
                        {
                            const int kx = ket.comp_x[0][k];
                            const int ky = ket.comp_y[0][k];
                            const int kz = ket.comp_z[0][k];
                            for (int l_idx = 0; l_idx < ket.dims_cart[1];
                                 l_idx++)
                            {
                                const int lx_l = ket.comp_x[1][l_idx];
                                const int ly_l = ket.comp_y[1][l_idx];
                                const int lz_l = ket.comp_z[1][l_idx];
                                float val = 0.0f;
                                for (int mux = 0; mux <= ix + jx; mux++)
                                {
                                    const float ex = prim.E_bra[0][ix][jx][mux];
                                    if (ex == 0.0f) continue;
                                    for (int muy = 0; muy <= iy + jy; muy++)
                                    {
                                        const float ey =
                                            prim.E_bra[1][iy][jy][muy];
                                        if (ey == 0.0f) continue;
                                        for (int muz = 0; muz <= iz + jz; muz++)
                                        {
                                            const float ez =
                                                prim.E_bra[2][iz][jz][muz];
                                            const float e_bra_val = ex * ey * ez;
                                            if (e_bra_val == 0.0f) continue;
                                            for (int nux = 0; nux <= kx + lx_l;
                                                 nux++)
                                            {
                                                const float dx =
                                                    E_ket[0][kx][lx_l][nux];
                                                if (dx == 0.0f) continue;
                                                for (int nuy = 0;
                                                     nuy <= ky + ly_l; nuy++)
                                                {
                                                    const float dy =
                                                        E_ket[1][ky][ly_l]
                                                              [nuy];
                                                    if (dy == 0.0f) continue;
                                                    for (int nuz = 0;
                                                         nuz <= kz + lz_l;
                                                         nuz++)
                                                    {
                                                        const float dz =
                                                            E_ket[2][kz][lz_l]
                                                                  [nuz];
                                                        const float sign_val =
                                                            ((nux + nuy + nuz) % 2 ==
                                                             0)
                                                                ? 1.0f
                                                                : -1.0f;
                                                        val +=
                                                            e_bra_val * dx * dy *
                                                            dz *
                                                            HR[HR_IDX_RUNTIME(
                                                                mux + nux,
                                                                muy + nuy,
                                                                muz + nuz, 0,
                                                                hr_base)] *
                                                            sign_val;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                shell_eri[QC_Shell_Buffer_Index(
                                    i, j, k, l_idx, dims_cart[1], dims_cart[2],
                                    dims_cart[3])] += val * n_abcd;
                            }
                        }
                    }
                }
            }
        }
    }

    if (is_spherical)
    {
        QC_Cart2Sph_Shell_ERI(cart2sph_mat, nao_sph, off_cart, off_eff,
                              dims_cart, dims_sph, shell_eri, shell_tmp);
    }

    for (int i = 0; i < dims_eff[0]; i++)
    {
        const float ni = norms[off_eff[0] + i];
        for (int j = 0; j < dims_eff[1]; j++)
        {
            const float nj = norms[off_eff[1] + j];
            for (int k = 0; k < dims_eff[2]; k++)
            {
                const float nk = norms[off_eff[2] + k];
                for (int l_idx = 0; l_idx < dims_eff[3]; l_idx++)
                {
                    const float nl = norms[off_eff[3] + l_idx];
                    const int idx =
                        QC_Shell_Buffer_Index(i, j, k, l_idx, dims_eff[1],
                                              dims_eff[2], dims_eff[3]);
                    shell_eri[idx] *= ni * nj * nk * nl;
                }
            }
        }
    }
    return true;
}

static inline void QC_Profile_Vector_CPU(const char* name, const int n,
                                         const float* x)
{
    double norm2 = 0.0;
    float max_abs = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < n; i++)
    {
        const float v = x[i];
        const float av = fabsf(v);
        norm2 += (double)v * (double)v;
        if (av > max_abs)
        {
            max_abs = av;
            max_idx = i;
        }
    }
    printf("Build_Fock CPU Vector         | %s norm=%.6e | max_abs=%.6e @%d\n",
           name, sqrt(norm2), (double)max_abs, max_idx);
}

static inline void QC_Profile_Matrix_CPU(const char* name, const int nao,
                                         const float* M)
{
    double frob2 = 0.0;
    float max_abs = 0.0f;
    float max_asym = 0.0f;
    int max_abs_i = 0, max_abs_j = 0;
    int max_asym_i = 0, max_asym_j = 0;
    for (int i = 0; i < nao; i++)
    {
        for (int j = 0; j < nao; j++)
        {
            const float v = M[i * nao + j];
            const float av = fabsf(v);
            frob2 += (double)v * (double)v;
            if (av > max_abs)
            {
                max_abs = av;
                max_abs_i = i;
                max_abs_j = j;
            }
            if (j > i)
            {
                const float asym = fabsf(v - M[j * nao + i]);
                if (asym > max_asym)
                {
                    max_asym = asym;
                    max_asym_i = i;
                    max_asym_j = j;
                }
            }
        }
    }
    printf(
        "Build_Fock CPU Matrix         | %s frob=%.6e | max_abs=%.6e @(%d,%d) | "
        "max_asym=%.6e @(%d,%d)\n",
        name, sqrt(frob2), (double)max_abs, max_abs_i, max_abs_j,
        (double)max_asym, max_asym_i, max_asym_j);
}

static inline void QC_Build_Fock_Direct_CPU(
    const QC_INTEGRAL_TASKS& task_ctx, const int nbas, const int* atm,
    const int* bas, const float* env, const int* ao_offsets_cart,
    const int* ao_offsets_sph, const float* norms,
    const float* shell_pair_bounds, const float* pair_density_coul,
    const float* pair_density_exx_a, const float* pair_density_exx_b,
    const float shell_screen_tol, const float* P_coul, const float* P_exx_a,
    const float* P_exx_b, const float exx_scale_a, const float exx_scale_b,
    const int nao, const int nao_sph, const int is_spherical,
    const float* cart2sph_mat, float* F_a, float* F_b, float* global_hr_pool,
    int hr_base, int hr_size, int shell_buf_size, float prim_screen_tol,
    const int fock_thread_count, const bool profile_build_fock)
{
    const int n_pairs = task_ctx.n_shell_pairs;
    if (n_pairs <= 0) return;

    std::vector<QC_Shell_Pair_Meta_CPU> pair_meta((size_t)n_pairs);
    for (int pair_id = 0; pair_id < n_pairs; pair_id++)
    {
        QC_Init_Shell_Pair_Meta_CPU(task_ctx.h_shell_pairs[pair_id], atm, bas,
                                    env, ao_offsets_cart, ao_offsets_sph,
                                    is_spherical, pair_meta[(size_t)pair_id]);
    }

    std::vector<float> shell_max_exx_a((size_t)nbas, 0.0f);
    std::vector<float> shell_max_exx_b((size_t)nbas, 0.0f);
    for (int pair_id = 0; pair_id < n_pairs; pair_id++)
    {
        const QC_ONE_E_TASK& pair = task_ctx.h_shell_pairs[pair_id];
        const float exx_a = pair_density_exx_a[pair_id];
        shell_max_exx_a[(size_t)pair.x] =
            fmaxf(shell_max_exx_a[(size_t)pair.x], exx_a);
        shell_max_exx_a[(size_t)pair.y] =
            fmaxf(shell_max_exx_a[(size_t)pair.y], exx_a);
        if (pair_density_exx_b != NULL)
        {
            const float exx_b = pair_density_exx_b[pair_id];
            shell_max_exx_b[(size_t)pair.x] =
                fmaxf(shell_max_exx_b[(size_t)pair.x], exx_b);
            shell_max_exx_b[(size_t)pair.y] =
                fmaxf(shell_max_exx_b[(size_t)pair.y], exx_b);
        }
    }

    std::vector<float> anchor_activity((size_t)n_pairs, 0.0f);
    std::vector<int> sorted_pair_ids((size_t)n_pairs, 0);
    for (int pair_id = 0; pair_id < n_pairs; pair_id++)
    {
        const QC_ONE_E_TASK& pair = task_ctx.h_shell_pairs[pair_id];
        const float exx_anchor_a =
            exx_scale_a == 0.0f
                ? 0.0f
                : exx_scale_a * fmaxf(shell_max_exx_a[(size_t)pair.x],
                                      shell_max_exx_a[(size_t)pair.y]);
        const float exx_anchor_b =
            (pair_density_exx_b == NULL || exx_scale_b == 0.0f)
                ? 0.0f
                : exx_scale_b * fmaxf(shell_max_exx_b[(size_t)pair.x],
                                      shell_max_exx_b[(size_t)pair.y]);
        anchor_activity[(size_t)pair_id] =
            shell_pair_bounds[pair_id] *
            QC_Max4(pair_density_coul[pair_id], exx_anchor_a, exx_anchor_b,
                    0.0f);
        sorted_pair_ids[(size_t)pair_id] = pair_id;
    }
    std::sort(sorted_pair_ids.begin(), sorted_pair_ids.end(),
              [shell_pair_bounds](const int lhs, const int rhs)
              { return shell_pair_bounds[lhs] > shell_pair_bounds[rhs]; });
    std::vector<int> sorted_activity_ids = sorted_pair_ids;
    std::sort(sorted_activity_ids.begin(), sorted_activity_ids.end(),
              [&anchor_activity](const int lhs, const int rhs)
              { return anchor_activity[(size_t)lhs] >
                       anchor_activity[(size_t)rhs]; });

    const float max_bound = shell_pair_bounds[sorted_pair_ids.front()];
    const float max_activity = anchor_activity[(size_t)sorted_activity_ids.front()];
    const int nao2 = nao * nao;
    double total_candidate_time = 0.0;
    double total_exact_screen_time = 0.0;
    double total_eri_time = 0.0;
    double total_jk_time = 0.0;
    long long total_anchor_pairs = 0;
    long long total_candidate_pairs = 0;
    long long total_exact_screen_calls = 0;
    long long total_screen_pass_quartets = 0;
    long long total_eri_quartets = 0;
    long long total_ao_unique_quartets = 0;
    float total_max_abs_eri = 0.0f;
    float total_max_ratio_eri = 0.0f;
    int total_max_abs_shells[4] = {0, 0, 0, 0};
    int total_max_abs_aos[4] = {0, 0, 0, 0};
    int total_max_ratio_shells[4] = {0, 0, 0, 0};
    int total_max_ratio_aos[4] = {0, 0, 0, 0};

#pragma omp parallel num_threads(fock_thread_count)
    {
        const int tid = omp_get_thread_num();
        float* F_a_accum = F_a + (size_t)tid * (size_t)nao2;
        float* F_b_accum =
            (F_b != NULL) ? (F_b + (size_t)tid * (size_t)nao2) : NULL;
        float* task_pool =
            global_hr_pool + (size_t)tid * (size_t)(hr_size + 2 * shell_buf_size);
        float* HR = task_pool;
        float* shell_eri = task_pool + hr_size;
        float* shell_tmp = shell_eri + shell_buf_size;
        std::vector<int> partner_marks((size_t)n_pairs, -1);
        std::vector<int> candidate_partners;
        candidate_partners.reserve(256);
        std::vector<QC_Bra_Prim_Cache_CPU> bra_prims;
        double thread_candidate_time = 0.0;
        double thread_exact_screen_time = 0.0;
        double thread_eri_time = 0.0;
        double thread_jk_time = 0.0;
        long long thread_anchor_pairs = 0;
        long long thread_candidate_pairs = 0;
        long long thread_exact_screen_calls = 0;
        long long thread_screen_pass_quartets = 0;
        long long thread_eri_quartets = 0;
        long long thread_ao_unique_quartets = 0;
        float thread_max_abs_eri = 0.0f;
        float thread_max_ratio_eri = 0.0f;
        int thread_max_abs_shells[4] = {0, 0, 0, 0};
        int thread_max_abs_aos[4] = {0, 0, 0, 0};
        int thread_max_ratio_shells[4] = {0, 0, 0, 0};
        int thread_max_ratio_aos[4] = {0, 0, 0, 0};

#pragma omp for schedule(dynamic)
        for (int pair_ij = 0; pair_ij < n_pairs; pair_ij++)
        {
            const QC_Shell_Pair_Meta_CPU& bra_meta = pair_meta[(size_t)pair_ij];
            const float activity_ij = anchor_activity[(size_t)pair_ij];
            if (fmaxf(activity_ij * max_bound,
                      shell_pair_bounds[pair_ij] * max_activity) <
                shell_screen_tol)
                continue;
            thread_anchor_pairs++;

            const double candidate_t0 =
                profile_build_fock ? omp_get_wtime() : 0.0;
            candidate_partners.clear();
            const int stamp = pair_ij;

            if (activity_ij > 0.0f)
            {
                const float bound_threshold = shell_screen_tol / activity_ij;
                const int bound_count = QC_Count_Active_Partners_By_Bound(
                    sorted_pair_ids, shell_pair_bounds, bound_threshold);
                for (int rank = 0; rank < bound_count; rank++)
                {
                    const int pair_kl = sorted_pair_ids[(size_t)rank];
                    if (pair_kl > pair_ij ||
                        partner_marks[(size_t)pair_kl] == stamp)
                        continue;
                    partner_marks[(size_t)pair_kl] = stamp;
                    candidate_partners.push_back(pair_kl);
                }
            }

            const float activity_threshold =
                shell_screen_tol / shell_pair_bounds[pair_ij];
            const int activity_count = QC_Count_Active_Partners_By_Activity(
                sorted_activity_ids, anchor_activity.data(), activity_threshold);
            for (int rank = 0; rank < activity_count; rank++)
            {
                const int pair_kl = sorted_activity_ids[(size_t)rank];
                if (pair_kl > pair_ij || partner_marks[(size_t)pair_kl] == stamp)
                    continue;
                partner_marks[(size_t)pair_kl] = stamp;
                candidate_partners.push_back(pair_kl);
            }
            if (profile_build_fock)
            {
                thread_candidate_time += omp_get_wtime() - candidate_t0;
            }
            thread_candidate_pairs += (long long)candidate_partners.size();
            QC_Build_Bra_Prim_Cache_CPU(bra_meta, env, prim_screen_tol,
                                        bra_prims);
            if (bra_prims.empty()) continue;

            for (const int pair_kl : candidate_partners)
            {
                const double exact_t0 =
                    profile_build_fock ? omp_get_wtime() : 0.0;
                const float exact_screen = QC_Exact_Quartet_Screen_CPU(
                    task_ctx, pair_ij, pair_kl, shell_pair_bounds,
                    pair_density_coul, pair_density_exx_a, pair_density_exx_b,
                    exx_scale_a, exx_scale_b);
                if (profile_build_fock)
                {
                    thread_exact_screen_time += omp_get_wtime() - exact_t0;
                }
                thread_exact_screen_calls++;
                if (exact_screen < shell_screen_tol) continue;
                thread_screen_pass_quartets++;

                const QC_ONE_E_TASK& ij = task_ctx.h_shell_pairs[pair_ij];
                const QC_ONE_E_TASK& kl = task_ctx.h_shell_pairs[pair_kl];
                const QC_Shell_Pair_Meta_CPU& ket_meta =
                    pair_meta[(size_t)pair_kl];
                const float quartet_bound =
                    shell_pair_bounds[pair_ij] * shell_pair_bounds[pair_kl];
                int dims_eff[4];
                int off_eff[4];
                const double eri_t0 =
                    profile_build_fock ? omp_get_wtime() : 0.0;
                const bool eri_ok =
                    QC_Compute_Shell_Quartet_ERI_Buffer_CPU_BraCached(
                        bra_meta, ket_meta, env, norms, is_spherical,
                        cart2sph_mat, nao_sph, bra_prims, HR, shell_eri,
                        shell_tmp, hr_base, shell_buf_size, prim_screen_tol,
                        dims_eff, off_eff);
                if (profile_build_fock)
                {
                    thread_eri_time += omp_get_wtime() - eri_t0;
                }
                if (!eri_ok) continue;
                thread_eri_quartets++;

                const double jk_t0 =
                    profile_build_fock ? omp_get_wtime() : 0.0;
                for (int i = 0; i < dims_eff[0]; i++)
                {
                    const int p = off_eff[0] + i;
                    for (int j = 0; j < dims_eff[1]; j++)
                    {
                        const int q = off_eff[1] + j;
                        for (int k = 0; k < dims_eff[2]; k++)
                        {
                            const int r = off_eff[2] + k;
                            for (int l_idx = 0; l_idx < dims_eff[3]; l_idx++)
                            {
                                const int s = off_eff[3] + l_idx;
                                if (ij.x == ij.y && q > p) continue;
                                if (kl.x == kl.y && s > r) continue;

                                const int pq_pair = QC_AO_Pair_Index(p, q);
                                const int rs_pair = QC_AO_Pair_Index(r, s);
                                if (ij.x == kl.x && ij.y == kl.y &&
                                    rs_pair > pq_pair)
                                    continue;

                                const float val = shell_eri[QC_Shell_Buffer_Index(
                                    i, j, k, l_idx, dims_eff[1], dims_eff[2],
                                    dims_eff[3])];
                                if (val == 0.0f) continue;
                                if (profile_build_fock)
                                {
                                    const float abs_val = fabsf(val);
                                    if (abs_val > thread_max_abs_eri)
                                    {
                                        thread_max_abs_eri = abs_val;
                                        thread_max_abs_shells[0] = ij.x;
                                        thread_max_abs_shells[1] = ij.y;
                                        thread_max_abs_shells[2] = kl.x;
                                        thread_max_abs_shells[3] = kl.y;
                                        thread_max_abs_aos[0] = p;
                                        thread_max_abs_aos[1] = q;
                                        thread_max_abs_aos[2] = r;
                                        thread_max_abs_aos[3] = s;
                                    }
                                    const float ratio =
                                        abs_val / fmaxf(quartet_bound, 1e-30f);
                                    if (ratio > thread_max_ratio_eri)
                                    {
                                        thread_max_ratio_eri = ratio;
                                        thread_max_ratio_shells[0] = ij.x;
                                        thread_max_ratio_shells[1] = ij.y;
                                        thread_max_ratio_shells[2] = kl.x;
                                        thread_max_ratio_shells[3] = kl.y;
                                        thread_max_ratio_aos[0] = p;
                                        thread_max_ratio_aos[1] = q;
                                        thread_max_ratio_aos[2] = r;
                                        thread_max_ratio_aos[3] = s;
                                    }
                                }
                                thread_ao_unique_quartets++;
                                QC_Accumulate_Fock_Unique_Quartet(
                                    p, q, r, s, val, nao, P_coul, P_exx_a,
                                    P_exx_b, exx_scale_a, exx_scale_b,
                                    F_a_accum, F_b_accum);
                            }
                        }
                    }
                }
                if (profile_build_fock)
                {
                    thread_jk_time += omp_get_wtime() - jk_t0;
                }
            }
        }

#pragma omp critical
        {
            total_candidate_time += thread_candidate_time;
            total_exact_screen_time += thread_exact_screen_time;
            total_eri_time += thread_eri_time;
            total_jk_time += thread_jk_time;
            total_anchor_pairs += thread_anchor_pairs;
            total_candidate_pairs += thread_candidate_pairs;
            total_exact_screen_calls += thread_exact_screen_calls;
            total_screen_pass_quartets += thread_screen_pass_quartets;
            total_eri_quartets += thread_eri_quartets;
            total_ao_unique_quartets += thread_ao_unique_quartets;
            if (thread_max_abs_eri > total_max_abs_eri)
            {
                total_max_abs_eri = thread_max_abs_eri;
                for (int i = 0; i < 4; i++)
                {
                    total_max_abs_shells[i] = thread_max_abs_shells[i];
                    total_max_abs_aos[i] = thread_max_abs_aos[i];
                }
            }
            if (thread_max_ratio_eri > total_max_ratio_eri)
            {
                total_max_ratio_eri = thread_max_ratio_eri;
                for (int i = 0; i < 4; i++)
                {
                    total_max_ratio_shells[i] = thread_max_ratio_shells[i];
                    total_max_ratio_aos[i] = thread_max_ratio_aos[i];
                }
            }
        }
    }

    if (profile_build_fock)
    {
        printf(
            "Build_Fock CPU Direct Profile | anchor_pairs=%lld | candidate_pairs=%lld | "
            "exact_calls=%lld | exact_pass=%lld | eri_quartets=%lld | "
            "ao_unique=%lld\n",
            total_anchor_pairs, total_candidate_pairs, total_exact_screen_calls,
            total_screen_pass_quartets, total_eri_quartets,
            total_ao_unique_quartets);
        printf(
            "Build_Fock CPU Direct Time    | candidate=%.6f s | exact=%.6f s | "
            "eri=%.6f s | jk=%.6f s\n",
            total_candidate_time, total_exact_screen_time, total_eri_time,
            total_jk_time);
        printf(
            "Build_Fock CPU ERI Check      | max_abs=%.6e | shells=(%d,%d|%d,%d) | "
            "aos=(%d,%d|%d,%d)\n",
            (double)total_max_abs_eri, total_max_abs_shells[0],
            total_max_abs_shells[1], total_max_abs_shells[2],
            total_max_abs_shells[3], total_max_abs_aos[0],
            total_max_abs_aos[1], total_max_abs_aos[2], total_max_abs_aos[3]);
        printf(
            "Build_Fock CPU Schwarz Check  | max_ratio=%.6e | shells=(%d,%d|%d,%d) | "
            "aos=(%d,%d|%d,%d)\n",
            (double)total_max_ratio_eri, total_max_ratio_shells[0],
            total_max_ratio_shells[1], total_max_ratio_shells[2],
            total_max_ratio_shells[3], total_max_ratio_aos[0],
            total_max_ratio_aos[1], total_max_ratio_aos[2],
            total_max_ratio_aos[3]);
    }
}

static __global__ void QC_Reduce_Thread_Fock_Kernel(const int total,
                                                    const int n_threads,
                                                    const float* F_thread,
                                                    float* F_out)
{
    SIMPLE_DEVICE_FOR(idx, total)
    {
        float sum = F_out[idx];
        for (int tid = 0; tid < n_threads; tid++)
        {
            sum += F_thread[(size_t)tid * (size_t)total + (size_t)idx];
        }
        F_out[idx] = sum;
    }
}
#endif
