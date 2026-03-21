#pragma once

static __global__ void QC_DIIS_Init_System_Kernel(const int n, const int m,
                                                  double* B, double* rhs)
{
    for (int i = 0; i < n; i++)
    {
        rhs[i] = 0.0;
        for (int j = 0; j < n; j++) B[i * n + j] = 0.0;
    }
    rhs[m] = -1.0;
    for (int i = 0; i < m; i++)
    {
        B[i * n + m] = -1.0;
        B[m * n + i] = -1.0;
    }
    B[m * n + m] = 0.0;
}

static __global__ void QC_DIIS_Set_B_From_Accum_Kernel(const int n, const int i,
                                                       const int j,
                                                       const double reg,
                                                       const double* d_accum,
                                                       double* B)
{
    const double v = d_accum[0] + ((i == j) ? reg : 0.0);
    B[i * n + j] = v;
    B[j * n + i] = v;
}

static __global__ void QC_DIIS_Solve_Linear_System_Kernel(const int n,
                                                          double* A, double* b,
                                                          int* info)
{
    info[0] = 0;
    for (int k = 0; k < n; k++)
    {
        int pivot = k;
        double max_abs = fabs(A[k * n + k]);
        for (int i = k + 1; i < n; i++)
        {
            const double v = fabs(A[i * n + k]);
            if (v > max_abs)
            {
                max_abs = v;
                pivot = i;
            }
        }
        if (max_abs < 1e-18)
        {
            info[0] = k + 1;
            return;
        }
        if (pivot != k)
        {
            for (int j = k; j < n; j++)
            {
                const double tmp = A[k * n + j];
                A[k * n + j] = A[pivot * n + j];
                A[pivot * n + j] = tmp;
            }
            const double tb = b[k];
            b[k] = b[pivot];
            b[pivot] = tb;
        }
        const double diag = A[k * n + k];
        for (int i = k + 1; i < n; i++)
        {
            const double factor = A[i * n + k] / diag;
            A[i * n + k] = 0.0;
            for (int j = k + 1; j < n; j++)
                A[i * n + j] -= factor * A[k * n + j];
            b[i] -= factor * b[k];
        }
    }
    for (int i = n - 1; i >= 0; i--)
    {
        double sum = b[i];
        for (int j = i + 1; j < n; j++) sum -= A[i * n + j] * b[j];
        const double diag = A[i * n + i];
        if (fabs(diag) < 1e-18)
        {
            info[0] = i + 1;
            return;
        }
        b[i] = sum / diag;
    }
}

// Compute DIIS error e = FPS - SPF in double precision
static void QC_Build_DIIS_Error_Double(int nao, const double* d_F,
                                       const float* d_P, const float* d_S,
                                       double* d_err)
{
    const int nao2 = nao * nao;
    std::vector<double> dP(nao2), dS(nao2);
    std::vector<double> t1(nao2), t2(nao2), t3(nao2);
    for (int i = 0; i < nao2; i++)
    {
        dP[i] = (double)d_P[i];
        dS[i] = (double)d_S[i];
    }
    const double one = 1.0, zero = 0.0;
    // t1 = F * P
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nao, nao, nao, one, d_F, nao, dP.data(), nao,
                zero, t1.data(), nao);
    // t2 = FP * S
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nao, nao, nao, one, t1.data(), nao, dS.data(), nao,
                zero, t2.data(), nao);
    // t1 = S * P
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nao, nao, nao, one, dS.data(), nao, dP.data(), nao,
                zero, t1.data(), nao);
    // t3 = SP * F
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nao, nao, nao, one, t1.data(), nao, d_F, nao,
                zero, t3.data(), nao);
    // err = FPS - SPF
    for (int i = 0; i < nao2; i++)
        d_err[i] = t2[i] - t3[i];
}

static void QC_DIIS_History_Push_Double(int nao2, int diis_space,
                                        int& hist_count, int& hist_head,
                                        double** d_f_hist, double** d_e_hist,
                                        const double* d_f_new,
                                        const double* d_e_new)
{
    if (diis_space <= 0) return;
    const int bytes = sizeof(double) * nao2;
    int write_idx = 0;
    if (hist_count < diis_space)
    {
        write_idx = (hist_head + hist_count) % diis_space;
        hist_count++;
    }
    else
    {
        write_idx = hist_head;
        hist_head = (hist_head + 1) % diis_space;
        hist_count = diis_space;
    }
    memcpy(d_f_hist[write_idx], d_f_new, bytes);
    memcpy(d_e_hist[write_idx], d_e_new, bytes);
}

// Dot product of two double arrays
static double QC_Double_Dot(int n, const double* a, const double* b)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

static bool QC_DIIS_Extrapolate_Double(int nao, int diis_space, int hist_count,
                                       int hist_head, double** d_f_hist,
                                       double** d_e_hist, double reg,
                                       double* d_f_out, double* d_B,
                                       double* d_rhs, int* d_info)
{
    if (hist_count < 2 || diis_space <= 0) return false;
    const int m = std::min(hist_count, diis_space);
    if (m < 2) return false;
    const int n = m + 1;
    const int nao2 = nao * nao;
    auto hist_idx = [&](int logical_idx) -> int
    { return (hist_head + logical_idx) % diis_space; };

    // Build B matrix and rhs
    for (int i = 0; i < n; i++)
    {
        d_rhs[i] = 0.0;
        for (int j = 0; j < n; j++) d_B[i * n + j] = 0.0;
    }
    d_rhs[m] = -1.0;
    for (int i = 0; i < m; i++)
    {
        d_B[i * n + m] = -1.0;
        d_B[m * n + i] = -1.0;
    }
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            double v = QC_Double_Dot(nao2, d_e_hist[hist_idx(i)],
                                     d_e_hist[hist_idx(j)]);
            if (i == j) v += reg;
            d_B[i * n + j] = v;
            d_B[j * n + i] = v;
        }
    }

    // Solve using eigenvalue decomposition (PySCF approach)
    // This handles ill-conditioned B matrices by filtering small eigenvalues
    d_info[0] = 0;
    {
        // dsyev workspace query and solve (B is symmetric)
        // B is stored row-major; dsyev expects col-major
        // For symmetric matrix, row-major = col-major (transpose of symmetric = itself)
        std::vector<double> H(n * n);
        memcpy(H.data(), d_B, sizeof(double) * n * n);
        std::vector<double> w(n);
        int lwork_q = -1;
        double wq;
        LAPACKE_dsyev_work(LAPACK_COL_MAJOR, 'V', 'U', n,
                           H.data(), n, w.data(), &wq, lwork_q);
        int lwork = (int)wq;
        std::vector<double> work(lwork);
        int info = LAPACKE_dsyev_work(LAPACK_COL_MAJOR, 'V', 'U', n,
                                      H.data(), n, w.data(),
                                      work.data(), lwork);
        if (info != 0) { d_info[0] = info; return false; }

        // H now contains eigenvectors (col-major)
        // c = V * (1/w) * (V^T * g), filtering |w| < 1e-14
        // g = d_rhs = [0, 0, ..., -1]
        std::vector<double> c(n, 0.0);
        for (int k = 0; k < n; k++)
        {
            if (fabs(w[k]) < 1e-14) continue;
            // v_k^T * g
            double vg = 0.0;
            for (int i = 0; i < n; i++)
                vg += H[k * n + i] * d_rhs[i];  // col-major: H[i, k] = H[k*n+i]
            double coeff = vg / w[k];
            for (int i = 0; i < n; i++)
                c[i] += coeff * H[k * n + i];
        }
        memcpy(d_rhs, c.data(), sizeof(double) * n);
    }

    // Print DIIS coefficients and B matrix diagonal
    printf("DIIS extrap: m=%d, coeffs=[", m);
    for (int i = 0; i < m; i++) printf("%.6f%s", d_rhs[i], i<m-1?", ":"");
    printf("], B_diag=[");
    for (int i = 0; i < m; i++) printf("%.6e%s", d_B[i*(m+1)+i], i<m-1?", ":"");
    printf("]\n");
    fflush(stdout);

    // Extrapolate: F_out = sum_i c[i] * F_hist[i]
    memset(d_f_out, 0, sizeof(double) * nao2);
    for (int i = 0; i < m; i++)
    {
        double c = d_rhs[i];
        const double* fh = d_f_hist[hist_idx(i)];
        for (int idx = 0; idx < nao2; idx++)
            d_f_out[idx] += c * fh[idx];
    }
    return true;
}

static void QC_DIIS_Reset(int& hist_count, int& hist_head)
{
    hist_count = 0;
    hist_head = 0;
}

void QUANTUM_CHEMISTRY::Apply_DIIS(int iter)
{
    if (!scf_ws.use_diis || (iter + 1) < scf_ws.diis_start_iter) return;

#ifndef USE_GPU
    // CPU path: full double DIIS
    // Use d_F_double for DIIS (double Fock from Build_Fock reduce)
    double* dF = scf_ws.d_F_double;
    if (!dF) return;  // fallback: no double Fock available

    QC_Build_DIIS_Error_Double(mol.nao, dF, scf_ws.d_P, scf_ws.d_S,
                               scf_ws.d_diis_err);
    {
        double enorm = 0;
        for (int i = 0; i < (int)mol.nao2; i++)
            enorm += scf_ws.d_diis_err[i] * scf_ws.d_diis_err[i];
        printf("DIIS error norm=%.6e, F_max=%.6e\n",
               sqrt(enorm), [&]{double mx=0; for(int i=0;i<(int)mol.nao2;i++) mx=fmax(mx,fabs(dF[i])); return mx;}());
        fflush(stdout);
    }
    QC_DIIS_History_Push_Double(
        (int)mol.nao2, scf_ws.diis_space, scf_ws.diis_hist_count,
        scf_ws.diis_hist_head, scf_ws.d_diis_f_hist.data(),
        scf_ws.d_diis_e_hist.data(), dF, scf_ws.d_diis_err);
    if (scf_ws.diis_hist_count >= 2)
    {
        if (QC_DIIS_Extrapolate_Double(
                mol.nao, scf_ws.diis_space, scf_ws.diis_hist_count,
                scf_ws.diis_hist_head, scf_ws.d_diis_f_hist.data(),
                scf_ws.d_diis_e_hist.data(), scf_ws.diis_reg, dF,
                scf_ws.d_diis_B, scf_ws.d_diis_rhs, scf_ws.d_diis_info))
        {
            // Cast back to float d_F for energy computation
            for (int i = 0; i < (int)mol.nao2; i++)
                scf_ws.d_F[i] = (float)dF[i];
        }
    }

    if (!scf_ws.unrestricted) return;

    double* dFb = scf_ws.d_F_b_double;
    if (!dFb) return;
    QC_Build_DIIS_Error_Double(mol.nao, dFb, scf_ws.d_P_b, scf_ws.d_S,
                               scf_ws.d_diis_err);
    QC_DIIS_History_Push_Double(
        (int)mol.nao2, scf_ws.diis_space, scf_ws.diis_hist_count_b,
        scf_ws.diis_hist_head_b, scf_ws.d_diis_f_hist_b.data(),
        scf_ws.d_diis_e_hist_b.data(), dFb, scf_ws.d_diis_err);
    if (scf_ws.diis_hist_count_b >= 2)
    {
        if (QC_DIIS_Extrapolate_Double(
                mol.nao, scf_ws.diis_space, scf_ws.diis_hist_count_b,
                scf_ws.diis_hist_head_b, scf_ws.d_diis_f_hist_b.data(),
                scf_ws.d_diis_e_hist_b.data(), scf_ws.diis_reg, dFb,
                scf_ws.d_diis_B, scf_ws.d_diis_rhs, scf_ws.d_diis_info))
        {
            for (int i = 0; i < (int)mol.nao2; i++)
                scf_ws.d_F_b[i] = (float)dFb[i];
        }
    }
#else
    // GPU path: keep float DIIS (unchanged for now)
    // ... would need separate implementation
#endif
}
