#pragma once

#ifndef USE_GPU
// CPU path: double-precision matrix operations for Fp=X^T*F*X, C=X*eigvec, P=C*C^T
// to avoid float32 sgemm accumulation errors that corrupt the density matrix.

static inline void QC_MatMul_Double_CPU(const int m, const int n, const int k,
                                        const float* A, const float* B,
                                        float* C_out)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (int p = 0; p < k; p++)
                sum += (double)A[i * k + p] * (double)B[p * n + j];
            C_out[i * n + j] = (float)sum;
        }
}

// A is row-major, B is col-major (eigenvectors from LAPACK)
static inline void QC_MatMul_RowCol_Double_CPU(const int m, const int n,
                                               const int k, const float* A_row,
                                               const float* B_col,
                                               float* C_out)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (int p = 0; p < k; p++)
                sum += (double)A_row[i * k + p] * (double)B_col[j * k + p];
            C_out[i * n + j] = (float)sum;
        }
}

static inline void QC_Build_Density_Double_CPU(const int nao, const int n_occ,
                                               const float density_factor,
                                               const float* C_row,
                                               float* P_new)
{
    const int nao2 = nao * nao;
    const double factor = (double)density_factor;
    for (int i = 0; i < nao; i++)
        for (int j = 0; j <= i; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < n_occ; k++)
                sum += (double)C_row[i * nao + k] * (double)C_row[j * nao + k];
            const float val = (float)(factor * sum);
            P_new[i * nao + j] = val;
            P_new[j * nao + i] = val;
        }
}
#endif

void QUANTUM_CHEMISTRY::Diagonalize_And_Build_Density()
{
#ifndef USE_GPU
    // CPU: use double-precision matmul for Fp = X^T * F * X
    const int nao = mol.nao;
    const int nao2 = mol.nao2;

    // Tmp = F * X (row * row)
    QC_MatMul_Double_CPU(nao, nao, nao, scf_ws.d_F, scf_ws.d_X, scf_ws.d_Tmp);
    // Fp = X^T * Tmp = X^T * F * X
    // X is row-major, so X^T[i,k] = X[k,i] = d_X[k*nao+i]
    for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < nao; k++)
                sum += (double)scf_ws.d_X[k * nao + i] *
                       (double)scf_ws.d_Tmp[k * nao + j];
            scf_ws.d_Fp[i * nao + j] = (float)sum;
        }

    // Use dsyevd (double) instead of ssyevd for eigenvalue decomposition
    {
        std::vector<double> dFp(nao2), dW(nao);
        for (int i = 0; i < nao2; i++) dFp[i] = (double)scf_ws.d_Fp[i];
        int lwork_d = -1, liwork_d = -1;
        double work_query;
        lapack_int iwork_query;
        LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, 'V', 'U', (lapack_int)nao,
                            dFp.data(), (lapack_int)nao, dW.data(),
                            &work_query, lwork_d, &iwork_query, liwork_d);
        lwork_d = (int)work_query;
        liwork_d = iwork_query;
        std::vector<double> dwork(lwork_d);
        std::vector<lapack_int> diwork(liwork_d);
        int info_d = (int)LAPACKE_dsyevd_work(
            LAPACK_COL_MAJOR, 'V', 'U', (lapack_int)nao, dFp.data(),
            (lapack_int)nao, dW.data(), dwork.data(), (lapack_int)lwork_d,
            diwork.data(), (lapack_int)liwork_d);
        (void)info_d;
        for (int i = 0; i < nao2; i++) scf_ws.d_Fp[i] = (float)dFp[i];
        for (int i = 0; i < nao; i++) scf_ws.d_W[i] = (float)dW[i];
    }

    // C = X * eigvec (eigvec in col-major from LAPACK)
    QC_MatMul_RowCol_Double_CPU(nao, nao, nao, scf_ws.d_X, scf_ws.d_Fp,
                                scf_ws.d_C);
    // P_new = occ_factor * C_occ * C_occ^T
    QC_Build_Density_Double_CPU(nao, scf_ws.n_alpha, scf_ws.occ_factor,
                                scf_ws.d_C, scf_ws.d_P_new);

    if (!scf_ws.unrestricted) return;

    QC_MatMul_Double_CPU(nao, nao, nao, scf_ws.d_F_b, scf_ws.d_X,
                         scf_ws.d_Tmp);
    for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < nao; k++)
                sum += (double)scf_ws.d_X[k * nao + i] *
                       (double)scf_ws.d_Tmp[k * nao + j];
            scf_ws.d_Fp_b[i * nao + j] = (float)sum;
        }
    {
        std::vector<double> dFp(nao2), dW(nao);
        for (int i = 0; i < nao2; i++) dFp[i] = (double)scf_ws.d_Fp_b[i];
        int lwork_d = -1, liwork_d = -1;
        double work_query;
        lapack_int iwork_query;
        LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, 'V', 'U', (lapack_int)nao,
                            dFp.data(), (lapack_int)nao, dW.data(),
                            &work_query, lwork_d, &iwork_query, liwork_d);
        lwork_d = (int)work_query;
        liwork_d = iwork_query;
        std::vector<double> dwork(lwork_d);
        std::vector<lapack_int> diwork(liwork_d);
        LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, 'V', 'U', (lapack_int)nao,
                            dFp.data(), (lapack_int)nao, dW.data(),
                            dwork.data(), (lapack_int)lwork_d, diwork.data(),
                            (lapack_int)liwork_d);
        for (int i = 0; i < nao2; i++) scf_ws.d_Fp_b[i] = (float)dFp[i];
        for (int i = 0; i < nao; i++) scf_ws.d_W[i] = (float)dW[i];
    }

    QC_MatMul_RowCol_Double_CPU(nao, nao, nao, scf_ws.d_X, scf_ws.d_Fp_b,
                                scf_ws.d_C_b);
    QC_Build_Density_Double_CPU(nao, scf_ws.n_beta, 1.0f, scf_ws.d_C_b,
                                scf_ws.d_P_b_new);
#else
    // GPU: keep original sgemm path
    QC_MatMul_RowRow_Blas(blas_handle, mol.nao, mol.nao, mol.nao, scf_ws.d_F,
                          scf_ws.d_X, scf_ws.d_Tmp);
    QC_MatMul_RowRow_Blas(blas_handle, mol.nao, mol.nao, mol.nao, scf_ws.d_X,
                          scf_ws.d_Tmp, scf_ws.d_Fp);

    deviceMemcpy(scf_ws.d_Work, scf_ws.d_Fp, sizeof(float) * mol.nao2,
                 deviceMemcpyDeviceToDevice);
    QC_Diagonalize(solver_handle, mol.nao, scf_ws.d_Work, scf_ws.d_W,
                   scf_ws.d_solver_work, scf_ws.lwork, scf_ws.d_solver_iwork,
                   scf_ws.liwork, scf_ws.d_info);
    deviceMemcpy(scf_ws.d_Fp, scf_ws.d_Work, sizeof(float) * mol.nao2,
                 deviceMemcpyDeviceToDevice);

    QC_MatMul_RowCol_Blas(blas_handle, mol.nao, mol.nao, mol.nao, scf_ws.d_X,
                          scf_ws.d_Fp, scf_ws.d_C);
    QC_Build_Density_Blas(blas_handle, mol.nao, scf_ws.n_alpha,
                          scf_ws.occ_factor, scf_ws.d_C, scf_ws.d_P_new);

    if (!scf_ws.unrestricted) return;

    QC_MatMul_RowRow_Blas(blas_handle, mol.nao, mol.nao, mol.nao, scf_ws.d_F_b,
                          scf_ws.d_X, scf_ws.d_Tmp);
    QC_MatMul_RowRow_Blas(blas_handle, mol.nao, mol.nao, mol.nao, scf_ws.d_X,
                          scf_ws.d_Tmp, scf_ws.d_Fp_b);

    deviceMemcpy(scf_ws.d_Work, scf_ws.d_Fp_b, sizeof(float) * mol.nao2,
                 deviceMemcpyDeviceToDevice);
    QC_Diagonalize(solver_handle, mol.nao, scf_ws.d_Work, scf_ws.d_W,
                   scf_ws.d_solver_work, scf_ws.lwork, scf_ws.d_solver_iwork,
                   scf_ws.liwork, scf_ws.d_info);
    deviceMemcpy(scf_ws.d_Fp_b, scf_ws.d_Work, sizeof(float) * mol.nao2,
                 deviceMemcpyDeviceToDevice);

    QC_MatMul_RowCol_Blas(blas_handle, mol.nao, mol.nao, mol.nao, scf_ws.d_X,
                          scf_ws.d_Fp_b, scf_ws.d_C_b);
    QC_Build_Density_Blas(blas_handle, mol.nao, scf_ws.n_beta, 1.0f,
                          scf_ws.d_C_b, scf_ws.d_P_b_new);
#endif
}
