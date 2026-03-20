#pragma once

#ifndef USE_GPU
// CPU path: double-precision matrix operations for Fp=X^T*F*X, C=X*eigvec, P=C*C^T
// to avoid float32 sgemm accumulation errors that corrupt the density matrix.

// F(float) * X(double) -> Tmp(float), all in double arithmetic
static inline void QC_MatMul_FX_CPU(const int m, const int n, const int k,
                                    const float* F, const double* X,
                                    float* C_out)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (int p = 0; p < k; p++)
                sum += (double)F[i * k + p] * X[p * n + j];
            C_out[i * n + j] = (float)sum;
        }
}

// X(double, row-major) * eigvec(float, col-major from LAPACK) -> C(float)
static inline void QC_MatMul_XEig_CPU(const int m, const int n,
                                      const int k, const double* X_row,
                                      const float* eig_col, float* C_out)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (int p = 0; p < k; p++)
                sum += X_row[i * k + p] * (double)eig_col[j * k + p];
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
    // CPU: use double-precision throughout Fp = X^T * F * X
    // Reconstruct F in double from thread-private double buffers to avoid
    // the float32 cast in d_F which loses precision
    const int nao = mol.nao;
    const int nao2 = mol.nao2;

    // Build double-precision F by reducing thread double buffers directly,
    // bypassing the float32 d_F which lost precision in the reduce cast.
    // F = H_core(+Vxc) + sum_threads(d_F_thread[tid])
    // d_F was set to H_core(+Vxc) before the reduce added thread contributions.
    // Reconstruct in double: F_double = H_core(+Vxc) + sum(thread_doubles)
    std::vector<double> dF(nao2);
    for (int idx = 0; idx < nao2; idx++)
    {
        // H_core(+Vxc) was stored into d_F by QC_Init_Fock_Kernel before
        // the thread buffers were added. d_H_core is float.
        double sum = (double)scf_ws.d_H_core[idx];
        for (int tid = 0; tid < scf_ws.fock_thread_count; tid++)
            sum += scf_ws.d_F_thread[(size_t)tid * nao2 + idx];
        dF[idx] = sum;
    }

    // Fp = X^T * F_double * X (all in double, X promoted from float)
    std::vector<double> Tmp_d(nao2), Fp_d(nao2);
    // Tmp_d = F_double * X
    for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < nao; k++)
                sum += dF[i * nao + k] * (double)scf_ws.d_X[k * nao + j];
            Tmp_d[i * nao + j] = sum;
        }
    // Fp_d = X^T * Tmp_d
    for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < nao; k++)
                sum += (double)scf_ws.d_X[k * nao + i] * Tmp_d[k * nao + j];
            Fp_d[i * nao + j] = sum;
            scf_ws.d_Fp[i * nao + j] = (float)sum;
        }

    // Use dsyevd (double). Fp_d is row-major but LAPACK expects col-major.
    // For symmetric matrix: row-major upper = col-major lower, so use 'L'.
    {
        std::vector<double> dW(nao);
        int lwork_d = -1, liwork_d = -1;
        double work_query;
        lapack_int iwork_query;
        LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, 'V', 'L', (lapack_int)nao,
                            Fp_d.data(), (lapack_int)nao, dW.data(),
                            &work_query, lwork_d, &iwork_query, liwork_d);
        lwork_d = (int)work_query;
        liwork_d = iwork_query;
        std::vector<double> dwork(lwork_d);
        std::vector<lapack_int> diwork(liwork_d);
        int info_d = (int)LAPACKE_dsyevd_work(
            LAPACK_COL_MAJOR, 'V', 'L', (lapack_int)nao, Fp_d.data(),
            (lapack_int)nao, dW.data(), dwork.data(), (lapack_int)lwork_d,
            diwork.data(), (lapack_int)liwork_d);
        (void)info_d;
        // Eigenvectors are now in col-major layout (transposed from our
        // row-major perspective), which is what QC_MatMul_RowCol expects
        for (int i = 0; i < nao2; i++) scf_ws.d_Fp[i] = (float)Fp_d[i];
        for (int i = 0; i < nao; i++) scf_ws.d_W[i] = (float)dW[i];
    }

    // C = X * eigvec (X is double, eigvec in col-major from LAPACK is float)
    QC_MatMul_XEig_CPU(nao, nao, nao, scf_ws.d_X, scf_ws.d_Fp, scf_ws.d_C);
    // P_new = occ_factor * C_occ * C_occ^T
    QC_Build_Density_Double_CPU(nao, scf_ws.n_alpha, scf_ws.occ_factor,
                                scf_ws.d_C, scf_ws.d_P_new);
    // Dump matrices for debugging
    {
        static int call = 0;
        if (++call == 2)
        {
            FILE* fp;
            fp = fopen("/tmp/sponge_F.bin","wb"); fwrite(scf_ws.d_F,4,nao2,fp); fclose(fp);
            fp = fopen("/tmp/sponge_X.bin","wb"); fwrite(scf_ws.d_X,sizeof(double),nao2,fp); fclose(fp);
            fp = fopen("/tmp/sponge_C.bin","wb"); fwrite(scf_ws.d_C,4,nao2,fp); fclose(fp);
            fp = fopen("/tmp/sponge_Pnew.bin","wb"); fwrite(scf_ws.d_P_new,4,nao2,fp); fclose(fp);
            fp = fopen("/tmp/sponge_S.bin","wb"); fwrite(scf_ws.d_S,4,nao2,fp); fclose(fp);
            printf("DUMPED iter2 to /tmp/sponge_*.bin (nao=%d)\n", nao);
        }
    }

    // Diagnostic: Tr[P_new * S] should equal n_electrons
    {
        double tr = 0.0;
        float pmax = 0.0f;
        for (int i = 0; i < nao; i++)
            for (int j = 0; j < nao; j++)
            {
                tr += (double)scf_ws.d_P_new[i * nao + j] *
                      (double)scf_ws.d_S[i * nao + j];
                pmax = fmaxf(pmax, fabsf(scf_ws.d_P_new[i * nao + j]));
            }
        printf("DiagDensity | n_alpha=%d | occ_factor=%.1f | Tr[P_new*S]=%.6f | P_new_max=%.6e | nao=%d\n",
               scf_ws.n_alpha, scf_ws.occ_factor, tr, (double)pmax, nao);
        fflush(stdout);
    }

    if (!scf_ws.unrestricted) return;

    QC_MatMul_FX_CPU(nao, nao, nao, scf_ws.d_F_b, scf_ws.d_X,
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
        std::vector<double> dFpb(nao2), dW(nao);
        for (int i = 0; i < nao; i++)
            for (int j = 0; j < nao; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < nao; k++)
                    sum += (double)scf_ws.d_X[k * nao + i] *
                           (double)scf_ws.d_Tmp[k * nao + j];
                dFpb[i * nao + j] = sum;
            }
        int lwork_d = -1, liwork_d = -1;
        double work_query;
        lapack_int iwork_query;
        LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, 'V', 'L', (lapack_int)nao,
                            dFpb.data(), (lapack_int)nao, dW.data(),
                            &work_query, lwork_d, &iwork_query, liwork_d);
        lwork_d = (int)work_query;
        liwork_d = iwork_query;
        std::vector<double> dwork(lwork_d);
        std::vector<lapack_int> diwork(liwork_d);
        LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, 'V', 'L', (lapack_int)nao,
                            dFpb.data(), (lapack_int)nao, dW.data(),
                            dwork.data(), (lapack_int)lwork_d, diwork.data(),
                            (lapack_int)liwork_d);
        for (int i = 0; i < nao2; i++) scf_ws.d_Fp_b[i] = (float)dFpb[i];
        for (int i = 0; i < nao; i++) scf_ws.d_W[i] = (float)dW[i];
    }

    QC_MatMul_XEig_CPU(nao, nao, nao, scf_ws.d_X, scf_ws.d_Fp_b,
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
