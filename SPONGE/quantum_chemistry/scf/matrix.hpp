#pragma once

static inline int QC_Diagonalize_Workspace_Size(SOLVER_HANDLE solver_handle,
                                                int n, float* mat, float* w,
                                                float** work_ptr,
                                                void** iwork_ptr, int* lwork,
                                                int* liwork)
{
#ifdef USE_GPU
    if (work_ptr == NULL || iwork_ptr == NULL || lwork == NULL ||
        liwork == NULL || mat == NULL || w == NULL)
        return -1;

    *liwork = 0;
    int stat = (int)deviceSolverSsyevdBufferSize(
        solver_handle, DEVICE_EIG_MODE_VECTOR, DEVICE_FILL_MODE_UPPER, n, mat,
        n, w, lwork);
    if (stat != 0 || *lwork <= 0) return (stat != 0) ? stat : -2;

    if (*work_ptr != NULL)
    {
        deviceFree(*work_ptr);
        *work_ptr = NULL;
    }
    Device_Malloc_Safely((void**)work_ptr, sizeof(float) * (int)(*lwork));
    deviceMemset(*work_ptr, 0, sizeof(float) * (int)(*lwork));

    if (*iwork_ptr != NULL)
    {
        deviceFree(*iwork_ptr);
        *iwork_ptr = NULL;
    }
    return 0;
#elif defined(USE_MKL) || defined(USE_OPENBLAS)
    (void)solver_handle;
    if (work_ptr == NULL || iwork_ptr == NULL || lwork == NULL ||
        liwork == NULL || mat == NULL || w == NULL)
        return -1;
    if (n <= 0)
    {
        *lwork = 0;
        *liwork = 0;
        if (*work_ptr != NULL)
        {
            deviceFree(*work_ptr);
            *work_ptr = NULL;
        }
        if (*iwork_ptr != NULL)
        {
            deviceFree(*iwork_ptr);
            *iwork_ptr = NULL;
        }
        return 0;
    }

    float work_opt = 0.0f;
    lapack_int iwork_opt = 0;
    lapack_int info = LAPACKE_ssyevd_work(
        LAPACK_COL_MAJOR, 'V', 'U', (lapack_int)n, mat, (lapack_int)n, w,
        &work_opt, (lapack_int)-1, &iwork_opt, (lapack_int)-1);
    if (info != 0) return (int)info;
    *lwork = (int)(work_opt + 0.5f);
    if (*lwork < 1) *lwork = 1;
    *liwork = (int)iwork_opt;
    if (*liwork < 0) *liwork = 0;

    if (*work_ptr != NULL)
    {
        deviceFree(*work_ptr);
        *work_ptr = NULL;
    }
    Device_Malloc_Safely((void**)work_ptr, sizeof(float) * (int)(*lwork));
    deviceMemset(*work_ptr, 0, sizeof(float) * (int)(*lwork));

    if (*iwork_ptr != NULL)
    {
        deviceFree(*iwork_ptr);
        *iwork_ptr = NULL;
    }
    if (*liwork > 0)
    {
        Device_Malloc_Safely(iwork_ptr, sizeof(lapack_int) * (int)(*liwork));
        deviceMemset(*iwork_ptr, 0, sizeof(lapack_int) * (int)(*liwork));
    }

    return 0;
#else
    (void)solver_handle;
    (void)n;
    (void)mat;
    (void)w;
    (void)work_ptr;
    (void)iwork_ptr;
    (void)lwork;
    (void)liwork;
    return -1;
#endif
}

static inline void QC_Diagonalize(SOLVER_HANDLE solver_handle, int n,
                                  float* mat, float* w, float* work, int lwork,
                                  void* iwork, int liwork, int* info)
{
#ifdef USE_GPU
    deviceSolverSsyevd(solver_handle, DEVICE_EIG_MODE_VECTOR,
                       DEVICE_FILL_MODE_UPPER, n, mat, n, w, work, lwork, info);
#elif defined(USE_MKL) || defined(USE_OPENBLAS)
    *info = (int)LAPACKE_ssyevd_work(
        LAPACK_COL_MAJOR, 'V', 'U', (lapack_int)n, mat, (lapack_int)n, w, work,
        (lapack_int)lwork, (lapack_int*)iwork, (lapack_int)liwork);
#else
    *info = -1;
#endif
}

static inline void QC_MatMul_RowRow_Blas(BLAS_HANDLE blas_handle, const int m,
                                         const int n, const int kdim,
                                         const float* A_row, const float* B_row,
                                         float* C_row)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_N, n, m, kdim,
                    &alpha, B_row, n, A_row, kdim, &beta, C_row, n);
}

static inline void QC_MatMul_RowCol_Blas(BLAS_HANDLE blas_handle, const int m,
                                         const int n, const int kdim,
                                         const float* A_row, const float* B_col,
                                         float* C_row)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_N, n, m, kdim,
                    &alpha, B_col, kdim, A_row, kdim, &beta, C_row, n);
}

static inline void QC_Build_Density_Blas(BLAS_HANDLE blas_handle, const int nao,
                                         const int n_occ,
                                         const float density_factor,
                                         const float* C_row, float* P_new_row)
{
    const int nao2 = (int)nao * (int)nao;
    deviceMemset(P_new_row, 0, sizeof(float) * nao2);
    if (n_occ <= 0 || density_factor == 0.0f) return;

    const float alpha = density_factor;
    const float beta = 0.0f;
    deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_N, nao, nao,
                    n_occ, &alpha, C_row, nao, C_row, nao, &beta, P_new_row,
                    nao);
}

static __global__ void QC_Elec_Energy_Accumulate_Kernel(const int nao2,
                                                        const float* P,
                                                        const float* H_core,
                                                        const float* F,
                                                        double* out_sum)
{
    SIMPLE_DEVICE_FOR(idx, nao2)
    {
        atomicAdd(out_sum, 0.5 * (double)P[idx] *
                               ((double)H_core[idx] + (double)F[idx]));
    }
}

static __global__ void QC_Mat_Dot_Accumulate_Kernel(const int nao2,
                                                    const float* A,
                                                    const float* B,
                                                    double* out_sum)
{
    SIMPLE_DEVICE_FOR(idx, nao2)
    {
        atomicAdd(out_sum, (double)A[idx] * (double)B[idx]);
    }
}

static __global__ void QC_Add_Matrix_Kernel(const int n, const float* A,
                                            const float* B, float* C)
{
    SIMPLE_DEVICE_FOR(idx, n) { C[idx] = A[idx] + B[idx]; }
}

static __global__ void QC_Sub_Matrix_Kernel(const int n, const float* A,
                                            const float* B, float* C)
{
    SIMPLE_DEVICE_FOR(idx, n) { C[idx] = A[idx] - B[idx]; }
}

static __global__ void QC_Add_Scaled_Matrix_From_Double_Kernel(
    const int n, const int coeff_idx, const double* coeff, const float* A,
    float* C)
{
    SIMPLE_DEVICE_FOR(idx, n)
    {
        C[idx] += (float)(coeff[coeff_idx] * (double)A[idx]);
    }
}

// ====================== Double-precision wrappers ======================

// Double-precision eigendecomposition workspace query
static inline int QC_Diagonalize_Double_Workspace_Size(
    SOLVER_HANDLE solver_handle, int n, double* mat, double* w,
    double** work_ptr, int* lwork)
{
    if (n <= 0) { *lwork = 0; return 0; }
    int stat = (int)deviceSolverDsyevdBufferSize(
        solver_handle, DEVICE_EIG_MODE_VECTOR, DEVICE_FILL_MODE_UPPER,
        n, mat, n, w, lwork);
    if (stat != 0 || *lwork <= 0) return (stat != 0) ? stat : -2;
    if (*work_ptr) { deviceFree(*work_ptr); *work_ptr = NULL; }
    Device_Malloc_Safely((void**)work_ptr, sizeof(double) * (*lwork));
    deviceMemset(*work_ptr, 0, sizeof(double) * (*lwork));
    return 0;
}

// Double-precision eigendecomposition
static inline void QC_Diagonalize_Double(
    SOLVER_HANDLE solver_handle, int n, double* mat, double* w,
    double* work, int lwork, int* info)
{
    deviceSolverDsyevd(solver_handle, DEVICE_EIG_MODE_VECTOR,
                       DEVICE_FILL_MODE_UPPER, n, mat, n, w, work, lwork, info);
}

// Double row-major C = A * B
static inline void QC_Dgemm_NN(BLAS_HANDLE handle, int m, int n, int k,
                                const double* A, int lda,
                                const double* B, int ldb,
                                double* C, int ldc)
{
    const double one = 1.0, zero = 0.0;
    // Col-major BLAS: C_col = B_col * A_col to get row-major C = A * B
    deviceBlasDgemm(handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_N,
                    n, m, k, &one, B, ldb, A, lda, &zero, C, ldc);
}

// Double row-major C = A^T * B
static inline void QC_Dgemm_TN(BLAS_HANDLE handle, int m, int n, int k,
                                const double* A, int lda,
                                const double* B, int ldb,
                                double* C, int ldc)
{
    const double one = 1.0, zero = 0.0;
    deviceBlasDgemm(handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T,
                    n, m, k, &one, B, ldb, A, lda, &zero, C, ldc);
}

// Double row-major C = A * B^T
static inline void QC_Dgemm_NT(BLAS_HANDLE handle, int m, int n, int k,
                                const double* A, int lda,
                                const double* B, int ldb,
                                double* C, int ldc)
{
    const double one = 1.0, zero = 0.0;
    deviceBlasDgemm(handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_N,
                    n, m, k, &one, B, ldb, A, lda, &zero, C, ldc);
}

// ====================== Double-precision device kernels ======================

// Float -> Double promotion kernel
static __global__ void QC_Float_To_Double_Kernel(const int n, const float* src,
                                                  double* dst)
{
    SIMPLE_DEVICE_FOR(i, n) { dst[i] = (double)src[i]; }
}

// Double -> Float cast kernel
static __global__ void QC_Double_To_Float_Kernel(const int n, const double* src,
                                                  float* dst)
{
    SIMPLE_DEVICE_FOR(i, n) { dst[i] = (float)src[i]; }
}

// Double level shift: dF[i] += ls * (dS[i] - 0.5 * dSPS[i])
static __global__ void QC_Level_Shift_Kernel(const int n, const double ls,
                                              const double* dS,
                                              const double* dSPS, double* dF)
{
    SIMPLE_DEVICE_FOR(i, n) { dF[i] += ls * (dS[i] - 0.5 * dSPS[i]); }
}

// Copy eigenvector columns for canonical orthogonalization X build
static __global__ void QC_Build_X_Canonical_Kernel(
    const int nao, const int nao_eff, const double* eigvec_col,
    const double* eigval, const double lindep_thresh, double* X_row)
{
    SIMPLE_DEVICE_FOR(i, nao)
    {
        int col = 0;
        for (int k = 0; k < nao; k++)
        {
            if (eigval[k] < lindep_thresh) continue;
            X_row[i * nao + col] = eigvec_col[i + k * nao] / sqrt(eigval[k]);
            col++;
        }
    }
}

// Copy rectangular double to padded float: dst[i*nao+j] = src[i*ne+j] for j<ne
static __global__ void QC_Rect_Double_To_Padded_Float_Kernel(
    const int nao, const int ne, const double* src, float* dst)
{
    SIMPLE_DEVICE_FOR(idx, nao * nao)
    {
        int i = idx / nao;
        int j = idx % nao;
        dst[idx] = (j < ne) ? (float)src[i * ne + j] : 0.0f;
    }
}

// Double dot product with atomic accumulation
static __global__ void QC_Double_Dot_Kernel(const int n, const double* A,
                                             const double* B, double* out_sum)
{
    SIMPLE_DEVICE_FOR(i, n) { atomicAdd(out_sum, A[i] * B[i]); }
}

// Double weighted accumulation: dst[i] += coeff * src[i]
static __global__ void QC_Double_Axpy_Kernel(const int n, const double coeff,
                                              const double* src, double* dst)
{
    SIMPLE_DEVICE_FOR(i, n) { dst[i] += coeff * src[i]; }
}

// Double subtract: dst[i] = A[i] - B[i]
static __global__ void QC_Double_Sub_Kernel(const int n, const double* A,
                                             const double* B, double* dst)
{
    SIMPLE_DEVICE_FOR(i, n) { dst[i] = A[i] - B[i]; }
}

// Float to double with atomic accumulation (for density history push)
static __global__ void QC_Float_To_Double_Copy_Kernel(const int n,
                                                       const float* src,
                                                       double* dst)
{
    SIMPLE_DEVICE_FOR(i, n) { dst[i] = (double)src[i]; }
}
