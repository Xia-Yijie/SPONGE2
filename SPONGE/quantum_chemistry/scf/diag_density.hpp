#pragma once

void QUANTUM_CHEMISTRY::Diagonalize_And_Build_Density()
{
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
}
