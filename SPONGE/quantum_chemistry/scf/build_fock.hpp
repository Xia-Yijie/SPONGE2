#pragma once

static __global__ void QC_Build_Fock_Kernel(
    const int nao, const float* H_core, const float* P_coul, const float* P_exx,
    const float* ERI, const float exx_scale, const float* Vxc,
    const int use_vxc, float* F)
{
    SIMPLE_DEVICE_FOR(idx, nao * nao)
    {
        int i = idx / nao;
        int j = idx - i * nao;
        double fij = (double)H_core[i * nao + j];
        for (int k = 0; k < nao; k++)
        {
            for (int l = 0; l < nao; l++)
            {
                int id1 = ((int)i * nao * nao * nao) + (int)j * nao * nao +
                          k * nao + l;
                int id2 = ((int)i * nao * nao * nao) + (int)k * nao * nao +
                          j * nao + l;
                fij += (double)P_coul[k * nao + l] * (double)ERI[id1] -
                       (double)exx_scale * (double)P_exx[k * nao + l] *
                           (double)ERI[id2];
            }
        }
        if (use_vxc) fij += (double)Vxc[i * nao + j];
        F[i * nao + j] = (float)fij;
    }
}

void QUANTUM_CHEMISTRY::Build_Fock()
{
    const int threads = 256;
    const int total = mol.nao2;

    if (dft.enable_dft)
    {
        if (scf_ws.unrestricted)
        {
            QC_Build_DFT_VXC_UKS(
                blas_handle, method, mol.is_spherical, mol.nao_cart, mol.nao,
                dft.max_grid_size, dft.grid_batch_size, mol.nbas,
                dft.d_grid_coords, dft.d_grid_weights, cart2sph.d_cart2sph_mat,
                mol.d_centers, mol.d_l_list, mol.d_exps, mol.d_coeffs,
                mol.d_shell_offsets, mol.d_shell_sizes, mol.d_ao_offsets,
                scf_ws.d_norms, scf_ws.d_P, scf_ws.d_P_b, dft.d_ao_vals_cart,
                dft.d_ao_grad_x_cart, dft.d_ao_grad_y_cart,
                dft.d_ao_grad_z_cart, dft.d_ao_vals, dft.d_ao_grad_x,
                dft.d_ao_grad_y, dft.d_ao_grad_z, dft.d_exc_total, dft.d_Vxc,
                dft.d_Vxc_beta);
        }
        else
        {
            QC_Build_DFT_VXC(
                blas_handle, method, mol.is_spherical, mol.nao_cart, mol.nao,
                dft.max_grid_size, dft.grid_batch_size, mol.nbas,
                dft.d_grid_coords, dft.d_grid_weights, cart2sph.d_cart2sph_mat,
                mol.d_centers, mol.d_l_list, mol.d_exps, mol.d_coeffs,
                mol.d_shell_offsets, mol.d_shell_sizes, mol.d_ao_offsets,
                scf_ws.d_norms, scf_ws.d_P, dft.d_ao_vals_cart,
                dft.d_ao_grad_x_cart, dft.d_ao_grad_y_cart,
                dft.d_ao_grad_z_cart, dft.d_ao_vals, dft.d_ao_grad_x,
                dft.d_ao_grad_y, dft.d_ao_grad_z, dft.d_rho, dft.d_sigma,
                dft.d_exc, dft.d_vrho, dft.d_vsigma, dft.d_exc_total,
                dft.d_Vxc);
        }
    }

    Launch_Device_Kernel(
        QC_Build_Fock_Kernel, (total + threads - 1) / threads, threads, 0, 0,
        mol.nao, scf_ws.d_H_core, scf_ws.d_P_coul, scf_ws.d_P, scf_ws.d_ERI,
        scf_ws.unrestricted ? dft.exx_fraction : (0.5f * dft.exx_fraction),
        dft.d_Vxc, dft.enable_dft, scf_ws.d_F);

    if (scf_ws.unrestricted)
    {
        Launch_Device_Kernel(QC_Build_Fock_Kernel,
                             (total + threads - 1) / threads, threads, 0, 0,
                             mol.nao, scf_ws.d_H_core, scf_ws.d_P_coul,
                             scf_ws.d_P_b, scf_ws.d_ERI, dft.exx_fraction,
                             dft.d_Vxc_beta, dft.enable_dft, scf_ws.d_F_b);
    }
}
