#pragma once

static __global__ void QC_Mix_Density_Kernel(const int nao2, const int iter,
                                             const float mix,
                                             const float* P_new_row,
                                             float* P_row)
{
    SIMPLE_DEVICE_FOR(idx, nao2)
    {
        if (iter == 0)
            P_row[idx] = P_new_row[idx];
        else
            P_row[idx] = (1.0f - mix) * P_row[idx] + mix * P_new_row[idx];
    }
}

static __global__ void QC_Density_Diff_Accumulate_Kernel(const int nao2,
                                                         const float* P_new_row,
                                                         const float* P_row,
                                                         double* out_sum)
{
    SIMPLE_DEVICE_FOR(idx, nao2)
    {
        const double diff = (double)P_new_row[idx] - (double)P_row[idx];
        atomicAdd(out_sum, diff * diff);
    }
}

bool QUANTUM_CHEMISTRY::Mix_And_Check_Convergence(int iter)
{
    const int mix_threads = 256;
    const int mix_blocks = (mol.nao2 + mix_threads - 1) / mix_threads;
    double h_trace_p_old = 0.0;
    double h_trace_p_new = 0.0;
    double h_trace_p_mix = 0.0;
    double h_trace_p_old_b = 0.0;
    double h_trace_p_new_b = 0.0;
    double h_trace_p_mix_b = 0.0;

    if (scf_ws.print_iter)
    {
        deviceMemset(scf_ws.d_diis_accum, 0, sizeof(double));
        Launch_Device_Kernel(QC_Mat_Dot_Accumulate_Kernel, mix_blocks,
                             mix_threads, 0, 0, mol.nao2, scf_ws.d_P,
                             scf_ws.d_S, scf_ws.d_diis_accum);
        deviceMemcpy(&h_trace_p_old, scf_ws.d_diis_accum, sizeof(double),
                     deviceMemcpyDeviceToHost);

        deviceMemset(scf_ws.d_diis_accum, 0, sizeof(double));
        Launch_Device_Kernel(QC_Mat_Dot_Accumulate_Kernel, mix_blocks,
                             mix_threads, 0, 0, mol.nao2, scf_ws.d_P_new,
                             scf_ws.d_S, scf_ws.d_diis_accum);
        deviceMemcpy(&h_trace_p_new, scf_ws.d_diis_accum, sizeof(double),
                     deviceMemcpyDeviceToHost);

        if (scf_ws.unrestricted)
        {
            deviceMemset(scf_ws.d_diis_accum, 0, sizeof(double));
            Launch_Device_Kernel(QC_Mat_Dot_Accumulate_Kernel, mix_blocks,
                                 mix_threads, 0, 0, mol.nao2, scf_ws.d_P_b,
                                 scf_ws.d_S, scf_ws.d_diis_accum);
            deviceMemcpy(&h_trace_p_old_b, scf_ws.d_diis_accum,
                         sizeof(double), deviceMemcpyDeviceToHost);

            deviceMemset(scf_ws.d_diis_accum, 0, sizeof(double));
            Launch_Device_Kernel(QC_Mat_Dot_Accumulate_Kernel, mix_blocks,
                                 mix_threads, 0, 0, mol.nao2, scf_ws.d_P_b_new,
                                 scf_ws.d_S, scf_ws.d_diis_accum);
            deviceMemcpy(&h_trace_p_new_b, scf_ws.d_diis_accum,
                         sizeof(double), deviceMemcpyDeviceToHost);
        }
    }

    deviceMemset(scf_ws.d_density_residual, 0, sizeof(double));
    Launch_Device_Kernel(QC_Density_Diff_Accumulate_Kernel, mix_blocks,
                         mix_threads, 0, 0, mol.nao2, scf_ws.d_P_new,
                         scf_ws.d_P, scf_ws.d_density_residual);
    if (scf_ws.unrestricted)
    {
        Launch_Device_Kernel(QC_Density_Diff_Accumulate_Kernel, mix_blocks,
                             mix_threads, 0, 0, mol.nao2, scf_ws.d_P_b_new,
                             scf_ws.d_P_b, scf_ws.d_density_residual);
    }

    Launch_Device_Kernel(QC_Mix_Density_Kernel,
                         mix_blocks, mix_threads, 0, 0, (int)mol.nao2, iter,
                         scf_ws.density_mixing, scf_ws.d_P_new, scf_ws.d_P);

    if (scf_ws.unrestricted)
    {
        Launch_Device_Kernel(QC_Mix_Density_Kernel, mix_blocks, mix_threads, 0,
                             0, mol.nao2, iter, scf_ws.density_mixing,
                             scf_ws.d_P_b_new, scf_ws.d_P_b);
        Launch_Device_Kernel(QC_Add_Matrix_Kernel, mix_blocks, mix_threads, 0,
                             0, (int)mol.nao2, scf_ws.d_P, scf_ws.d_P_b,
                             scf_ws.d_Ptot);
    }

    if (scf_ws.print_iter)
    {
        deviceMemset(scf_ws.d_diis_accum, 0, sizeof(double));
        Launch_Device_Kernel(QC_Mat_Dot_Accumulate_Kernel, mix_blocks,
                             mix_threads, 0, 0, mol.nao2, scf_ws.d_P,
                             scf_ws.d_S, scf_ws.d_diis_accum);
        deviceMemcpy(&h_trace_p_mix, scf_ws.d_diis_accum, sizeof(double),
                     deviceMemcpyDeviceToHost);

        if (scf_ws.unrestricted)
        {
            deviceMemset(scf_ws.d_diis_accum, 0, sizeof(double));
            Launch_Device_Kernel(QC_Mat_Dot_Accumulate_Kernel, mix_blocks,
                                 mix_threads, 0, 0, mol.nao2, scf_ws.d_P_b,
                                 scf_ws.d_S, scf_ws.d_diis_accum);
            deviceMemcpy(&h_trace_p_mix_b, scf_ws.d_diis_accum,
                         sizeof(double), deviceMemcpyDeviceToHost);
        }
    }

    if (scf_ws.print_iter && CONTROLLER::MPI_rank == 0)
    {
        double h_energy = 0.0;
        double h_delta_e = 0.0;
        double h_density_residual = 0.0;
        deviceMemcpy(&h_energy, scf_ws.d_scf_energy, sizeof(double),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&h_delta_e, scf_ws.d_delta_e, sizeof(double),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&h_density_residual, scf_ws.d_density_residual,
                     sizeof(double), deviceMemcpyDeviceToHost);
        const double denom =
            (double)mol.nao2 * (scf_ws.unrestricted ? 2.0 : 1.0);
        const double density_rms = sqrt(h_density_residual / fmax(denom, 1.0));
        printf(
            "SCF Iter %3d | E(Ha)=%.12f | dE(Ha)=%+.6e | dP_rms=%.6e\n",
            iter + 1, h_energy, h_delta_e, density_rms);
        printf(
            "              Tr[P_old S]=%.6f | Tr[P_new S]=%.6f | Tr[P_mix S]=%.6f\n",
            h_trace_p_old + h_trace_p_old_b, h_trace_p_new + h_trace_p_new_b,
            h_trace_p_mix + h_trace_p_mix_b);
        fflush(stdout);
    }

    int h_converged = 0;
    deviceMemcpy(&h_converged, scf_ws.d_converged, sizeof(int),
                 deviceMemcpyDeviceToHost);
    return h_converged != 0;
}
