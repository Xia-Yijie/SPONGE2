#include "gradient/grad_one_e.hpp"
#include "gradient/grad_workspace.h"
#include "gradient/gradient.hpp"
#include "quantum_chemistry.h"

void QUANTUM_CHEMISTRY::Compute_Gradient(VECTOR* frc, const VECTOR box_length)
{
    if (!is_initialized) return;
    const int natm = mol.natm;
    const int nao = mol.nao;
    const int nao2 = mol.nao2;

    // 清零梯度累加器
    deviceMemset(grad_ws.d_grad, 0, sizeof(double) * natm * 3);

    // DEBUG helper
    auto print_grad = [&](const char* label)
    {
        if (CONTROLLER::MPI_rank == 0)
        {
            std::vector<double> h_grad(natm * 3);
            deviceMemcpy(h_grad.data(), grad_ws.d_grad,
                         sizeof(double) * natm * 3, deviceMemcpyDeviceToHost);
            fprintf(stdout, "%s:\n", label);
            for (int i = 0; i < natm; i++)
                fprintf(stdout, "  Atom %d: %12.8f %12.8f %12.8f\n", i,
                        h_grad[i * 3], h_grad[i * 3 + 1],
                        h_grad[i * 3 + 2]);
            fflush(stdout);
        }
    };

    // 1. 构建能量加权密度矩阵 W
    {
        // 复用 alpha 的 Fock 矩阵缓冲作为临时 D 矩阵
        float* d_D_tmp = scf_ws.alpha.d_F;
        QC_Build_Energy_Weighted_Density(
            blas_handle, nao, scf_ws.runtime.n_alpha,
            scf_ws.runtime.occ_factor, scf_ws.alpha.d_C, scf_ws.ortho.d_W,
            grad_ws.d_W_density, d_D_tmp);

        if (scf_ws.runtime.unrestricted && grad_ws.d_W_density_beta)
        {
            float* d_D_tmp_b = scf_ws.beta.d_F;
            QC_Build_Energy_Weighted_Density(
                blas_handle, nao, scf_ws.runtime.n_beta, 1.0f,
                scf_ws.beta.d_C, scf_ws.ortho.d_W,
                grad_ws.d_W_density_beta, d_D_tmp_b);
        }
    }

    // 2. 核排斥梯度
    {
        const int threads = 256;
        const VECTOR box_bohr(box_length.x * CONSTANT_ANGSTROM_TO_BOHR,
                              box_length.y * CONSTANT_ANGSTROM_TO_BOHR,
                              box_length.z * CONSTANT_ANGSTROM_TO_BOHR);
        Launch_Device_Kernel(QC_Nuclear_Gradient_Kernel,
                             (natm + threads - 1) / threads, threads, 0, 0,
                             natm, mol.d_Z, mol.d_atm, mol.d_env, box_bohr,
                             grad_ws.d_grad);
    }

    print_grad("After nuclear only");

    // 3. 单电子积分导数: Tr[P·dH/dR] - Tr[W·dS/dR]
    {
        // P 和 W 需要在笛卡尔基下；当前为球谐基
        // 使用 P_coul (对 restricted = alpha.d_P, 对 unrestricted = Ptot)
        const float* d_P_use = scf_ws.direct.d_P_coul;
        const float* d_W_use = grad_ws.d_W_density;
        // TODO: UHF 时 W = W_alpha + W_beta 需要先合并

        const int chunk_size = ONE_E_BATCH_SIZE;
        for (int i = 0; i < task_ctx.topo.n_1e_tasks; i += chunk_size)
        {
            int current_chunk =
                std::min(chunk_size, task_ctx.topo.n_1e_tasks - i);
            QC_ONE_E_TASK* task_ptr = task_ctx.buffers.d_1e_tasks + i;
            Launch_Device_Kernel(
                OneE_Grad_Kernel, (current_chunk + 63) / 64, 64, 0, 0,
                current_chunk, task_ptr, mol.d_centers, mol.d_l_list,
                mol.d_exps, mol.d_coeffs, mol.d_shell_offsets,
                mol.d_shell_sizes, mol.d_ao_offsets, mol.d_atm, mol.d_env,
                mol.natm, mol.nao, grad_ws.d_shell_atom, d_P_use, d_W_use,
                scf_ws.ortho.d_norms, grad_ws.d_grad);
        }
    }

    // 4. 双电子积分导数: Tr[P·dG/dR]
    // TODO: Phase 3 实现 grad_eri.hpp

    // 5. DFT XC 网格梯度
    // TODO: Phase 4 实现 grad_xc.hpp

    print_grad("After nuclear + 1e gradient");

    // 6. 将梯度写入 MD 力数组
    {
        const int threads = 256;
        Launch_Device_Kernel(QC_Writeback_Gradient_Kernel,
                             (natm + threads - 1) / threads, threads, 0, 0,
                             natm, d_atom_local, grad_ws.d_grad, frc);
    }
}
