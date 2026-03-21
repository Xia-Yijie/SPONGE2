#pragma once

#include "accumulate_energy.hpp"
#include "apply_diis.hpp"
#include "build_fock.hpp"
#include "diag_density.hpp"
#include "mix_converge.hpp"
#include "pre_scf.hpp"
#include "workspace.hpp"

void QUANTUM_CHEMISTRY::Solve_SCF(const VECTOR* crd, const VECTOR box_length,
                                  bool need_energy, int md_step)
{
    if (!is_initialized) return;

    Update_Coordinates_From_MD(crd, box_length);
    if (dft.enable_dft) Update_DFT_Grid();

    Reset_SCF_State();
    Compute_OneE_Integrals();
    if (need_energy) Compute_Nuclear_Repulsion(box_length);
    Prepare_Integrals();
    Build_Overlap_X();

    for (int iter = 0; iter < scf_ws.max_scf_iter; ++iter)
    {
        Build_Fock();
        Accumulate_SCF_Energy(iter);
        Apply_DIIS(iter);
        Diagonalize_And_Build_Density();
        if (Mix_And_Check_Convergence(iter, md_step)) break;
    }
}
