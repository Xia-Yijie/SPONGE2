#pragma once

#include "dft.hpp"

static void QC_Eval_AO_Cart_Grad(const QC_MOLECULE& mol, float x, float y,
                                 float z, std::vector<float>& ao,
                                 std::vector<float>& gx, std::vector<float>& gy,
                                 std::vector<float>& gz)
{
    const int nao = mol.nao_cart;
    std::fill(ao.begin(), ao.end(), 0.0f);
    std::fill(gx.begin(), gx.end(), 0.0f);
    std::fill(gy.begin(), gy.end(), 0.0f);
    std::fill(gz.begin(), gz.end(), 0.0f);

    for (int ish = 0; ish < mol.nbas; ish++)
    {
        const int l = mol.h_l_list[ish];
        const int ncart = (l + 1) * (l + 2) / 2;
        const int ao_off = mol.h_ao_offsets[ish];
        const VECTOR c = mol.h_centers[ish];
        const float dx = x - c.x;
        const float dy = y - c.y;
        const float dz = z - c.z;
        const float r2 = dx * dx + dy * dy + dz * dz;

        float px[6], py[6], pz[6];
        px[0] = 1.0f;
        py[0] = 1.0f;
        pz[0] = 1.0f;
        for (int k = 1; k <= 5; k++)
        {
            px[k] = px[k - 1] * dx;
            py[k] = py[k - 1] * dy;
            pz[k] = pz[k - 1] * dz;
        }

        for (int ip = 0; ip < mol.h_shell_sizes[ish]; ip++)
        {
            const int pidx = mol.h_shell_offsets[ish] + ip;
            const float alpha = mol.h_exps[pidx];
            const float coeff = mol.h_coeffs[pidx];
            const float e = coeff * expf(-alpha * r2);
            if (fabsf(e) < 1e-20f) continue;

            for (int ic = 0; ic < ncart; ic++)
            {
                int lx, ly, lz;
                QC_Get_Lxyz_Host(l, ic, lx, ly, lz);
                const float poly = px[lx] * py[ly] * pz[lz];
                const float val = e * poly;

                const float dpoly_x =
                    ((lx > 0) ? (float)lx * px[lx - 1] : 0.0f) * py[ly] *
                    pz[lz];
                const float dpoly_y =
                    px[lx] * ((ly > 0) ? (float)ly * py[ly - 1] : 0.0f) *
                    pz[lz];
                const float dpoly_z =
                    px[lx] * py[ly] *
                    ((lz > 0) ? (float)lz * pz[lz - 1] : 0.0f);

                const float gxv = e * (dpoly_x - 2.0f * alpha * dx * poly);
                const float gyv = e * (dpoly_y - 2.0f * alpha * dy * poly);
                const float gzv = e * (dpoly_z - 2.0f * alpha * dz * poly);

                const int i = ao_off + ic;
                ao[i] += val;
                gx[i] += gxv;
                gy[i] += gyv;
                gz[i] += gzv;
            }
        }
    }
}
