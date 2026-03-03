#ifndef QC_STRUCTURE_DFT_H
#define QC_STRUCTURE_DFT_H

#include "../../common.h"

// DFT configuration and long-lived buffers shared by CPU/GPU paths.
struct QC_DFT
{
    // Hybrid functional exact-exchange fraction.
    float exx_fraction = 1.0f;
    // Whether DFT/UKS code path is enabled (0/1).
    int enable_dft = 0;
    // Molecular grid controls.
    int dft_radial_points = 60;
    int dft_angular_points = 194;

    // Grid storage and capacity.
    std::vector<float> h_grid_coords;   // [max_grid_capacity * 3]
    std::vector<float> h_grid_weights;  // [max_grid_capacity]
    float* d_grid_coords = NULL;        // [max_grid_capacity * 3]
    float* d_grid_weights = NULL;       // [max_grid_capacity]
    int max_grid_capacity = 0;
    int max_grid_size = 0;
    int grid_batch_size = 8192;

    // AO values on DFT grid (batched).
    float* d_ao_vals = NULL;
    float* d_ao_grad_x = NULL;
    float* d_ao_grad_y = NULL;
    float* d_ao_grad_z = NULL;
    // Cartesian intermediates for spherical basis transform.
    float* d_ao_vals_cart = NULL;
    float* d_ao_grad_x_cart = NULL;
    float* d_ao_grad_y_cart = NULL;
    float* d_ao_grad_z_cart = NULL;

    // Density/functional intermediates.
    double* d_rho = NULL;
    double* d_sigma = NULL;
    double* d_exc = NULL;
    double* d_vrho = NULL;
    double* d_vsigma = NULL;

    // XC outputs.
    float* d_Vxc = NULL;
    float* d_Vxc_beta = NULL;
    double* d_exc_total = NULL;
};

#endif
