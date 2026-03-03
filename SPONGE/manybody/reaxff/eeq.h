#ifndef REAXFF_EEQ_H
#define REAXFF_EEQ_H

#include "../../common.h"
#include "../../control.h"

struct REAXFF_EEQ
{
    int is_initialized = 0;

    int atom_numbers = 0;
    int atom_type_numbers = 0;

    float* h_chi = NULL;
    float* h_eta = NULL;
    float* h_gamma = NULL;

    float* d_chi = NULL;
    float* d_eta = NULL;
    float* d_gamma = NULL;

    int* h_atom_type = NULL;
    int* d_atom_type = NULL;

    // Solver temporary arrays (pre-allocated)
    float* d_b = NULL;
    float* d_r = NULL;
    float* d_p = NULL;
    float* d_Ap = NULL;
    float* d_q = NULL;

    // Additional temporary arrays for CG
    float* d_s = NULL;
    float* d_t = NULL;
    float* d_temp_sum = NULL;

    // Convergence parameters
    float tolerance = 1e-6f;
    int max_iter = 1000;

    float h_energy = 0.0f;

    void Initial(CONTROLLER* controller, int atom_numbers,
                 const char* parameter_in_file, const char* type_in_file);
    void Calculate_Charges(int atom_numbers, float* d_charge,
                           const VECTOR* d_crd, const LTMatrix3 cell,
                           const LTMatrix3 rcell, const ATOM_GROUP* fnl_d_nl,
                           float cutoff, float* d_energy = NULL,
                           VECTOR* frc = NULL, int need_virial = 0,
                           LTMatrix3* atom_virial = NULL);
    void Step_Print(CONTROLLER* controller);
    void Print_Charges(const float* d_charge);
};

#endif
