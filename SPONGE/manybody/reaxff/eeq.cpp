#include "eeq.h"

#define COULOMB_CONSTANT (332.05221729f)

void REAXFF_EEQ::Initial(CONTROLLER* controller, int atom_numbers,
                         const char* parameter_in_file,
                         const char* type_in_file)
{
    if (parameter_in_file == NULL || type_in_file == NULL)
    {
        controller->printf(
            "REAXFF_EEQ IS NOT INITIALIZED (missing input files)\n\n");
        return;
    }

    this->atom_numbers = atom_numbers;
    controller->printf("START INITIALIZING REAXFF_EEQ\n");

    FILE* fp_p;
    Open_File_Safely(&fp_p, parameter_in_file, "r");
    char line[1024];
    auto throw_bad_format = [&](const char* file_name, const char* reason)
    {
        char error_msg[1024];
        sprintf(error_msg, "Reason:\n\t%s in file %s\n", reason, file_name);
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "REAXFF_EEQ::Initial", error_msg);
    };
    auto read_line_or_throw =
        [&](FILE* file, const char* file_name, const char* stage)
    {
        if (fgets(line, 1024, file) == NULL)
        {
            char reason[512];
            sprintf(reason, "failed to read %s", stage);
            throw_bad_format(file_name, reason);
        }
    };

    read_line_or_throw(fp_p, parameter_in_file, "parameter header line 1");
    read_line_or_throw(fp_p, parameter_in_file, "general parameter count line");
    int n_gen_params = 0;
    if (sscanf(line, "%d", &n_gen_params) != 1)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of general parameters");
    }

    for (int i = 0; i < n_gen_params; i++)
    {
        read_line_or_throw(fp_p, parameter_in_file, "general parameter block");
    }

    read_line_or_throw(fp_p, parameter_in_file, "atom type count line");
    int n_atom_types = 0;
    if (sscanf(line, "%d", &n_atom_types) != 1)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of atom types");
    }
    this->atom_type_numbers = n_atom_types;

    read_line_or_throw(fp_p, parameter_in_file, "atom type header line 1");
    read_line_or_throw(fp_p, parameter_in_file, "atom type header line 2");
    read_line_or_throw(fp_p, parameter_in_file, "atom type header line 3");

    std::map<std::string, int> type_map;
    Malloc_Safely((void**)&h_chi, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_eta, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_gamma, sizeof(float) * n_atom_types);

    for (int i = 0; i < n_atom_types; i++)
    {
        char atom_name[16];
        float dummy;

        read_line_or_throw(fp_p, parameter_in_file,
                           "atom type parameter line 1");
        if (sscanf(line, "%s %f %f %f %f %f %f %f %f", atom_name, &dummy,
                   &dummy, &dummy, &dummy, &dummy, &h_gamma[i], &dummy,
                   &dummy) != 9)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 1 for type index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        type_map[std::string(atom_name)] = i;

        read_line_or_throw(fp_p, parameter_in_file,
                           "atom type parameter line 2");
        if (sscanf(line, "%f %f %f %f %f %f %f", &dummy, &dummy, &dummy, &dummy,
                   &dummy, &h_chi[i], &h_eta[i]) != 7)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 2 for type index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }

        h_chi[i] *= CONSTANT_EV_TO_KCAL_MOL;
        h_eta[i] *= CONSTANT_EV_TO_KCAL_MOL * 2.0f;

        read_line_or_throw(fp_p, parameter_in_file,
                           "atom type parameter line 3");
        read_line_or_throw(fp_p, parameter_in_file,
                           "atom type parameter line 4");
    }
    fclose(fp_p);

    FILE* fp_t;
    Open_File_Safely(&fp_t, type_in_file, "r");
    int check_atom_numbers = 0;
    read_line_or_throw(fp_t, type_in_file, "atom number line");
    if (sscanf(line, "%d", &check_atom_numbers) != 1)
    {
        throw_bad_format(type_in_file, "failed to parse atom numbers");
    }
    if (check_atom_numbers != atom_numbers)
    {
        char reason[512];
        sprintf(reason, "atom numbers (%d) does not match system (%d)",
                check_atom_numbers, atom_numbers);
        throw_bad_format(type_in_file, reason);
    }

    Malloc_Safely((void**)&h_atom_type, sizeof(int) * atom_numbers);
    for (int i = 0; i < atom_numbers; i++)
    {
        char type_name[16];
        read_line_or_throw(fp_t, type_in_file, "atom type entry line");
        if (sscanf(line, "%s", type_name) != 1)
        {
            char reason[512];
            sprintf(reason, "failed to parse atom type at index %d", i + 1);
            throw_bad_format(type_in_file, reason);
        }
        if (type_map.find(std::string(type_name)) == type_map.end())
        {
            char reason[512];
            sprintf(reason, "atom type %s not found in parameter file %s",
                    type_name, parameter_in_file);
            throw_bad_format(type_in_file, reason);
        }
        h_atom_type[i] = type_map[std::string(type_name)];
    }
    fclose(fp_t);

    Device_Malloc_And_Copy_Safely((void**)&d_chi, h_chi,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_eta, h_eta,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_gamma, h_gamma,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                  sizeof(int) * atom_numbers);

    Device_Malloc_Safely((void**)&d_b, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_r, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_p, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_Ap, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_q, sizeof(float) * atom_numbers);

    Device_Malloc_Safely((void**)&d_s, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_t, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_temp_sum, sizeof(float));

    is_initialized = 1;
    controller->Step_Print_Initial("REAXFF_EEQ", "%14.7e");
    controller->printf("END INITIALIZING REAXFF_EEQ\n\n");
}

static __global__ void EEQ_Matrix_Vector_Multiply(
    int atom_numbers, const VECTOR* crd, const int* atom_types,
    const float* eta, const float* gamma, const float* p, float* Ap,
    const ATOM_GROUP* nl, const LTMatrix3 cell, const LTMatrix3 rcell,
    float cutoff)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        int type_i = atom_types[i];
        float gamma_i = gamma[type_i];

        float sum = eta[type_i] * p[i];

        ATOM_GROUP nl_i = nl[i];
        VECTOR ri = crd[i];
        for (int j_idx = 0; j_idx < nl_i.atom_numbers; j_idx++)
        {
            int atom_j = nl_i.atom_serial[j_idx];

            int type_j = atom_types[atom_j];
            float gamma_j = gamma[type_j];

            VECTOR rj = crd[atom_j];
            VECTOR drij = Get_Periodic_Displacement(ri, rj, cell, rcell);
            float r2 = drij.x * drij.x + drij.y * drij.y + drij.z * drij.z;
            float r = sqrtf(r2);

            if (r < cutoff)
            {
                float x = r / cutoff;
                float x2 = x * x;
                float x4 = x2 * x2;
                float x5 = x4 * x;
                float x6 = x5 * x;
                float x7 = x6 * x;
                float taper =
                    20.0f * x7 - 70.0f * x6 + 84.0f * x5 - 35.0f * x4 + 1.0f;

                float gamma_ij = sqrtf(gamma_i * gamma_j);
                float r_shield = powf(
                    powf(r, 3.0f) + powf(1.0f / gamma_ij, 3.0f), 1.0f / 3.0f);

                float interaction_strength =
                    taper * (COULOMB_CONSTANT / r_shield);

                float contrib_to_i = interaction_strength * p[atom_j];

                float contrib_to_j = interaction_strength * p[i];

                sum += contrib_to_i;

                atomicAdd(&Ap[atom_j], contrib_to_j);
            }
        }

        atomicAdd(&Ap[i], sum);
    }
}

static __global__ void Vector_Update_P(int n, float* p, const float* r,
                                       float beta)
{
    SIMPLE_DEVICE_FOR(i, n) { p[i] = r[i] + beta * p[i]; }
}

static __global__ void Vector_Update_X_R(int n, float* x, float* r,
                                         const float* p, const float* Ap,
                                         float alpha)
{
    SIMPLE_DEVICE_FOR(i, n)
    {
        x[i] += alpha * p[i];
        r[i] -= alpha * Ap[i];
    }
}

static __global__ void Setup_B_Chi(int n, float* b, const int* atom_types,
                                   const float* chi)
{
    SIMPLE_DEVICE_FOR(i, n) { b[i] = -chi[atom_types[i]]; }
}

static __global__ void Setup_B_One(int n, float* b)
{
    SIMPLE_DEVICE_FOR(i, n) { b[i] = 1.0f; }
}

static __global__ void Vector_Scale_Add(int n, float* q, const float* t,
                                        const float* s, float mu)
{
    SIMPLE_DEVICE_FOR(i, n) { q[i] = t[i] + mu * s[i]; }
}

static __global__ void EEQ_Convert_Charge_Unit(int n, float* q_out,
                                               const float* q_in, float scale)
{
    SIMPLE_DEVICE_FOR(i, n) { q_out[i] = q_in[i] * scale; }
}

static __global__ void Elementwise_Multiply(int n, float* out, const float* a,
                                            const float* b)
{
    SIMPLE_DEVICE_FOR(i, n) { out[i] = a[i] * b[i]; }
}

static __global__ void EEQ_Distribute_Energy_Kernel(
    int n, float* d_energy, const float* d_charge, const int* atom_types,
    const float* d_chi, const float* d_eta, const float* d_Aq)
{
    SIMPLE_DEVICE_FOR(i, n)
    {
        int type_i = atom_types[i];
        float qi = d_charge[i];

        float e_pol_i = d_chi[type_i] * qi + 0.5f * d_eta[type_i] * qi * qi;

        float e_ele_i = 0.5f * qi * (d_Aq[i] - d_eta[type_i] * qi);

        float en_i = e_pol_i + e_ele_i;
        atomicAdd(&d_energy[i], en_i);
    }
}

static __global__ void EEQ_Calculate_Force_Kernel(
    int atom_numbers, const VECTOR* crd, const int* atom_types,
    const float* gamma, const float* d_charge, VECTOR* frc,
    const ATOM_GROUP* nl, const LTMatrix3 cell, const LTMatrix3 rcell,
    float cutoff, LTMatrix3* atom_virial)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        int type_i = atom_types[i];
        float gamma_i = gamma[type_i];
        float qi = d_charge[i];
        if (fabsf(qi) >= 1e-10f)
        {
            ATOM_GROUP nl_i = nl[i];
            VECTOR ri = crd[i];

            for (int j_idx = 0; j_idx < nl_i.atom_numbers; j_idx++)
            {
                int atom_j = nl_i.atom_serial[j_idx];
                if (atom_j <= i) continue;

                float qj = d_charge[atom_j];
                if (fabsf(qj) < 1e-10f) continue;

                int type_j = atom_types[atom_j];
                float gamma_j = gamma[type_j];

                VECTOR rj = crd[atom_j];
                VECTOR drij = Get_Periodic_Displacement(ri, rj, cell, rcell);
                float r2 = drij.x * drij.x + drij.y * drij.y + drij.z * drij.z;
                float r = sqrtf(r2);

                if (r < cutoff)
                {
                    SADfloat<1> r_sad(r, 0);

                    SADfloat<1> x = r_sad / SADfloat<1>(cutoff);
                    SADfloat<1> x2 = x * x;
                    SADfloat<1> x4 = x2 * x2;
                    SADfloat<1> x5 = x4 * x;
                    SADfloat<1> x6 = x5 * x;
                    SADfloat<1> x7 = x6 * x;
                    SADfloat<1> taper = 20.0f * x7 - 70.0f * x6 + 84.0f * x5 -
                                        35.0f * x4 + 1.0f;

                    float gamma_ij = sqrtf(gamma_i * gamma_j);
                    SADfloat<1> r_shield =
                        powf(powf(r_sad, 3.0f) + powf(1.0f / gamma_ij, 3.0f),
                             1.0f / 3.0f);

                    SADfloat<1> H = taper * (COULOMB_CONSTANT / r_shield);

                    float force_mag = -qi * qj * H.dval[0] / r;

                    float fx = force_mag * drij.x;
                    float fy = force_mag * drij.y;
                    float fz = force_mag * drij.z;

                    atomicAdd(&frc[i].x, fx);
                    atomicAdd(&frc[i].y, fy);
                    atomicAdd(&frc[i].z, fz);
                    atomicAdd(&frc[atom_j].x, -fx);
                    atomicAdd(&frc[atom_j].y, -fy);
                    atomicAdd(&frc[atom_j].z, -fz);
                    if (atom_virial)
                    {
                        VECTOR fij = {fx, fy, fz};
                        atomicAdd(atom_virial + i,
                                  Get_Virial_From_Force_Dis(fij, drij));
                    }
                }
            }
        }
    }
}

static __global__ void EEQ_Calculate_Epol_Kernel(int n, float* out,
                                                 const int* types,
                                                 const float* chi,
                                                 const float* eta,
                                                 const float* q)
{
    SIMPLE_DEVICE_FOR(i, n)
    {
        int t = types[i];
        out[i] = chi[t] * q[i] + 0.5f * eta[t] * q[i] * q[i];
    }
}

static __global__ void EEQ_Calculate_Eele_Kernel(int n, float* out,
                                                 const int* types,
                                                 const float* eta,
                                                 const float* q,
                                                 const float* Aq)
{
    SIMPLE_DEVICE_FOR(i, n)
    {
        out[i] = 0.5f * q[i] * (Aq[i] - eta[types[i]] * q[i]);
    }
}

void REAXFF_EEQ::Calculate_Charges(int atom_numbers, float* d_charge,
                                   const VECTOR* d_crd, const LTMatrix3 cell,
                                   const LTMatrix3 rcell,
                                   const ATOM_GROUP* fnl_d_nl, float cutoff,
                                   float* d_energy, VECTOR* frc,
                                   int need_virial, LTMatrix3* atom_virial)
{
    if (!is_initialized || fnl_d_nl == NULL) return;

    dim3 blockSize = {CONTROLLER::device_max_thread};
    dim3 gridSize = {(atom_numbers + blockSize.x - 1) / blockSize.x};

    auto solve = [&](float* x, float* b_in)
    {
        deviceMemset(x, 0, sizeof(float) * atom_numbers);
        deviceMemcpy(d_r, b_in, sizeof(float) * atom_numbers,
                     deviceMemcpyDeviceToDevice);
        deviceMemcpy(d_p, d_r, sizeof(float) * atom_numbers,
                     deviceMemcpyDeviceToDevice);

        float r_dot_r_old = 0, r_dot_r_new = 0;

        Launch_Device_Kernel(Elementwise_Multiply, gridSize, blockSize, 0, NULL,
                             atom_numbers, d_q, d_r, d_r);

        Sum_Of_List(d_q, d_temp_sum, atom_numbers);
        deviceMemcpy(&r_dot_r_old, d_temp_sum, sizeof(float),
                     deviceMemcpyDeviceToHost);

        for (int iter = 0; iter < max_iter; iter++)
        {
            deviceMemset(d_Ap, 0, sizeof(float) * atom_numbers);

            Launch_Device_Kernel(EEQ_Matrix_Vector_Multiply, gridSize,
                                 blockSize, 0, NULL, atom_numbers, d_crd,
                                 d_atom_type, d_eta, d_gamma, d_p, d_Ap,
                                 fnl_d_nl, cell, rcell, cutoff);

            float p_dot_Ap = 0;
            Launch_Device_Kernel(Elementwise_Multiply, gridSize, blockSize, 0,
                                 NULL, atom_numbers, d_q, d_p, d_Ap);

            Sum_Of_List(d_q, d_temp_sum, atom_numbers);
            deviceMemcpy(&p_dot_Ap, d_temp_sum, sizeof(float),
                         deviceMemcpyDeviceToHost);

            float alpha = r_dot_r_old / p_dot_Ap;

            Launch_Device_Kernel(Vector_Update_X_R, gridSize, blockSize, 0,
                                 NULL, atom_numbers, x, d_r, d_p, d_Ap, alpha);

            Launch_Device_Kernel(Elementwise_Multiply, gridSize, blockSize, 0,
                                 NULL, atom_numbers, d_q, d_r, d_r);

            Sum_Of_List(d_q, d_temp_sum, atom_numbers);
            deviceMemcpy(&r_dot_r_new, d_temp_sum, sizeof(float),
                         deviceMemcpyDeviceToHost);

            if (sqrtf(r_dot_r_new) < tolerance) break;

            float beta = r_dot_r_new / r_dot_r_old;

            Launch_Device_Kernel(Vector_Update_P, gridSize, blockSize, 0, NULL,
                                 atom_numbers, d_p, d_r, beta);

            r_dot_r_old = r_dot_r_new;
        }
    };

    Launch_Device_Kernel(Setup_B_Chi, gridSize, blockSize, 0, NULL,
                         atom_numbers, d_b, d_atom_type, d_chi);
    solve(d_t, d_b);

    Launch_Device_Kernel(Setup_B_One, gridSize, blockSize, 0, NULL,
                         atom_numbers, d_b);
    solve(d_s, d_b);

    float sum_t = 0, sum_s = 0;
    Sum_Of_List(d_t, d_temp_sum, atom_numbers);
    deviceMemcpy(&sum_t, d_temp_sum, sizeof(float), deviceMemcpyDeviceToHost);
    Sum_Of_List(d_s, d_temp_sum, atom_numbers);
    deviceMemcpy(&sum_s, d_temp_sum, sizeof(float), deviceMemcpyDeviceToHost);

    float Qtot = 0;
    float mu = (Qtot - sum_t) / sum_s;

    Launch_Device_Kernel(Vector_Scale_Add, gridSize, blockSize, 0, NULL,
                         atom_numbers, d_q, d_t, d_s, mu);

    deviceMemset(d_Ap, 0, sizeof(float) * atom_numbers);
    Launch_Device_Kernel(EEQ_Matrix_Vector_Multiply, gridSize, blockSize, 0,
                         NULL, atom_numbers, d_crd, d_atom_type, d_eta, d_gamma,
                         d_q, d_Ap, fnl_d_nl, cell, rcell, cutoff);

    Launch_Device_Kernel(EEQ_Calculate_Epol_Kernel, gridSize, blockSize, 0,
                         NULL, atom_numbers, d_r, d_atom_type, d_chi, d_eta,
                         d_q);

    float sum_epol = 0;
    Sum_Of_List(d_r, d_temp_sum, atom_numbers);
    deviceMemcpy(&sum_epol, d_temp_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);

    Launch_Device_Kernel(EEQ_Calculate_Eele_Kernel, gridSize, blockSize, 0,
                         NULL, atom_numbers, d_r, d_atom_type, d_eta, d_q,
                         d_Ap);

    float sum_eele = 0;
    Sum_Of_List(d_r, d_temp_sum, atom_numbers);
    deviceMemcpy(&sum_eele, d_temp_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);

    h_energy = sum_epol + sum_eele;

    if (d_energy != NULL)
    {
        Launch_Device_Kernel(EEQ_Distribute_Energy_Kernel, gridSize, blockSize,
                             0, NULL, atom_numbers, d_energy, d_q, d_atom_type,
                             d_chi, d_eta, d_Ap);
    }

    if (frc != NULL)
    {
        Launch_Device_Kernel(EEQ_Calculate_Force_Kernel, gridSize, blockSize, 0,
                             NULL, atom_numbers, d_crd, d_atom_type, d_gamma,
                             d_q, frc, fnl_d_nl, cell, rcell, cutoff,
                             need_virial ? atom_virial : NULL);
    }

    Launch_Device_Kernel(EEQ_Convert_Charge_Unit, gridSize, blockSize, 0, NULL,
                         atom_numbers, d_charge, d_q,
                         CONSTANT_SPONGE_CHARGE_SCALE);
}

void REAXFF_EEQ::Step_Print(CONTROLLER* controller)

{
    if (!is_initialized) return;
    controller->Step_Print("REAXFF_EEQ", h_energy, true);
}

void REAXFF_EEQ::Print_Charges(const float* d_charge)
{
    if (!is_initialized) return;
    float* h_q = NULL;
    Malloc_Safely((void**)&h_q, sizeof(float) * atom_numbers);
    deviceMemcpy(h_q, d_charge, sizeof(float) * atom_numbers,
                 deviceMemcpyDeviceToHost);

    FILE* fp = fopen("eeq_charges.txt", "w");
    if (fp)
    {
        for (int i = 0; i < atom_numbers; i++)
        {
            float q_elementary = h_q[i] / CONSTANT_SPONGE_CHARGE_SCALE;
            fprintf(fp, "%d %.6f\n", i + 1, q_elementary);
        }
        fclose(fp);
    }
    free(h_q);
}
