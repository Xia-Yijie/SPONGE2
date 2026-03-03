#include "bond_soft.h"

static __global__ void Soft_Bond_Force_With_Atom_Energy_And_Virial_device(
    const int bond_numbers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const int local_atom_numbers, const int* atom_a,
    const int* atom_b, const float* bond_k, const float* bond_r0,
    const int* AB_mask, VECTOR* frc, int need_atom_energy, float* atom_energy,
    float* bond_ene, int need_virial, LTMatrix3* atom_virial,
    int need_dH_dlambda, float* dH_dlambda, const float lambda,
    const float alpha)
{
#ifdef USE_GPU
    int bond_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (bond_i < bond_numbers)
#else
#pragma omp parallel for
    for (int bond_i = 0; bond_i < bond_numbers; bond_i++)
#endif
    {
        int atom_i = atom_a[bond_i];
        int atom_j = atom_b[bond_i];

        float k = bond_k[bond_i];
        float r0 = bond_r0[bond_i];
        int ABmask = AB_mask[bond_i];
        float tmp_lambda = (ABmask == 0 ? 1.0 - lambda : lambda);
        float tmp_lambda_ = 1.0 - tmp_lambda;

        VECTOR dr =
            Get_Periodic_Displacement(crd[atom_i], crd[atom_j], cell, rcell);
        float abs_r = norm3df(dr.x, dr.y, dr.z);
        float r_1 = 1. / abs_r;

        float temp_denominator =
            1 + alpha * tmp_lambda_ * (abs_r - r0) * (abs_r - r0);
        float tempf = 2 * k * tmp_lambda * (abs_r - r0) / temp_denominator /
                      temp_denominator;

        VECTOR f = tempf * r_1 * dr;

        if (atom_i < local_atom_numbers)
        {
            atomicAdd(&frc[atom_i].x, -f.x);
            atomicAdd(&frc[atom_i].y, -f.y);
            atomicAdd(&frc[atom_i].z, -f.z);
            if (need_virial)
            {
                atomicAdd(atom_virial + atom_i,
                          Get_Virial_From_Force_Dis(-f, dr));
            }
            if (need_atom_energy)
            {
                float ene = tmp_lambda * k * (abs_r - r0) * (abs_r - r0) /
                            temp_denominator;
                atomicAdd(&atom_energy[atom_i], ene);
                bond_ene[bond_i] = ene;
            }
            if (need_dH_dlambda)
            {
                float tmp_sign = (ABmask == 0 ? -1.0 : 1.0);
                float tmp_lambda_ = 1.0 - tmp_lambda;
                float tempf2 = (abs_r - r0) * (abs_r - r0);
                float temp_denominator =
                    1.0 / (1 + alpha * tmp_lambda_ * tempf2);
                float dH_dlambda_abs = k * tempf2 * temp_denominator *
                                       temp_denominator * (1 + alpha * tempf2);
                dH_dlambda[bond_i] = dH_dlambda_abs * tmp_sign;
            }
        }

        if (atom_j < local_atom_numbers)
        {
            atomicAdd(&frc[atom_j].x, f.x);
            atomicAdd(&frc[atom_j].y, f.y);
            atomicAdd(&frc[atom_j].z, f.z);
        }
    }
}

void BOND_SOFT::Initial(CONTROLLER* controller, CONECT* connectivity,
                        PAIR_DISTANCE* con_dis, const char* module_name)
{
    controller[0].printf("START INITIALIZING BOND SOFT:\n");
    if (module_name == NULL)
    {
        strcpy(this->module_name, "bond_soft");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (controller[0].Command_Exist(this->module_name, "in_file"))
    {
        if (controller[0].Command_Exist("lambda_bond"))
        {
            this->lambda = atof(controller[0].Command("lambda_bond"));
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorMissingCommand, "BOND_SOFT::Initial",
                "Reason:\n\t'lambda_bond' is required for the calculation of "
                "bond_soft.\n");
        }
        if (controller[0].Command_Exist("soft_bond_alpha"))
        {
            this->alpha = atof(controller[0].Command("soft_bond_alpha"));
        }
        else
        {
            printf(
                "Warning: FEP alpha of soft bond missing for the calculation "
                "of SOFT BOND, set to default value 0.0.\n");
            this->alpha = 0.0;
        }
        FILE* fp = NULL;
        Open_File_Safely(
            &fp, controller[0].Command(this->module_name, "in_file"), "r");
        int toscan = fscanf(fp, "%d", &soft_bond_numbers);
        controller[0].printf("    soft_bond_numbers is %d\n",
                             soft_bond_numbers);
        Memory_Allocate();
        for (int i = 0; i < soft_bond_numbers; i++)
        {
            toscan = fscanf(fp, "%d %d %f %f %d", h_atom_a + i, h_atom_b + i,
                            h_k + i, h_r0 + i, h_ABmask + i);
        }
        fclose(fp);
        Parameter_Host_To_Device();
        is_initialized = 1;
    }

    if (is_initialized && !is_controller_printf_initialized)
    {
        for (int i = 0; i < soft_bond_numbers; i += 1)
        {
            connectivity[0][h_atom_a[i]].insert(h_atom_b[i]);
            connectivity[0][h_atom_b[i]].insert(h_atom_a[i]);
            if (h_atom_a[i] < h_atom_b[i])
            {
                con_dis[0][std::pair<int, int>(h_atom_a[i], h_atom_b[i])] =
                    h_r0[i];
            }
            else
            {
                con_dis[0][std::pair<int, int>(h_atom_b[i], h_atom_a[i])] =
                    h_r0[i];
            }
        }
        controller[0].Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n",
                             last_modify_date);
    }

    controller[0].printf("END INITIALIZING SOFT BOND\n\n");
}

void BOND_SOFT::Memory_Allocate()
{
    if (!Malloc_Safely((void**)&(this->h_atom_a),
                       sizeof(int) * this->soft_bond_numbers))
        printf(
            "        Error occurs when malloc BOND_SOFT::h_atom_a in "
            "BOND_SOFT::Memory_Allocate");
    if (!Malloc_Safely((void**)&(this->h_atom_b),
                       sizeof(int) * this->soft_bond_numbers))
        printf(
            "        Error occurs when malloc BOND_SOFT::h_atom_b in "
            "BOND_SOFT::Memory_Allocate");
    if (!Malloc_Safely((void**)&(this->h_k),
                       sizeof(float) * this->soft_bond_numbers))
        printf(
            "        Error occurs when malloc BOND_SOFT::h_k in "
            "BOND_SOFT::Memory_Allocate");
    if (!Malloc_Safely((void**)&(this->h_r0),
                       sizeof(float) * this->soft_bond_numbers))
        printf(
            "        Error occurs when malloc BOND_SOFT::h_r0 in "
            "BOND_SOFT::Memory_Allocate");
    if (!Malloc_Safely((void**)&(this->h_ABmask),
                       sizeof(int) * this->soft_bond_numbers))
        printf(
            "        Error occurs when malloc BOND_SOFT::h_ABmask in "
            "BOND_SOFT::Memory_Allocate");
    if (!Malloc_Safely((void**)&(this->h_soft_bond_ene),
                       sizeof(float) * this->soft_bond_numbers))
        printf(
            "        Error occurs when malloc BOND_SOFT::h_soft_bond_ene in "
            "BOND_SOFT::Memory_Allocate");
    if (!Malloc_Safely((void**)&(this->h_sigma_of_soft_bond_ene),
                       sizeof(float)))
        printf(
            "        Error occurs when malloc "
            "BOND_SOFT::h_sigma_of_soft_bond_ene in "
            "BOND_SOFT::Memory_Allocate");
    if (!Malloc_Safely((void**)&(this->h_soft_bond_dH_dlambda),
                       sizeof(float) * this->soft_bond_numbers))
        printf(
            "        Error occurs when malloc "
            "BOND_SOFT::h_soft_bond_dH_dlambda in BOND_SOFT::Memory_Allocate");
    if (!Malloc_Safely((void**)&(this->h_sigma_of_dH_dlambda), sizeof(float)))
        printf(
            "        Error occurs when malloc BOND_SOFT::h_sigma_of_dH_dlambda "
            "in BOND_SOFT::Memory_Allocate");

    if (!Device_Malloc_Safely((void**)&this->d_atom_a,
                              sizeof(int) * this->soft_bond_numbers))
        printf(
            "        Error occurs when CUDA malloc BOND_SOFT::d_atom_a in "
            "BOND_SOFT::Memory_Allocate");
    if (!Device_Malloc_Safely((void**)&this->d_atom_b,
                              sizeof(int) * this->soft_bond_numbers))
        printf(
            "        Error occurs when CUDA malloc BOND_SOFT::d_atom_b in "
            "BOND_SOFT::Memory_Allocate");
    if (!Device_Malloc_Safely((void**)&this->d_k,
                              sizeof(float) * this->soft_bond_numbers))
        printf(
            "        Error occurs when CUDA malloc BOND_SOFT::d_k in "
            "BOND_SOFT::Memory_Allocate");
    if (!Device_Malloc_Safely((void**)&this->d_r0,
                              sizeof(float) * this->soft_bond_numbers))
        printf(
            "        Error occurs when CUDA malloc BOND_SOFT::d_r0 in "
            "BOND_SOFT::Memory_Allocate");
    if (!Device_Malloc_Safely((void**)&this->d_ABmask,
                              sizeof(int) * this->soft_bond_numbers))
        printf(
            "         Error occurs when CUDA malloc BOND_SOFT::d_ABmask in "
            "BOND_SOFT::Memory_Allocate");
    if (!Device_Malloc_Safely((void**)&this->d_soft_bond_ene,
                              sizeof(float) * this->soft_bond_numbers))
        printf(
            "        Error occurs when CUDA malloc BOND_SOFT::d_bond_ene in "
            "BOND_SOFT::Memory_Allocate");
    if (!Device_Malloc_Safely((void**)&this->d_sigma_of_soft_bond_ene,
                              sizeof(float)))
        printf(
            "        Error occurs when CUDA malloc "
            "BOND_SOFT::d_sigma_of_bond_ene in BOND_SOFT::Memory_Allocate");
    if (!Device_Malloc_Safely((void**)&this->d_soft_bond_dH_dlambda,
                              sizeof(float) * this->soft_bond_numbers))
        printf(
            "        Error occurs when CUDA malloc "
            "BOND_SOFT::d_soft_bond_dH_dlambda in BOND_SOFT::Memory_Allocate");
    if (!Device_Malloc_Safely((void**)&this->d_sigma_of_dH_dlambda,
                              sizeof(float)))
        printf(
            "        Error occurs when CUDA malloc "
            "BOND_SOFT::d_sigma_of_dH_dlambda in BOND_SOFT::Memory_Allocate");

    Device_Malloc_Safely((void**)&this->d_atom_a_local,
                         sizeof(int) * this->soft_bond_numbers);
    Device_Malloc_Safely((void**)&this->d_atom_b_local,
                         sizeof(int) * this->soft_bond_numbers);
    Device_Malloc_Safely((void**)&this->d_ABmask_local,
                         sizeof(int) * this->soft_bond_numbers);
    Device_Malloc_Safely((void**)&this->d_k_local,
                         sizeof(float) * this->soft_bond_numbers);
    Device_Malloc_Safely((void**)&this->d_r0_local,
                         sizeof(float) * this->soft_bond_numbers);
    Device_Malloc_Safely((void**)&this->d_num_bond_local, sizeof(int));
    deviceMemset(this->d_num_bond_local, 0, sizeof(int));
}

void BOND_SOFT::Parameter_Host_To_Device()
{
    deviceMemcpy(this->d_atom_a, this->h_atom_a,
                 sizeof(int) * this->soft_bond_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(this->d_atom_b, this->h_atom_b,
                 sizeof(int) * this->soft_bond_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(this->d_k, this->h_k, sizeof(float) * this->soft_bond_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(this->d_r0, this->h_r0,
                 sizeof(float) * this->soft_bond_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(this->d_ABmask, this->h_ABmask,
                 sizeof(float) * this->soft_bond_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(this->d_soft_bond_ene, this->h_soft_bond_ene,
                 sizeof(float) * this->soft_bond_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(this->d_sigma_of_soft_bond_ene, this->h_sigma_of_soft_bond_ene,
                 sizeof(float), deviceMemcpyHostToDevice);
}

void BOND_SOFT::Clear()
{
    if (is_initialized)
    {
        Free_Single_Device_Pointer((void**)&this->d_atom_a);
        Free_Single_Device_Pointer((void**)&this->d_atom_b);
        Free_Single_Device_Pointer((void**)&this->d_k);
        Free_Single_Device_Pointer((void**)&this->d_r0);
        Free_Single_Device_Pointer((void**)&this->d_ABmask);
        Free_Single_Device_Pointer((void**)&this->d_soft_bond_ene);
        Free_Single_Device_Pointer((void**)&this->d_sigma_of_soft_bond_ene);
        Free_Single_Device_Pointer((void**)&this->d_ABmask_local);
        Free_Single_Device_Pointer((void**)&this->d_atom_a_local);
        Free_Single_Device_Pointer((void**)&this->d_atom_b_local);
        Free_Single_Device_Pointer((void**)&this->d_k_local);
        Free_Single_Device_Pointer((void**)&this->d_r0_local);
        free(this->h_atom_a);
        free(this->h_atom_b);
        free(this->h_k);
        free(this->h_r0);
        free(this->h_soft_bond_ene);
        free(this->h_sigma_of_soft_bond_ene);

        h_atom_a = NULL;
        d_atom_a = NULL;
        h_atom_b = NULL;
        d_atom_b = NULL;
        d_ABmask = NULL;
        d_k = NULL;
        h_k = NULL;
        d_r0 = NULL;
        h_r0 = NULL;
        h_ABmask = NULL;

        h_soft_bond_ene = NULL;
        d_soft_bond_ene = NULL;
        d_sigma_of_soft_bond_ene = NULL;
        h_sigma_of_soft_bond_ene = NULL;

        is_initialized = 0;
    }
}

void BOND_SOFT::Soft_Bond_Force_With_Atom_Energy_And_Virial(
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell, VECTOR* frc,
    int need_atom_energy, float* atom_energy, int need_virial,
    LTMatrix3* atom_virial, int need_dH_dlambda)
{
    if (is_initialized)
    {
        Launch_Device_Kernel(
            Soft_Bond_Force_With_Atom_Energy_And_Virial_device,
            (num_bond_local + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, num_bond_local, crd, cell,
            rcell, local_atom_numbers, this->d_atom_a_local,
            this->d_atom_b_local, this->d_k_local, this->d_r0_local,
            this->d_ABmask_local, frc, need_atom_energy, atom_energy,
            this->d_soft_bond_ene, need_virial, atom_virial, need_dH_dlambda,
            this->d_soft_bond_dH_dlambda, this->lambda, this->alpha);
    }
}

static __global__ void get_local_device(
    int bond_numbers, int* d_atom_a, int* d_atom_b, char* atom_local_label,
    int* atom_local_id, int* d_atom_a_local, int* d_atom_b_local, float* d_k,
    float* d_r0, int* d_ABmask, float* d_k_local, float* d_r0_local,
    int* d_ABmask_local, int* d_num_bond_local)
{
#ifdef USE_GPU
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx != 0) return;
#endif
    d_num_bond_local[0] = 0;
    for (int i = 0; i < bond_numbers; i++)
    {
        if (atom_local_label[d_atom_a[i]] == 1 ||
            atom_local_label[d_atom_b[i]] == 1)
        {
            d_atom_a_local[d_num_bond_local[0]] = atom_local_id[d_atom_a[i]];
            d_atom_b_local[d_num_bond_local[0]] = atom_local_id[d_atom_b[i]];
            d_ABmask_local[d_num_bond_local[0]] = d_ABmask[i];
            d_k_local[d_num_bond_local[0]] = d_k[i];
            d_r0_local[d_num_bond_local[0]] = d_r0[i];
            d_num_bond_local[0]++;
        }
    }
}

void BOND_SOFT::Get_Local(int* atom_local, int local_atom_numbers,
                          int ghost_numbers, char* atom_local_label,
                          int* atom_local_id)
{
    if (!is_initialized) return;
    num_bond_local = 0;
    this->local_atom_numbers = local_atom_numbers;
    Launch_Device_Kernel(
        get_local_device, 1, 1, 0, NULL, this->soft_bond_numbers,
        this->d_atom_a, this->d_atom_b, atom_local_label, atom_local_id,
        this->d_atom_a_local, this->d_atom_b_local, this->d_k, this->d_r0,
        this->d_ABmask, this->d_k_local, this->d_r0_local, this->d_ABmask_local,
        this->d_num_bond_local);
    deviceMemcpy(&this->num_bond_local, this->d_num_bond_local, sizeof(int),
                 deviceMemcpyDeviceToHost);
}

void BOND_SOFT::Step_Print(CONTROLLER* controller)
{
    if (is_initialized && CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Sum_Of_List(d_soft_bond_ene, d_sigma_of_soft_bond_ene,
                    num_bond_local);  // 局部求和
        deviceMemcpy(h_sigma_of_soft_bond_ene, d_sigma_of_soft_bond_ene,
                     sizeof(float), deviceMemcpyDeviceToHost);
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, h_sigma_of_soft_bond_ene, 1, MPI_FLOAT,
                      MPI_SUM, CONTROLLER::pp_comm);
#endif
        controller->Step_Print(this->module_name, h_sigma_of_soft_bond_ene,
                               true);
    }
}

float BOND_SOFT::Get_Partial_H_Partial_Lambda(CONTROLLER* controller)
{
    if (is_initialized && CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Sum_Of_List(d_soft_bond_dH_dlambda, d_sigma_of_dH_dlambda,
                    num_bond_local);  // 局部求和
        deviceMemcpy(h_sigma_of_dH_dlambda, d_sigma_of_dH_dlambda,
                     sizeof(float), deviceMemcpyDeviceToHost);
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, h_sigma_of_dH_dlambda, 1, MPI_FLOAT,
                      MPI_SUM, CONTROLLER::pp_comm);
#endif
        return h_sigma_of_dH_dlambda[0];
    }
    else
    {
        return NAN;
    }
}
