#include "dihedral.h"

static __global__ void Dihedral_Force_With_Atom_Energy_And_Virial_Device(
    const int dihedral_numbers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const int local_atom_numbers, const int* atom_a,
    const int* atom_b, const int* atom_c, const int* atom_d, const int* ipn,
    const float* pk, const float* gamc, const float* gams, const float* pn,
    VECTOR* frc, int need_atom_energy, float* ene, float* di_ene,
    int need_virial, LTMatrix3* virial)
{
#ifdef USE_GPU
    int dihedral_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (dihedral_i < dihedral_numbers)
#else
#pragma omp parallel for
    for (int dihedral_i = 0; dihedral_i < dihedral_numbers; dihedral_i++)
#endif
    {
        int atom_i = atom_a[dihedral_i];
        int atom_j = atom_b[dihedral_i];
        int atom_k = atom_c[dihedral_i];
        int atom_l = atom_d[dihedral_i];

        int temp_ipn = ipn[dihedral_i];

        float temp_pk = pk[dihedral_i];
        float temp_pn = pn[dihedral_i];
        float temp_gamc = gamc[dihedral_i];
        float temp_gams = gams[dihedral_i];

        VECTOR drij =
            Get_Periodic_Displacement(crd[atom_i], crd[atom_j], cell, rcell);
        VECTOR drkj =
            Get_Periodic_Displacement(crd[atom_k], crd[atom_j], cell, rcell);
        VECTOR drkl =
            Get_Periodic_Displacement(crd[atom_k], crd[atom_l], cell, rcell);

        VECTOR r1 = drij ^ drkj;
        VECTOR r2 = drkl ^ drkj;

        float r1_1 = rnorm3df(r1.x, r1.y, r1.z);
        float r2_1 = rnorm3df(r2.x, r2.y, r2.z);
        float r1_2 = r1_1 * r1_1;
        float r2_2 = r2_1 * r2_1;
        float r1_1_r2_1 = r1_1 * r2_1;
        // PHI, pay attention to the var NAME
        float phi = r1 * r2 * r1_1_r2_1;
        phi = fmaxf(-0.999999, fminf(phi, 0.999999));
        phi = acosf(phi);

        float sign = (r2 ^ r1) * drkj;
        phi = copysignf(phi, sign);

        phi = CONSTANT_Pi - phi;

        float nphi = temp_pn * phi;

        float cos_phi = cosf(phi);
        float sin_phi = sinf(phi);
        float cos_nphi = cosf(nphi);
        float sin_nphi = sinf(nphi);

        // Here and following var name "phi" corespongding to the declaration of
        // phi aka, the var with the comment line "PHI, pay attention to the var
        // NAME" The real dihedral = Pi - ArcCos(so-called "phi") d(real
        // dihedral) = 1/sin(real dihedral) * d(so-called  "phi")
        float dE_dphi;
        if (fabsf(sin_phi) < 1e-6)
        {
            temp_ipn *= (((temp_ipn - 1) & 1) ^ 1);
            dE_dphi = temp_gamc * (temp_pn - temp_ipn + temp_ipn * cos_phi);
        }
        else
            dE_dphi = temp_pn * (temp_gamc * sin_nphi - temp_gams * cos_nphi) /
                      sin_phi;

        VECTOR dphi_dr1 = r1_1_r2_1 * r2 + cos_phi * r1_2 * r1;
        VECTOR dphi_dr2 = r1_1_r2_1 * r1 + cos_phi * r2_2 * r2;

        VECTOR dE_dri = dE_dphi * drkj ^ dphi_dr1;
        VECTOR dE_drl = dE_dphi * dphi_dr2 ^ drkj;
        VECTOR dE_drj_part = dE_dphi * ((drij ^ dphi_dr1) + (drkl ^ dphi_dr2));

        VECTOR fi = dE_dri;
        VECTOR fj = dE_drj_part - dE_dri;
        VECTOR fk = -dE_drl - dE_drj_part;
        VECTOR fl = dE_drl;
        if (atom_i < local_atom_numbers)
        {
            atomicAdd(&frc[atom_i].x, fi.x);
            atomicAdd(&frc[atom_i].y, fi.y);
            atomicAdd(&frc[atom_i].z, fi.z);
            if (need_atom_energy)
            {
                atomicAdd(&ene[atom_i], (temp_pk + cos_nphi * temp_gamc +
                                         sin_nphi * temp_gams));
                di_ene[dihedral_i] =
                    (temp_pk + cos_nphi * temp_gamc + sin_nphi * temp_gams);
            }

            if (need_virial)
            {
                atomicAdd(virial + atom_i,
                          Get_Virial_From_Force_Dis(dE_drl, drkl) +
                              Get_Virial_From_Force_Dis(dE_dri, drij) +
                              Get_Virial_From_Force_Dis(dE_drj_part, drkj));
            }
        }

        if (atom_j < local_atom_numbers)
        {
            atomicAdd(&frc[atom_j].x, fj.x);
            atomicAdd(&frc[atom_j].y, fj.y);
            atomicAdd(&frc[atom_j].z, fj.z);
        }
        if (atom_k < local_atom_numbers)
        {
            atomicAdd(&frc[atom_k].x, fk.x);
            atomicAdd(&frc[atom_k].y, fk.y);
            atomicAdd(&frc[atom_k].z, fk.z);
        }
        if (atom_l < local_atom_numbers)
        {
            atomicAdd(&frc[atom_l].x, fl.x);
            atomicAdd(&frc[atom_l].y, fl.y);
            atomicAdd(&frc[atom_l].z, fl.z);
        }
    }
}

void DIHEDRAL::Initial(CONTROLLER* controller, const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "dihedral");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }

    char file_name_suffix[CHAR_LENGTH_MAX];
    sprintf(file_name_suffix, "in_file");

    if (controller->Command_Exist(this->module_name, file_name_suffix))
    {
        controller->printf("START INITIALIZING DIHEDRAL (%s_%s):\n",
                           this->module_name, file_name_suffix);

        FILE* fp = NULL;
        Open_File_Safely(
            &fp, controller->Command(this->module_name, file_name_suffix), "r");

        int scanf_ret = fscanf(fp, "%d", &dihedral_numbers);
        controller->printf("    dihedral_numbers is %d\n", dihedral_numbers);
        Memory_Allocate();

        float temp;
        for (int i = 0; i < dihedral_numbers; i++)
        {
            scanf_ret =
                fscanf(fp, "%d %d %d %d %d %f %f", h_atom_a + i, h_atom_b + i,
                       h_atom_c + i, h_atom_d + i, h_ipn + i, h_pk + i, &temp);
            h_pn[i] = (float)h_ipn[i];
            h_gamc[i] = cosf(temp) * h_pk[i];
            h_gams[i] = sinf(temp) * h_pk[i];
        }
        fclose(fp);
        Parameter_Host_To_Device();
        is_initialized = 1;
    }
    else if (controller->Command_Exist("amber_parm7") && module_name == NULL)
    {
        controller->printf("START INITIALIZING DIHEDRAL (amber_parm7):\n");
        Read_Information_From_AMBERFILE(controller->Command("amber_parm7"),
                                        controller[0]);
        if (dihedral_numbers > 0) is_initialized = 1;
    }
    else
    {
        controller->printf("DIHEDRAL IS NOT INITIALIZED\n\n");
    }

    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = 1;
        controller->printf("    structure last modify date is %d\n",
                           last_modify_date);
    }
    if (is_initialized)
    {
        controller->printf("END INITIALIZING DIHEDRAL\n\n");
    }
}

void DIHEDRAL::Read_Information_From_AMBERFILE(const char* file_name,
                                               CONTROLLER controller)
{
    float *phase_type_cpu = NULL, *pk_type_cpu = NULL, *pn_type_cpu = NULL;
    int dihedral_type_numbers = 0, dihedral_with_hydrogen = 0;
    FILE* parm = NULL;
    Open_File_Safely(&parm, file_name, "r");
    char temps[CHAR_LENGTH_MAX];
    char temp_first_str[CHAR_LENGTH_MAX];
    char temp_second_str[CHAR_LENGTH_MAX];
    int i, tempi;
    float tempf, tempf2;
    controller.printf("    Reading dihedral information from AMBER file:\n");
    while (true)
    {
        if (!fgets(temps, CHAR_LENGTH_MAX, parm)) break;
        if (sscanf(temps, "%s %s", temp_first_str, temp_second_str) != 2)
        {
            continue;
        }
        if (strcmp(temp_first_str, "%FLAG") == 0 &&
            strcmp(temp_second_str, "POINTERS") == 0)
        {
            char* get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);

            for (i = 0; i < 6; i++) int scanf_ret = fscanf(parm, "%d", &tempi);

            int scanf_ret = fscanf(parm, "%d", &dihedral_with_hydrogen);
            scanf_ret = fscanf(parm, "%d", &this->dihedral_numbers);
            this->dihedral_numbers += dihedral_with_hydrogen;

            controller.printf("        dihedral numbers is %d\n",
                              this->dihedral_numbers);

            this->Memory_Allocate();

            for (i = 0; i < 9; i++) scanf_ret = fscanf(parm, "%d", &tempi);

            scanf_ret = fscanf(parm, "%d", &dihedral_type_numbers);
            controller.printf("        dihedral type numbers is %d\n",
                              dihedral_type_numbers);

            Malloc_Safely((void**)&phase_type_cpu,
                          sizeof(float) * dihedral_type_numbers);
            Malloc_Safely((void**)&pk_type_cpu,
                          sizeof(float) * dihedral_type_numbers);
            Malloc_Safely((void**)&pn_type_cpu,
                          sizeof(float) * dihedral_type_numbers);
        }
        if (strcmp(temp_first_str, "%FLAG") == 0 &&
            strcmp(temp_second_str, "DIHEDRAL_FORCE_CONSTANT") == 0)
        {
            controller.printf("        read dihedral force constant\n");
            char* get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            for (i = 0; i < dihedral_type_numbers; i++)
                int scanf_ret = fscanf(parm, "%f", &pk_type_cpu[i]);
        }

        if (strcmp(temp_first_str, "%FLAG") == 0 &&
            strcmp(temp_second_str, "DIHEDRAL_PHASE") == 0)
        {
            controller.printf("        read dihedral phase\n");
            char* get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            for (i = 0; i < dihedral_type_numbers; i++)
                int scanf_ret = fscanf(parm, "%f", &phase_type_cpu[i]);
        }

        if (strcmp(temp_first_str, "%FLAG") == 0 &&
            strcmp(temp_second_str, "DIHEDRAL_PERIODICITY") == 0)
        {
            controller.printf("        read dihedral periodicity\n");
            char* get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            for (i = 0; i < dihedral_type_numbers; i++)
                int scanf_ret = fscanf(parm, "%f", &pn_type_cpu[i]);
        }

        if (strcmp(temp_first_str, "%FLAG") == 0 &&
            strcmp(temp_second_str, "DIHEDRALS_INC_HYDROGEN") == 0)
        {
            char* get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            for (i = 0; i < dihedral_with_hydrogen; i++)
            {
                int scanf_ret = fscanf(parm, "%d\n", &this->h_atom_a[i]);
                scanf_ret = fscanf(parm, "%d\n", &this->h_atom_b[i]);
                scanf_ret = fscanf(parm, "%d\n", &this->h_atom_c[i]);
                scanf_ret = fscanf(parm, "%d\n", &this->h_atom_d[i]);
                scanf_ret = fscanf(parm, "%d\n", &tempi);
                this->h_atom_a[i] /= 3;
                this->h_atom_b[i] /= 3;
                this->h_atom_c[i] /= 3;

                this->h_atom_d[i] = abs(this->h_atom_d[i] / 3);
                tempi -= 1;
                this->h_pk[i] = pk_type_cpu[tempi];

                tempf = phase_type_cpu[tempi];
                if (abs(tempf - CONSTANT_Pi) <= 0.001) tempf = CONSTANT_Pi;

                tempf2 = cosf(tempf);
                if (fabsf(tempf2) < 1e-6) tempf2 = 0;
                this->h_gamc[i] = tempf2 * this->h_pk[i];

                tempf2 = sinf(tempf);
                if (fabsf(tempf2) < 1e-6) tempf2 = 0;
                this->h_gams[i] = tempf2 * this->h_pk[i];

                this->h_pn[i] = fabsf(pn_type_cpu[tempi]);
                this->h_ipn[i] = (int)(this->h_pn[i] + 0.001);
            }
        }
        if (strcmp(temp_first_str, "%FLAG") == 0 &&
            strcmp(temp_second_str, "DIHEDRALS_WITHOUT_HYDROGEN") == 0)
        {
            char* get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            for (i = dihedral_with_hydrogen; i < this->dihedral_numbers; i++)
            {
                int scanf_ret = fscanf(parm, "%d\n", &this->h_atom_a[i]);
                scanf_ret = fscanf(parm, "%d\n", &this->h_atom_b[i]);
                scanf_ret = fscanf(parm, "%d\n", &this->h_atom_c[i]);
                scanf_ret = fscanf(parm, "%d\n", &this->h_atom_d[i]);
                scanf_ret = fscanf(parm, "%d\n", &tempi);
                this->h_atom_a[i] /= 3;
                this->h_atom_b[i] /= 3;
                this->h_atom_c[i] /= 3;

                this->h_atom_d[i] = abs(this->h_atom_d[i] / 3);
                tempi -= 1;
                this->h_pk[i] = pk_type_cpu[tempi];

                tempf = phase_type_cpu[tempi];
                if (abs(tempf - CONSTANT_Pi) <= 0.001) tempf = CONSTANT_Pi;

                tempf2 = cosf(tempf);
                if (fabsf(tempf2) < 1e-6) tempf2 = 0;
                this->h_gamc[i] = tempf2 * this->h_pk[i];

                tempf2 = sinf(tempf);
                if (fabsf(tempf2) < 1e-6) tempf2 = 0;
                this->h_gams[i] = tempf2 * this->h_pk[i];

                this->h_pn[i] = fabsf(pn_type_cpu[tempi]);
                this->h_ipn[i] = (int)(this->h_pn[i] + 0.001);
            }
        }
    }

    for (int i = 0; i < this->dihedral_numbers; ++i)
    {
        if (this->h_atom_c[i] < 0) this->h_atom_c[i] *= -1;
    }

    controller.printf("    End reading dihedral information from AMBER file\n");

    fclose(parm);
    free(pn_type_cpu);
    free(phase_type_cpu);
    free(pk_type_cpu);

    Parameter_Host_To_Device();
    if (dihedral_type_numbers > 0) is_initialized = 1;
}

void DIHEDRAL::Memory_Allocate()
{
    Malloc_Safely((void**)&h_atom_a, sizeof(int) * dihedral_numbers);
    Malloc_Safely((void**)&h_atom_b, sizeof(int) * dihedral_numbers);
    Malloc_Safely((void**)&h_atom_c, sizeof(int) * dihedral_numbers);
    Malloc_Safely((void**)&h_atom_d, sizeof(int) * dihedral_numbers);
    Malloc_Safely((void**)&h_ipn, sizeof(int) * dihedral_numbers);
    Malloc_Safely((void**)&h_pk, sizeof(float) * dihedral_numbers);
    Malloc_Safely((void**)&h_gamc, sizeof(float) * dihedral_numbers);
    Malloc_Safely((void**)&h_gams, sizeof(float) * dihedral_numbers);
    Malloc_Safely((void**)&h_pn, sizeof(float) * dihedral_numbers);
    Malloc_Safely((void**)&h_dihedral_ene, sizeof(float) * dihedral_numbers);
    memset(h_dihedral_ene, 0, sizeof(float) * dihedral_numbers);
    Malloc_Safely((void**)&h_sigma_of_dihedral_ene, sizeof(float));
    memset(h_sigma_of_dihedral_ene, 0, sizeof(float));
}

void DIHEDRAL::Parameter_Host_To_Device()
{
    Device_Malloc_Safely((void**)&d_atom_a, sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_atom_b, sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_atom_c, sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_atom_d, sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_pk, sizeof(float) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_pn, sizeof(float) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_ipn, sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_gamc, sizeof(float) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_gams, sizeof(float) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_dihedral_ene,
                         sizeof(float) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_sigma_of_dihedral_ene, sizeof(float));

    Device_Malloc_Safely((void**)&d_atom_a_local,
                         sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_atom_b_local,
                         sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_atom_c_local,
                         sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_atom_d_local,
                         sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_pk_local, sizeof(float) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_pn_local, sizeof(float) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_ipn_local, sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_gamc_local,
                         sizeof(float) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_gams_local,
                         sizeof(float) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_num_dihe_local, sizeof(int));
    deviceMemset(d_num_dihe_local, 0, sizeof(int));

    deviceMemcpy(d_atom_a, h_atom_a, sizeof(int) * dihedral_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_atom_b, h_atom_b, sizeof(int) * dihedral_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_atom_c, h_atom_c, sizeof(int) * dihedral_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_atom_d, h_atom_d, sizeof(int) * dihedral_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_pk, h_pk, sizeof(float) * dihedral_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_pn, h_pn, sizeof(float) * dihedral_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_ipn, h_ipn, sizeof(int) * dihedral_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_gamc, h_gamc, sizeof(float) * dihedral_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_gams, h_gams, sizeof(float) * dihedral_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemset(d_dihedral_ene, 0, sizeof(float) * dihedral_numbers);
    deviceMemset(d_sigma_of_dihedral_ene, 0, sizeof(float));
}

static __global__ void get_local_device(
    int dihedral_numbers, int* d_atom_a, int* d_atom_b, int* d_atom_c,
    int* d_atom_d, char* atom_local_label, int* atom_local_id,
    int* d_atom_a_local, int* d_atom_b_local, int* d_atom_c_local,
    int* d_atom_d_local, int* d_ipn, float* d_pk, float* d_gamc, float* d_gams,
    float* d_pn, int* d_ipn_local, float* d_pk_local, float* d_gamc_local,
    float* d_gams_local, float* d_pn_local, int* d_num_dihe_local)
{
#ifdef USE_GPU
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx != 0) return;
#endif
    d_num_dihe_local[0] = 0;
    // 判断第i个dihedral中原子a是否在local中，如果是则需要在本domain中计算
    for (int i = 0; i < dihedral_numbers; i++)
    {
        if (atom_local_label[d_atom_a[i]] == 1 ||
            atom_local_label[d_atom_b[i]] == 1 ||
            atom_local_label[d_atom_c[i]] == 1 ||
            atom_local_label[d_atom_d[i]] == 1)
        {
            d_atom_a_local[d_num_dihe_local[0]] = atom_local_id[d_atom_a[i]];
            d_atom_b_local[d_num_dihe_local[0]] = atom_local_id[d_atom_b[i]];
            d_atom_c_local[d_num_dihe_local[0]] = atom_local_id[d_atom_c[i]];
            d_atom_d_local[d_num_dihe_local[0]] = atom_local_id[d_atom_d[i]];
            d_ipn_local[d_num_dihe_local[0]] = d_ipn[i];
            d_pk_local[d_num_dihe_local[0]] = d_pk[i];
            d_gamc_local[d_num_dihe_local[0]] = d_gamc[i];
            d_gams_local[d_num_dihe_local[0]] = d_gams[i];
            d_pn_local[d_num_dihe_local[0]] = d_pn[i];
            d_num_dihe_local[0]++;
        }
    }
}

void DIHEDRAL::Get_Local(int* atom_local, int local_atom_numbers,
                         int ghost_numbers, char* atom_local_label,
                         int* atom_local_id)
{
    if (!is_initialized) return;
    num_dihe_local = 0;
    this->local_atom_numbers = local_atom_numbers;
    Launch_Device_Kernel(
        get_local_device, 1, 1, 0, NULL, dihedral_numbers, d_atom_a, d_atom_b,
        d_atom_c, d_atom_d, atom_local_label, atom_local_id, d_atom_a_local,
        d_atom_b_local, d_atom_c_local, d_atom_d_local, d_ipn, d_pk, d_gamc,
        d_gams, d_pn, d_ipn_local, d_pk_local, d_gamc_local, d_gams_local,
        d_pn_local, d_num_dihe_local);
    deviceMemcpy(&num_dihe_local, d_num_dihe_local, sizeof(int),
                 deviceMemcpyDeviceToHost);
}

void DIHEDRAL::Dihedral_Force_With_Atom_Energy_And_Virial(
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell, VECTOR* frc,
    int need_atom_energy, float* atom_energy, int need_virial,
    LTMatrix3* atom_virial)
{
    if (is_initialized)  // 修改：删除MPI_rank==0判断，求和变为局部求和，加入判断是否需要计算atom_energy和virial
    {
        Launch_Device_Kernel(
            Dihedral_Force_With_Atom_Energy_And_Virial_Device,
            (dihedral_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, this->num_dihe_local, crd,
            cell, rcell, this->local_atom_numbers, this->d_atom_a_local,
            this->d_atom_b_local, this->d_atom_c_local, this->d_atom_d_local,
            this->d_ipn_local, this->d_pk_local, this->d_gamc_local,
            this->d_gams_local, this->d_pn_local, frc, need_atom_energy,
            atom_energy, d_dihedral_ene, need_virial, atom_virial);
    }
}

void DIHEDRAL::Step_Print(CONTROLLER* controller, bool print_sum)
{
    if (is_initialized && CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Sum_Of_List(d_dihedral_ene, d_sigma_of_dihedral_ene,
                    num_dihe_local);  // 修改为局部求和
        deviceMemcpy(h_sigma_of_dihedral_ene, d_sigma_of_dihedral_ene,
                     sizeof(float), deviceMemcpyDeviceToHost);
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, h_sigma_of_dihedral_ene, 1, MPI_FLOAT,
                      MPI_SUM, CONTROLLER::pp_comm);
#endif
        if (CONTROLLER::MPI_rank == 0)
        {
            controller->Step_Print(this->module_name, h_sigma_of_dihedral_ene,
                                   print_sum);
        }
    }
}
