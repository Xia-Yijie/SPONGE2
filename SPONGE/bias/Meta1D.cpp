#include "Meta1D.h"

static float BSpline_Interpolation_4(int i, float x)  // in fact need 1.-x
{
    if (i == -1)
    {
        return 1. / 6. * x * x * x;
    }
    else if (i == 0)
    {
        return -0.5 * x * x * x + 0.5 * x * x + 0.5 * x + 1. / 6.;
    }
    else if (i == 1)
    {
        return 0.5 * x * x * x - x * x + 2. / 3.;
    }
    else if (i == 2)
    {
        return -1. / 6. * x * x * x + 0.5 * x * x - 0.5 * x + 1. / 6.;
    }
    else
    {
        extern CONTROLLER controller;
        controller.Throw_SPONGE_Error(
            spongeErrorOverflow, "BSpline_Interpolation_4",
            "Reason:\n\tThe input of the BSpline overflowed\n");
        return 0.;
    }
}

static float DBSpline_Interpolation_4(int i,
                                      float x)  // in fact need 1.-x, and is -D
{
    if (i == -1)
    {
        return 3. / 6. * x * x;
    }
    else if (i == 0)
    {
        return -0.5 * 3. * x * x + 0.5 * 2. * x + 0.5;
    }
    else if (i == 1)
    {
        return 0.5 * 3. * x * x - 2. * x;
    }
    else if (i == 2)
    {
        return -3. / 6. * x * x + 0.5 * 2. * x - 0.5;
    }
    else
    {
        extern CONTROLLER controller;
        controller.Throw_SPONGE_Error(
            spongeErrorOverflow, "BSpline_Interpolation_4",
            "Reason:\n\tThe input of the BSpline overflowed\n");
        return 0.;
    }
}

void META1D::Initial(CONTROLLER* controller,
                     COLLECTIVE_VARIABLE_CONTROLLER* cv_controller,
                     char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "meta1d");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (!cv_controller->Command_Exist(this->module_name, "CV"))
    {
        controller->printf("META1D IS NOT INITIALIZED\n\n");
        return;
    }
    controller->printf("START INITIALIZING META1D:\n");
    cv = cv_controller->Ask_For_CV(this->module_name, 1)[0];

    sprintf(read_potential_file_name, "meta1d_potential.txt");
    sprintf(write_potential_file_name, "meta1d_potential.txt");
    if (controller->Command_Exist("default_in_file_prefix"))
    {
        sprintf(read_potential_file_name, "%s_meta1d_potential.txt",
                controller->Command("default_in_file_prefix"));
    }
    else
    {
        sprintf(read_potential_file_name, "meta1d_potential.txt");
    }
    if (controller->Command_Exist("default_out_file_prefix"))
    {
        sprintf(write_potential_file_name, "%s_meta1d_potential.txt",
                controller->Command("default_out_file_prefix"));
    }
    else
    {
        sprintf(write_potential_file_name, "meta1d_potential.txt");
    }

    cv_period = 0;
    if (cv_controller->Command_Exist(this->module_name, "CV_period"))
    {
        float* temp_value =
            cv_controller->Ask_For_Float_Parameter("meta1d", "CV_period");
        cv_period = temp_value[0];
        free(temp_value);
    }

    bool potential_loaded = false;
    if (cv_controller->Command_Exist(this->module_name, "potential_in_file"))
    {
        strcpy(read_potential_file_name,
               cv_controller
                   ->Ask_For_String_Parameter(this->module_name,
                                              "potential_in_file")[0]
                   .c_str());
        Read_Potential(controller);
        potential_loaded = true;
    }
    else
    {
        FILE* temp_file = fopen(read_potential_file_name, "r");
        if (temp_file != NULL)
        {
            fclose(temp_file);
            Read_Potential(controller);
            potential_loaded = true;
        }
    }
    if (!potential_loaded)
    {
        int check_necessary_inpu_exist = 0;
        if (cv_controller->Command_Exist(this->module_name, "CV_minimal"))
        {
            float* temp_value =
                cv_controller->Ask_For_Float_Parameter("meta1d", "CV_minimal");
            cv_min = temp_value[0];
            free(temp_value);
            check_necessary_inpu_exist += 1;
        }
        if (cv_controller->Command_Exist(this->module_name, "CV_maximum"))
        {
            float* temp_value =
                cv_controller->Ask_For_Float_Parameter("meta1d", "CV_maximum");
            cv_max = temp_value[0];
            free(temp_value);
            check_necessary_inpu_exist += 1;
        }
        dcv = 0.01;
        if (cv_controller->Command_Exist(this->module_name, "dCV"))
        {
            float* temp_value =
                cv_controller->Ask_For_Float_Parameter("meta1d", "dCV");
            dcv = temp_value[0];
            free(temp_value);
        }
        if ((float)(cv_max - cv_min) / dcv < 1.0)
        {
            check_necessary_inpu_exist = -1;
        }
        if (check_necessary_inpu_exist != 2)
        {
            char error_reason[CHAR_LENGTH_MAX];
            sprintf(
                error_reason,
                "Reason:\n\tthe required inputs are not complete or incorrect \
(potential_in_file or (CV_minimal, CV_maximum))\n");
            controller->Throw_SPONGE_Error(spongeErrorMissingCommand,
                                           "META1D::Initial", error_reason);
        }

        grid_numbers = (float)(cv_max - cv_min) / dcv + 1.;
        dcv = (float)(cv_max - cv_min) / (grid_numbers - 1);

        Malloc_Safely((void**)&potential_list, sizeof(float) * grid_numbers);
        for (int i = 0; i < grid_numbers; i = i + 1)
        {
            potential_list[i] = 0;
        }
    }
    if (cv_controller->Command_Exist(this->module_name, "height"))
    {
        float* temp_value =
            cv_controller->Ask_For_Float_Parameter("meta1d", "height");
        height = temp_value[0];
        free(temp_value);
    }
    if (cv_controller->Command_Exist(this->module_name, "sigma"))
    {
        float* temp_value =
            cv_controller->Ask_For_Float_Parameter("meta1d", "sigma");
        sigma = temp_value[0];
        free(temp_value);
    }
    if (cv_controller->Command_Exist(this->module_name, "wall_height"))
    {
        float* temp_value =
            cv_controller->Ask_For_Float_Parameter("meta1d", "wall_height");
        border_potential_height = temp_value[0];
        free(temp_value);
    }
    if (cv_controller->Command_Exist(this->module_name, "welltemp_factor"))
    {
        float* temp_value =
            cv_controller->Ask_For_Float_Parameter("meta1d", "welltemp_factor");
        welltemp_factor = temp_value[0];
        free(temp_value);
    }
    if (cv_controller->Command_Exist(this->module_name, "potential_out_file"))
    {
        strcpy(write_potential_file_name,
               cv_controller
                   ->Ask_For_String_Parameter(this->module_name,
                                              "potential_out_file")[0]
                   .c_str());
    }
    if (cv_controller->Command_Exist(this->module_name,
                                     "potential_update_interval"))
    {
        int* temp_value = cv_controller->Ask_For_Int_Parameter(
            "meta1d", "potential_update_interval");
        potential_update_interval = temp_value[0];
        free(temp_value);
    }
    else
    {
        controller->printf(
            "    Potential update interval is set to "
            "write_information_interval by default\n");
        if (controller->Command_Exist("write_information_interval"))
        {
            potential_update_interval =
                atoi(controller->Command("write_information_interval"));
        }
        else
        {
            potential_update_interval = 1000;
        }
    }
    controller->Step_Print_Initial(this->module_name, "%.2f");
    controller->printf("    potential output file: %s\n",
                       write_potential_file_name);
    is_initialized = 1;
    controller->printf("END INITIALIZING META1D\n\n");
}

void META1D::Write_Potential()
{
    if (!is_initialized)
    {
        return;
    }
    // 在最后一个MPI进程上写文件，这也是CV与BIAS模块所在的进程
    if (CONTROLLER::MPI_rank == CONTROLLER::MPI_size - 1)
    {
        FILE* temp_file = NULL;
        Open_File_Safely(&temp_file, write_potential_file_name, "w");
        fprintf(temp_file, "system name and description\n");
        fprintf(temp_file, "%f %f %f\n", cv_min, cv_max, dcv);
        fprintf(temp_file, "%d\n", grid_numbers);
        for (int i = 0; i < grid_numbers; i = i + 1)
        {
            fprintf(temp_file, "%f %f\n", (float)dcv * i + cv_min,
                    potential_list[i]);
        }
        fclose(temp_file);
    }
}

void META1D::Read_Potential(CONTROLLER* controller)
{
    FILE* temp_file = NULL;
    Open_File_Safely(&temp_file, read_potential_file_name, "r");
    char temp_char[256];
    char* get_val = fgets(temp_char, 256, temp_file);
    int scanf_ret = fscanf(temp_file, "%f %f %f\n", &cv_min, &cv_max, &dcv);
    if (scanf_ret != 3)
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "META1D::Read_Potential",
                                       "Reason:\n\tbad potential input file\n");
    }
    controller->printf(
        "    CV_minimal = %f\n    CV_maximum = %f\n    dCV = %f\n", cv_min,
        cv_max, dcv);
    scanf_ret = fscanf(temp_file, "%d\n", &grid_numbers);
    if (scanf_ret != 1)
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "META1D::Read_Potential",
                                       "Reason:\n\tbad potential input file\n");
    }
    Malloc_Safely((void**)&potential_list, sizeof(float) * grid_numbers);
    float temp_float;
    for (int i = 0; i < grid_numbers; i = i + 1)
    {
        scanf_ret =
            fscanf(temp_file, "%f %f\n", &temp_float, &potential_list[i]);
        if (scanf_ret != 2)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "META1D::Read_Potential",
                "Reason:\n\tbad potential input file\n");
        }
    }

    fclose(temp_file);
}

static __global__ void Add_Frc(const int atom_numbers, VECTOR* frc,
                               VECTOR* cv_grad, float dheight_dcv)
{
#ifdef USE_GPU
    for (int i = blockIdx.x + blockDim.x * threadIdx.x; i < atom_numbers;
         i += gridDim.x * blockDim.x)
#else
#pragma omp parallel for
    for (int i = 0; i < atom_numbers; i++)
#endif
    {
        frc[i] = frc[i] - dheight_dcv * cv_grad[i];
    }
}

static __global__ void Add_Potential(float* d_potential, const float to_add)
{
    d_potential[0] += to_add;
}

static __global__ void Add_Virial(LTMatrix3* d_virial, const float dU_dCV,
                                  const LTMatrix3* cv_virial)
{
    d_virial[0] = d_virial[0] - dU_dCV * cv_virial[0];
}

void META1D::Meta_Force_With_Energy_And_Virial(int atom_numbers, VECTOR* frc,
                                               int need_potential,
                                               int need_pressure,
                                               float* d_potential,
                                               LTMatrix3* d_virial)
{
    if (!is_initialized)
    {
        return;
    }
    float dU_dCV = -DPotential(cv->value) / dcv;
    Launch_Device_Kernel(Add_Frc, 20, 256, 0, NULL, atom_numbers, frc,
                         cv->crd_grads, dU_dCV);
    if (need_potential)
    {
        Launch_Device_Kernel(Add_Potential, 1, 1, 0, NULL, d_potential,
                             this->Potential(this->cv->value));
    }
    if (need_pressure)
    {
        Launch_Device_Kernel(Add_Virial, 1, 1, 0, NULL, d_virial, dU_dCV,
                             cv->virial);
    }
}

float META1D::Potential(float x)
{
    if (!is_initialized)
    {
        return NAN;
    }
    float temp_pos = (float)(x - cv_min) / dcv;
    int pos = (float)temp_pos;
    float scale = 1. - temp_pos + (float)pos;
    if (pos >= 1 && pos <= grid_numbers - 3)
    {
        return BSpline_Interpolation_4(-1, scale) * potential_list[pos - 1] +
               BSpline_Interpolation_4(0, scale) * potential_list[pos] +
               BSpline_Interpolation_4(1, scale) * potential_list[pos + 1] +
               BSpline_Interpolation_4(2, scale) * potential_list[pos + 2];
    }
    else if (pos == 0)
    {
        return BSpline_Interpolation_4(-1, scale) * border_potential_height +
               BSpline_Interpolation_4(0, scale) * potential_list[0] +
               BSpline_Interpolation_4(1, scale) * potential_list[1] +
               BSpline_Interpolation_4(2, scale) * potential_list[2];
    }
    else if (pos == grid_numbers - 2)
    {
        return BSpline_Interpolation_4(-1, scale) *
                   potential_list[grid_numbers - 3] +
               BSpline_Interpolation_4(0, scale) *
                   potential_list[grid_numbers - 2] +
               BSpline_Interpolation_4(1, scale) *
                   potential_list[grid_numbers - 1] +
               BSpline_Interpolation_4(2, scale) * border_potential_height;
    }
    else
    {
        extern CONTROLLER controller;
        char error_reason[CHAR_LENGTH_MAX];
        sprintf(error_reason,
                "Reason:\n\tInput of x=%f out of range (%f, %f)\n", x, cv_min,
                cv_max);
        controller.Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                      "META1D::Potential", error_reason);
        return 0.;
    }
}

float META1D::DPotential(float x)
{
    if (!is_initialized)
    {
        return NAN;
    }
    float temp_pos = (float)(x - cv_min) / dcv;
    int pos = (float)temp_pos;
    float scale = 1. - temp_pos + (float)pos;
    if (pos >= 1 && pos <= grid_numbers - 3)
    {
        return DBSpline_Interpolation_4(-1, scale) * potential_list[pos - 1] +
               DBSpline_Interpolation_4(0, scale) * potential_list[pos] +
               DBSpline_Interpolation_4(1, scale) * potential_list[pos + 1] +
               DBSpline_Interpolation_4(2, scale) * potential_list[pos + 2];
    }
    else if (pos == 0)
    {
        return DBSpline_Interpolation_4(-1, scale) * border_potential_height +
               DBSpline_Interpolation_4(0, scale) * potential_list[0] +
               DBSpline_Interpolation_4(1, scale) * potential_list[1] +
               DBSpline_Interpolation_4(2, scale) * potential_list[2];
    }
    else if (pos == grid_numbers - 2)
    {
        return DBSpline_Interpolation_4(-1, scale) *
                   potential_list[grid_numbers - 3] +
               DBSpline_Interpolation_4(0, scale) *
                   potential_list[grid_numbers - 2] +
               DBSpline_Interpolation_4(1, scale) *
                   potential_list[grid_numbers - 1] +
               DBSpline_Interpolation_4(2, scale) * border_potential_height;
    }
    else
    {
        extern CONTROLLER controller;
        char error_reason[CHAR_LENGTH_MAX];
        sprintf(error_reason,
                "Reason:\n\tInput of x=%f out of range (%f, %f)\n", x, cv_min,
                cv_max);
        controller.Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                      "META1D::DPotential", error_reason);
        return 0.;
    }
}

void META1D::AddPotential(int steps)
{
    if (!is_initialized)
    {
        return;
    }
    if (steps % potential_update_interval == 0)
    {
        for (int i = 0; i < grid_numbers; i = i + 1)
        {
            float pos = (float)i * dcv + cv_min;
            float delta_cv = pos - cv->value;
            if (cv_period > 0)
            {
                delta_cv -= floorf(delta_cv / cv_period + 0.5) * cv_period;
            }
            float add = height * 1. / sqrtf(2. * 3.141592654) / sigma *
                        expf(-delta_cv * delta_cv / 2. / sigma / sigma);
            potential_list[i] =
                potential_list[i] +
                add * expf(-potential_list[i] / welltemp_factor);
        }
    }
}

void META1D::Do_Metadynamics(int atom_numbers, VECTOR* crd, LTMatrix3 cell,
                             LTMatrix3 rcell, int step, int need_potential,
                             int need_pressure, VECTOR* frc, float* d_potential,
                             LTMatrix3* d_virial)
{
    if (this->is_initialized)
    {
        int need = CV_NEED_CPU_VALUE | CV_NEED_CRD_GRADS;
        if (need_pressure)
        {
            need |= CV_NEED_VIRIAL;
        }
        this->cv->Compute(atom_numbers, crd, cell, rcell, need, step);
        this->Meta_Force_With_Energy_And_Virial(atom_numbers, frc,
                                                need_potential, need_pressure,
                                                d_potential, d_virial);
        this->AddPotential(step);
    }
}

void META1D::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized)
    {
        return;
    }
    float meta_ene = 0.0f;
    if (CONTROLLER::MPI_size == 1 && CONTROLLER::PM_MPI_size == 1)
    {
        meta_ene = this->Potential(this->cv->value);
        controller->Step_Print(this->module_name, meta_ene);
        return;
    }
#ifdef USE_MPI
    if (CONTROLLER::MPI_rank == CONTROLLER::MPI_size - 1)
    {
        meta_ene = this->Potential(this->cv->value);
        MPI_Send(&meta_ene, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    if (CONTROLLER::MPI_rank == 0)
    {
        MPI_Recv(&meta_ene, 1, MPI_FLOAT, CONTROLLER::MPI_size - 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        controller->Step_Print(this->module_name, meta_ene);
    }
#endif
}
