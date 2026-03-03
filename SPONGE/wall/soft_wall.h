#pragma once
#include "../common.h"
#include "../control.h"
#include "../third_party/jit/jit.hpp"

struct SOFT_WALL
{
    char module_name[CHAR_LENGTH_MAX];
    std::string source_code;
    JIT_Function force_function;
    float* item_energy;
    float *sum_energy, h_sum_energy;
    void Compile(CONTROLLER* controller);
    void Initial(int atom_numbers);
    void Compute_Force(int atom_numbers, VECTOR* crd, VECTOR* frc,
                       int need_potential, float* atom_energy);
};

struct SOFT_WALLS
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    std::vector<SOFT_WALL*> forces;
    void Initial(CONTROLLER* controller, int atom_numbers,
                 const char* module_name = NULL);
    void Compute_Force(int atom_numbers, VECTOR* crd, VECTOR* frc,
                       int need_potential, float* atom_energy);
    void Step_Print(CONTROLLER* controller);
};
