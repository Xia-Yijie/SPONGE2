#pragma once

struct NVE_iteration
{
    MD_INFORMATION* md_info =
        NULL;  // 指向自己主结构体的指针，以方便调用主结构体的信息
    float max_velocity = -1;
    void Leap_Frog(const int atom_numbers, VECTOR* vel, VECTOR* crd,
                   VECTOR* frc, const float* inverse_mass, const float dt);
    void Initial(CONTROLLER* controller, MD_INFORMATION* md_info);
};
