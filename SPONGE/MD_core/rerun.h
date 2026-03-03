#pragma once

struct RERUN_information
{
    MD_INFORMATION* md_info =
        NULL;  // 指向自己主结构体的指针，以方便调用主结构体的信息
    FILE* traj_file = NULL;
    FILE* box_file = NULL;
    FILE* vel_file = NULL;
    LTMatrix3 g;  // 盒子变化的速度
    int need_box_update = 0;
    int start_frame = 0;
    int strip_frame = 0;
    void Initial(CONTROLLER* controller, MD_INFORMATION* md_info);
    bool Iteration(int strip = -1);
};
