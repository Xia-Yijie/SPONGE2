#pragma once

struct update_group_information
{
    int ug_numbers = 0;                        // 更新组的数目
    CONECT connectivity;                       // 更新组的边
    ATOM_GROUP* ug = NULL;                     // 更新组的指针数组
    ATOM_GROUP* d_ug = NULL;                   // 更新组的指针数组（设备端）
    void Initial_Edge(int atom_numbers);       // 初始化更新组的边
    void Read_Update_Group(int atom_numbers);  // 从边转化为组，并上传至设备端
    void Copy_UG_To_Device();                  // 将更新组的指针数组上传至设备端
};  // 更新组信息
