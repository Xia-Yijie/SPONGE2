#pragma once

void MD_INFORMATION::non_bond_information::Initial(CONTROLLER* controller,
                                                   MD_INFORMATION* md_info)
{
    if (controller[0].Command_Exist("skin"))
    {
        controller->Check_Float(
            "skin", "MD_INFORMATION::non_bond_information::Initial");
        skin = atof(controller[0].Command("skin"));
    }
    else
    {
        skin = 2.0;
    }
    controller->printf("    skin set to %.2f Angstrom\n", skin);

    if (controller[0].Command_Exist("cutoff"))
    {
        controller->Check_Float(
            "cutoff", "MD_INFORMATION::non_bond_information::Initial");
        cutoff = atof(controller[0].Command("cutoff"));
    }
    else
    {
        cutoff = 10.0;
    }
    controller->printf("    cutoff set to %.2f Angstrom\n", cutoff);
    /*===========================
    读取排除表相关信息
    ============================*/
    if (controller[0].Command_Exist("exclude_in_file"))
    {
        FILE* fp = NULL;
        controller->printf("    Start reading excluded list:\n");
        Open_File_Safely(&fp, controller[0].Command("exclude_in_file"), "r");

        int atom_numbers = 0;
        int scanf_ret =
            fscanf(fp, "%d %d", &atom_numbers, &excluded_atom_numbers);
        if (scanf_ret != 2)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "MD_INFORMATION::non_bond_information::Initial",
                "The format of exclude_in_file is not right\n");
        }
        if (md_info->atom_numbers > 0 && md_info->atom_numbers != atom_numbers)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorConflictingCommand,
                "MD_INFORMATION::non_bond_information::Initial",
                ATOM_NUMBERS_DISMATCH);
        }
        else if (md_info->atom_numbers == 0)
        {
            md_info->atom_numbers = atom_numbers;
        }
        controller->printf("        excluded list total length is %d\n",
                           excluded_atom_numbers);

        Malloc_Safely((void**)&h_excluded_list_start,
                      sizeof(int) * atom_numbers);
        Malloc_Safely((void**)&h_excluded_numbers, sizeof(int) * atom_numbers);
        Malloc_Safely((void**)&h_excluded_list,
                      sizeof(int) * excluded_atom_numbers * 2);
        int count = 0;
        for (int i = 0; i < atom_numbers; i++)
        {
            scanf_ret = fscanf(fp, "%d", &h_excluded_numbers[i]);
            if (scanf_ret != 1)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat,
                    "MD_INFORMATION::non_bond_information::Initial",
                    "Reason:\n\tThe format of exclude_in_file is not right\n");
            }
            h_excluded_list_start[i] = count;
            for (int j = 0; j < h_excluded_numbers[i]; j++)
            {
                scanf_ret = fscanf(fp, "%d", &h_excluded_list[count]);
                if (scanf_ret != 1)
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorBadFileFormat,
                        "MD_INFORMATION::non_bond_information::Initial",
                        "Reason:\n\tThe format of exclude_in_file is not "
                        "right\n");
                }
                count++;
            }
        }
        if (count != excluded_atom_numbers)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "MD_INFORMATION::non_bond_information::Initial",
                "Reason:\n\tThe format of exclude_in_file is not right "
                "(excluded_atom_numbers is not right)\n");
        }
        Device_Malloc_And_Copy_Safely((void**)&d_excluded_list_start,
                                      h_excluded_list_start,
                                      sizeof(int) * atom_numbers);
        Device_Malloc_And_Copy_Safely((void**)&d_excluded_numbers,
                                      h_excluded_numbers,
                                      sizeof(int) * atom_numbers);
        Device_Malloc_And_Copy_Safely((void**)&d_excluded_list, h_excluded_list,
                                      sizeof(int) * excluded_atom_numbers * 2);
        controller->printf("    End reading excluded list\n\n");
        fclose(fp);
    }
    else if (controller[0].Command_Exist("amber_parm7"))
    {
        /*===========================
        从parm中读取排除表相关信息
        ============================*/
        FILE* parm = NULL;
        Open_File_Safely(&parm, controller[0].Command("amber_parm7"), "r");
        controller->printf(
            "    Start reading excluded list from AMBER file:\n");
        while (true)
        {
            char temps[CHAR_LENGTH_MAX];
            char temp_first_str[CHAR_LENGTH_MAX];
            char temp_second_str[CHAR_LENGTH_MAX];
            if (!fgets(temps, CHAR_LENGTH_MAX, parm))
            {
                break;
            }
            if (sscanf(temps, "%s %s", temp_first_str, temp_second_str) != 2)
            {
                continue;
            }
            if (strcmp(temp_first_str, "%FLAG") == 0 &&
                strcmp(temp_second_str, "POINTERS") == 0)
            {
                char* get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);

                int atom_numbers = 0;
                int scanf_ret = fscanf(parm, "%d\n", &atom_numbers);
                if (scanf_ret != 1)
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorBadFileFormat,
                        "MD_INFORMATION::non_bond_information::Initial",
                        "The format of amber_parm7 is not right\n");
                }
                if (md_info->atom_numbers > 0 &&
                    md_info->atom_numbers != atom_numbers)
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorConflictingCommand,
                        "MD_INFORMATION::non_bond_information::Initial",
                        ATOM_NUMBERS_DISMATCH);
                }
                else if (md_info->atom_numbers == 0)
                {
                    md_info->atom_numbers = atom_numbers;
                }
                Malloc_Safely((void**)&h_excluded_list_start,
                              sizeof(int) * atom_numbers);
                Malloc_Safely((void**)&h_excluded_numbers,
                              sizeof(int) * atom_numbers);
                for (int i = 0; i < 9; i = i + 1)
                {
                    scanf_ret = fscanf(parm, "%d\n", &excluded_atom_numbers);
                    if (scanf_ret != 1)
                    {
                        controller->Throw_SPONGE_Error(
                            spongeErrorBadFileFormat,
                            "MD_INFORMATION::non_bond_information::Initial",
                            "The format of amber_parm7 is not right\n");
                    }
                }
                scanf_ret = fscanf(parm, "%d\n", &excluded_atom_numbers);
                if (scanf_ret != 1)
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorBadFileFormat,
                        "MD_INFORMATION::non_bond_information::Initial",
                        "The format of amber_parm7 is not right\n");
                }
                controller->printf("        excluded list total length is %d\n",
                                   excluded_atom_numbers);
                Malloc_Safely((void**)&h_excluded_list,
                              sizeof(int) * excluded_atom_numbers * 2);
            }

            // read atom_excluded_number for every atom
            if (strcmp(temp_first_str, "%FLAG") == 0 &&
                strcmp(temp_second_str, "NUMBER_EXCLUDED_ATOMS") == 0)
            {
                char* get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
                for (int i = 0; i < md_info->atom_numbers; i = i + 1)
                {
                    int scanf_ret =
                        fscanf(parm, "%d\n", &h_excluded_numbers[i]);
                    if (scanf_ret != 1)
                    {
                        controller->Throw_SPONGE_Error(
                            spongeErrorBadFileFormat,
                            "MD_INFORMATION::non_bond_information::Initial",
                            "The format of amber_parm7 is not right\n");
                    }
                }
            }
            // read every atom's excluded atom list
            if (strcmp(temp_first_str, "%FLAG") == 0 &&
                strcmp(temp_second_str, "EXCLUDED_ATOMS_LIST") == 0)
            {
                int count = 0;
                int lin = 0;
                char* get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
                for (int i = 0; i < md_info->atom_numbers; i = i + 1)
                {
                    h_excluded_list_start[i] = count;
                    for (int j = 0; j < h_excluded_numbers[i]; j = j + 1)
                    {
                        int scanf_ret = fscanf(parm, "%d\n", &lin);
                        if (scanf_ret != 1)
                        {
                            controller->Throw_SPONGE_Error(
                                spongeErrorBadFileFormat,
                                "MD_INFORMATION::non_bond_information::Initial",
                                "The format of amber_parm7 is not right\n");
                        }
                        if (lin == 0)
                        {
                            h_excluded_numbers[i] = 0;
                            break;
                        }
                        else
                        {
                            h_excluded_list[count] = lin - 1;
                            count = count + 1;
                        }
                    }
                    if (h_excluded_numbers[i] > 0)
                        qsort(h_excluded_list + h_excluded_list_start[i],
                              h_excluded_numbers[i], sizeof(int),
                              [](const void* a, const void* b) -> int
                              { return *((int*)a) - *((int*)b); });
                }
            }
        }

        Device_Malloc_And_Copy_Safely((void**)&d_excluded_list_start,
                                      h_excluded_list_start,
                                      sizeof(int) * md_info->atom_numbers);
        Device_Malloc_And_Copy_Safely((void**)&d_excluded_numbers,
                                      h_excluded_numbers,
                                      sizeof(int) * md_info->atom_numbers);
        Device_Malloc_And_Copy_Safely((void**)&d_excluded_list, h_excluded_list,
                                      sizeof(int) * excluded_atom_numbers * 2);
        controller->printf("    End reading excluded list from AMBER file\n\n");
        fclose(parm);
    }
    else
    {
        int atom_numbers = md_info->atom_numbers;
        excluded_atom_numbers = 0;
        controller->printf("    Set all atom exclude no atoms as default\n");

        Malloc_Safely((void**)&h_excluded_list_start,
                      sizeof(int) * atom_numbers);
        Malloc_Safely((void**)&h_excluded_numbers, sizeof(int) * atom_numbers);
        Malloc_Safely((void**)&h_excluded_list,
                      sizeof(int) * excluded_atom_numbers * 2);

        int count = 0;
        for (int i = 0; i < atom_numbers; i++)
        {
            h_excluded_numbers[i] = 0;
            h_excluded_list_start[i] = count;
            for (int j = 0; j < h_excluded_numbers[i]; j++)
            {
                h_excluded_list[count] = 0;
                count++;
            }
        }
        Device_Malloc_And_Copy_Safely((void**)&d_excluded_list_start,
                                      h_excluded_list_start,
                                      sizeof(int) * md_info->atom_numbers);
        Device_Malloc_And_Copy_Safely((void**)&d_excluded_numbers,
                                      h_excluded_numbers,
                                      sizeof(int) * md_info->atom_numbers);
        Device_Malloc_And_Copy_Safely((void**)&d_excluded_list, h_excluded_list,
                                      sizeof(int) * excluded_atom_numbers * 2);
    }
}

void MD_INFORMATION::non_bond_information::Excluded_List_Reform(
    int atom_numbers)
{
    std::map<int, std::vector<int>> temp;
    int *new_list, *new_list_tail;
    Malloc_Safely((void**)&new_list, sizeof(int) * excluded_atom_numbers * 2);
    excluded_atom_numbers = 0;
    new_list_tail = new_list;
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
    {
        std::vector<int> initializer;
        temp[atom_i] = initializer;
    }
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
    {
        int* excluded_list_j = h_excluded_list + h_excluded_list_start[atom_i];
        for (int atom_j : temp[atom_i])
        {
            new_list_tail[0] = atom_j;
            new_list_tail += 1;
        }
        for (int j = 0; j < h_excluded_numbers[atom_i]; j++)
        {
            int atom_j = excluded_list_j[j];
            new_list_tail[0] = atom_j;
            new_list_tail += 1;
            temp[atom_j].push_back(atom_i);
        }
        h_excluded_numbers[atom_i] += temp[atom_i].size();

        h_excluded_list_start[atom_i] = excluded_atom_numbers;

        excluded_atom_numbers += h_excluded_numbers[atom_i];
    }
    // 释放host上的旧排除表
    free(h_excluded_list);
    h_excluded_list = new_list;

#ifndef USE_CPU
    // 释放device上的旧排除表
    if (d_excluded_list != NULL) deviceFree(d_excluded_list);
    if (d_excluded_list_start != NULL) deviceFree(d_excluded_list_start);
    if (d_excluded_numbers != NULL) deviceFree(d_excluded_numbers);
#endif
    // 重新分配device上的排除表
    Device_Malloc_And_Copy_Safely((void**)&d_excluded_list, h_excluded_list,
                                  sizeof(int) * excluded_atom_numbers * 2);
    // 重新分配device上的排除表起点
    Device_Malloc_And_Copy_Safely((void**)&d_excluded_list_start,
                                  h_excluded_list_start,
                                  sizeof(int) * atom_numbers);
    // 重新分配device上的排除表每个原子排除数
    Device_Malloc_And_Copy_Safely((void**)&d_excluded_numbers,
                                  h_excluded_numbers,
                                  sizeof(int) * atom_numbers);
}