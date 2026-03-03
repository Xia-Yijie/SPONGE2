#pragma once

void MD_INFORMATION::RERUN_information::Initial(CONTROLLER* controller,
                                                MD_INFORMATION* md_info)
{
    this->md_info = md_info;
    if (md_info->mode == RERUN)
    {
        controller->printf("    Start initializing rerun:\n");
        std::string filename;
        if (!controller->Command_Exist(TRAJ_COMMAND))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorMissingCommand,
                "MD_INFORMATION::RERUN_information::Initial",
                "Reason:\n\tno trajectory information found (command 'crd' "
                "required)");
        }
        filename = controller->Command(TRAJ_COMMAND);
        if (controller->MPI_rank == 0)
        {
            Open_File_Safely(&traj_file, filename.c_str(), "rb");

            controller->printf(
                "        Open rerun coordinate trajectory '%s'\n",
                filename.c_str());
            controller->Set_File_Buffer(traj_file,
                                        sizeof(VECTOR) * md_info->atom_numbers);
            if (!controller->Command_Exist(BOX_TRAJ_COMMAND))
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorMissingCommand,
                    "MD_INFORMATION::RERUN_information::Initial",
                    "Reason:\n\tno box information found (command 'box' "
                    "required)");
            }
            filename = controller->Command(BOX_TRAJ_COMMAND);

            Open_File_Safely(&box_file, filename.c_str(), "r");

            controller->printf("        Open rerun box trajectory '%s'\n",
                               filename.c_str());
            controller->Set_File_Buffer(box_file, sizeof(char) * 50);
            vel_file = NULL;
            if (controller->Command_Exist(VEL_TRAJ_COMMAND))
            {
                filename = controller->Command(VEL_TRAJ_COMMAND);

                Open_File_Safely(&vel_file, filename.c_str(), "r");

                controller->printf(
                    "        Open rerun velocity trajectory '%s'\n",
                    filename.c_str());
                controller->Set_File_Buffer(
                    vel_file, sizeof(VECTOR) * md_info->atom_numbers);
            }
        }
        start_frame = 0;
        if (controller->Command_Exist("rerun_start"))
        {
            controller->Check_Float(
                "rerun_start", "MD_INFORMATION::RERUN_information::Initial");
            start_frame = atoi(controller->Command("rerun_start"));
        }
        strip_frame = 0;
        if (controller->Command_Exist("rerun_strip"))
        {
            controller->Check_Float(
                "rerun_strip", "MD_INFORMATION::RERUN_information::Initial");
            strip_frame = atoi(controller->Command("rerun_strip"));
        }
        need_box_update = 0;
        if (controller->Command_Exist("rerun_need_box_update"))
        {
            controller->Check_Float(
                "rerun_need_box_update",
                "MD_INFORMATION::RERUN_information::Initial");
            need_box_update =
                atoi(controller->Command("rerun_need_box_update"));
        }
        md_info->sys.step_limit = INT_MAX;
        controller->printf("    End initializing rerun\n\n");
    }
}

bool MD_INFORMATION::RERUN_information::Iteration(int strip)
{
    int n, nvel;
    int scanf_box;
    if (strip < 0)
    {
        strip = this->strip_frame;
    }
    VECTOR old_box_length = md_info->sys.box_length;
    VECTOR old_box_angle = md_info->sys.box_angle;
    if (CONTROLLER::MPI_rank == 0)
    {
        for (int i = 0; i < strip; i++)
        {
            n = fread(this->md_info->coordinate, sizeof(VECTOR),
                      this->md_info->atom_numbers, traj_file);
            scanf_box =
                fscanf(box_file, "%f %f %f %f %f %f",
                       &md_info->sys.box_length.x, &md_info->sys.box_length.y,
                       &md_info->sys.box_length.z, &md_info->sys.box_angle.x,
                       &md_info->sys.box_angle.y, &md_info->sys.box_angle.z);
            if (vel_file != NULL)
            {
                nvel = fread(this->md_info->velocity, sizeof(VECTOR),
                             this->md_info->atom_numbers, vel_file);
            }
            md_info->sys.steps += 1;
        }
        n = fread(this->md_info->coordinate, sizeof(VECTOR),
                  this->md_info->atom_numbers, traj_file);
        scanf_box =
            fscanf(box_file, "%f %f %f %f %f %f", &md_info->sys.box_length.x,
                   &md_info->sys.box_length.y, &md_info->sys.box_length.z,
                   &md_info->sys.box_angle.x, &md_info->sys.box_angle.y,
                   &md_info->sys.box_angle.z);
        nvel = this->md_info->atom_numbers;
        if (vel_file != NULL)
        {
            nvel = fread(this->md_info->velocity, sizeof(VECTOR),
                         this->md_info->atom_numbers, vel_file);
        }
        if (n != this->md_info->atom_numbers || scanf_box != 6 ||
            nvel != this->md_info->atom_numbers)
        {
            md_info->sys.step_limit = md_info->sys.steps;
            // close files
            if (traj_file != NULL)
            {
                fclose(traj_file);
                traj_file = NULL;
            }
            if (box_file != NULL)
            {
                fclose(box_file);
                box_file = NULL;
            }
            if (vel_file != NULL)
            {
                fclose(vel_file);
                vel_file = NULL;
            }
        }
    }
#ifdef USE_MPI
    MPI_Bcast(&md_info->sys.step_limit, sizeof(int), MPI_BYTE, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(this->md_info->coordinate,
              sizeof(VECTOR) * this->md_info->atom_numbers, MPI_BYTE, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(this->md_info->velocity,
              sizeof(VECTOR) * this->md_info->atom_numbers, MPI_BYTE, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(&md_info->sys.box_length, sizeof(VECTOR), MPI_BYTE, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(&md_info->sys.box_angle, sizeof(VECTOR), MPI_BYTE, 0,
              MPI_COMM_WORLD);
#endif
    bool box_changed =
        fabsf(old_box_length.x - md_info->sys.box_length.x) > 1e-4 ||
        fabsf(old_box_length.y - md_info->sys.box_length.y) > 1e-4 ||
        fabsf(old_box_length.z - md_info->sys.box_length.z) > 1e-4 ||
        fabsf(old_box_angle.x - md_info->sys.box_angle.x) > 1e-4 ||
        fabsf(old_box_angle.y - md_info->sys.box_angle.y) > 1e-4 ||
        fabsf(old_box_angle.z - md_info->sys.box_angle.z) > 1e-4;
    if (box_changed && md_info->pbc.pbc)
    {
        // 通过盒子变化计算g
        LTMatrix3 new_cell = md_info->pbc.Get_Cell(md_info->sys.box_length,
                                                   md_info->sys.box_angle);
        g = (1 / md_info->dt) *
            (new_cell * inv(md_info->pbc.cell) - LTMatrix3(1, 0, 1, 0, 0, 1));
    }
    deviceMemcpy(this->md_info->crd, this->md_info->coordinate,
                 sizeof(VECTOR) * this->md_info->atom_numbers,
                 deviceMemcpyHostToDevice);
    if (vel_file != NULL)
        deviceMemcpy(this->md_info->vel, this->md_info->velocity,
                     sizeof(VECTOR) * this->md_info->atom_numbers,
                     deviceMemcpyHostToDevice);
    return box_changed;
}
