#pragma once

#include <string>

namespace Xponge
{
namespace Amber
{

struct Parmchk2Options
{
    int ffset = 1;
    bool print_all = false;
    bool print_dihedral_contain_X = true;
    std::string datapath;
};

void Generate_Gaff_Frcmod(const std::string& input_mol2,
                          const std::string& output_frcmod,
                          const Parmchk2Options& options);

}  // namespace Amber
}  // namespace Xponge
