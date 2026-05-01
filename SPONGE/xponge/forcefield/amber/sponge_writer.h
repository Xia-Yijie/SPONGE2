#pragma once

#include <string>
#include <vector>

#include "../../model.h"
#include "parameters.h"

namespace Xponge
{
namespace Amber
{

struct SpongeInputOptions
{
    std::string output_dir = ".";
    std::string prefix = "xponge";
    std::vector<double> box = {950.0, 950.0, 950.0, 90.0, 90.0, 90.0};
    bool connect_residue_tails = false;
    bool prefix_files = false;
    bool write_mdin = true;
    bool write_atom_metadata = false;
    double charge_scale = 1.0;
    std::string cmap_source;
};

void Save_Gaff_Sponge_Input(const Molecule& molecule,
                            const GaffParameters& parameters,
                            const SpongeInputOptions& options);

}
}
