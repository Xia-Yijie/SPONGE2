#pragma once

#include <iosfwd>
#include <string>

#include "assignment.h"

namespace Xponge
{
namespace Assign
{

Assignment Get_Assignment_From_Mol2(const std::string& filename);
Assignment Get_Assignment_From_Mol2(std::istream& input,
                                    const std::string& source_name);

void Save_Assignment_As_Mol2(const Assignment& assignment,
                             const std::string& filename,
                             const std::string& atomtype = "sybyl");
void Save_Assignment_As_Mol2(const Assignment& assignment,
                             std::ostream& output,
                             const std::string& atomtype = "sybyl");

}
}
