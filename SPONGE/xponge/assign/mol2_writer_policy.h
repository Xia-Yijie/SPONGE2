#pragma once

#include <string>

#include "assignment.h"

namespace Xponge
{
namespace Assign
{

enum class Mol2BondOutputPolicy
{
    Preserve,
    AmberGaffAntechamber
};

std::string Atom_Type_For_Output(const Assignment& assignment,
                                 std::size_t atom,
                                 const std::string& atomtype);

bool Is_Antechamber_Aromatic_Output_Type(const std::string& atom_type);

std::string Preserve_Bond_Type_For_Output(const Assignment& assignment,
                                          std::size_t atom1,
                                          int atom2,
                                          int order);

Mol2BondOutputPolicy Mol2_Bond_Output_Policy_From_Atom_Type(
    const std::string& atomtype);

std::string Mol2_Bond_Type_For_Output(const Assignment& assignment,
                                      std::size_t atom1,
                                      int atom2,
                                      int order,
                                      const std::string& atomtype,
                                      Mol2BondOutputPolicy policy);

}
}
