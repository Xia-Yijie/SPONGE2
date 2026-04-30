#include "mol2_writer_policy.h"

#include <stdexcept>

namespace Xponge
{
namespace Assign
{
std::string Atom_Type_For_Output(const Assignment& assignment,
                                 std::size_t atom,
                                 const std::string& atomtype)
{
    const auto& atoms = assignment.atoms();
    const auto& atom_types = assignment.atom_types();
    if (!atomtype.empty() && atomtype != "sybyl")
    {
        if (atom < atom_types.size() && !atom_types[atom].empty())
        {
            return atom_types[atom];
        }
        throw std::runtime_error("atom type has not been assigned");
    }
    return atoms[atom].element + atoms[atom].element_detail;
}

bool Is_Antechamber_Aromatic_Output_Type(const std::string& atom_type)
{
    return atom_type == "C.ar" || atom_type == "N.ar" || atom_type == "ca" ||
           atom_type == "cp" || atom_type == "pb" || atom_type == "nb";
}

namespace
{

bool Has_Marker(const Assignment& assignment,
                std::size_t atom1,
                int atom2,
                const std::string& marker)
{
    const auto& outer = assignment.bond_markers()[atom1];
    const auto inner_it = outer.find(atom2);
    return inner_it != outer.end() &&
           inner_it->second.find(marker) != inner_it->second.end();
}

bool Has_Any_Marker(const Assignment& assignment,
                    std::size_t atom1,
                    int atom2,
                    const std::string& marker1,
                    const std::string& marker2)
{
    const auto& outer = assignment.bond_markers()[atom1];
    const auto inner_it = outer.find(atom2);
    if (inner_it == outer.end())
    {
        return false;
    }
    const auto& set = inner_it->second;
    return set.find(marker1) != set.end() ||
           set.find(marker2) != set.end();
}

std::string Amber_Gaff_Antechamber_Bond_Type_For_Output(
    const Assignment& assignment,
    std::size_t atom1,
    int atom2,
    int order,
    const std::string& atomtype)
{
    const bool is_aromatic =
        Has_Any_Marker(assignment, atom1, atom2, "ar", "mol2_ar");
    const bool ar_single =
        Has_Marker(assignment, atom1, atom2, "gaff_ar_single");
    const bool ar_double =
        Has_Marker(assignment, atom1, atom2, "gaff_ar_double");

    if (is_aromatic)
    {
        if (ar_single || ar_double)
        {
            const std::string type1 =
                Atom_Type_For_Output(assignment, atom1, atomtype);
            const std::string type2 = Atom_Type_For_Output(
                assignment, static_cast<std::size_t>(atom2), atomtype);
            if (Is_Antechamber_Aromatic_Output_Type(type1) &&
                Is_Antechamber_Aromatic_Output_Type(type2))
            {
                return "ar";
            }
            return ar_double ? "2" : "1";
        }
        return "ar";
    }
    if (Has_Any_Marker(assignment, atom1, atom2, "am", "mol2_am"))
    {
        return "1";
    }
    if (order < 0)
    {
        return "un";
    }
    return std::to_string(order);
}

}  // namespace

std::string Preserve_Bond_Type_For_Output(const Assignment& assignment,
                                          std::size_t atom1,
                                          int atom2,
                                          int order)
{
    if (Has_Any_Marker(assignment, atom1, atom2, "ar", "mol2_ar"))
    {
        return "ar";
    }
    if (Has_Any_Marker(assignment, atom1, atom2, "am", "mol2_am"))
    {
        return "am";
    }
    if (order < 0)
    {
        return "un";
    }
    return std::to_string(order);
}

Mol2BondOutputPolicy Mol2_Bond_Output_Policy_From_Atom_Type(
    const std::string& atomtype)
{
    return atomtype == "gaff" ? Mol2BondOutputPolicy::AmberGaffAntechamber
                              : Mol2BondOutputPolicy::Preserve;
}

std::string Mol2_Bond_Type_For_Output(const Assignment& assignment,
                                      std::size_t atom1,
                                      int atom2,
                                      int order,
                                      const std::string& atomtype,
                                      Mol2BondOutputPolicy policy)
{
    switch (policy)
    {
    case Mol2BondOutputPolicy::Preserve:
        return Preserve_Bond_Type_For_Output(assignment, atom1, atom2, order);
    case Mol2BondOutputPolicy::AmberGaffAntechamber:
        return Amber_Gaff_Antechamber_Bond_Type_For_Output(
            assignment, atom1, atom2, order, atomtype);
    }
    throw std::runtime_error("unknown mol2 bond output policy");
}

}  // namespace Assign
}  // namespace Xponge
