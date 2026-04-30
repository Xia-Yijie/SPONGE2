#include "mol2_reader.h"

#include <fstream>
#include <iomanip>
#include <cstdio>
#include <sstream>
#include <stdexcept>

#include "mol2_writer_policy.h"

namespace Xponge
{
namespace Assign
{
namespace
{

std::string Trim(const std::string& value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
    {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

}  // namespace

Assignment Get_Assignment_From_Mol2(const std::string& filename)
{
    std::ifstream input(filename);
    if (!input)
    {
        throw std::runtime_error("failed to open mol2 file: " + filename);
    }
    return Get_Assignment_From_Mol2(input, filename);
}

Assignment Get_Assignment_From_Mol2(std::istream& input,
                                    const std::string& source_name)
{
    Assignment assignment;
    bool has_molecule = false;
    std::string section;
    std::string line;

    while (std::getline(input, line))
    {
        const std::string stripped = Trim(line);
        if (stripped.empty())
        {
            continue;
        }
        if (stripped.rfind("@<TRIPOS>", 0) == 0)
        {
            section = stripped.substr(9);
            continue;
        }
        if (section == "MOLECULE")
        {
            if (!has_molecule)
            {
                assignment.set_name(stripped);
                has_molecule = true;
            }
            continue;
        }
        if (section == "ATOM")
        {
            std::istringstream iss(stripped);
            int atom_id = 0;
            std::string atom_name;
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            std::string atom_type;
            int residue_id = 0;
            std::string residue_name;
            double charge = 0.0;
            if (!(iss >> atom_id >> atom_name >> x >> y >> z >> atom_type >>
                  residue_id >> residue_name >> charge))
            {
                throw std::runtime_error("invalid mol2 atom line in " +
                                         source_name + ": " + stripped);
            }
            assignment.Add_Atom(atom_type, x, y, z, atom_name, charge);
            continue;
        }
        if (section == "UNITY_ATOM_ATTR")
        {
            std::istringstream iss(stripped);
            int atom = 0;
            int attr_count = 0;
            if (!(iss >> atom >> attr_count))
            {
                throw std::runtime_error("invalid UNITY_ATOM_ATTR line in " +
                                         source_name + ": " + stripped);
            }
            for (int i = 0; i < attr_count; ++i)
            {
                if (!std::getline(input, line))
                {
                    throw std::runtime_error(
                        "unexpected EOF in UNITY_ATOM_ATTR block");
                }
                std::istringstream attr_line(Trim(line));
                std::string attr;
                int value = 0;
                attr_line >> attr >> value;
                if (attr != "charge")
                {
                    throw std::runtime_error("unsupported UNITY_ATOM_ATTR: " +
                                             attr);
                }
                assignment.atoms().at(static_cast<std::size_t>(atom - 1))
                    .formal_charge = value;
            }
            continue;
        }
        if (section == "BOND")
        {
            std::istringstream iss(stripped);
            int bond_id = 0;
            int atom1 = 0;
            int atom2 = 0;
            std::string bond_type;
            if (!(iss >> bond_id >> atom1 >> atom2 >> bond_type))
            {
                throw std::runtime_error("invalid mol2 bond line in " +
                                         source_name + ": " + stripped);
            }

            if (bond_type.size() == 1 && bond_type[0] >= '1' &&
                bond_type[0] <= '9')
            {
                assignment.Add_Bond(atom1 - 1, atom2 - 1, bond_type[0] - '0');
            }
            else if (bond_type == "ar")
            {
                assignment.Add_Bond(atom1 - 1, atom2 - 1, -1);
                assignment.Add_Bond_Marker(atom1 - 1, atom2 - 1, "mol2_ar");
            }
            else if (bond_type == "am")
            {
                assignment.Add_Bond(atom1 - 1, atom2 - 1, -1);
                assignment.Add_Bond_Marker(atom1 - 1, atom2 - 1, "mol2_am");
            }
            else if (bond_type == "un")
            {
                assignment.Add_Bond(atom1 - 1, atom2 - 1, -1);
            }
            else
            {
                throw std::runtime_error("unsupported mol2 bond type: " +
                                         bond_type);
            }
        }
    }

    if (!has_molecule)
    {
        throw std::runtime_error("file is not a mol2 file: " + source_name);
    }
    return assignment;
}

void Save_Assignment_As_Mol2(const Assignment& assignment,
                             const std::string& filename,
                             const std::string& atomtype)
{
    std::ofstream output(filename);
    if (!output)
    {
        throw std::runtime_error("failed to open mol2 output file: " +
                                 filename);
    }
    Save_Assignment_As_Mol2(assignment, output, atomtype);
}

void Save_Assignment_As_Mol2(const Assignment& assignment,
                             std::ostream& output,
                             const std::string& atomtype)
{
    std::vector<std::string> bond_lines;
    const auto& bonds = assignment.bonds();
    const Mol2BondOutputPolicy bond_policy =
        Mol2_Bond_Output_Policy_From_Atom_Type(atomtype);

    for (const auto& bond : assignment.bond_order())
    {
        const int atom1 = bond.first;
        const int atom2 = bond.second;
        if (atom1 < 0 || atom2 < 0 ||
            static_cast<std::size_t>(atom1) >= bonds.size() ||
            bonds[atom1].find(atom2) == bonds[atom1].end())
        {
            continue;
        }

        const std::string bond_type = Mol2_Bond_Type_For_Output(
            assignment, static_cast<std::size_t>(atom1), atom2,
            bonds[atom1].at(atom2), atomtype, bond_policy);
        char bond_buffer[128];
        std::snprintf(bond_buffer, sizeof(bond_buffer), "%6d%6d %-4s\n",
                      atom1 + 1, atom2 + 1, bond_type.c_str());
        bond_lines.emplace_back(bond_buffer);
    }

    output << "@<TRIPOS>MOLECULE\n";
    output << assignment.name() << "\n";
    char line_buffer[512];
    std::snprintf(line_buffer, sizeof(line_buffer), "%5zu%6zu%6d%6d%6d\n",
                  assignment.atom_numbers(), bond_lines.size(), 1, 0, 0);
    output << line_buffer;
    output << "SMALL\ndc\n\n\n";
    output << "@<TRIPOS>ATOM\n";

    const auto& atoms = assignment.atoms();
    for (std::size_t i = 0; i < atoms.size(); ++i)
    {
        const auto& atom = atoms[i];
        std::snprintf(
            line_buffer, sizeof(line_buffer),
            "%7zu %-8s %10.4f %10.4f %10.4f %-2s %9d %-4s %13.6f\n",
            i + 1, (atom.name.empty() ? atom.element : atom.name).c_str(),
            atom.coordinate.x, atom.coordinate.y, atom.coordinate.z,
            Atom_Type_For_Output(assignment, i, atomtype).c_str(), 1,
            assignment.name().c_str(), atom.charge);
        output << line_buffer;
    }

    output << "@<TRIPOS>BOND\n";
    for (std::size_t i = 0; i < bond_lines.size(); ++i)
    {
        output << std::setw(6) << (i + 1) << bond_lines[i];
    }
    output << "@<TRIPOS>SUBSTRUCTURE\n";
    std::snprintf(line_buffer, sizeof(line_buffer),
                  "%6d %-3s %9d TEMP              0 ****  ****    0 ROOT\n",
                  1, assignment.name().c_str(), 1);
    output << line_buffer;
}

}  // namespace Assign
}  // namespace Xponge
