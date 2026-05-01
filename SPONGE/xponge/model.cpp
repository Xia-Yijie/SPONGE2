#include "model.h"

#include <algorithm>
#include <stdexcept>

namespace Xponge
{

std::vector<ModelAtom> Molecule::Atoms() const
{
    std::vector<ModelAtom> atoms;
    for (const auto& residue : residues)
    {
        atoms.insert(atoms.end(), residue.atoms.begin(), residue.atoms.end());
    }
    return atoms;
}

namespace
{

int Residue_Atom_Index(const ResidueType& residue, const std::string& name)
{
    for (std::size_t i = 0; i < residue.atoms.size(); ++i)
    {
        if (residue.atoms[i].name == name)
        {
            return static_cast<int>(i);
        }
    }
    throw std::runtime_error("missing atom " + name + " in residue " +
                             residue.name);
}

}  // namespace

std::vector<std::pair<int, int>> Molecule::Bonds(
    bool connect_residue_tails) const
{
    std::vector<std::pair<int, int>> bonds;
    std::vector<int> offsets;
    int offset = 0;
    for (const auto& residue : residues)
    {
        offsets.push_back(offset);
        for (const auto& bond : residue.bonds)
        {
            int i = bond.first + offset;
            int j = bond.second + offset;
            if (i > j)
            {
                std::swap(i, j);
            }
            bonds.push_back({i, j});
        }
        offset += static_cast<int>(residue.atoms.size());
    }
    if (connect_residue_tails)
    {
        for (std::size_t i = 0; i + 1 < residues.size(); ++i)
        {
            const auto& left = residues[i];
            const auto& right = residues[i + 1];
            if (left.tail.empty() || right.head.empty())
            {
                continue;
            }
            int a = offsets[i] + Residue_Atom_Index(left, left.tail);
            int b = offsets[i + 1] + Residue_Atom_Index(right, right.head);
            if (a > b)
            {
                std::swap(a, b);
            }
            bonds.push_back({a, b});
        }
    }
    std::sort(bonds.begin(), bonds.end());
    bonds.erase(std::unique(bonds.begin(), bonds.end()), bonds.end());
    return bonds;
}

std::string Molecule::Forcefield() const
{
    std::string forcefield;
    for (const auto& residue : residues)
    {
        if (residue.forcefield.empty())
        {
            continue;
        }
        if (forcefield.empty())
        {
            forcefield = residue.forcefield;
        }
        else if (forcefield != residue.forcefield)
        {
            throw std::runtime_error(
                "molecule mixes residue types from multiple force fields");
        }
    }
    return forcefield;
}

}  // namespace Xponge
