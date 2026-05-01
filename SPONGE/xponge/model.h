#pragma once

#include <string>
#include <utility>
#include <vector>

namespace Xponge
{

struct ModelAtom
{
    std::string name;
    std::string type;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double charge = 0.0;
    double mass = 0.0;
    std::string lj_type;
};

struct ResidueType
{
    std::string name = "MOL";
    std::vector<ModelAtom> atoms;
    std::vector<std::pair<int, int>> bonds;
    std::string forcefield;
    std::string head;
    std::string tail;
};

struct Molecule
{
    std::vector<ResidueType> residues;

    std::vector<ModelAtom> Atoms() const;
    std::vector<std::pair<int, int>> Bonds(bool connect_residue_tails) const;
    std::string Forcefield() const;
};

}
