#include "assignment.h"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <utility>

namespace Xponge
{
namespace Assign
{

Assignment::Assignment(std::string name) : name_(std::move(name)) {}

const std::string& Assignment::name() const noexcept
{
    return name_;
}

void Assignment::set_name(std::string name)
{
    name_ = std::move(name);
}

std::size_t Assignment::atom_numbers() const noexcept
{
    return atoms_.size();
}

bool Assignment::built() const noexcept
{
    return built_;
}

void Assignment::set_built(bool built) noexcept
{
    built_ = built;
    if (!built_)
    {
        kekulized_ = false;
    }
}

bool Assignment::kekulized() const noexcept
{
    return kekulized_;
}

void Assignment::set_kekulized(bool kekulized) noexcept
{
    kekulized_ = kekulized;
}

const std::vector<Atom>& Assignment::atoms() const noexcept
{
    return atoms_;
}

std::vector<Atom>& Assignment::atoms() noexcept
{
    return atoms_;
}

const std::vector<BondMap>& Assignment::bonds() const noexcept
{
    return bonds_;
}

std::vector<BondMap>& Assignment::bonds() noexcept
{
    return bonds_;
}

const std::vector<std::pair<int, int>>& Assignment::bond_order() const noexcept
{
    return bond_order_;
}

const std::vector<AtomMarker>& Assignment::atom_markers() const noexcept
{
    return atom_markers_;
}

std::vector<AtomMarker>& Assignment::atom_markers() noexcept
{
    return atom_markers_;
}

const std::vector<BondMarkerMap>& Assignment::bond_markers() const noexcept
{
    return bond_markers_;
}

std::vector<BondMarkerMap>& Assignment::bond_markers() noexcept
{
    return bond_markers_;
}

const std::vector<std::string>& Assignment::atom_types() const noexcept
{
    return atom_types_;
}

std::vector<std::string>& Assignment::atom_types() noexcept
{
    return atom_types_;
}

int Assignment::Add_Atom(const std::string& element,
                         double x,
                         double y,
                         double z,
                         std::string name,
                         double charge)
{
    Atom atom;
    const std::size_t dot = element.find('.');
    if (dot == std::string::npos)
    {
        atom.element = element;
    }
    else
    {
        atom.element = element.substr(0, dot);
        atom.element_detail = element.substr(dot);
    }

    atom.name = std::move(name);
    atom.coordinate = {x, y, z};
    atom.charge = charge;

    atoms_.push_back(std::move(atom));
    bonds_.emplace_back();
    atom_markers_.emplace_back();
    bond_markers_.emplace_back();
    atom_types_.emplace_back();
    set_built(false);
    return static_cast<int>(atoms_.size() - 1);
}

void Assignment::Add_Bond(int atom1, int atom2, int order)
{
    Check_Atom_Index(atom1);
    Check_Atom_Index(atom2);
    if (atom1 == atom2)
    {
        throw std::invalid_argument("cannot add a bond from an atom to itself");
    }

    if (bonds_[atom1].find(atom2) == bonds_[atom1].end())
    {
        bond_order_.push_back({atom1, atom2});
    }
    bonds_[atom1][atom2] = order;
    bonds_[atom2][atom1] = order;
    bond_markers_[atom1][atom2];
    bond_markers_[atom2][atom1];
    set_built(false);
}

void Assignment::Delete_Bond(int atom1, int atom2)
{
    Check_Atom_Index(atom1);
    Check_Atom_Index(atom2);
    bonds_[atom1].erase(atom2);
    bonds_[atom2].erase(atom1);
    bond_order_.erase(
        std::remove_if(bond_order_.begin(), bond_order_.end(),
                       [atom1, atom2](const std::pair<int, int>& bond) {
                           return (bond.first == atom1 && bond.second == atom2) ||
                                  (bond.first == atom2 && bond.second == atom1);
                       }),
        bond_order_.end());
    bond_markers_[atom1].erase(atom2);
    bond_markers_[atom2].erase(atom1);
    set_built(false);
}

void Assignment::Add_Atom_Marker(int atom, const std::string& marker)
{
    Check_Atom_Index(atom);
    ++atom_markers_[atom][marker];
}

void Assignment::Add_Bond_Marker(int atom1,
                                 int atom2,
                                 const std::string& marker,
                                 bool only1)
{
    Check_Bond_Index(atom1, atom2);
    bond_markers_[atom1][atom2].insert(marker);
    ++atom_markers_[atom1][marker];
    if (!only1)
    {
        bond_markers_[atom2][atom1].insert(marker);
        ++atom_markers_[atom2][marker];
    }
}

bool Assignment::Atom_Judge(int atom, const std::string& mask) const
{
    Check_Atom_Index(atom);
    const std::size_t digit = mask.find_first_of("0123456789");
    if (digit == std::string::npos || digit == 0)
    {
        throw std::invalid_argument("atom mask should be like C3 or Cl1");
    }

    const std::string element = mask.substr(0, digit);
    const int degree = std::stoi(mask.substr(digit));
    return atoms_[atom].element == element &&
           static_cast<int>(bonds_[atom].size()) == degree;
}

bool Assignment::Atom_Judge(int atom, const char* element, int degree) const
{
    Check_Atom_Index(atom);
    return atoms_[atom].element == element &&
           static_cast<int>(bonds_[atom].size()) == degree;
}

bool Assignment::Check_Connectivity() const
{
    if (atoms_.empty())
    {
        return true;
    }

    std::vector<bool> visited(atoms_.size(), false);
    std::queue<int> queue;
    queue.push(0);
    visited[0] = true;

    std::size_t visited_count = 0;
    while (!queue.empty())
    {
        const int atom = queue.front();
        queue.pop();
        ++visited_count;
        for (const auto& neighbor : bonds_[atom])
        {
            const int next = neighbor.first;
            if (!visited[next])
            {
                visited[next] = true;
                queue.push(next);
            }
        }
    }
    return visited_count == atoms_.size();
}

void Assignment::Check_Atom_Index(int atom) const
{
    if (atom < 0 || static_cast<std::size_t>(atom) >= atoms_.size())
    {
        throw std::out_of_range("atom index is out of range");
    }
}

void Assignment::Check_Bond_Index(int atom1, int atom2) const
{
    Check_Atom_Index(atom1);
    Check_Atom_Index(atom2);
    if (bonds_[atom1].find(atom2) == bonds_[atom1].end() ||
        bonds_[atom2].find(atom1) == bonds_[atom2].end())
    {
        throw std::out_of_range("bond does not exist");
    }
}

}
}
