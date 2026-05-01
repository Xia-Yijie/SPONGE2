#pragma once

#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace Xponge
{
namespace Assign
{

struct Coordinate
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Atom
{
    std::string element;
    std::string name;
    std::string element_detail;
    Coordinate coordinate;
    double charge = 0.0;
    int formal_charge = 0;
};

using AtomMarker = std::map<std::string, int>;
using BondMap = std::map<int, int>;
using BondMarkerMap = std::map<int, std::set<std::string>>;

class Assignment
{
public:
    explicit Assignment(std::string name = "ASN");

    const std::string& name() const noexcept;
    void set_name(std::string name);

    std::size_t atom_numbers() const noexcept;

    bool built() const noexcept;
    void set_built(bool built) noexcept;

    bool kekulized() const noexcept;
    void set_kekulized(bool kekulized) noexcept;

    const std::vector<Atom>& atoms() const noexcept;
    std::vector<Atom>& atoms() noexcept;

    const std::vector<BondMap>& bonds() const noexcept;
    std::vector<BondMap>& bonds() noexcept;
    const std::vector<std::pair<int, int>>& bond_order() const noexcept;

    const std::vector<AtomMarker>& atom_markers() const noexcept;
    std::vector<AtomMarker>& atom_markers() noexcept;

    const std::vector<BondMarkerMap>& bond_markers() const noexcept;
    std::vector<BondMarkerMap>& bond_markers() noexcept;

    const std::vector<std::string>& atom_types() const noexcept;
    std::vector<std::string>& atom_types() noexcept;

    int Add_Atom(const std::string& element,
                 double x,
                 double y,
                 double z,
                 std::string name = "",
                 double charge = 0.0);
    void Add_Bond(int atom1, int atom2, int order = -1);
    void Delete_Bond(int atom1, int atom2);
    void Add_Atom_Marker(int atom, const std::string& marker);
    void Add_Bond_Marker(int atom1,
                         int atom2,
                         const std::string& marker,
                         bool only1 = false);
    bool Atom_Judge(int atom, const std::string& mask) const;
    bool Atom_Judge(int atom, const char* element, int degree) const;
    bool Check_Connectivity() const;

private:
    void Check_Atom_Index(int atom) const;
    void Check_Bond_Index(int atom1, int atom2) const;

    std::string name_;
    std::vector<Atom> atoms_;
    std::vector<BondMap> bonds_;
    std::vector<std::pair<int, int>> bond_order_;
    std::vector<AtomMarker> atom_markers_;
    std::vector<BondMarkerMap> bond_markers_;
    std::vector<std::string> atom_types_;
    bool built_ = false;
    bool kekulized_ = false;
};

}
}
