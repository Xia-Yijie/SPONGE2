#include "gaff_typing.h"

#include <algorithm>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "mol2_writer_policy.h"

namespace Xponge
{
namespace Assign
{
namespace
{

std::string Sybyl_Type(const Atom& atom)
{
    return atom.element + atom.element_detail;
}

bool Is_Element(const Assignment& assignment, int atom, const std::string& e)
{
    return assignment.atoms()[atom].element == e;
}

int Degree(const Assignment& assignment, int atom)
{
    return static_cast<int>(assignment.bonds()[atom].size());
}

const std::set<std::string>& Electronegative_Elements()
{
    static const std::set<std::string> kSet{
        "N", "O", "F", "Cl", "Br", "S", "I"};
    return kSet;
}

std::string Input_Bond_Label(const Assignment& assignment,
                             int atom1,
                             int atom2,
                             int order)
{
    return Preserve_Bond_Type_For_Output(
        assignment, static_cast<std::size_t>(atom1), atom2, order);
}

bool Has_Bond_Order(const Assignment& assignment, int atom, int order)
{
    for (const auto& item : assignment.bonds()[atom])
    {
        if (item.second == order)
        {
            return true;
        }
    }
    return false;
}

int Count_Bond_Order(const Assignment& assignment, int atom, int order)
{
    int count = 0;
    for (const auto& item : assignment.bonds()[atom])
    {
        if (item.second == order)
        {
            ++count;
        }
    }
    return count;
}

bool Has_Single_And_Double_Bond(const Assignment& assignment, int atom)
{
    return Has_Bond_Order(assignment, atom, 1) &&
           Has_Bond_Order(assignment, atom, 2);
}

bool Has_Bond_Label(const Assignment& assignment, int atom, const char* label)
{
    for (const auto& item : assignment.bonds()[atom])
    {
        if (Input_Bond_Label(assignment, atom, item.first, item.second) ==
            label)
        {
            return true;
        }
    }
    return false;
}

bool Has_Neighbor_Element(const Assignment& assignment,
                          int atom,
                          const std::set<std::string>& elements)
{
    for (const auto& item : assignment.bonds()[atom])
    {
        if (elements.find(assignment.atoms()[item.first].element) !=
            elements.end())
        {
            return true;
        }
    }
    return false;
}

int Count_Neighbor_Element(const Assignment& assignment,
                           int atom,
                           const std::set<std::string>& elements)
{
    int count = 0;
    for (const auto& item : assignment.bonds()[atom])
    {
        if (elements.find(assignment.atoms()[item.first].element) !=
            elements.end())
        {
            ++count;
        }
    }
    return count;
}

bool Has_Neighbor_Type(const Assignment& assignment,
                       const std::vector<std::string>& sybyl_table,
                       int atom,
                       const std::set<std::string>& sybyl_types)
{
    for (const auto& item : assignment.bonds()[atom])
    {
        if (sybyl_types.find(sybyl_table[item.first]) != sybyl_types.end())
        {
            return true;
        }
    }
    return false;
}

bool Has_Carbonyl_Neighbor(const Assignment& assignment, int atom)
{
    for (const auto& item : assignment.bonds()[atom])
    {
        const int neighbor = item.first;
        if (!Is_Element(assignment, neighbor, "C"))
        {
            continue;
        }
        for (const auto& item2 : assignment.bonds()[neighbor])
        {
            if (item2.first == atom)
            {
                continue;
            }
            if (item2.second == 2 &&
                (Is_Element(assignment, item2.first, "O") ||
                 Is_Element(assignment, item2.first, "S")))
            {
                return true;
            }
        }
    }
    return false;
}

bool Is_Carbonyl_Carbon(const Assignment& assignment, int atom)
{
    if (!Is_Element(assignment, atom, "C"))
    {
        return false;
    }
    for (const auto& item : assignment.bonds()[atom])
    {
        if (item.second == 2 &&
            (Is_Element(assignment, item.first, "O") ||
             Is_Element(assignment, item.first, "S")))
        {
            return true;
        }
    }
    return false;
}

bool Has_Noncarbonyl_Sp2_Carbon_Neighbor(const Assignment& assignment,
                                         const std::vector<std::string>& sybyl_table,
                                         int atom)
{
    for (const auto& item : assignment.bonds()[atom])
    {
        if (sybyl_table[item.first] == "C.2" &&
            !Is_Carbonyl_Carbon(assignment, item.first))
        {
            return true;
        }
    }
    return false;
}

int Shortest_Path_Excluding_Bond(const Assignment& assignment,
                                 int start,
                                 int target,
                                 int exclude_a,
                                 int exclude_b)
{
    std::vector<int> distance(assignment.atom_numbers(), -1);
    std::queue<int> queue;
    distance[start] = 0;
    queue.push(start);
    while (!queue.empty())
    {
        const int current = queue.front();
        queue.pop();
        for (const auto& item : assignment.bonds()[current])
        {
            const int next = item.first;
            if ((current == exclude_a && next == exclude_b) ||
                (current == exclude_b && next == exclude_a))
            {
                continue;
            }
            if (distance[next] >= 0)
            {
                continue;
            }
            distance[next] = distance[current] + 1;
            if (next == target)
            {
                return distance[next];
            }
            queue.push(next);
        }
    }
    return -1;
}

std::vector<int> Shortest_Path_Excluding_Atom(const Assignment& assignment,
                                              int start,
                                              int target,
                                              int excluded)
{
    std::vector<int> parent(assignment.atom_numbers(), -1);
    std::queue<int> queue;
    parent[start] = start;
    queue.push(start);
    while (!queue.empty())
    {
        const int current = queue.front();
        queue.pop();
        for (const auto& item : assignment.bonds()[current])
        {
            const int next = item.first;
            if (next == excluded || parent[next] >= 0)
            {
                continue;
            }
            parent[next] = current;
            if (next == target)
            {
                std::vector<int> path;
                for (int atom = target; atom != start; atom = parent[atom])
                {
                    path.push_back(atom);
                }
                path.push_back(start);
                std::reverse(path.begin(), path.end());
                return path;
            }
            queue.push(next);
        }
    }
    return {};
}

int Smallest_Ring_Size(const Assignment& assignment, int atom)
{
    int best = 0;
    const auto& bonds = assignment.bonds()[atom];
    for (auto it1 = bonds.begin(); it1 != bonds.end(); ++it1)
    {
        for (auto it2 = std::next(it1); it2 != bonds.end(); ++it2)
        {
            const int path = Shortest_Path_Excluding_Bond(
                assignment, it1->first, it2->first, atom, it1->first);
            if (path < 0)
            {
                continue;
            }
            const int ring = path + 2;
            if (best == 0 || ring < best)
            {
                best = ring;
            }
        }
    }
    return best;
}

bool Is_Aromatic_Input(const Assignment& assignment, int atom)
{
    return Has_Bond_Label(assignment, atom, "ar");
}

bool Is_Planar_Ring_Atom(const Assignment& assignment, int atom)
{
    return assignment.Atom_Judge(atom, "C", 3) ||
           assignment.Atom_Judge(atom, "N", 2) ||
           assignment.Atom_Judge(atom, "N", 3) ||
           assignment.Atom_Judge(atom, "O", 2) ||
           assignment.Atom_Judge(atom, "S", 2) ||
           assignment.Atom_Judge(atom, "P", 2) ||
           assignment.Atom_Judge(atom, "P", 3);
}

bool Is_In_Planar_Ring(const Assignment& assignment, int atom)
{
    const auto& bonds = assignment.bonds()[atom];
    for (auto it1 = bonds.begin(); it1 != bonds.end(); ++it1)
    {
        for (auto it2 = std::next(it1); it2 != bonds.end(); ++it2)
        {
            const std::vector<int> path = Shortest_Path_Excluding_Atom(
                assignment, it1->first, it2->first, atom);
            if (path.empty())
            {
                continue;
            }
            const std::size_t ring_size = path.size() + 1;
            if (ring_size < 4 || ring_size > 8)
            {
                continue;
            }
            bool planar = Is_Planar_Ring_Atom(assignment, atom);
            for (const int ring_atom : path)
            {
                planar = planar && Is_Planar_Ring_Atom(assignment, ring_atom);
            }
            if (planar)
            {
                return true;
            }
        }
    }
    return false;
}

int Count_Rings_Of_Size(const Assignment& assignment, int atom, int size)
{
    std::set<std::string> rings;
    const auto& bonds = assignment.bonds()[atom];
    for (auto it1 = bonds.begin(); it1 != bonds.end(); ++it1)
    {
        for (auto it2 = std::next(it1); it2 != bonds.end(); ++it2)
        {
            const std::vector<int> path = Shortest_Path_Excluding_Atom(
                assignment, it1->first, it2->first, atom);
            if (path.empty() || static_cast<int>(path.size() + 1) != size)
            {
                continue;
            }
            std::vector<int> ring = path;
            ring.push_back(atom);
            std::sort(ring.begin(), ring.end());
            std::string key;
            for (const int ring_atom : ring)
            {
                key += std::to_string(ring_atom) + ",";
            }
            rings.insert(key);
        }
    }
    return static_cast<int>(rings.size());
}

bool Is_Aromatic_Ring_Atom(const Assignment& assignment,
                           const std::vector<std::string>& sybyl_table,
                           int atom)
{
    const std::string& sybyl = sybyl_table[atom];
    return Is_Aromatic_Input(assignment, atom) ||
           sybyl == "C.ar" || sybyl == "N.ar";
}

bool Is_Conjugating_Degree(const Assignment& assignment, int atom)
{
    return assignment.Atom_Judge(atom, "C", 3) ||
           assignment.Atom_Judge(atom, "C", 2) ||
           assignment.Atom_Judge(atom, "N", 2) ||
           assignment.Atom_Judge(atom, "P", 2) ||
           ((assignment.Atom_Judge(atom, "S", 3) ||
             assignment.Atom_Judge(atom, "S", 4) ||
             assignment.Atom_Judge(atom, "P", 3) ||
             assignment.Atom_Judge(atom, "P", 4)) &&
            Has_Bond_Order(assignment, atom, 2));
}

bool Has_Single_Conjugating_Neighbor(const Assignment& assignment, int atom)
{
    for (const auto& item : assignment.bonds()[atom])
    {
        if (item.second == 1 &&
            Is_Conjugating_Degree(assignment, item.first))
        {
            return true;
        }
    }
    return false;
}

bool Has_Next_Conjugating_Neighbor(const Assignment& assignment, int atom)
{
    for (const auto& item : assignment.bonds()[atom])
    {
        const int neighbor = item.first;
        if (!Is_Conjugating_Degree(assignment, neighbor))
        {
            continue;
        }
        for (const auto& item2 : assignment.bonds()[neighbor])
        {
            if (item2.first != atom &&
                Is_Conjugating_Degree(assignment, item2.first))
            {
                return true;
            }
        }
    }
    return false;
}

bool Is_Output_Aromatic_Bond(const Assignment& assignment, int atom1, int atom2)
{
    const auto& atom_types = assignment.atom_types();
    if (atom1 < 0 || atom2 < 0 ||
        static_cast<std::size_t>(atom1) >= atom_types.size() ||
        static_cast<std::size_t>(atom2) >= atom_types.size())
    {
        return false;
    }
    return Is_Antechamber_Aromatic_Output_Type(atom_types[atom1]) &&
           Is_Antechamber_Aromatic_Output_Type(atom_types[atom2]);
}

struct Typing_Context
{
    explicit Typing_Context(const Assignment& assignment)
        : assignment(assignment),
          sybyl(assignment.atom_numbers()),
          smallest_ring(assignment.atom_numbers(), -1),
          six_membered_ring_count(assignment.atom_numbers(), -1),
          aromatic_ring(assignment.atom_numbers(), -1),
          planar_ring(assignment.atom_numbers(), -1),
          aromatic_input(assignment.atom_numbers(), -1)
    {
        for (std::size_t i = 0; i < assignment.atom_numbers(); ++i)
        {
            sybyl[i] = Sybyl_Type(assignment.atoms()[i]);
        }
    }

    int Smallest_Ring(int atom) const
    {
        int& value = smallest_ring[atom];
        if (value < 0)
        {
            value = Smallest_Ring_Size(assignment, atom);
        }
        return value;
    }

    int Six_Membered_Ring_Count(int atom) const
    {
        int& value = six_membered_ring_count[atom];
        if (value < 0)
        {
            value = Count_Rings_Of_Size(assignment, atom, 6);
        }
        return value;
    }

    bool Aromatic_Ring(int atom) const
    {
        int& value = aromatic_ring[atom];
        if (value < 0)
        {
            value = Is_Aromatic_Ring_Atom(assignment, sybyl, atom) ? 1 : 0;
        }
        return value != 0;
    }

    bool Planar_Ring(int atom) const
    {
        int& value = planar_ring[atom];
        if (value < 0)
        {
            value = Is_In_Planar_Ring(assignment, atom) ? 1 : 0;
        }
        return value != 0;
    }

    bool Aromatic_Input(int atom) const
    {
        int& value = aromatic_input[atom];
        if (value < 0)
        {
            value = Is_Aromatic_Input(assignment, atom) ? 1 : 0;
        }
        return value != 0;
    }

    const Assignment& assignment;
    std::vector<std::string> sybyl;
    mutable std::vector<int> smallest_ring;
    mutable std::vector<int> six_membered_ring_count;
    mutable std::vector<int> aromatic_ring;
    mutable std::vector<int> planar_ring;
    mutable std::vector<int> aromatic_input;
};

std::string Type_Hydrogen(const Assignment& assignment, int atom)
{
    const int heavy = assignment.bonds()[atom].begin()->first;
    const auto& heavy_atom = assignment.atoms()[heavy];
    if (heavy_atom.element == "N")
    {
        return "hn";
    }
    if (heavy_atom.element == "O")
    {
        return "ho";
    }
    if (heavy_atom.element == "S")
    {
        return "hs";
    }
    if (heavy_atom.element == "P")
    {
        return "hp";
    }
    if (heavy_atom.element != "C")
    {
        return "ha";
    }

    if (assignment.Atom_Judge(heavy, "C", 4))
    {
        for (const auto& item : assignment.bonds()[heavy])
        {
            if (assignment.Atom_Judge(item.first, "N", 4))
            {
                return "hx";
            }
        }
        const int electronegative = Count_Neighbor_Element(
            assignment, heavy, Electronegative_Elements());
        if (electronegative >= 3)
        {
            return "h3";
        }
        if (electronegative == 2)
        {
            return "h2";
        }
        if (electronegative == 1)
        {
            return "h1";
        }
        return "hc";
    }

    const int electronegative = Count_Neighbor_Element(
        assignment, heavy, Electronegative_Elements());
    if (electronegative >= 2)
    {
        return "h5";
    }
    if (electronegative == 1)
    {
        return "h4";
    }
    return "ha";
}

std::string Type_Carbon(const Typing_Context& context, int atom)
{
    const auto& assignment = context.assignment;
    const int ring = context.Smallest_Ring(atom);
    if (assignment.Atom_Judge(atom, "C", 4))
    {
        if (ring == 3)
        {
            return "cx";
        }
        if (ring == 4)
        {
            return "cy";
        }
        return "c3";
    }
    if (assignment.Atom_Judge(atom, "C", 3) && context.Aromatic_Ring(atom))
    {
        bool all_neighbors_aromatic_heavy = true;
        int aromatic_bonds = 0;
        for (const auto& item : assignment.bonds()[atom])
        {
            if (Is_Element(assignment, item.first, "H"))
            {
                all_neighbors_aromatic_heavy = false;
                continue;
            }
            if (!context.Aromatic_Ring(item.first))
            {
                all_neighbors_aromatic_heavy = false;
            }
            if (Input_Bond_Label(assignment, atom, item.first, item.second) ==
                "ar")
            {
                ++aromatic_bonds;
            }
        }
        if (all_neighbors_aromatic_heavy && aromatic_bonds == 2 &&
            context.Six_Membered_Ring_Count(atom) == 1)
        {
            return "cp";
        }
        return "ca";
    }
    if (assignment.Atom_Judge(atom, "C", 3))
    {
        for (const auto& item : assignment.bonds()[atom])
        {
            if (item.second == 2 &&
                (Is_Element(assignment, item.first, "O") ||
                 Is_Element(assignment, item.first, "S")))
            {
                return "c";
            }
        }
        if (Count_Bond_Order(assignment, atom, 1) == 3)
        {
            for (const auto& item : assignment.bonds()[atom])
            {
                if ((Is_Element(assignment, item.first, "O") ||
                     Is_Element(assignment, item.first, "S")) &&
                    Degree(assignment, item.first) == 1)
                {
                    return "c";
                }
            }
        }
        if (Has_Single_And_Double_Bond(assignment, atom) &&
            context.Planar_Ring(atom))
        {
            return "cc";
        }
        if (Has_Bond_Order(assignment, atom, 2) &&
            Has_Single_Conjugating_Neighbor(assignment, atom))
        {
            return "ce";
        }
        if (ring == 3)
        {
            return "cu";
        }
        if (ring == 4)
        {
            return "cv";
        }
        return "c2";
    }
    if (assignment.Atom_Judge(atom, "C", 2))
    {
        if (Has_Bond_Order(assignment, atom, 1) &&
            Has_Bond_Order(assignment, atom, 3))
        {
            for (const auto& item : assignment.bonds()[atom])
            {
                if (item.second != 1)
                {
                    continue;
                }
                if (assignment.Atom_Judge(item.first, "C", 3) ||
                    assignment.Atom_Judge(item.first, "C", 2) ||
                    assignment.Atom_Judge(item.first, "N", 2) ||
                    assignment.Atom_Judge(item.first, "P", 2) ||
                    assignment.Atom_Judge(item.first, "N", 1))
                {
                    return "cg";
                }
            }
        }
        return "c1";
    }
    if (assignment.Atom_Judge(atom, "C", 1))
    {
        return "c1";
    }
    return "c3";
}

std::string Type_Oxygen(const Typing_Context& context, int atom)
{
    const auto& assignment = context.assignment;
    const std::string& sybyl = context.sybyl[atom];
    if (sybyl == "O.2" || sybyl == "O.co2")
    {
        return "o";
    }
    if (Has_Neighbor_Element(assignment, atom, {"H"}))
    {
        return "oh";
    }
    const int ring = context.Smallest_Ring(atom);
    if (ring == 3)
    {
        return "op";
    }
    if (ring == 4)
    {
        return "oq";
    }
    return "os";
}

std::string Type_Nitrogen(const Typing_Context& context, int atom)
{
    const auto& assignment = context.assignment;
    const std::string& sybyl = context.sybyl[atom];
    const int ring = context.Smallest_Ring(atom);
    if (assignment.Atom_Judge(atom, "N", 2) && context.Aromatic_Ring(atom))
    {
        return "nb";
    }
    if (assignment.Atom_Judge(atom, "N", 1))
    {
        return "n1";
    }
    if (assignment.Atom_Judge(atom, "N", 4))
    {
        return ring == 3 ? "nk" : (ring == 4 ? "nl" : "n4");
    }
    if (sybyl == "N.am")
    {
        return ring == 4 ? "nj" : "n";
    }
    if (sybyl == "N.pl3")
    {
        int terminal_oxygen = 0;
        for (const auto& item : assignment.bonds()[atom])
        {
            if (context.sybyl[item.first] == "O.2" ||
                assignment.Atom_Judge(item.first, "O", 1))
            {
                ++terminal_oxygen;
            }
        }
        if (terminal_oxygen >= 2)
        {
            return "no";
        }
        return "na";
    }
    if (assignment.Atom_Judge(atom, "N", 3))
    {
        int terminal_oxygen = 0;
        for (const auto& item : assignment.bonds()[atom])
        {
            if (assignment.Atom_Judge(item.first, "O", 1))
            {
                ++terminal_oxygen;
            }
        }
        if (terminal_oxygen >= 2)
        {
            return "no";
        }
        if (Has_Carbonyl_Neighbor(assignment, atom))
        {
            return ring == 3 ? "ni" : (ring == 4 ? "nj" : "n");
        }
        if (context.Aromatic_Input(atom))
        {
            return "na";
        }
        if (Has_Neighbor_Type(assignment, context.sybyl, atom, {"C.ar", "N.ar"}))
        {
            return "nh";
        }
        for (const auto& item : assignment.bonds()[atom])
        {
            const std::string& neighbor = context.sybyl[item.first];
            if (neighbor == "N.pl3" ||
                (neighbor == "N.am" &&
                 Has_Noncarbonyl_Sp2_Carbon_Neighbor(assignment, context.sybyl, item.first)) ||
                ((neighbor == "C.2" || neighbor == "N.2") &&
                 Has_Bond_Order(assignment, item.first, 2)))
            {
                return "nh";
            }
        }
        if (ring == 3)
        {
            return "np";
        }
        if (ring == 4)
        {
            return "nq";
        }
        return "n3";
    }
    if (assignment.Atom_Judge(atom, "N", 2))
    {
        bool single_npl3 = false;
        bool double_sulfur_substituted_carbon = false;
        for (const auto& item : assignment.bonds()[atom])
        {
            if (item.second == 1 &&
                context.sybyl[item.first] == "N.pl3")
            {
                single_npl3 = true;
            }
            if (item.second == 2 &&
                context.sybyl[item.first] == "C.2" &&
                Count_Neighbor_Element(assignment, item.first, {"S"}) >= 2)
            {
                double_sulfur_substituted_carbon = true;
            }
        }
        if (single_npl3 && double_sulfur_substituted_carbon)
        {
            return "n2";
        }
        if ((Has_Bond_Order(assignment, atom, 1) &&
             Has_Bond_Order(assignment, atom, 3)) ||
            Count_Bond_Order(assignment, atom, 2) == 2)
        {
            return "n1";
        }
        if (Has_Single_And_Double_Bond(assignment, atom) &&
            (context.Aromatic_Input(atom) ||
             context.Planar_Ring(atom)) &&
            (Has_Single_Conjugating_Neighbor(assignment, atom) ||
             Has_Next_Conjugating_Neighbor(assignment, atom)))
        {
            return "nc";
        }
        if (Has_Single_And_Double_Bond(assignment, atom) &&
            Has_Single_Conjugating_Neighbor(assignment, atom))
        {
            return "ne";
        }
        return "n2";
    }
    return "n3";
}

std::string Type_Sulfur(const Typing_Context& context, int atom)
{
    const auto& assignment = context.assignment;
    const std::string& sybyl = context.sybyl[atom];
    const int ring = context.Smallest_Ring(atom);
    if (sybyl == "S.2")
    {
        return "s";
    }
    if (sybyl == "S.o")
    {
        return Has_Single_Conjugating_Neighbor(assignment, atom) ? "sx" : "s4";
    }
    if (sybyl == "S.o2")
    {
        return Has_Neighbor_Type(assignment, context.sybyl, atom,
                                 {"C.ar", "C.2"})
                   ? "sy"
                   : "s6";
    }
    if (sybyl == "S.3")
    {
        if (Degree(assignment, atom) >= 4)
        {
            return "s6";
        }
        if (Has_Neighbor_Element(assignment, atom, {"H"}))
        {
            return "sh";
        }
        if (ring == 3)
        {
            return "sp";
        }
        if (ring == 4)
        {
            return "sq";
        }
        return Degree(assignment, atom) == 3 ? "s4" : "ss";
    }
    return "s6";
}

std::string Type_Phosphorus(const Assignment& assignment, int atom)
{
    const int degree = Degree(assignment, atom);
    if (degree <= 2)
    {
        return "p2";
    }
    if (degree == 3)
    {
        if (Has_Bond_Order(assignment, atom, 2))
        {
            return Has_Neighbor_Element(assignment, atom, {"O", "S"}) ? "p4"
                                                                      : "px";
        }
        return "p3";
    }
    if (degree == 4 && Has_Bond_Order(assignment, atom, 2))
    {
        return Has_Single_Conjugating_Neighbor(assignment, atom) ? "py" : "p5";
    }
    return "p5";
}

std::string Type_Atom(const Typing_Context& context, int atom)
{
    const auto& assignment = context.assignment;
    const auto& element = assignment.atoms()[atom].element;
    if (element == "H")
    {
        return Type_Hydrogen(assignment, atom);
    }
    if (element == "C")
    {
        return Type_Carbon(context, atom);
    }
    if (element == "O")
    {
        return Type_Oxygen(context, atom);
    }
    if (element == "N")
    {
        return Type_Nitrogen(context, atom);
    }
    if (element == "S")
    {
        return Type_Sulfur(context, atom);
    }
    if (element == "P")
    {
        return Type_Phosphorus(assignment, atom);
    }
    if (element == "F")
    {
        return "f";
    }
    if (element == "Cl")
    {
        return "cl";
    }
    if (element == "Br")
    {
        return "br";
    }
    if (element == "I")
    {
        return "i";
    }
    return element;
}

void Apply_Bond_Label(Assignment& assignment,
                      int atom1,
                      int atom2,
                      const std::string& label)
{
    auto& bonds = assignment.bonds();
    auto& markers = assignment.bond_markers();
    markers[atom1][atom2].clear();
    markers[atom2][atom1].clear();
    if (label == "ar")
    {
        bonds[atom1][atom2] = -1;
        bonds[atom2][atom1] = -1;
        markers[atom1][atom2].insert("mol2_ar");
        markers[atom2][atom1].insert("mol2_ar");
    }
    else if (label == "am")
    {
        bonds[atom1][atom2] = 1;
        bonds[atom2][atom1] = 1;
    }
    else if (label == "un")
    {
        bonds[atom1][atom2] = -1;
        bonds[atom2][atom1] = -1;
    }
    else
    {
        const int order = std::stoi(label);
        bonds[atom1][atom2] = order;
        bonds[atom2][atom1] = order;
    }
}

void Normalize_Gaff_Bond_Types(Assignment& assignment)
{
    auto& bonds = assignment.bonds();
    auto& markers = assignment.bond_markers();
    for (const auto& bond : assignment.bond_order())
    {
        const int i = bond.first;
        const int j = bond.second;
        if (bonds[i].find(j) == bonds[i].end())
        {
            continue;
        }
        const std::string label = Input_Bond_Label(assignment, i, j, bonds[i][j]);
        if (label != "ar")
        {
            continue;
        }
        bool adjacent_single = false;
        bool adjacent_double = false;
        const auto inspect_neighbors = [&](int self, int other) {
            for (const auto& item : bonds[self])
            {
                if (item.first == other ||
                    Is_Output_Aromatic_Bond(assignment, self, item.first))
                {
                    continue;
                }
                const auto& neighbor_markers = markers[self][item.first];
                if (item.second == 2 ||
                    neighbor_markers.find("gaff_ar_double") !=
                        neighbor_markers.end())
                {
                    adjacent_double = true;
                }
                if (neighbor_markers.find("gaff_ar_single") !=
                    neighbor_markers.end())
                {
                    adjacent_single = true;
                }
            }
        };
        inspect_neighbors(i, j);
        inspect_neighbors(j, i);
        if (adjacent_single && !adjacent_double)
        {
            markers[i][j].insert("gaff_ar_double");
            markers[j][i].insert("gaff_ar_double");
        }
        else
        {
            markers[i][j].insert("gaff_ar_single");
            markers[j][i].insert("gaff_ar_single");
        }
    }

    for (const auto& bond : assignment.bond_order())
    {
        const int i = bond.first;
        const int j = bond.second;
        if (bonds[i].find(j) == bonds[i].end())
        {
            continue;
        }
        const std::string label = Input_Bond_Label(assignment, i, j, bonds[i][j]);
        if (label == "am")
        {
            Apply_Bond_Label(assignment, i, j, "1");
        }
    }
}

void Propagate_Paired_Types(const Assignment& assignment,
                            std::vector<std::string>& atom_types,
                            const std::set<std::string>& base_types,
                            const std::map<std::string, std::string>& paired)
{
    std::vector<int> color(atom_types.size(), 0);
    std::vector<int> eligible(atom_types.size(), 0);
    int eligible_count = 0;
    for (std::size_t i = 0; i < atom_types.size(); ++i)
    {
        if (base_types.find(atom_types[i]) != base_types.end())
        {
            eligible[i] = 1;
            if (eligible_count == 0)
            {
                color[i] = 1;
            }
            ++eligible_count;
        }
    }
    if (eligible_count == 0)
    {
        return;
    }

    for (int iter = 0; iter < eligible_count - 1; ++iter)
    {
        bool propagated = false;
        for (std::size_t atom = 0; atom < assignment.bonds().size(); ++atom)
        {
            for (const auto& item : assignment.bonds()[atom])
            {
                const int neighbor = item.first;
                if (static_cast<int>(atom) >= neighbor)
                {
                    continue;
                }
                if (eligible[atom] + eligible[neighbor] != 2)
                {
                    continue;
                }
                if (!propagated && color[atom] == 0 &&
                    color[neighbor] == 0)
                {
                    color[atom] = 1;
                }
                if (color[atom] == 0 && color[neighbor] != 0)
                {
                    propagated = true;
                    color[atom] =
                        item.second == 1 ? color[neighbor] : -color[neighbor];
                }
                if (color[neighbor] == 0 && color[atom] != 0)
                {
                    propagated = true;
                    color[neighbor] =
                        item.second == 1 ? color[atom] : -color[atom];
                }
            }
        }
        if (!propagated)
        {
            break;
        }
    }

    for (std::size_t i = 0; i < atom_types.size(); ++i)
    {
        if (color[i] != -1)
        {
            continue;
        }
        const auto type = paired.find(atom_types[i]);
        if (type != paired.end())
        {
            atom_types[i] = type->second;
        }
    }
}

void Apply_Antechamber_Paired_Adjustments(const Assignment& assignment,
                                          std::vector<std::string>& atom_types)
{
    Propagate_Paired_Types(
        assignment, atom_types, {"cc", "ce", "cg", "pc", "pe", "nc", "ne"},
        {{"cc", "cd"}, {"ce", "cf"}, {"cg", "ch"}, {"pc", "pd"},
         {"pe", "pf"}, {"nc", "nd"}, {"ne", "nf"}});
    Propagate_Paired_Types(assignment, atom_types, {"cp"}, {{"cp", "cq"}});
}

}

void Determine_Gaff_Atom_Type(Assignment& assignment)
{
    auto& atom_types = assignment.atom_types();
    const Typing_Context context(assignment);
    atom_types.resize(assignment.atom_numbers());
    for (std::size_t i = 0; i < assignment.atom_numbers(); ++i)
    {
        atom_types[i] =
            Type_Atom(context, static_cast<int>(i));
    }
    Apply_Antechamber_Paired_Adjustments(assignment, atom_types);
    Normalize_Gaff_Bond_Types(assignment);
}

}
}
