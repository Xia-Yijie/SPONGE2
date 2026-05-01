#include "parmchk2.h"
#include "string_utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace Xponge
{
namespace Amber
{
namespace
{

struct CorrEntry
{
    int target_idx = -1;
    double bl = 0.0;
    double blf = 0.0;
    double cba = 0.0;
    double cbaf = 0.0;
    double ba = 0.0;
    double baf = 0.0;
    double ctor = 0.0;
    double tor = 0.0;
    double improper = 0.0;
    int type = 0;
};

struct ParmEntry
{
    std::string atomtype;
    int improper_flag = 0;
    int group_id = 0;
    double mass = 0.0;
    int equtype_flag = 0;
    int atomic_num = 0;
    std::vector<int> equa;
    std::vector<CorrEntry> corr;
};

struct ParmTable
{
    std::vector<ParmEntry> entries;
    std::map<std::string, int> by_name;
    double wt_BL = 0.5;
    double wt_BLF = 0.5;
    double wt_BA = 0.5;
    double wt_BAF = 0.5;
    double wt_X = 10.0;
    double wt_X3 = 30.0;
    double wt_BA_CTR = 10.0;
    double wt_TOR_CTR = 10.0;
    double wt_IMPROPER = 10.0;
    double wt_GROUP = 100.0;
    double wt_EQUTYPE = 20.0;
    double default_BL = 20.0;
    double default_BLF = 30.0;
    double default_BA = 4.0;
    double default_BAF = 15.0;
    double default_BA_CTR = 13.0;
    double default_BAF_CTR = 17.0;
    double default_TOR = 87.0;
    double default_TOR_CTR = 65.0;
    double default_FRACT1 = 0.5;
    double default_FRACT2 = 0.5;
    double threshold_BA = 20.0;
};

struct BlbaTables
{
    double pc = 4.5;
    std::map<std::pair<int, int>, std::pair<double, double>> bl;
    std::map<int, std::pair<double, double>> ba;
};

struct GaffAtom
{
    std::string name;
    double mass = 0.0;
    double pol = 0.0;
};

struct GaffBond
{
    std::string name1;
    std::string name2;
    double force = 0.0;
    double length = 0.0;
};

struct GaffAngle
{
    std::string name1;
    std::string name2;
    std::string name3;
    double force = 0.0;
    double angle = 0.0;
};

struct GaffTorsion
{
    std::string name1;
    std::string name2;
    std::string name3;
    std::string name4;
    int mul = 0;
    double force = 0.0;
    double phase = 0.0;
    double fterm = 0.0;
    int num_X = 0;
};

struct GaffImproper
{
    std::string name1;
    std::string name2;
    std::string name3;
    std::string name4;
    double force = 0.0;
    double phase = 0.0;
    double fterm = 0.0;
    int num_X = 0;
};

struct GaffVdw
{
    std::string name;
    double rstar = 0.0;
    double epsilon = 0.0;
};

struct GaffTables
{
    std::vector<GaffAtom> atoms;
    std::vector<GaffBond> bonds;
    std::vector<GaffAngle> angles;
    std::vector<GaffTorsion> torsions;
    std::vector<GaffImproper> impropers;
    std::vector<GaffVdw> vdws;
    std::map<std::string, std::string> vdw_equivalents;
};

struct Mol2Atom
{
    std::string name;
    std::string gaff_type;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    std::string element;
    int atomic_num = 0;
};

struct Mol2Molecule
{
    std::vector<Mol2Atom> atoms;
    std::vector<std::pair<int, int>> bonds;

    std::vector<std::vector<int>> Neighbors() const
    {
        std::vector<std::vector<int>> nbrs(atoms.size());
        for (const auto& bond : bonds)
        {
            nbrs[bond.first].push_back(bond.second);
            nbrs[bond.second].push_back(bond.first);
        }
        return nbrs;
    }
};

using detail::Read_File;
using detail::Split_Atoms_Words;
using detail::Split_Lines;
using detail::Starts_With;
using detail::Trim;
using detail::Words;

double Parse_Double(const std::string& value)
{
    return std::stod(value);
}

int Parse_Int(const std::string& value)
{
    return std::stoi(value);
}

bool Try_Parse_Float_After_Prefix(const std::string& line,
                                  std::size_t offset,
                                  double* value)
{
    std::string rest = line.substr(offset);
    rest = Trim(rest);
    if (rest.empty())
    {
        return false;
    }
    const auto space = rest.find_first_of(" \t");
    const std::string token =
        space == std::string::npos ? rest : rest.substr(0, space);
    try
    {
        *value = std::stod(token);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

int Count_X(const std::vector<std::string>& atoms)
{
    return static_cast<int>(
        std::count(atoms.begin(), atoms.end(), std::string("X")));
}

std::string Join_Path(const std::string& directory, const std::string& file)
{
    if (directory.empty())
    {
        return file;
    }
    const char last = directory[directory.size() - 1];
    if (last == '/' || last == '\\')
    {
        return directory + file;
    }
    return directory + "/" + file;
}

std::string Format_Fixed(double value, int width, int precision)
{
    std::ostringstream out;
    out << std::fixed << std::setw(width) << std::setprecision(precision)
        << value;
    return out.str();
}

std::string Format_Int(int value, int width)
{
    std::ostringstream out;
    out << std::setw(width) << value;
    return out.str();
}

std::string Left(const std::string& value, int width)
{
    std::ostringstream out;
    out << std::left << std::setw(width) << value << std::right;
    return out.str();
}

std::string Right(const std::string& value, int width)
{
    std::ostringstream out;
    out << std::right << std::setw(width) << value;
    return out.str();
}

std::map<std::string, std::string> Element_By_Gaff()
{
    return {
        {"h1", "H"}, {"h2", "H"}, {"h3", "H"}, {"h4", "H"}, {"h5", "H"},
        {"ha", "H"}, {"hc", "H"}, {"hn", "H"}, {"ho", "H"}, {"hp", "H"},
        {"hs", "H"}, {"hw", "H"}, {"hx", "H"}, {"c", "C"},  {"c1", "C"},
        {"c2", "C"}, {"c3", "C"}, {"ca", "C"}, {"cc", "C"}, {"cd", "C"},
        {"ce", "C"}, {"cf", "C"}, {"cg", "C"}, {"ch", "C"}, {"cp", "C"},
        {"cq", "C"}, {"cu", "C"}, {"cv", "C"}, {"cx", "C"}, {"cy", "C"},
        {"cz", "C"}, {"n", "N"},  {"n1", "N"}, {"n2", "N"}, {"n3", "N"},
        {"n4", "N"}, {"na", "N"}, {"nb", "N"}, {"nc", "N"}, {"nd", "N"},
        {"ne", "N"}, {"nf", "N"}, {"nh", "N"}, {"no", "N"}, {"ni", "N"},
        {"nj", "N"}, {"nk", "N"}, {"nl", "N"}, {"nm", "N"}, {"nn", "N"},
        {"np", "N"}, {"nq", "N"}, {"n7", "N"}, {"n8", "N"}, {"n9", "N"},
        {"o", "O"},  {"oh", "O"}, {"os", "O"}, {"op", "O"}, {"oq", "O"},
        {"ow", "O"}, {"f", "F"},  {"cl", "Cl"}, {"br", "Br"}, {"i", "I"},
        {"p2", "P"}, {"p3", "P"}, {"p4", "P"}, {"p5", "P"}, {"pb", "P"},
        {"pc", "P"}, {"pd", "P"}, {"pe", "P"}, {"pf", "P"}, {"px", "P"},
        {"py", "P"}, {"s", "S"},  {"s2", "S"}, {"s4", "S"}, {"s6", "S"},
        {"sh", "S"}, {"ss", "S"}, {"sx", "S"}, {"sy", "S"}, {"si", "Si"}};
}

std::map<std::string, int> Atomic_Number()
{
    return {{"H", 1},   {"He", 2},  {"Li", 3},  {"Be", 4},  {"B", 5},
            {"C", 6},   {"N", 7},   {"O", 8},   {"F", 9},   {"Ne", 10},
            {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"AL", 13}, {"Si", 14},
            {"SI", 14}, {"P", 15},  {"S", 16},  {"Cl", 17}, {"CL", 17},
            {"Ar", 18}, {"K", 19},  {"Ca", 20}, {"Sc", 21}, {"Ti", 22},
            {"V", 23},  {"Cr", 24}, {"Mn", 25}, {"Fe", 26}, {"Co", 27},
            {"Ni", 28}, {"Cu", 29}, {"Zn", 30}, {"Ga", 31}, {"Ge", 32},
            {"As", 33}, {"Se", 34}, {"Br", 35}, {"BR", 35}, {"Kr", 36},
            {"Rb", 37}, {"Sr", 38}, {"Y", 39},  {"Zr", 40}, {"Nb", 41},
            {"Mo", 42}, {"Tc", 43}, {"Ru", 44}, {"Rh", 45}, {"Pd", 46},
            {"Ag", 47}, {"Cd", 48}, {"In", 49}, {"Sn", 50}, {"Sb", 51},
            {"Te", 52}, {"I", 53},  {"Xe", 54}, {"Cs", 55}, {"Ba", 56},
            {"Pt", 78}, {"Au", 79}, {"Hg", 80}, {"Tl", 81}, {"Pb", 82}};
}

std::string Lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](char c) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    });
    return value;
}

std::string Gaff_To_Element(const std::string& gaff_type,
                            const std::string& atom_name)
{
    static const auto by_gaff = Element_By_Gaff();
    static const auto atomic = Atomic_Number();
    const auto by_gaff_it = by_gaff.find(Lower(gaff_type));
    if (by_gaff_it != by_gaff.end())
    {
        return by_gaff_it->second;
    }
    if (atomic.find(gaff_type) != atomic.end())
    {
        return gaff_type;
    }
    std::string leading;
    for (char c : atom_name)
    {
        if (std::isalpha(static_cast<unsigned char>(c)))
        {
            leading.push_back(c);
            if (leading.size() == 2)
            {
                break;
            }
        }
    }
    if (atomic.find(leading) != atomic.end())
    {
        return leading;
    }
    if (!leading.empty() &&
        atomic.find(leading.substr(0, 1)) != atomic.end())
    {
        return leading.substr(0, 1);
    }
    return "";
}

Mol2Molecule Read_Mol2(const std::string& path)
{
    static const auto atomic = Atomic_Number();
    Mol2Molecule mol;
    std::string section;
    for (const auto& raw : Split_Lines(Read_File(path)))
    {
        if (Starts_With(raw, "@<TRIPOS>"))
        {
            section = Trim(raw.substr(9));
            continue;
        }
        if (section == "ATOM")
        {
            const auto words = Words(raw);
            if (words.size() < 6)
            {
                continue;
            }
            Mol2Atom atom;
            atom.name = words[1];
            atom.x = Parse_Double(words[2]);
            atom.y = Parse_Double(words[3]);
            atom.z = Parse_Double(words[4]);
            atom.gaff_type = words[5];
            atom.element = Gaff_To_Element(atom.gaff_type, atom.name);
            const auto atomic_it = atomic.find(atom.element);
            atom.atomic_num = atomic_it == atomic.end() ? 0 : atomic_it->second;
            mol.atoms.push_back(atom);
        }
        else if (section == "BOND")
        {
            const auto words = Words(raw);
            if (words.size() < 3)
            {
                continue;
            }
            int i = Parse_Int(words[1]) - 1;
            int j = Parse_Int(words[2]) - 1;
            if (i > j)
            {
                std::swap(i, j);
            }
            mol.bonds.push_back({i, j});
        }
    }
    return mol;
}

ParmTable Load_Parmchk_Dat(const std::string& path)
{
    ParmTable table;
    std::vector<std::vector<std::string>> pending_equa;
    std::vector<std::vector<std::pair<std::string, CorrEntry>>> pending_corr;
    std::vector<std::string>* current_equa = nullptr;
    std::vector<std::pair<std::string, CorrEntry>>* current_corr = nullptr;
    const std::vector<std::pair<std::string, double ParmTable::*>> key_map = {
        {"WEIGHT_BL", &ParmTable::wt_BL},
        {"WEIGHT_BLF", &ParmTable::wt_BLF},
        {"WEIGHT_BA", &ParmTable::wt_BA},
        {"WEIGHT_BAF", &ParmTable::wt_BAF},
        {"WEIGHT_X", &ParmTable::wt_X},
        {"WEIGHT_X3", &ParmTable::wt_X3},
        {"WEIGHT_BA_CTR", &ParmTable::wt_BA_CTR},
        {"WEIGHT_TOR_CTR", &ParmTable::wt_TOR_CTR},
        {"WEIGHT_IMPROPER", &ParmTable::wt_IMPROPER},
        {"WEIGHT_GROUP", &ParmTable::wt_GROUP},
        {"WEIGHT_EQUTYPE", &ParmTable::wt_EQUTYPE},
        {"THRESHOLD_BA", &ParmTable::threshold_BA},
        {"DEFAULT_BL", &ParmTable::default_BL},
        {"DEFAULT_BLF", &ParmTable::default_BLF},
        {"DEFAULT_BA", &ParmTable::default_BA},
        {"DEFAULT_BAF", &ParmTable::default_BAF},
        {"DEFAULT_BA_CTR", &ParmTable::default_BA_CTR},
        {"DEFAULT_BAF_CTR", &ParmTable::default_BAF_CTR},
        {"DEFAULT_TOR", &ParmTable::default_TOR},
        {"DEFAULT_TOR_CTR", &ParmTable::default_TOR_CTR},
        {"DEFAULT_FRACT1", &ParmTable::default_FRACT1},
        {"DEFAULT_FRACT2", &ParmTable::default_FRACT2}};

    for (const auto& raw : Split_Lines(Read_File(path)))
    {
        const std::string line = Trim(raw);
        if (line.empty() || line[0] == '#' || line[0] == '-')
        {
            continue;
        }
        const auto words = Words(line);
        if (Starts_With(line, "PARM"))
        {
            ParmEntry entry;
            entry.atomtype = words[1];
            entry.improper_flag = Parse_Int(words[2]);
            entry.group_id = Parse_Int(words[3]);
            entry.mass = Parse_Double(words[4]);
            entry.equtype_flag = Parse_Int(words[5]);
            entry.atomic_num = Parse_Int(words[6]);
            table.by_name[entry.atomtype] =
                static_cast<int>(table.entries.size());
            table.entries.push_back(entry);
            pending_equa.push_back({entry.atomtype});
            pending_corr.push_back({{entry.atomtype, CorrEntry()}});
            pending_corr.back().back().second.type = 0;
            current_equa = &pending_equa.back();
            current_corr = &pending_corr.back();
        }
        else if (Starts_With(line, "EQUA"))
        {
            current_equa->push_back(words[1]);
            CorrEntry ce;
            ce.type = 1;
            current_corr->push_back({words[1], ce});
        }
        else if (Starts_With(line, "CORR"))
        {
            CorrEntry ce;
            ce.bl = Parse_Double(words[2]);
            ce.blf = Parse_Double(words[3]);
            ce.cba = Parse_Double(words[4]);
            ce.cbaf = Parse_Double(words[5]);
            ce.ba = Parse_Double(words[6]);
            ce.baf = Parse_Double(words[7]);
            ce.ctor = Parse_Double(words[8]);
            ce.tor = Parse_Double(words[9]);
            ce.improper = Parse_Double(words[10]);
            ce.type = 2;
            current_corr->push_back({words[1], ce});
        }
        else
        {
            for (const auto& item : key_map)
            {
                if (Starts_With(line, item.first))
                {
                    double value = 0.0;
                    if (Try_Parse_Float_After_Prefix(line, item.first.size(),
                                                     &value))
                    {
                        table.*(item.second) = value;
                    }
                }
            }
        }
    }

    for (std::size_t i = 0; i < table.entries.size(); ++i)
    {
        ParmEntry& entry = table.entries[i];
        for (const auto& name : pending_equa[i])
        {
            const auto it = table.by_name.find(name);
            if (it != table.by_name.end())
            {
                entry.equa.push_back(it->second);
            }
        }
        for (auto item : pending_corr[i])
        {
            const auto it = table.by_name.find(item.first);
            if (it == table.by_name.end())
            {
                continue;
            }
            CorrEntry ce = item.second;
            ce.target_idx = it->second;
            if (ce.type == 2)
            {
                if (ce.bl < 0) ce.bl = table.default_BL;
                if (ce.blf < 0) ce.blf = table.default_BLF;
                if (ce.cba < 0) ce.cba = table.default_BA_CTR;
                if (ce.cbaf < 0) ce.cbaf = table.default_BAF_CTR;
                if (ce.ba < 0) ce.ba = table.default_BA;
                if (ce.baf < 0) ce.baf = table.default_BAF;
                if (ce.ctor < 0) ce.ctor = table.default_TOR_CTR;
                if (ce.tor < 0) ce.tor = table.default_TOR;
                if (ce.improper < 0) ce.improper = 0.0;
                ce.ctor = ce.ctor * table.default_FRACT1 +
                          ce.improper * table.default_FRACT2;
            }
            entry.corr.push_back(ce);
        }
    }
    return table;
}

BlbaTables Load_Blba(const std::string& path)
{
    BlbaTables blba;
    for (const auto& raw : Split_Lines(Read_File(path)))
    {
        const std::string line = Trim(raw);
        if (line.empty())
        {
            continue;
        }
        const auto words = Words(line);
        if (words.empty() || words[0] != "PARM")
        {
            continue;
        }
        if (words[1] == "PC")
        {
            blba.pc = Parse_Double(words[2]);
        }
        else if (words[1] == "BL")
        {
            const int z1 = Parse_Int(words[3]);
            const int z2 = Parse_Int(words[5]);
            const auto value =
                std::make_pair(Parse_Double(words[6]), Parse_Double(words[7]));
            blba.bl[{z1, z2}] = value;
            blba.bl[{z2, z1}] = value;
        }
        else if (words[1] == "BA")
        {
            blba.ba[Parse_Int(words[3])] =
                {Parse_Double(words[4]), Parse_Double(words[5])};
        }
    }
    return blba;
}

GaffTables Load_Gaff_Dat(const std::string& path)
{
    GaffTables g;
    const auto lines = Split_Lines(Read_File(path));
    std::size_t idx = 1;
    while (idx < lines.size() && !Trim(lines[idx]).empty())
    {
        const auto words = Words(lines[idx]);
        g.atoms.push_back(
            {words[0], Parse_Double(words[1]),
             words.size() >= 3 ? Parse_Double(words[2]) : 0.0});
        ++idx;
    }
    ++idx;
    if (idx < lines.size())
    {
        ++idx;
    }
    while (idx < lines.size() && Trim(lines[idx]).empty())
    {
        ++idx;
    }
    while (idx < lines.size() && !Trim(lines[idx]).empty())
    {
        auto split = Split_Atoms_Words(lines[idx], 5);
        g.bonds.push_back({split.first[0], split.first[1],
                           Parse_Double(split.second[0]),
                           Parse_Double(split.second[1])});
        ++idx;
    }
    ++idx;
    while (idx < lines.size() && !Trim(lines[idx]).empty())
    {
        auto split = Split_Atoms_Words(lines[idx], 8);
        g.angles.push_back({split.first[0], split.first[1], split.first[2],
                            Parse_Double(split.second[0]),
                            Parse_Double(split.second[1])});
        ++idx;
    }
    ++idx;
    std::vector<std::string> last_atoms;
    while (idx < lines.size() && !Trim(lines[idx]).empty())
    {
        auto split = Split_Atoms_Words(lines[idx], 11, &last_atoms);
        last_atoms = split.first;
        g.torsions.push_back(
            {split.first[0], split.first[1], split.first[2], split.first[3],
             Parse_Int(split.second[0]), Parse_Double(split.second[1]),
             Parse_Double(split.second[2]), Parse_Double(split.second[3]),
             Count_X(split.first)});
        ++idx;
    }
    ++idx;
    std::stable_sort(g.torsions.begin(), g.torsions.end(),
                     [](const GaffTorsion& a, const GaffTorsion& b) {
                         return (a.num_X > 0) < (b.num_X > 0);
                     });
    while (idx < lines.size() && !Trim(lines[idx]).empty())
    {
        auto split = Split_Atoms_Words(lines[idx], 11);
        int num_X = Count_X(split.first);
        std::string a1 = split.first[0];
        std::string a2 = split.first[1];
        std::string a3 = split.first[2];
        std::string a4 = split.first[3];
        if (num_X == 0)
        {
            std::vector<std::string> outer = {a1, a2, a4};
            std::sort(outer.begin(), outer.end());
            a1 = outer[0];
            a2 = outer[1];
            a4 = outer[2];
        }
        else
        {
            if (a1 == "X")
            {
                if (a2 != "X" && a4 != "X" && a2 > a4) std::swap(a2, a4);
            }
            else if (a2 == "X")
            {
                if (a1 != "X" && a4 != "X" && a1 > a4) std::swap(a1, a4);
            }
            else if (a4 == "X")
            {
                if (a1 != "X" && a2 != "X" && a1 > a2) std::swap(a1, a2);
            }
        }
        g.impropers.push_back({a1, a2, a3, a4,
                               Parse_Double(split.second[0]),
                               Parse_Double(split.second[1]),
                               Parse_Double(split.second[2]), num_X});
        ++idx;
    }
    ++idx;
    while (idx < lines.size() && !Trim(lines[idx]).empty())
    {
        ++idx;
    }
    ++idx;
    while (idx < lines.size() && !Trim(lines[idx]).empty())
    {
        const auto words = Words(lines[idx]);
        for (std::size_t i = 1; i < words.size(); ++i)
        {
            g.vdw_equivalents[words[i]] = words[0];
        }
        ++idx;
    }
    while (idx < lines.size() && !Starts_With(lines[idx], "MOD4"))
    {
        ++idx;
    }
    ++idx;
    while (idx < lines.size() && !Trim(lines[idx]).empty())
    {
        const auto words = Words(lines[idx]);
        g.vdws.push_back(
            {words[0], Parse_Double(words[1]), Parse_Double(words[2])});
        ++idx;
    }
    std::map<std::string, GaffVdw> primary;
    for (const auto& v : g.vdws)
    {
        primary[v.name] = v;
    }
    for (const auto& item : g.vdw_equivalents)
    {
        if (primary.find(item.first) == primary.end() &&
            primary.find(item.second) != primary.end())
        {
            GaffVdw v = primary[item.second];
            v.name = item.first;
            g.vdws.push_back(v);
        }
    }
    return g;
}

double Equtype_Penalty(const ParmTable& pt, int c1, int c2, int s1, int s2)
{
    const int n1 = pt.entries[c1].equtype_flag;
    const int n2 = pt.entries[c2].equtype_flag;
    if (n1 == 0 && n2 == 0)
    {
        return 0.0;
    }
    const int n3 = pt.entries[s1].equtype_flag;
    const int n4 = pt.entries[s2].equtype_flag;
    const int tn1 = std::abs(n1) + std::abs(n2);
    const int tn2 = std::abs(n3) + std::abs(n4);
    if ((tn1 == 3) != (tn2 == 3))
    {
        return pt.wt_EQUTYPE;
    }
    if (n1 + n2 == 0 && n3 < 0 && n4 < 0)
    {
        return 0.5 * pt.wt_EQUTYPE;
    }
    return 0.0;
}

class Engine
{
public:
    Engine(Mol2Molecule mol,
           ParmTable pt,
           GaffTables gaff,
           BlbaTables blba,
           bool allparm,
           bool output_improper)
        : mol_(std::move(mol)),
          pt_(std::move(pt)),
          gaff_(std::move(gaff)),
          blba_(std::move(blba)),
          allparm_(allparm),
          output_improper_(output_improper)
    {
        bondparm_ = gaff_.bonds;
        angleparm_ = gaff_.angles;
        torsionparm_ = gaff_.torsions;
        improperparm_ = gaff_.impropers;
        vdwparm_ = gaff_.vdws;
        bond_snapshot_ = bondparm_.size();
        angle_snapshot_ = angleparm_.size();
        torsion_snapshot_ = torsionparm_.size();
        improper_snapshot_ = improperparm_.size();
        vdw_snapshot_ = vdwparm_.size();
        for (const auto& atom : gaff_.atoms)
        {
            gaff_atom_by_name_[atom.name] = atom;
        }
    }

    std::vector<std::string> Run()
    {
        Assign_Parmid();
        Improper_Id2();
        std::vector<std::string> out;
        Extend(out, Chk_Atomtype());
        Extend(out, Chk_Bond());
        Extend(out, Chk_Angle());
        Extend(out, Chk_Torsion());
        Extend(out, Chk_Improper());
        Extend(out, Chk_Vdw());
        return out;
    }

private:
    static void Extend(std::vector<std::string>& target,
                       const std::vector<std::string>& source)
    {
        target.insert(target.end(), source.begin(), source.end());
    }

    void Assign_Parmid()
    {
        for (const auto& atom : mol_.atoms)
        {
            const auto it = pt_.by_name.find(atom.gaff_type);
            if (it != pt_.by_name.end())
            {
                parmids_.push_back(it->second);
                continue;
            }
            ParmEntry entry;
            entry.atomtype = atom.gaff_type;
            entry.mass = gaff_atom_by_name_.count(atom.gaff_type)
                             ? gaff_atom_by_name_[atom.gaff_type].mass
                             : 0.0;
            entry.atomic_num = atom.atomic_num;
            entry.equa = {static_cast<int>(pt_.entries.size())};
            CorrEntry ce;
            ce.target_idx = static_cast<int>(pt_.entries.size());
            ce.type = 0;
            entry.corr = {ce};
            pt_.by_name[entry.atomtype] = static_cast<int>(pt_.entries.size());
            pt_.entries.push_back(entry);
            parmids_.push_back(pt_.by_name[atom.gaff_type]);
        }
    }

    void Improper_Id2()
    {
        const auto nbrs = mol_.Neighbors();
        for (std::size_t i = 0; i < mol_.atoms.size(); ++i)
        {
            const int pid = parmids_[i];
            if (pt_.entries[pid].improper_flag != 1 || nbrs[i].size() < 3)
            {
                continue;
            }
            impropers_.push_back(
                {nbrs[i][0], nbrs[i][1], static_cast<int>(i), nbrs[i][2]});
        }
    }

    std::vector<std::string> Chk_Atomtype()
    {
        std::vector<std::string> lines = {"Remark line goes here", "MASS"};
        std::set<std::string> emitted;
        for (std::size_t i = 0; i < mol_.atoms.size(); ++i)
        {
            const std::string& type = mol_.atoms[i].gaff_type;
            if (!emitted.insert(type).second)
            {
                continue;
            }
            const auto ga = gaff_atom_by_name_.find(type);
            if (ga != gaff_atom_by_name_.end())
            {
                if (allparm_)
                {
                    lines.push_back(Left(type, 2) + " " +
                                    Format_Fixed(ga->second.mass, -8, 3) +
                                    "   " +
                                    Format_Fixed(ga->second.pol, 8, 3));
                }
                continue;
            }
            const std::string sub = Find_Atomtype_Substitute(parmids_[i]);
            const double mass = pt_.entries[parmids_[i]].mass;
            if (!sub.empty())
            {
                const double pol = gaff_atom_by_name_[sub].pol;
                gaff_atom_by_name_[type] = {type, mass, pol};
                lines.push_back(Left(type, 2) + " " +
                                Format_Fixed(mass, -8, 3) + "   " +
                                Format_Fixed(pol, 8, 3) +
                                "               same as " + sub + " ");
            }
            else
            {
                gaff_atom_by_name_[type] = {type, mass, 0.0};
                lines.push_back(Left(type, 2) + " " +
                                Format_Fixed(mass, -8, 3) +
                                "      0.000               ATTN, no polarizability parameter");
            }
        }
        return lines;
    }

    std::string Find_Atomtype_Substitute(int pid)
    {
        const auto& entry = pt_.entries[pid];
        for (int sub_idx : entry.equa)
        {
            const std::string& name = pt_.entries[sub_idx].atomtype;
            if (gaff_atom_by_name_.find(name) != gaff_atom_by_name_.end())
            {
                return name;
            }
        }
        for (const auto& ce : entry.corr)
        {
            if (ce.type <= 1)
            {
                continue;
            }
            const std::string& name = pt_.entries[ce.target_idx].atomtype;
            if (gaff_atom_by_name_.find(name) != gaff_atom_by_name_.end())
            {
                return name;
            }
        }
        return "";
    }

    std::vector<std::string> Chk_Bond()
    {
        std::vector<std::string> lines = {"", "BOND"};
        std::set<std::pair<std::string, std::string>> seen;
        for (const auto& bond : mol_.bonds)
        {
            std::string n1 = mol_.atoms[bond.first].gaff_type;
            std::string n2 = mol_.atoms[bond.second].gaff_type;
            if (n1 > n2)
            {
                std::swap(n1, n2);
            }
            if (!seen.insert({n1, n2}).second)
            {
                continue;
            }
            const auto line =
                Chk_One_Bond(n1, n2, parmids_[bond.first], parmids_[bond.second]);
            if (!line.empty())
            {
                lines.push_back(line);
            }
        }
        return lines;
    }

    GaffBond* Bond_Lookup(const std::string& n1,
                          const std::string& n2,
                          std::size_t limit)
    {
        for (std::size_t k = 0; k < limit; ++k)
        {
            auto& b = bondparm_[k];
            if ((b.name1 == n1 && b.name2 == n2) ||
                (b.name1 == n2 && b.name2 == n1))
            {
                return &b;
            }
        }
        return nullptr;
    }

    std::string Chk_One_Bond(const std::string& n1,
                             const std::string& n2,
                             int pid1,
                             int pid2)
    {
        if (auto* b = Bond_Lookup(n1, n2, bond_snapshot_))
        {
            if (allparm_)
            {
                return Left(n1, 2) + "-" + Left(n2, 2) +
                       Format_Fixed(b->force, 8, 2) +
                       Format_Fixed(b->length, 8, 3);
            }
            return "";
        }
        const auto& e1 = pt_.entries[pid1].equa;
        const auto& e2 = pt_.entries[pid2].equa;
        for (std::size_t m = 0; m < e1.size(); ++m)
        {
            for (std::size_t n = 0; n < e2.size(); ++n)
            {
                if (m == 0 && n == 0) continue;
                const auto& t1 = pt_.entries[e1[m]].atomtype;
                const auto& t2 = pt_.entries[e2[n]].atomtype;
                if (auto* hit = Bond_Lookup(t1, t2, bond_snapshot_))
                {
                    const GaffBond copy = *hit;
                    bondparm_.push_back({n1, n2, copy.force, copy.length});
                    return Left(n1, 2) + "-" + Left(n2, 2) +
                           Format_Fixed(copy.force, 8, 2) +
                           Format_Fixed(copy.length, 8, 3) +
                           "       same as " + Right(copy.name1, 2) + "-" +
                           Right(copy.name2, 2) + ", penalty score=  0.0";
                }
            }
        }
        double best_score = std::numeric_limits<double>::infinity();
        GaffBond* best_hit = nullptr;
        for (const auto& ce1 : pt_.entries[pid1].corr)
        {
            for (const auto& ce2 : pt_.entries[pid2].corr)
            {
                if (ce1.type <= 1 && ce2.type <= 1) continue;
                const auto& t1 = pt_.entries[ce1.target_idx].atomtype;
                const auto& t2 = pt_.entries[ce2.target_idx].atomtype;
                auto* hit = Bond_Lookup(t1, t2, bond_snapshot_);
                if (!hit) continue;
                double score = ce1.bl * pt_.wt_BL + ce1.blf * pt_.wt_BLF +
                               ce2.bl * pt_.wt_BL + ce2.blf * pt_.wt_BLF;
                if (pt_.entries[ce1.target_idx].group_id !=
                    pt_.entries[ce2.target_idx].group_id)
                {
                    score += pt_.wt_GROUP;
                }
                if (score < best_score)
                {
                    best_score = score;
                    best_hit = hit;
                }
            }
        }
        if (best_hit)
        {
            const GaffBond copy = *best_hit;
            bondparm_.push_back({n1, n2, copy.force, copy.length});
            return Left(n1, 2) + "-" + Left(n2, 2) +
                   Format_Fixed(copy.force, 8, 2) +
                   Format_Fixed(copy.length, 8, 3) +
                   "       same as " + Right(copy.name1, 2) + "-" +
                   Right(copy.name2, 2) + ", penalty score=" +
                   Format_Fixed(best_score, 5, 1);
        }
        bondparm_.push_back({n1, n2, 0.0, 0.0});
        return Left(n1, 2) + "-" + Left(n2, 2) +
               "    0.00   0.000       ATTN, need revision";
    }

    std::vector<std::string> Chk_Angle()
    {
        std::vector<std::string> lines = {"", "ANGLE"};
        std::set<std::tuple<std::string, std::string, std::string>> seen;
        const auto nbrs = mol_.Neighbors();
        for (std::size_t j = 0; j < mol_.atoms.size(); ++j)
        {
            for (std::size_t ii = 0; ii < nbrs[j].size(); ++ii)
            {
                for (std::size_t kk = ii + 1; kk < nbrs[j].size(); ++kk)
                {
                    const int i = nbrs[j][ii];
                    const int k = nbrs[j][kk];
                    if (i == k) continue;
                    std::string n1 = mol_.atoms[i].gaff_type;
                    std::string n2 = mol_.atoms[j].gaff_type;
                    std::string n3 = mol_.atoms[k].gaff_type;
                    if (n1 > n3)
                    {
                        std::swap(n1, n3);
                    }
                    auto key = std::make_tuple(n1, n2, n3);
                    if (!seen.insert(key).second)
                    {
                        continue;
                    }
                    const auto line = Chk_One_Angle(n1, n2, n3, parmids_[i],
                                                    parmids_[j], parmids_[k]);
                    if (!line.empty())
                    {
                        lines.push_back(line);
                    }
                }
            }
        }
        return lines;
    }

    GaffAngle* Angle_Lookup(const std::string& n1,
                            const std::string& n2,
                            const std::string& n3,
                            std::size_t limit)
    {
        for (std::size_t k = 0; k < limit; ++k)
        {
            auto& a = angleparm_[k];
            if (a.name2 != n2)
            {
                continue;
            }
            if ((a.name1 == n1 && a.name3 == n3) ||
                (a.name1 == n3 && a.name3 == n1))
            {
                return &a;
            }
        }
        return nullptr;
    }

    std::string Chk_One_Angle(const std::string& n1,
                              const std::string& n2,
                              const std::string& n3,
                              int pid1,
                              int pid2,
                              int pid3)
    {
        if (auto* a = Angle_Lookup(n1, n2, n3, angle_snapshot_))
        {
            if (allparm_)
            {
                return Left(n1, 2) + "-" + Left(n2, 2) + "-" + Left(n3, 2) +
                       Format_Fixed(a->force, 9, 3) +
                       Format_Fixed(a->angle, 12, 3);
            }
            return "";
        }
        const auto& e1 = pt_.entries[pid1].equa;
        const auto& e2 = pt_.entries[pid2].equa;
        const auto& e3 = pt_.entries[pid3].equa;
        for (std::size_t m = 0; m < e1.size(); ++m)
        {
            for (std::size_t n = 0; n < e2.size(); ++n)
            {
                for (std::size_t o = 0; o < e3.size(); ++o)
                {
                    if (m == 0 && n == 0 && o == 0) continue;
                    const auto& t1 = pt_.entries[e1[m]].atomtype;
                    const auto& t2 = pt_.entries[e2[n]].atomtype;
                    const auto& t3 = pt_.entries[e3[o]].atomtype;
                    if (auto* hit = Angle_Lookup(t1, t2, t3, angle_snapshot_))
                    {
                        const GaffAngle copy = *hit;
                        angleparm_.push_back(
                            {n1, n2, n3, copy.force, copy.angle});
                        return Left(n1, 2) + "-" + Left(n2, 2) + "-" +
                               Left(n3, 2) + Format_Fixed(copy.force, 9, 3) +
                               Format_Fixed(copy.angle, 12, 3) +
                               "   same as " + Left(copy.name1, 2) + "-" +
                               Left(copy.name2, 2) + "-" +
                               Left(copy.name3, 2) +
                               ", penalty score=  0.0";
                    }
                }
            }
        }
        double best_score = std::numeric_limits<double>::infinity();
        GaffAngle* best_hit = nullptr;
        for (const auto& ce1 : pt_.entries[pid1].corr)
        {
            for (const auto& ce2 : pt_.entries[pid2].corr)
            {
                for (const auto& ce3 : pt_.entries[pid3].corr)
                {
                    if (ce1.type <= 1 && ce2.type <= 1 && ce3.type <= 1)
                        continue;
                    const auto& t1 = pt_.entries[ce1.target_idx].atomtype;
                    const auto& t2 = pt_.entries[ce2.target_idx].atomtype;
                    const auto& t3 = pt_.entries[ce3.target_idx].atomtype;
                    auto* hit = Angle_Lookup(t1, t2, t3, angle_snapshot_);
                    if (!hit) continue;
                    const double s1 =
                        ce1.ba * pt_.wt_BA + ce1.baf * pt_.wt_BAF;
                    const double s2 =
                        (ce2.cba * pt_.wt_BA + ce2.cbaf * pt_.wt_BAF) *
                        pt_.wt_BA_CTR;
                    const double s3 =
                        ce3.ba * pt_.wt_BA + ce3.baf * pt_.wt_BAF;
                    double score = s1 + s2 + s3 + pt_.wt_GROUP;
                    const int g1 = pt_.entries[ce1.target_idx].group_id;
                    const int g2 = pt_.entries[ce2.target_idx].group_id;
                    const int g3 = pt_.entries[ce3.target_idx].group_id;
                    if (g1 == g2 && g2 == g3)
                    {
                        score -= pt_.wt_GROUP;
                    }
                    if (score > pt_.threshold_BA) continue;
                    if (score < best_score)
                    {
                        best_score = score;
                        best_hit = hit;
                    }
                }
            }
        }
        if (best_hit)
        {
            const GaffAngle copy = *best_hit;
            angleparm_.push_back(
                {n1, n2, n3, copy.force, copy.angle});
            return Left(n1, 2) + "-" + Left(n2, 2) + "-" + Left(n3, 2) +
                   Format_Fixed(copy.force, 9, 3) +
                   Format_Fixed(copy.angle, 12, 3) +
                   "   same as " + Left(copy.name1, 2) + "-" +
                   Left(copy.name2, 2) + "-" + Left(copy.name3, 2) +
                   ", penalty score=" + Format_Fixed(best_score, 5, 1);
        }
        const auto emp = Empangle(n1, n2, n3, pid1, pid2, pid3);
        if (!emp.comment.empty())
        {
            angleparm_.push_back({n1, n2, n3, emp.force, emp.angle});
            return Left(n1, 2) + "-" + Left(n2, 2) + "-" + Left(n3, 2) +
                   Format_Fixed(emp.force, 9, 3) +
                   Format_Fixed(emp.angle, 12, 3) + "   " + emp.comment;
        }
        angleparm_.push_back({n1, n2, n3, 0.0, 0.0});
        return Left(n1, 2) + "-" + Left(n2, 2) + "-" + Left(n3, 2) +
               "    0.000       0.000   ATTN, need revision";
    }

    struct EmpangleResult
    {
        double force = 0.0;
        double angle = 0.0;
        std::string comment;
    };

    EmpangleResult Empangle(const std::string& n1,
                            const std::string& n2,
                            const std::string& n3,
                            int pid1,
                            int pid2,
                            int pid3)
    {
        if (auto value = Empangle_Compute(n1, n2, n3, pid1, pid2, pid3);
            value.first >= 0.0)
        {
            return {value.first, value.second,
                    "Calculated with empirical approach for " + n1 + "-" +
                        n2 + "-" + n3};
        }
        const auto& e1 = pt_.entries[pid1].equa;
        const auto& e2 = pt_.entries[pid2].equa;
        const auto& e3 = pt_.entries[pid3].equa;
        for (std::size_t m = 0; m < e1.size(); ++m)
        {
            for (std::size_t n = 0; n < e2.size(); ++n)
            {
                for (std::size_t o = 0; o < e3.size(); ++o)
                {
                    if (m == 0 && n == 0 && o == 0) continue;
                    const auto& t1 = pt_.entries[e1[m]].atomtype;
                    const auto& t2 = pt_.entries[e2[n]].atomtype;
                    const auto& t3 = pt_.entries[e3[o]].atomtype;
                    auto value = Empangle_Compute(t1, t2, t3, e1[m], e2[n],
                                                  e3[o]);
                    if (value.first >= 0.0)
                    {
                        return {value.first, value.second,
                                "Calculated using " + Right(t1, 2) + "-" +
                                    Right(t2, 2) + "-" + Right(t3, 2) +
                                    ", penalty score=  0.0"};
                    }
                }
            }
        }
        double best_score = std::numeric_limits<double>::infinity();
        EmpangleResult best;
        for (const auto& ce1 : pt_.entries[pid1].corr)
        {
            for (const auto& ce2 : pt_.entries[pid2].corr)
            {
                for (const auto& ce3 : pt_.entries[pid3].corr)
                {
                    if (ce1.type <= 1 && ce2.type <= 1 && ce3.type <= 1)
                        continue;
                    double score = ce1.ba * pt_.wt_BA + ce1.baf * pt_.wt_BAF +
                                   (ce2.cba * pt_.wt_BA +
                                    ce2.cbaf * pt_.wt_BAF) *
                                       pt_.wt_BA_CTR +
                                   ce3.ba * pt_.wt_BA +
                                   ce3.baf * pt_.wt_BAF + pt_.wt_GROUP;
                    const int g1 = pt_.entries[ce1.target_idx].group_id;
                    const int g2 = pt_.entries[ce2.target_idx].group_id;
                    const int g3 = pt_.entries[ce3.target_idx].group_id;
                    if (g1 == g2 && g2 == g3)
                    {
                        score -= pt_.wt_GROUP;
                    }
                    if (score >= best_score) continue;
                    const auto& t1 = pt_.entries[ce1.target_idx].atomtype;
                    const auto& t2 = pt_.entries[ce2.target_idx].atomtype;
                    const auto& t3 = pt_.entries[ce3.target_idx].atomtype;
                    auto value = Empangle_Compute(t1, t2, t3, ce1.target_idx,
                                                  ce2.target_idx,
                                                  ce3.target_idx);
                    if (value.first < 0.0) continue;
                    best_score = score;
                    best = {value.first, value.second,
                            "Calculated using " + Right(t1, 2) + "-" +
                                Right(t2, 2) + "-" + Right(t3, 2) +
                                ", penalty score=" +
                                Format_Fixed(score, 5, 1)};
                }
            }
        }
        return best;
    }

    std::pair<double, double> Empangle_Compute(const std::string& t1,
                                               const std::string& t2,
                                               const std::string& t3,
                                               int p1,
                                               int p2,
                                               int p3)
    {
        auto* a112 = Angle_Lookup(t1, t2, t1, angleparm_.size());
        auto* a223 = Angle_Lookup(t3, t2, t3, angleparm_.size());
        if (!a112 || !a223) return {-1.0, 0.0};
        auto* b12 = Bond_Lookup(t1, t2, bondparm_.size());
        auto* b23 = Bond_Lookup(t2, t3, bondparm_.size());
        if (!b12 || !b23) return {-1.0, 0.0};
        const auto z1 = blba_.ba.find(pt_.entries[p1].atomic_num);
        const auto z2 = blba_.ba.find(pt_.entries[p2].atomic_num);
        const auto z3 = blba_.ba.find(pt_.entries[p3].atomic_num);
        if (z1 == blba_.ba.end() || z2 == blba_.ba.end() ||
            z3 == blba_.ba.end())
        {
            return {-1.0, 0.0};
        }
        const double r1 = b12->length;
        const double r2 = b23->length;
        const double d = (r1 - r2) * (r1 - r2) / ((r1 + r2) * (r1 + r2));
        const double theta = 0.5 * (a112->angle + a223->angle);
        double k = 143.9 * z1->second.second * z2->second.first *
                   z3->second.second * std::exp(-2.0 * d) / (r1 + r2);
        k /= std::sqrt(theta * std::acos(-1.0) / 180.0);
        return {k, theta};
    }

    std::vector<std::string> Chk_Torsion()
    {
        std::vector<std::string> lines = {"", "DIHE"};
        std::set<std::tuple<std::string, std::string, std::string, std::string>>
            seen;
        const auto nbrs = mol_.Neighbors();
        for (std::size_t j = 0; j < mol_.atoms.size(); ++j)
        {
            for (int k : nbrs[j])
            {
                if (static_cast<int>(j) > k) continue;
                for (int i : nbrs[j])
                {
                    if (i == k) continue;
                    for (int l : nbrs[k])
                    {
                        if (l == static_cast<int>(j) || i == l) continue;
                        auto names = Torsion_Normalize(i, j, k, l);
                        if (!seen.insert(names).second) continue;
                        auto out = Chk_One_Torsion(
                            std::get<0>(names), std::get<1>(names),
                            std::get<2>(names), std::get<3>(names),
                            parmids_[i], parmids_[j], parmids_[k],
                            parmids_[l]);
                        Extend(lines, out);
                    }
                }
            }
        }
        return lines;
    }

    std::tuple<std::string, std::string, std::string, std::string>
    Torsion_Normalize(int i, std::size_t j, int k, int l)
    {
        std::string n1 = mol_.atoms[i].gaff_type;
        std::string n2 = mol_.atoms[j].gaff_type;
        std::string n3 = mol_.atoms[k].gaff_type;
        std::string n4 = mol_.atoms[l].gaff_type;
        if (n2 > n3)
        {
            return {n4, n3, n2, n1};
        }
        if (n2 == n3 && n1 > n4)
        {
            return {n4, n2, n3, n1};
        }
        return {n1, n2, n3, n4};
    }

    int Torsion_Lookup_Specific(const std::string& n1,
                                const std::string& n2,
                                const std::string& n3,
                                const std::string& n4,
                                std::size_t limit)
    {
        for (std::size_t k = 0; k < limit; ++k)
        {
            const auto& t = torsionparm_[k];
            if (t.num_X > 0) continue;
            if (t.name1 == n1 && t.name2 == n2 && t.name3 == n3 &&
                t.name4 == n4)
            {
                return static_cast<int>(k);
            }
            if (t.name1 == n4 && t.name2 == n3 && t.name3 == n2 &&
                t.name4 == n1)
            {
                return static_cast<int>(k);
            }
        }
        return -1;
    }

    int Torsion_Lookup_General(const std::string& n2,
                               const std::string& n3,
                               std::size_t limit)
    {
        for (std::size_t k = 0; k < limit; ++k)
        {
            const auto& t = torsionparm_[k];
            if (t.num_X == 0 || t.name1 != "X" || t.name4 != "X")
            {
                continue;
            }
            if ((t.name2 == n2 && t.name3 == n3) ||
                (t.name2 == n3 && t.name3 == n2))
            {
                return static_cast<int>(k);
            }
        }
        return -1;
    }

    std::string Format_Torsion(const std::string& n1,
                               const std::string& n2,
                               const std::string& n3,
                               const std::string& n4,
                               const GaffTorsion& t,
                               const std::string& comment)
    {
        return Left(n1, 2) + "-" + Left(n2, 2) + "-" + Left(n3, 2) + "-" +
               Left(n4, 2) + Format_Int(t.mul, 4) +
               Format_Fixed(t.force, 9, 3) + Format_Fixed(t.phase, 14, 3) +
               Format_Fixed(t.fterm, 16, 3) + "      " + comment;
    }

    std::vector<std::string> Emit_Torsion_Chain(int start_idx,
                                                const std::string& n1,
                                                const std::string& n2,
                                                const std::string& n3,
                                                const std::string& n4,
                                                const std::string& first_comment,
                                                const std::string& final_comment)
    {
        std::vector<std::string> lines;
        for (std::size_t idx = static_cast<std::size_t>(start_idx);
             idx < torsionparm_.size(); ++idx)
        {
            const auto& t = torsionparm_[idx];
            if (t.fterm < 0)
            {
                lines.push_back(Format_Torsion(
                    n1, n2, n3, n4, t,
                    first_comment.empty() ? final_comment : first_comment));
            }
            else
            {
                lines.push_back(
                    Format_Torsion(n1, n2, n3, n4, t, final_comment));
                torsionparm_.push_back({n1, n2, n3, n4, t.mul, t.force,
                                        t.phase, t.fterm, 0});
                break;
            }
        }
        return lines;
    }

    std::vector<std::string> Chk_One_Torsion(const std::string& n1,
                                             const std::string& n2,
                                             const std::string& n3,
                                             const std::string& n4,
                                             int pid1,
                                             int pid2,
                                             int pid3,
                                             int pid4)
    {
        int idx =
            Torsion_Lookup_Specific(n1, n2, n3, n4, torsion_snapshot_);
        if (idx >= 0)
        {
            return allparm_ ? Emit_Torsion_Chain(idx, n1, n2, n3, n4, "", "")
                            : std::vector<std::string>();
        }
        if (Torsion_Lookup_General(n2, n3, torsion_snapshot_) >= 0)
        {
            return {};
        }
        const auto& e1 = pt_.entries[pid1].equa;
        const auto& e2 = pt_.entries[pid2].equa;
        const auto& e3 = pt_.entries[pid3].equa;
        const auto& e4 = pt_.entries[pid4].equa;
        for (std::size_t m = 0; m < e1.size(); ++m)
        {
            for (std::size_t n = 0; n < e2.size(); ++n)
            {
                for (std::size_t p = 0; p < e3.size(); ++p)
                {
                    for (std::size_t q = 0; q < e4.size(); ++q)
                    {
                        if (m == 0 && n == 0 && p == 0 && q == 0) continue;
                        const auto& t1 = pt_.entries[e1[m]].atomtype;
                        const auto& t2 = pt_.entries[e2[n]].atomtype;
                        const auto& t3 = pt_.entries[e3[p]].atomtype;
                        const auto& t4 = pt_.entries[e4[q]].atomtype;
                        int hit = Torsion_Lookup_Specific(
                            t1, t2, t3, t4, torsion_snapshot_);
                        if (hit >= 0)
                        {
                            const std::string same =
                                "same as " + Left(t1, 2) + "-" + Left(t2, 2) +
                                "-" + Left(t3, 2) + "-" + Left(t4, 2);
                            return Emit_Torsion_Chain(
                                hit, n1, n2, n3, n4, same,
                                same + ", penalty score=  0.0");
                        }
                    }
                }
            }
        }
        for (std::size_t m = 0; m < e2.size(); ++m)
        {
            for (std::size_t p = 0; p < e3.size(); ++p)
            {
                if (m == 0 && p == 0) continue;
                const auto& t2 = pt_.entries[e2[m]].atomtype;
                const auto& t3 = pt_.entries[e3[p]].atomtype;
                int hit = Torsion_Lookup_General(t2, t3, torsion_snapshot_);
                if (hit >= 0)
                {
                    const auto& th = torsionparm_[hit];
                    const std::string same =
                        "same as X -" + Left(th.name2, 2) + "-" +
                        Left(th.name3, 2) + "-X ";
                    return Emit_Torsion_Chain(
                        hit, n1, n2, n3, n4, same,
                        same + ", penalty score=  0.0");
                }
            }
        }
        const auto& c1 = pt_.entries[pid1].corr;
        const auto& c2 = pt_.entries[pid2].corr;
        const auto& c3 = pt_.entries[pid3].corr;
        const auto& c4 = pt_.entries[pid4].corr;
        double best_score = std::numeric_limits<double>::infinity();
        int best_idx = -1;
        for (const auto& ce1 : c1)
            for (const auto& ce2 : c2)
                for (const auto& ce3 : c3)
                    for (const auto& ce4 : c4)
                    {
                        if (ce1.type <= 1 && ce2.type <= 1 &&
                            ce3.type <= 1 && ce4.type <= 1)
                            continue;
                        const auto& t1 = pt_.entries[ce1.target_idx].atomtype;
                        const auto& t2 = pt_.entries[ce2.target_idx].atomtype;
                        const auto& t3 = pt_.entries[ce3.target_idx].atomtype;
                        const auto& t4 = pt_.entries[ce4.target_idx].atomtype;
                        int hit = Torsion_Lookup_Specific(
                            t1, t2, t3, t4, torsion_snapshot_);
                        if (hit < 0) continue;
                        double score = ce1.tor +
                                       ce2.ctor * pt_.wt_TOR_CTR +
                                       ce3.ctor * pt_.wt_TOR_CTR + ce4.tor +
                                       pt_.wt_GROUP;
                        score += Equtype_Penalty(pt_, pid2, pid3,
                                                  ce2.target_idx,
                                                  ce3.target_idx);
                        const int g1 = pt_.entries[ce1.target_idx].group_id;
                        const int g2 = pt_.entries[ce2.target_idx].group_id;
                        const int g3 = pt_.entries[ce3.target_idx].group_id;
                        const int g4 = pt_.entries[ce4.target_idx].group_id;
                        if (g1 == g2 && g2 == g3 && g3 == g4)
                        {
                            score -= pt_.wt_GROUP;
                        }
                        if (score < best_score)
                        {
                            best_score = score;
                            best_idx = hit;
                        }
                    }
        if (best_idx >= 0)
        {
            const auto& h = torsionparm_[best_idx];
            const std::string same = "same as " + Left(h.name1, 2) + "-" +
                                     Left(h.name2, 2) + "-" +
                                     Left(h.name3, 2) + "-" +
                                     Left(h.name4, 2);
            return Emit_Torsion_Chain(best_idx, n1, n2, n3, n4, same,
                                      same + ", penalty score=" +
                                          Format_Fixed(best_score, 5, 1));
        }
        best_score = std::numeric_limits<double>::infinity();
        best_idx = -1;
        for (const auto& ce2 : c2)
            for (const auto& ce3 : c3)
            {
                if (ce2.type <= 1 && ce3.type <= 1) continue;
                const auto& t2 = pt_.entries[ce2.target_idx].atomtype;
                const auto& t3 = pt_.entries[ce3.target_idx].atomtype;
                int hit = Torsion_Lookup_General(t2, t3, torsion_snapshot_);
                if (hit < 0) continue;
                double score = ce2.ctor * pt_.wt_TOR_CTR +
                               ce3.ctor * pt_.wt_TOR_CTR + pt_.wt_GROUP;
                score += Equtype_Penalty(pt_, pid2, pid3, ce2.target_idx,
                                          ce3.target_idx);
                if (pt_.entries[ce2.target_idx].group_id ==
                    pt_.entries[ce3.target_idx].group_id)
                {
                    score -= pt_.wt_GROUP;
                }
                if (score < best_score)
                {
                    best_score = score;
                    best_idx = hit;
                }
            }
        if (best_idx >= 0)
        {
            const auto& h = torsionparm_[best_idx];
            const std::string same = "same as X -" + Left(h.name2, 2) + "-" +
                                     Left(h.name3, 2) + "-X ";
            return Emit_Torsion_Chain(best_idx, n1, n2, n3, n4, same,
                                      same + ", penalty score=" +
                                          Format_Fixed(best_score, 5, 1));
        }
        return {Left(n1, 2) + "-" + Left(n2, 2) + "-" + Left(n3, 2) + "-" +
                Left(n4, 2) +
                "   1    0.000         0.000           0.000      ATTN, need revision"};
    }

    std::vector<std::string> Chk_Improper()
    {
        std::vector<std::string> lines = {"", "IMPROPER"};
        for (const auto& item : impropers_)
        {
            std::string t1 = mol_.atoms[std::get<0>(item)].gaff_type;
            std::string t2 = mol_.atoms[std::get<1>(item)].gaff_type;
            std::string t3 = mol_.atoms[std::get<2>(item)].gaff_type;
            std::string t4 = mol_.atoms[std::get<3>(item)].gaff_type;
            std::vector<std::string> outer = {t1, t2, t4};
            std::sort(outer.begin(), outer.end());
            const auto line = Chk_One_Improper(
                outer[0], outer[1], t3, outer[2], parmids_[std::get<0>(item)],
                parmids_[std::get<1>(item)], parmids_[std::get<2>(item)],
                parmids_[std::get<3>(item)]);
            if (!line.empty())
            {
                lines.push_back(line);
            }
        }
        return lines;
    }

    int Improper_Lookup_Specific(const std::string& n1,
                                 const std::string& n2,
                                 const std::string& n3,
                                 const std::string& n4,
                                 std::size_t limit)
    {
        for (std::size_t k = 0; k < limit; ++k)
        {
            const auto& ip = improperparm_[k];
            if (ip.num_X != 0) continue;
            if (ip.name1 == n1 && ip.name2 == n2 && ip.name3 == n3 &&
                ip.name4 == n4)
            {
                return static_cast<int>(k);
            }
        }
        return -1;
    }

    std::vector<std::pair<int, double>> General_Improper_Matches(
        const std::string& n1,
        const std::string& n2,
        const std::string& n3,
        const std::string& n4,
        std::size_t limit)
    {
        std::vector<std::pair<int, double>> hits;
        for (std::size_t k = 0; k < limit; ++k)
        {
            const auto& ip = improperparm_[k];
            if (ip.num_X == 0) continue;
            if (!(ip.name1 == n1 || ip.name1 == "X")) continue;
            if (!(ip.name2 == n2 || ip.name2 == "X")) continue;
            if (!(ip.name3 == n3 || ip.name3 == "X")) continue;
            if (!(ip.name4 == n4 || ip.name4 == "X")) continue;
            double score = 0.0;
            if (ip.name1 == "X") score += pt_.wt_X;
            if (ip.name2 == "X") score += pt_.wt_X;
            if (ip.name3 == "X") score += pt_.wt_X3;
            if (ip.name4 == "X") score += pt_.wt_X;
            hits.push_back({static_cast<int>(k), score});
        }
        return hits;
    }

    std::pair<int, double> Improper_Match_General(const std::string& n1,
                                                  const std::string& n2,
                                                  const std::string& n3,
                                                  const std::string& n4,
                                                  std::size_t limit)
    {
        auto hits = General_Improper_Matches(n1, n2, n3, n4, limit);
        if (hits.empty()) return {-1, 0.0};
        return *std::min_element(hits.begin(), hits.end(),
                                 [](const auto& a, const auto& b) {
                                     return a.second < b.second;
                                 });
    }

    std::pair<int, double> Improper_Match_General_First(
        const std::string& n1,
        const std::string& n2,
        const std::string& n3,
        const std::string& n4,
        std::size_t limit)
    {
        auto hits = General_Improper_Matches(n1, n2, n3, n4, limit);
        return hits.empty() ? std::make_pair(-1, 0.0) : hits.front();
    }

    std::string Format_Improper(const std::string& n1,
                                const std::string& n2,
                                const std::string& n3,
                                const std::string& n4,
                                const GaffImproper& ip,
                                const std::string& suffix = "")
    {
        return Left(n1, 2) + "-" + Left(n2, 2) + "-" + Left(n3, 2) + "-" +
               Left(n4, 2) + " " + Format_Fixed(ip.force, 11, 1) +
               Format_Fixed(ip.phase, 15, 1) + Format_Fixed(ip.fterm, 12, 1) +
               suffix;
    }

    void Append_Improper_Copy(int src_idx,
                              const std::string& n1,
                              const std::string& n2,
                              const std::string& n3,
                              const std::string& n4)
    {
        const auto& ip = improperparm_[src_idx];
        improperparm_.push_back(
            {n1, n2, n3, n4, ip.force, ip.phase, ip.fterm, 0});
    }

    std::string Chk_One_Improper(const std::string& n1,
                                 const std::string& n2,
                                 const std::string& n3,
                                 const std::string& n4,
                                 int pid1,
                                 int pid2,
                                 int pid3,
                                 int pid4)
    {
        int idx = Improper_Lookup_Specific(n1, n2, n3, n4,
                                           improperparm_.size());
        if (idx >= 0)
        {
            return allparm_ ? Format_Improper(n1, n2, n3, n4,
                                              improperparm_[idx])
                            : "";
        }
        const auto& e1 = pt_.entries[pid1].equa;
        const auto& e2 = pt_.entries[pid2].equa;
        const auto& e3 = pt_.entries[pid3].equa;
        const auto& e4 = pt_.entries[pid4].equa;
        for (std::size_t m = 0; m < e1.size(); ++m)
            for (std::size_t n = 0; n < e2.size(); ++n)
                for (std::size_t p = 0; p < e3.size(); ++p)
                    for (std::size_t q = 0; q < e4.size(); ++q)
                    {
                        if (m == 0 && n == 0 && p == 0 && q == 0) continue;
                        const auto& t1 = pt_.entries[e1[m]].atomtype;
                        const auto& t2 = pt_.entries[e2[n]].atomtype;
                        const auto& t3 = pt_.entries[e3[p]].atomtype;
                        const auto& t4 = pt_.entries[e4[q]].atomtype;
                        int hit = Improper_Lookup_Specific(
                            t1, t2, t3, t4, improper_snapshot_);
                        if (hit >= 0)
                        {
                            const GaffImproper ip = improperparm_[hit];
                            Append_Improper_Copy(hit, n1, n2, n3, n4);
                            if (output_improper_ || allparm_)
                            {
                                return Format_Improper(
                                    n1, n2, n3, n4, ip,
                                    "          Same as " + Left(t1, 2) + "-" +
                                        Left(t2, 2) + "-" + Left(t3, 2) +
                                        "-" + Left(t4, 2) +
                                        ", penalty score=  0.0)");
                            }
                            return "";
                        }
                    }
        auto gen = Improper_Match_General(n1, n2, n3, n4, improper_snapshot_);
        if (gen.first >= 0)
        {
            const GaffImproper ip = improperparm_[gen.first];
            Append_Improper_Copy(gen.first, n1, n2, n3, n4);
            if (output_improper_ || allparm_)
            {
                return Format_Improper(
                    n1, n2, n3, n4, ip,
                    "          Using general improper torsional angle " +
                        Right(ip.name1, 2) + "-" + Right(ip.name2, 2) + "-" +
                        Right(ip.name3, 2) + "-" + Right(ip.name4, 2) +
                        ", penalty score=" + Format_Fixed(gen.second, 5, 1) +
                        ")");
            }
            return "";
        }
        for (std::size_t m = 0; m < e1.size(); ++m)
            for (std::size_t n = 0; n < e2.size(); ++n)
                for (std::size_t p = 0; p < e3.size(); ++p)
                    for (std::size_t q = 0; q < e4.size(); ++q)
                    {
                        if (m == 0 && n == 0 && p == 0 && q == 0) continue;
                        const auto& t1 = pt_.entries[e1[m]].atomtype;
                        const auto& t2 = pt_.entries[e2[n]].atomtype;
                        const auto& t3 = pt_.entries[e3[p]].atomtype;
                        const auto& t4 = pt_.entries[e4[q]].atomtype;
                        auto first = Improper_Match_General_First(
                            t1, t2, t3, t4, improper_snapshot_);
                        if (first.first < 0) continue;
                        const GaffImproper ip = improperparm_[first.first];
                        Append_Improper_Copy(first.first, n1, n2, n3, n4);
                        if (output_improper_ || allparm_)
                        {
                            return Format_Improper(
                                n1, n2, n3, n4, ip,
                                "          Same as " + Right(ip.name1, 2) +
                                    "-" + Right(ip.name2, 2) + "-" +
                                    Right(ip.name3, 2) + "-" +
                                    Right(ip.name4, 2) +
                                    ", penalty score=" +
                                    Format_Fixed(first.second, 5, 1) +
                                    " (use general term))");
                        }
                        return "";
                    }
        const auto& c1 = pt_.entries[pid1].corr;
        const auto& c2 = pt_.entries[pid2].corr;
        const auto& c3 = pt_.entries[pid3].corr;
        const auto& c4 = pt_.entries[pid4].corr;
        double best_score = std::numeric_limits<double>::infinity();
        int best_idx = -1;
        for (const auto& ce1 : c1)
            for (const auto& ce2 : c2)
                for (const auto& ce3 : c3)
                    for (const auto& ce4 : c4)
                    {
                        if (ce1.type <= 1 && ce2.type <= 1 &&
                            ce3.type <= 1 && ce4.type <= 1)
                            continue;
                        const auto& t1 = pt_.entries[ce1.target_idx].atomtype;
                        const auto& t2 = pt_.entries[ce2.target_idx].atomtype;
                        const auto& t3 = pt_.entries[ce3.target_idx].atomtype;
                        const auto& t4 = pt_.entries[ce4.target_idx].atomtype;
                        int hit = Improper_Lookup_Specific(
                            t1, t2, t3, t4, improper_snapshot_);
                        if (hit < 0) continue;
                        double score = ce1.improper + ce2.improper +
                                       ce3.improper * pt_.wt_IMPROPER +
                                       ce4.improper + pt_.wt_GROUP;
                        const int g1 = pt_.entries[ce1.target_idx].group_id;
                        const int g2 = pt_.entries[ce2.target_idx].group_id;
                        const int g3 = pt_.entries[ce3.target_idx].group_id;
                        const int g4 = pt_.entries[ce4.target_idx].group_id;
                        if (g1 == g2 && g2 == g3 && g3 == g4)
                        {
                            score -= pt_.wt_GROUP;
                        }
                        if (score < best_score)
                        {
                            best_score = score;
                            best_idx = hit;
                        }
                    }
        if (best_idx >= 0)
        {
            const GaffImproper ip = improperparm_[best_idx];
            Append_Improper_Copy(best_idx, n1, n2, n3, n4);
            if (output_improper_ || allparm_)
            {
                return Format_Improper(
                    n1, n2, n3, n4, ip,
                    "          Same as " + Left(ip.name1, 2) + "-" +
                        Left(ip.name2, 2) + "-" + Left(ip.name3, 2) + "-" +
                        Left(ip.name4, 2) + ", penalty score=" +
                        Format_Fixed(best_score, 5, 1) + ")");
            }
            return "";
        }
        best_score = std::numeric_limits<double>::infinity();
        best_idx = -1;
        for (const auto& ce1 : c1)
            for (const auto& ce2 : c2)
                for (const auto& ce3 : c3)
                    for (const auto& ce4 : c4)
                    {
                        if (ce1.type <= 1 && ce2.type <= 1 &&
                            ce3.type <= 1 && ce4.type <= 1)
                            continue;
                        const auto& t1 = pt_.entries[ce1.target_idx].atomtype;
                        const auto& t2 = pt_.entries[ce2.target_idx].atomtype;
                        const auto& t3 = pt_.entries[ce3.target_idx].atomtype;
                        const auto& t4 = pt_.entries[ce4.target_idx].atomtype;
                        auto first = Improper_Match_General_First(
                            t1, t2, t3, t4, improper_snapshot_);
                        if (first.first < 0) continue;
                        double score = ce1.improper + ce2.improper +
                                       ce3.improper + pt_.wt_IMPROPER +
                                       ce4.improper + pt_.wt_GROUP +
                                       first.second;
                        const int g1 = pt_.entries[ce1.target_idx].group_id;
                        const int g2 = pt_.entries[ce2.target_idx].group_id;
                        const int g3 = pt_.entries[ce3.target_idx].group_id;
                        const int g4 = pt_.entries[ce4.target_idx].group_id;
                        if (g1 == g2 && g2 == g3 && g3 == g4)
                        {
                            score -= pt_.wt_GROUP;
                        }
                        if (score < best_score)
                        {
                            best_score = score;
                            best_idx = first.first;
                        }
                    }
        if (best_idx >= 0)
        {
            const GaffImproper ip = improperparm_[best_idx];
            Append_Improper_Copy(best_idx, n1, n2, n3, n4);
            if (output_improper_ || allparm_)
            {
                return Format_Improper(
                    n1, n2, n3, n4, ip,
                    "          Same as " + Left(ip.name1, 2) + "-" +
                        Left(ip.name2, 2) + "-" + Left(ip.name3, 2) + "-" +
                        Left(ip.name4, 2) + ", penalty score=" +
                        Format_Fixed(best_score, 5, 1) +
                        " (use general term))");
            }
            return "";
        }
        improperparm_.push_back({n1, n2, n3, n4, 1.1, 180.0, 2.0, 0});
        return Format_Improper(n1, n2, n3, n4, improperparm_.back(),
                               "          Using the default value");
    }

    std::vector<std::string> Chk_Vdw()
    {
        std::vector<std::string> lines = {"", "NONBON"};
        std::set<std::string> emitted;
        std::map<std::string, GaffVdw> vdw_by_name;
        for (std::size_t i = 0; i < vdw_snapshot_; ++i)
        {
            vdw_by_name[vdwparm_[i].name] = vdwparm_[i];
        }
        for (std::size_t i = 0; i < mol_.atoms.size(); ++i)
        {
            const auto& type = mol_.atoms[i].gaff_type;
            if (!emitted.insert(type).second) continue;
            const auto it = vdw_by_name.find(type);
            if (it != vdw_by_name.end())
            {
                if (allparm_)
                {
                    lines.push_back("  " + Left(type, 2) +
                                    Format_Fixed(it->second.rstar, 16, 4) +
                                    Format_Fixed(it->second.epsilon, 8, 4));
                }
                continue;
            }
            const std::string sub = Find_Vdw_Substitute(parmids_[i], vdw_by_name);
            if (!sub.empty())
            {
                const auto& vsub = vdw_by_name[sub];
                vdwparm_.push_back({type, vsub.rstar, vsub.epsilon});
                vdw_by_name[type] = vdwparm_.back();
                lines.push_back("  " + Left(type, 2) +
                                Format_Fixed(vsub.rstar, 16, 4) +
                                Format_Fixed(vsub.epsilon, 8, 4) +
                                "             same as " + Left(sub, 3));
            }
            else
            {
                vdwparm_.push_back({type, 0.0, 0.0});
                vdw_by_name[type] = vdwparm_.back();
                lines.push_back(
                    "  " + Left(type, 2) +
                    "          0.0000  0.0000             ATTN, need revision");
            }
        }
        lines.push_back("");
        lines.push_back("");
        return lines;
    }

    std::string Find_Vdw_Substitute(int pid,
                                    const std::map<std::string, GaffVdw>& vdw)
    {
        const auto& entry = pt_.entries[pid];
        for (int sub_idx : entry.equa)
        {
            const auto& name = pt_.entries[sub_idx].atomtype;
            if (vdw.find(name) != vdw.end())
            {
                return name;
            }
        }
        for (const auto& ce : entry.corr)
        {
            if (ce.type <= 1) continue;
            const auto& name = pt_.entries[ce.target_idx].atomtype;
            if (vdw.find(name) != vdw.end())
            {
                return name;
            }
        }
        return "";
    }

    Mol2Molecule mol_;
    ParmTable pt_;
    GaffTables gaff_;
    BlbaTables blba_;
    bool allparm_ = false;
    bool output_improper_ = true;
    std::vector<int> parmids_;
    std::vector<std::tuple<int, int, int, int>> impropers_;
    std::vector<GaffBond> bondparm_;
    std::vector<GaffAngle> angleparm_;
    std::vector<GaffTorsion> torsionparm_;
    std::vector<GaffImproper> improperparm_;
    std::vector<GaffVdw> vdwparm_;
    std::size_t bond_snapshot_ = 0;
    std::size_t angle_snapshot_ = 0;
    std::size_t torsion_snapshot_ = 0;
    std::size_t improper_snapshot_ = 0;
    std::size_t vdw_snapshot_ = 0;
    std::map<std::string, GaffAtom> gaff_atom_by_name_;
};

}  // namespace

void Generate_Gaff_Frcmod(const std::string& input_mol2,
                          const std::string& output_frcmod,
                          const Parmchk2Options& options)
{
    const std::string gaff_filename =
        options.ffset == 2 ? "gaff2.dat" : "gaff.dat";
    const std::string blba_filename =
        options.ffset == 2 ? "PARM_BLBA_GAFF2.DAT" : "PARM_BLBA_GAFF.DAT";
    ParmTable parm_table =
        Load_Parmchk_Dat(Join_Path(options.datapath, "PARMCHK.DAT"));
    GaffTables gaff = Load_Gaff_Dat(Join_Path(options.datapath, gaff_filename));
    BlbaTables blba = Load_Blba(Join_Path(options.datapath, blba_filename));
    Mol2Molecule mol = Read_Mol2(input_mol2);
    Engine engine(std::move(mol), std::move(parm_table), std::move(gaff),
                  std::move(blba), options.print_all,
                  options.print_dihedral_contain_X);
    const auto lines = engine.Run();
    std::ofstream output(output_frcmod.c_str());
    if (!output)
    {
        throw std::runtime_error("failed to open " + output_frcmod);
    }
    for (const auto& line : lines)
    {
        output << line << '\n';
    }
}

}  // namespace Amber
}  // namespace Xponge
