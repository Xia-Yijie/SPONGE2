#include "parameters.h"
#include "string_utils.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace Xponge
{
namespace Amber
{
namespace
{

using detail::Read_File;
using detail::Split_Atoms_Words;
using detail::Split_Lines;
using detail::Starts_With;
using detail::Words;

double Radians(double degree)
{
    return degree * std::acos(-1.0) / 180.0;
}

struct FrcmodRecord
{
    std::string flag;
    std::string line;
    std::vector<std::string> words;
    std::vector<std::string> atoms;
    std::vector<std::string> values;
};

std::vector<FrcmodRecord> Read_Frcmod_Records(const std::string& filename)
{
    std::vector<FrcmodRecord> records;
    std::string flag;
    std::vector<std::string> last_atoms;
    for (const auto& line : Split_Lines(Read_File(filename)))
    {
        if (Words(line).empty() || Starts_With(line, "Remark"))
        {
            continue;
        }
        auto words = Words(line);
        if (flag != "CMAP" && words.size() == 1)
        {
            flag = words[0];
            continue;
        }
        if (flag.empty())
        {
            continue;
        }
        FrcmodRecord record;
        record.flag = flag;
        record.line = line;
        record.words = words;
        if (Starts_With(flag, "BOND"))
        {
            auto split = Split_Atoms_Words(line, 5);
            record.atoms = split.first;
            record.values = split.second;
        }
        else if (Starts_With(flag, "ANGL"))
        {
            auto split = Split_Atoms_Words(line, 8);
            record.atoms = split.first;
            record.values = split.second;
        }
        else if (Starts_With(flag, "DIHE"))
        {
            auto split = Split_Atoms_Words(line, 11, &last_atoms);
            last_atoms = split.first;
            record.atoms = split.first;
            record.values = split.second;
        }
        else if (Starts_With(flag, "IMPROPER"))
        {
            auto split = Split_Atoms_Words(line, 11);
            record.atoms = split.first;
            record.values = split.second;
        }
        records.push_back(record);
    }
    return records;
}

std::string Repr_Double(double value)
{
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

struct CmapBlock
{
    std::vector<std::string> residues;
    CmapParameter parameter;
};

void Flush_Cmap_Block(const CmapBlock& block,
                      std::map<std::string, CmapParameter>* cmap)
{
    if (block.residues.empty())
    {
        return;
    }
    for (const auto& residue : block.residues)
    {
        (*cmap)["C-N-" + residue + "@XC-C-N"] = block.parameter;
    }
}

}  // namespace

std::pair<std::string, std::string> Canonical2(const std::string& a,
                                               const std::string& b)
{
    return a <= b ? std::make_pair(a, b) : std::make_pair(b, a);
}

std::tuple<std::string, std::string, std::string> Canonical3(
    const std::string& a,
    const std::string& b,
    const std::string& c)
{
    return std::make_tuple(a, b, c) <= std::make_tuple(c, b, a)
               ? std::make_tuple(a, b, c)
               : std::make_tuple(c, b, a);
}

GaffParameters Load_Gaff_Parameters(const std::string& dat_path,
                                    const std::string& frcmod_path)
{
    GaffParameters params;
    const auto lines = Split_Lines(Read_File(dat_path));
    std::size_t idx = 1;
    while (idx < lines.size() && !Words(lines[idx]).empty())
    {
        auto words = Words(lines[idx]);
        params.atom[words[0]] = {std::stod(words[1]), words[0]};
        ++idx;
    }
    idx += 2;
    while (idx < lines.size() && !Words(lines[idx]).empty())
    {
        auto split = Split_Atoms_Words(lines[idx], 5);
        params.bond[Canonical2(split.first[0], split.first[1])] =
            {std::stod(split.second[0]), std::stod(split.second[1])};
        ++idx;
    }
    ++idx;
    while (idx < lines.size() && !Words(lines[idx]).empty())
    {
        auto split = Split_Atoms_Words(lines[idx], 8);
        params.angle[Canonical3(split.first[0], split.first[1],
                                split.first[2])] = {
            std::stod(split.second[0]), Radians(std::stod(split.second[1]))};
        ++idx;
    }
    ++idx;
    std::vector<std::string> last_atoms;
    bool reset = true;
    while (idx < lines.size() && !Words(lines[idx]).empty())
    {
        auto split = Split_Atoms_Words(lines[idx], 11, &last_atoms);
        last_atoms = split.first;
        auto key = std::make_tuple(split.first[0], split.first[1],
                                   split.first[2], split.first[3]);
        if (reset || params.proper.find(key) == params.proper.end())
        {
            params.proper[key].clear();
        }
        params.proper[key].push_back(
            {std::stod(split.second[1]) / std::stoi(split.second[0]),
             Radians(std::stod(split.second[2])),
             std::abs(static_cast<int>(std::stod(split.second[3])))});
        reset = !(static_cast<int>(std::stod(split.second[3])) < 0);
        ++idx;
    }
    ++idx;
    while (idx < lines.size() && !Words(lines[idx]).empty())
    {
        auto split = Split_Atoms_Words(lines[idx], 11);
        params.improper[std::make_tuple(split.first[0], split.first[1],
                                        split.first[2], split.first[3])] = {
            std::stod(split.second[0]), Radians(std::stod(split.second[1])),
            static_cast<int>(std::stod(split.second[2]))};
        ++idx;
    }
    ++idx;
    while (idx < lines.size() && !Words(lines[idx]).empty())
    {
        auto words = Words(lines[idx]);
        if (params.atom.find(words[0]) != params.atom.end())
        {
            for (std::size_t i = 1; i < words.size(); ++i)
            {
                if (params.atom.find(words[i]) != params.atom.end())
                {
                    params.atom[words[i]].lj_type = words[0];
                }
            }
        }
        ++idx;
    }
    while (idx < lines.size() && !Starts_With(lines[idx], "MOD4"))
    {
        ++idx;
    }
    ++idx;
    while (idx < lines.size() && !Words(lines[idx]).empty())
    {
        auto words = Words(lines[idx]);
        params.lj[words[0]] = {std::stod(words[2]), std::stod(words[1])};
        ++idx;
    }
    if (!frcmod_path.empty())
    {
        bool frc_reset = true;
        for (const auto& record : Read_Frcmod_Records(frcmod_path))
        {
            if (Starts_With(record.flag, "MASS"))
            {
                params.atom[record.words[0]].mass = std::stod(record.words[1]);
                if (params.atom[record.words[0]].lj_type.empty())
                {
                    params.atom[record.words[0]].lj_type = record.words[0];
                }
            }
            else if (Starts_With(record.flag, "BOND"))
            {
                params.bond[Canonical2(record.atoms[0], record.atoms[1])] = {
                    std::stod(record.values[0]), std::stod(record.values[1])};
            }
            else if (Starts_With(record.flag, "ANGL"))
            {
                params.angle[Canonical3(record.atoms[0], record.atoms[1],
                                        record.atoms[2])] = {
                    std::stod(record.values[0]),
                    Radians(std::stod(record.values[1]))};
            }
            else if (Starts_With(record.flag, "DIHE"))
            {
                auto key = std::make_tuple(record.atoms[0], record.atoms[1],
                                           record.atoms[2], record.atoms[3]);
                if (frc_reset || params.proper.find(key) == params.proper.end())
                {
                    params.proper[key].clear();
                }
                params.proper[key].push_back(
                    {std::stod(record.values[1]) /
                         std::stoi(record.values[0]),
                     Radians(std::stod(record.values[2])),
                     std::abs(static_cast<int>(std::stod(record.values[3])))});
                frc_reset =
                    !(static_cast<int>(std::stod(record.values[3])) < 0);
            }
            else if (Starts_With(record.flag, "IMPROPER"))
            {
                params.improper[std::make_tuple(
                    record.atoms[0], record.atoms[1], record.atoms[2],
                    record.atoms[3])] = {std::stod(record.values[0]),
                                         Radians(std::stod(record.values[1])),
                                         static_cast<int>(
                                             std::stod(record.values[2]))};
            }
            else if (Starts_With(record.flag, "NONBON"))
            {
                params.lj[record.words[0]] = {std::stod(record.words[2]),
                                              std::stod(record.words[1])};
                if (params.atom[record.words[0]].lj_type.empty())
                {
                    params.atom[record.words[0]].lj_type = record.words[0];
                }
            }
        }
    }
    return params;
}

FrcmodXpongeData Load_Frcmod_As_Xponge_Data(const std::string& filename)
{
    std::map<std::string, std::string> atom_types;
    std::string bonds = "name  k[kcal/mol.A^-2]    b[A]\n";
    std::string angles = "name  k[kcal/mol.rad^-2]    b[degree]\n";
    std::string propers =
        "name  k[kcal/mol]    phi0[degree]    periodicity    reset\n";
    std::string impropers =
        "name  k[kcal/mol]    phi0[degree]    periodicity\n";
    std::string ljs = "name rmin[A]   epsilon[kcal/mol]\n";
    std::map<std::string, CmapParameter> cmap;
    std::string cmap_flag;
    CmapBlock cmap_block;
    int reset = 1;
    for (const auto& record : Read_Frcmod_Records(filename))
    {
        if (Starts_With(record.flag, "MASS"))
        {
            atom_types[record.words[0]] = record.words[1];
        }
        else if (Starts_With(record.flag, "BOND"))
        {
            bonds += record.atoms[0] + "-" + record.atoms[1] + "\t" +
                     record.values[0] + "\t" + record.values[1] + "\n";
        }
        else if (Starts_With(record.flag, "ANGL"))
        {
            angles += record.atoms[0] + "-" + record.atoms[1] + "-" +
                      record.atoms[2] + "\t" + record.values[0] + "\t" +
                      record.values[1] + "\n";
        }
        else if (Starts_With(record.flag, "DIHE"))
        {
            propers += record.atoms[0] + "-" + record.atoms[1] + "-" +
                       record.atoms[2] + "-" + record.atoms[3] + "\t" +
                       Repr_Double(std::stod(record.values[1]) /
                                   std::stoi(record.values[0])) +
                       "\t" + record.values[2] + "\t" +
                       std::to_string(
                           std::abs(static_cast<int>(
                               std::stod(record.values[3])))) +
                       "\t" + std::to_string(reset) + "\n";
            reset = static_cast<int>(std::stod(record.values[3])) < 0 ? 0 : 1;
        }
        else if (Starts_With(record.flag, "IMPROPER"))
        {
            impropers += record.atoms[0] + "-" + record.atoms[1] + "-" +
                         record.atoms[2] + "-" + record.atoms[3] + "\t" +
                         record.values[0] + "\t" + record.values[1] + "\t" +
                         std::to_string(
                             static_cast<int>(std::stod(record.values[2]))) +
                         "\n";
        }
        else if (Starts_With(record.flag, "NONBON"))
        {
            ljs += record.words[0] + "-" + record.words[0] + "\t" +
                   record.words[1] + "\t" + record.words[2] + "\n";
        }
        else if (Starts_With(record.flag, "CMAP"))
        {
            if (Starts_With(record.line, "%FLAG"))
            {
                if (record.line.find("CMAP_COUNT") != std::string::npos)
                {
                    Flush_Cmap_Block(cmap_block, &cmap);
                    cmap_block = CmapBlock();
                    cmap_block.parameter.resolution = 24;
                    cmap_flag = "CMAP_COUNT";
                }
                else if (record.line.find("CMAP_RESOLUTION") !=
                         std::string::npos)
                {
                    if (!record.words.empty())
                    {
                        cmap_block.parameter.resolution =
                            std::stoi(record.words.back());
                    }
                    cmap_flag = "CMAP_RESOLUTION";
                }
                else if (record.line.find("CMAP_RESLIST") != std::string::npos)
                {
                    cmap_flag = "CMAP_RESLIST";
                }
                else if (record.line.find("CMAP_TITLE") != std::string::npos)
                {
                    cmap_flag = "CMAP_TITLE";
                }
                else if (record.line.find("CMAP_PARAMETER") !=
                         std::string::npos)
                {
                    cmap_flag = "CMAP_PARAMETER";
                }
            }
            else if (cmap_flag == "CMAP_RESLIST")
            {
                cmap_block.residues.insert(cmap_block.residues.end(),
                                           record.words.begin(),
                                           record.words.end());
            }
            else if (cmap_flag == "CMAP_PARAMETER")
            {
                for (const auto& word : record.words)
                {
                    cmap_block.parameter.parameters.push_back(
                        std::stod(word));
                }
            }
        }
    }
    Flush_Cmap_Block(cmap_block, &cmap);
    std::string atoms = "name  mass  LJtype\n";
    for (const auto& item : atom_types)
    {
        atoms += item.first + "\t" + item.second + "\t" + item.first + "\n";
    }
    return {{atoms, bonds, angles, propers, impropers, ljs}, cmap};
}

}  // namespace Amber
}  // namespace Xponge
