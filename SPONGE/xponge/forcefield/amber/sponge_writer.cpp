#include "sponge_writer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace Xponge
{
namespace Amber
{
namespace
{

constexpr double LJ_SCALE_14 = 0.5;
constexpr double COULOMB_SCALE_14 = 1.0 / 1.2;

std::string Join_Path(const std::string& directory, const std::string& file)
{
    if (directory.empty() || directory == ".")
    {
        return directory == "." ? "./" + file : file;
    }
    const char last = directory[directory.size() - 1];
    if (last == '/' || last == '\\')
    {
        return directory + file;
    }
    return directory + "/" + file;
}

void Write_Lines(const std::string& path, const std::vector<std::string>& lines)
{
    std::ofstream output(path.c_str());
    if (!output)
    {
        throw std::runtime_error("failed to open " + path);
    }
    for (const auto& line : lines)
    {
        output << line << '\n';
    }
}

std::string Fixed(double value, int precision)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

std::string Sci(double value)
{
    std::ostringstream out;
    out << std::scientific << std::setprecision(8) << value;
    return out.str();
}

std::vector<std::set<int>> Bonds_By_Atom(
    int natom,
    const std::vector<std::pair<int, int>>& bonds)
{
    std::vector<std::set<int>> result(natom);
    for (const auto& bond : bonds)
    {
        result[bond.first].insert(bond.second);
        result[bond.second].insert(bond.first);
    }
    return result;
}

std::vector<std::map<int, std::set<int>>> Distance_Sets(
    const std::vector<std::set<int>>& bonds_by_atom)
{
    std::vector<std::map<int, std::set<int>>> result;
    for (std::size_t start = 0; start < bonds_by_atom.size(); ++start)
    {
        std::set<int> seen = {static_cast<int>(start)};
        std::deque<std::pair<int, int>> frontier = {
            {static_cast<int>(start), 0}};
        std::map<int, std::set<int>> by_depth;
        while (!frontier.empty())
        {
            auto current = frontier.front();
            frontier.pop_front();
            if (current.second == 4)
            {
                continue;
            }
            for (int next : bonds_by_atom[current.first])
            {
                if (seen.count(next))
                {
                    continue;
                }
                seen.insert(next);
                const int depth = current.second + 1;
                by_depth[depth].insert(next);
                frontier.push_back({next, depth});
            }
        }
        result.push_back(by_depth);
    }
    return result;
}

template <typename Table>
auto Match_Key(const Table& table,
               const std::tuple<std::string, std::string, std::string,
                                std::string>& atoms)
    -> typename Table::const_iterator
{
    auto direct = table.find(atoms);
    if (direct != table.end())
    {
        return direct;
    }
    auto reversed = std::make_tuple(std::get<3>(atoms), std::get<2>(atoms),
                                    std::get<1>(atoms), std::get<0>(atoms));
    auto reverse = table.find(reversed);
    if (reverse != table.end())
    {
        return reverse;
    }
    typename Table::const_iterator best = table.end();
    int best_score = -1;
    for (auto it = table.begin(); it != table.end(); ++it)
    {
        for (const auto& candidate : {atoms, reversed})
        {
            const std::array<std::string, 4> key = {
                std::get<0>(it->first), std::get<1>(it->first),
                std::get<2>(it->first), std::get<3>(it->first)};
            const std::array<std::string, 4> cand = {
                std::get<0>(candidate), std::get<1>(candidate),
                std::get<2>(candidate), std::get<3>(candidate)};
            bool ok = true;
            int score = 0;
            for (int i = 0; i < 4; ++i)
            {
                if (key[i] != "X" && key[i] != cand[i])
                {
                    ok = false;
                    break;
                }
                if (key[i] != "X")
                {
                    ++score;
                }
            }
            if (ok && score > best_score)
            {
                best_score = score;
                best = it;
            }
        }
    }
    return best;
}

auto Match_Improper_Key(
    const std::map<std::tuple<std::string, std::string, std::string,
                              std::string>,
                   ProperTerm>& table,
    const std::tuple<std::string, std::string, std::string, std::string>& atoms)
    -> decltype(table.begin())
{
    auto exact = table.find(atoms);
    if (exact != table.end())
    {
        return exact;
    }
    decltype(table.begin()) best = table.end();
    int best_score = -1;
    for (auto it = table.begin(); it != table.end(); ++it)
    {
        if ((std::get<0>(it->first) == "X" ||
             std::get<0>(it->first) == std::get<0>(atoms)) &&
            (std::get<1>(it->first) == "X" ||
             std::get<1>(it->first) == std::get<1>(atoms)) &&
            (std::get<2>(it->first) == "X" ||
             std::get<2>(it->first) == std::get<2>(atoms)) &&
            (std::get<3>(it->first) == "X" ||
             std::get<3>(it->first) == std::get<3>(atoms)))
        {
            int score = 0;
            if (std::get<0>(it->first) != "X") ++score;
            if (std::get<1>(it->first) != "X") ++score;
            if (std::get<2>(it->first) != "X") ++score;
            if (std::get<3>(it->first) != "X") ++score;
            if (score > best_score)
            {
                best_score = score;
                best = it;
            }
        }
    }
    return best;
}

std::pair<double, double> Pair_Lj_Coeff(const std::pair<double, double>& lj_a,
                                        const std::pair<double, double>& lj_b)
{
    const double epsilon = std::sqrt(lj_a.first * lj_b.first);
    const double rmin = lj_a.second + lj_b.second;
    return {epsilon * std::pow(rmin, 12), 2.0 * epsilon * std::pow(rmin, 6)};
}

void Copy_File(const std::string& source, const std::string& target)
{
    std::ifstream input(source.c_str(), std::ios::binary);
    std::ofstream output(target.c_str(), std::ios::binary);
    if (!input || !output)
    {
        throw std::runtime_error("failed to copy " + source + " to " + target);
    }
    output << input.rdbuf();
}

}  // namespace

void Save_Gaff_Sponge_Input(const Molecule& molecule,
                            const GaffParameters& parameters,
                            const SpongeInputOptions& options)
{
    const auto atoms = molecule.Atoms();
    const auto bonds = molecule.Bonds(options.connect_residue_tails);
    const int natom = static_cast<int>(atoms.size());
    const auto bonds_by_atom = Bonds_By_Atom(natom, bonds);
    const auto distances = Distance_Sets(bonds_by_atom);
    const std::string file_prefix = options.prefix_files ? options.prefix + "_" : "";

    std::vector<std::string> lines;
    lines.push_back(std::to_string(natom));
    for (const auto& atom : atoms)
    {
        lines.push_back(Fixed(parameters.atom.at(atom.type).mass, 8));
    }
    Write_Lines(Join_Path(options.output_dir, file_prefix + "mass.txt"), lines);

    lines = {std::to_string(natom)};
    for (const auto& atom : atoms)
    {
        lines.push_back(Fixed(atom.charge * options.charge_scale, 8));
    }
    Write_Lines(Join_Path(options.output_dir, file_prefix + "charge.txt"), lines);

    lines = {std::to_string(natom)};
    for (const auto& atom : atoms)
    {
        lines.push_back(Fixed(atom.x, 8) + " " + Fixed(atom.y, 8) + " " +
                        Fixed(atom.z, 8));
    }
    lines.push_back(Fixed(options.box[0], 8) + " " + Fixed(options.box[1], 8) +
                    " " + Fixed(options.box[2], 8) + " " +
                    Fixed(options.box[3], 8) + " " + Fixed(options.box[4], 8) +
                    " " + Fixed(options.box[5], 8));
    Write_Lines(Join_Path(options.output_dir, file_prefix + "coordinate.txt"), lines);
    if (options.write_atom_metadata)
    {
        lines = {std::to_string(natom)};
        for (const auto& atom : atoms) lines.push_back(atom.name);
        Write_Lines(Join_Path(options.output_dir, file_prefix + "atom_name.txt"),
                    lines);
        lines = {std::to_string(natom)};
        for (const auto& atom : atoms) lines.push_back(atom.type);
        Write_Lines(Join_Path(options.output_dir,
                              file_prefix + "atom_type_name.txt"),
                    lines);
        lines = {std::to_string(molecule.residues.size())};
        for (const auto& residue : molecule.residues) lines.push_back(residue.name);
        Write_Lines(Join_Path(options.output_dir, file_prefix + "resname.txt"),
                    lines);
    }

    lines = {std::to_string(natom) + " " +
             std::to_string(molecule.residues.size())};
    for (const auto& residue : molecule.residues)
    {
        lines.push_back(std::to_string(residue.atoms.size()));
    }
    Write_Lines(Join_Path(options.output_dir, file_prefix + "residue.txt"), lines);

    std::vector<std::tuple<int, int, double, double>> bond_rows;
    for (const auto& bond : bonds)
    {
        auto param = parameters.bond.at(
            Canonical2(atoms[bond.first].type, atoms[bond.second].type));
        bond_rows.push_back({bond.first, bond.second, param.first,
                             param.second});
    }
    std::sort(bond_rows.begin(), bond_rows.end());
    lines = {std::to_string(bond_rows.size())};
    for (const auto& row : bond_rows)
    {
        lines.push_back(std::to_string(std::get<0>(row)) + " " +
                        std::to_string(std::get<1>(row)) + " " +
                        Fixed(std::get<2>(row), 8) + " " +
                        Fixed(std::get<3>(row), 8));
    }
    Write_Lines(Join_Path(options.output_dir, file_prefix + "bond.txt"), lines);

    std::vector<std::tuple<int, int, int, double, double>> angle_rows;
    for (int j = 0; j < natom; ++j)
    {
        std::vector<int> neighbors(bonds_by_atom[j].begin(),
                                   bonds_by_atom[j].end());
        for (std::size_t a = 0; a < neighbors.size(); ++a)
        {
            for (std::size_t b = a + 1; b < neighbors.size(); ++b)
            {
                int i = neighbors[a];
                int k = neighbors[b];
                auto param = parameters.angle.at(Canonical3(
                    atoms[i].type, atoms[j].type, atoms[k].type));
                if (i > k)
                {
                    std::swap(i, k);
                }
                angle_rows.push_back({i, j, k, param.first, param.second});
            }
        }
    }
    std::sort(angle_rows.begin(), angle_rows.end());
    lines = {std::to_string(angle_rows.size())};
    for (const auto& row : angle_rows)
    {
        lines.push_back(std::to_string(std::get<0>(row)) + " " +
                        std::to_string(std::get<1>(row)) + " " +
                        std::to_string(std::get<2>(row)) + " " +
                        Fixed(std::get<3>(row), 8) + " " +
                        Fixed(std::get<4>(row), 8));
    }
    Write_Lines(Join_Path(options.output_dir, file_prefix + "angle.txt"), lines);

    std::vector<std::tuple<int, int, int, int, int, double, double>>
        dihedral_rows;
    std::set<std::tuple<int, int, int, int>> seen_dihedrals;
    for (int j = 0; j < natom; ++j)
    {
        for (int k : bonds_by_atom[j])
        {
            if (j > k) continue;
            for (int i : bonds_by_atom[j])
            {
                if (i == k) continue;
                for (int l : bonds_by_atom[k])
                {
                    if (l == j || i == l) continue;
                    auto ids = std::make_tuple(i, j, k, l);
                    auto rev = std::make_tuple(l, k, j, i);
                    auto canonical = std::min(ids, rev);
                    if (!seen_dihedrals.insert(canonical).second) continue;
                    auto types = std::make_tuple(
                        atoms[std::get<0>(canonical)].type,
                        atoms[std::get<1>(canonical)].type,
                        atoms[std::get<2>(canonical)].type,
                        atoms[std::get<3>(canonical)].type);
                    auto match = Match_Key(parameters.proper, types);
                    if (match == parameters.proper.end())
                    {
                        throw std::runtime_error("missing dihedral parameter");
                    }
                    for (const auto& term : match->second)
                    {
                        if (term.k != 0.0)
                        {
                            dihedral_rows.push_back(
                                {std::get<0>(canonical),
                                 std::get<1>(canonical),
                                 std::get<2>(canonical),
                                 std::get<3>(canonical), term.periodicity,
                                 term.k, term.phase});
                        }
                    }
                }
            }
        }
    }
    for (int center = 0; center < natom; ++center)
    {
        if (bonds_by_atom[center].size() != 3) continue;
        std::vector<int> neighbors(bonds_by_atom[center].begin(),
                                   bonds_by_atom[center].end());
        bool found = false;
        int best_score = -1;
        std::tuple<int, int, int, int> best_ids;
        ProperTerm best_term;
        std::sort(neighbors.begin(), neighbors.end());
        do
        {
            auto ids = std::make_tuple(neighbors[0], neighbors[1], center,
                                       neighbors[2]);
            auto types = std::make_tuple(
                atoms[std::get<0>(ids)].type, atoms[std::get<1>(ids)].type,
                atoms[std::get<2>(ids)].type, atoms[std::get<3>(ids)].type);
            auto match = Match_Improper_Key(parameters.improper, types);
            if (match == parameters.improper.end()) continue;
            int score = 0;
            if (std::get<0>(match->first) != "X") ++score;
            if (std::get<1>(match->first) != "X") ++score;
            if (std::get<2>(match->first) != "X") ++score;
            if (std::get<3>(match->first) != "X") ++score;
            if (!found || score > best_score)
            {
                found = true;
                best_score = score;
                best_ids = ids;
                best_term = match->second;
            }
        } while (std::next_permutation(neighbors.begin(), neighbors.end()));
        if (found && best_term.k != 0.0)
        {
            dihedral_rows.push_back(
                {std::get<0>(best_ids), std::get<1>(best_ids),
                 std::get<2>(best_ids), std::get<3>(best_ids),
                 best_term.periodicity, best_term.k, best_term.phase});
        }
    }
    std::sort(dihedral_rows.begin(), dihedral_rows.end());
    lines = {std::to_string(dihedral_rows.size())};
    for (const auto& row : dihedral_rows)
    {
        lines.push_back(std::to_string(std::get<0>(row)) + " " +
                        std::to_string(std::get<1>(row)) + " " +
                        std::to_string(std::get<2>(row)) + " " +
                        std::to_string(std::get<3>(row)) + " " +
                        std::to_string(std::get<4>(row)) + " " +
                        Fixed(std::get<5>(row), 8) + " " +
                        Fixed(std::get<6>(row), 8));
    }
    Write_Lines(Join_Path(options.output_dir, file_prefix + "dihedral.txt"), lines);
    Write_Lines(Join_Path(options.output_dir, file_prefix + "improper_dihedral.txt"), {"0"});

    std::vector<std::string> lj_types;
    std::map<std::string, int> lj_index;
    std::vector<int> atom_lj_indices;
    for (const auto& atom : atoms)
    {
        const auto lj_type = parameters.atom.at(atom.type).lj_type;
        if (lj_index.find(lj_type) == lj_index.end())
        {
            lj_index[lj_type] = static_cast<int>(lj_types.size());
            lj_types.push_back(lj_type);
        }
        atom_lj_indices.push_back(lj_index[lj_type]);
    }
    std::vector<std::pair<double, double>> lj_params;
    for (const auto& type : lj_types)
    {
        lj_params.push_back(parameters.lj.at(type));
    }
    std::vector<double> pair_a;
    std::vector<double> pair_b;
    for (std::size_t i = 0; i < lj_types.size(); ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            auto coeff = Pair_Lj_Coeff(lj_params[i], lj_params[j]);
            pair_a.push_back(coeff.first);
            pair_b.push_back(coeff.second);
        }
    }
    lines = {std::to_string(natom) + " " + std::to_string(lj_types.size())};
    for (double value : pair_a) lines.push_back(Sci(value));
    for (double value : pair_b) lines.push_back(Sci(value));
    for (int value : atom_lj_indices) lines.push_back(std::to_string(value));
    Write_Lines(Join_Path(options.output_dir, file_prefix + "LJ.txt"), lines);

    std::vector<std::tuple<int, int, double, double>> nb14_rows;
    for (int i = 0; i < natom; ++i)
    {
        const auto it = distances[i].find(3);
        if (it == distances[i].end()) continue;
        for (int j : it->second)
        {
            if (i < j)
            {
                nb14_rows.push_back({i, j, LJ_SCALE_14, COULOMB_SCALE_14});
            }
        }
    }
    lines = {std::to_string(nb14_rows.size())};
    for (const auto& row : nb14_rows)
    {
        lines.push_back(std::to_string(std::get<0>(row)) + " " +
                        std::to_string(std::get<1>(row)) + " " +
                        Fixed(std::get<2>(row), 8) + " " +
                        Fixed(std::get<3>(row), 8));
    }
    Write_Lines(Join_Path(options.output_dir, file_prefix + "nb14.txt"), lines);

    lines.clear();
    int excluded_total = 0;
    std::vector<std::string> excluded_rows;
    for (int i = 0; i < natom; ++i)
    {
        std::set<int> excluded;
        for (int depth : {1, 2, 3})
        {
            const auto it = distances[i].find(depth);
            if (it != distances[i].end())
            {
                for (int atom : it->second)
                {
                    if (atom > i) excluded.insert(atom);
                }
            }
        }
        excluded_total += static_cast<int>(excluded.size());
        std::string row = std::to_string(excluded.size());
        for (int atom : excluded)
        {
            row += " " + std::to_string(atom);
        }
        excluded_rows.push_back(row);
    }
    lines.push_back(std::to_string(natom) + " " +
                    std::to_string(excluded_total));
    lines.insert(lines.end(), excluded_rows.begin(), excluded_rows.end());
    Write_Lines(Join_Path(options.output_dir, file_prefix + "exclude.txt"), lines);
    if (!options.cmap_source.empty())
    {
        Copy_File(options.cmap_source,
                  Join_Path(options.output_dir, file_prefix + "cmap.txt"));
    }

    if (options.write_mdin)
    {
        Write_Lines(Join_Path(options.output_dir, "sponge.mdin"),
                    {"xponge GAFF single point",
                     "mode = nve",
                     "pbc = 0",
                     "step_limit = 0",
                     "dt = 0",
                     "cutoff = 999.0",
                     "coordinate_in_file = coordinate.txt",
                     "mass_in_file = mass.txt",
                     "charge_in_file = charge.txt",
                     "residue_in_file = residue.txt",
                     "exclude_in_file = exclude.txt",
                     "default_in_file_prefix = " + options.prefix,
                     "frc = frc.dat",
                     "print_pressure = 1",
                     "print_zeroth_frame = 1",
                     "write_mdout_interval = 1"});
    }

    if (!options.prefix_files)
    {
        for (const auto& name :
             {"bond", "angle", "dihedral", "improper_dihedral", "LJ", "nb14"})
        {
            Copy_File(Join_Path(options.output_dir, std::string(name) + ".txt"),
                      Join_Path(options.output_dir,
                                options.prefix + "_" + std::string(name) +
                                    ".txt"));
        }
    }
}

}  // namespace Amber
}  // namespace Xponge
