#pragma once

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../../utils/control/string.hpp"

namespace Xponge
{
namespace Amber
{
namespace detail
{

inline std::pair<std::vector<std::string>, std::vector<std::string>>
Split_Atoms_Words(const std::string& line,
                  std::size_t atom_field_width,
                  const std::vector<std::string>* last_atoms = nullptr)
{
    if (!line.empty() && line[0] == ' ')
    {
        if (last_atoms == nullptr)
        {
            throw std::runtime_error("missing previous atom field");
        }
        return {*last_atoms, string_words(line.substr(atom_field_width))};
    }
    const std::string atom_field =
        line.substr(0, std::min(atom_field_width, line.size()));
    std::vector<std::string> atoms;
    std::string atom;
    std::istringstream input(atom_field);
    while (std::getline(input, atom, '-'))
    {
        atoms.push_back(string_strip(atom));
    }
    return {atoms, line.size() > atom_field_width
                       ? string_words(line.substr(atom_field_width))
                       : std::vector<std::string>()};
}

}
}
}
