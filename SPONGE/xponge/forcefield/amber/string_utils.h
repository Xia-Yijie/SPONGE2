#pragma once

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../../utils/control/file.hpp"
#include "../../../utils/control/string.hpp"

namespace Xponge
{
namespace Amber
{
namespace detail
{

inline std::string Read_File(const std::string& path)
{
    return Read_File_To_String(path);
}

inline std::vector<std::string> Split_Lines(const std::string& text)
{
    return string_split_lines(text);
}

inline std::string Trim(const std::string& value)
{
    return string_strip(value);
}

inline std::vector<std::string> Words(const std::string& line)
{
    return string_words(line);
}

inline bool Starts_With(const std::string& value, const std::string& prefix)
{
    return string_starts_with(value, prefix);
}

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
        return {*last_atoms, Words(line.substr(atom_field_width))};
    }
    const std::string atom_field =
        line.substr(0, std::min(atom_field_width, line.size()));
    std::vector<std::string> atoms;
    std::string atom;
    std::istringstream input(atom_field);
    while (std::getline(input, atom, '-'))
    {
        atoms.push_back(Trim(atom));
    }
    return {atoms, line.size() > atom_field_width
                       ? Words(line.substr(atom_field_width))
                       : std::vector<std::string>()};
}

}  // namespace detail
}  // namespace Amber
}  // namespace Xponge
