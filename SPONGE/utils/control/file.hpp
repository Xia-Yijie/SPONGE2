#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

inline std::string Read_File_To_String(const std::string& path)
{
    std::ifstream stream(path, std::ios::in | std::ios::binary);
    if (!stream.is_open())
    {
        throw std::runtime_error("failed to open " + path);
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}
