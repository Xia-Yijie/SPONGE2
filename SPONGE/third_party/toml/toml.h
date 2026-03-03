#pragma once

#include <map>
#include <string>

namespace sponge::toml_wrap
{
bool ParseAndFlatten(const std::string& content, const std::string& source_path,
                     std::map<std::string, std::string>* parsed_commands,
                     std::string* error_message);
}
