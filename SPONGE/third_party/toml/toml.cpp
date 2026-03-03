#include "toml.h"

#include <sstream>

#if __has_include(<toml++/toml.hpp>)
#include <toml++/toml.hpp>
#elif __has_include(<toml++/toml.h>)
#include <toml++/toml.h>
#else
#error "tomlplusplus headers not found. Please install tomlplusplus."
#endif

namespace sponge::toml_wrap
{
namespace
{
std::string NodeValueToString(const toml::node& node,
                              const std::string& full_key,
                              std::string* error_message)
{
    if (auto val = node.value<std::string>())
    {
        return *val;
    }
    if (auto val = node.value<int64_t>())
    {
        return std::to_string(*val);
    }
    if (auto val = node.value<double>())
    {
        std::ostringstream oss;
        oss << *val;
        return oss.str();
    }
    if (auto val = node.value<bool>())
    {
        return *val ? "true" : "false";
    }
    if (const auto* arr = node.as_array())
    {
        std::ostringstream oss;
        bool first = true;
        for (const auto& item : *arr)
        {
            if (!first)
            {
                oss << ' ';
            }
            std::string item_value =
                NodeValueToString(item, full_key, error_message);
            if (!error_message->empty())
            {
                return "";
            }
            oss << item_value;
            first = false;
        }
        return oss.str();
    }
    *error_message = "unsupported TOML value type for '" + full_key + "'";
    return "";
}

bool FlattenTable(const toml::table& table, const std::string& prefix,
                  std::map<std::string, std::string>* parsed_commands,
                  std::string* error_message)
{
    for (const auto& [key, value] : table)
    {
        const std::string key_str(key.str());
        if (value.is_table())
        {
            const std::string next_prefix =
                prefix.empty() ? key_str : prefix + "_" + key_str;
            if (!FlattenTable(*value.as_table(), next_prefix, parsed_commands,
                              error_message))
            {
                return false;
            }
        }
        else
        {
            const std::string full_key =
                prefix.empty() ? key_str : prefix + "_" + key_str;
            std::string value_str =
                NodeValueToString(value, full_key, error_message);
            if (!error_message->empty())
            {
                return false;
            }
            (*parsed_commands)[full_key] = value_str;
        }
    }
    return true;
}
}  // namespace

bool ParseAndFlatten(const std::string& content, const std::string& source_path,
                     std::map<std::string, std::string>* parsed_commands,
                     std::string* error_message)
{
    if (parsed_commands == nullptr || error_message == nullptr)
    {
        return false;
    }
    parsed_commands->clear();
    error_message->clear();

#if TOML_EXCEPTIONS
    toml::table config;
    try
    {
        config = toml::parse(content, source_path);
    }
    catch (const toml::parse_error& err)
    {
        *error_message = err.description();
        return false;
    }
    return FlattenTable(config, "", parsed_commands, error_message);
#else
    toml::parse_result result = toml::parse(content, source_path);
    if (!result)
    {
        *error_message = result.error().description();
        return false;
    }
    return FlattenTable(result.table(), "", parsed_commands, error_message);
#endif
}
}  // namespace sponge::toml_wrap
