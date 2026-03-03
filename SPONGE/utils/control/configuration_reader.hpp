#pragma once

static inline void* skip_space_lines(char* buffer, FILE* fp)
{
    do
    {
        if (fgets(buffer, CHAR_LENGTH_MAX, fp) == NULL)
        {
            return NULL;
        }
    } while (strlen(buffer) < 2);
    return buffer;
}

// 用于读取复杂的配置文件
struct Configuration_Reader
{
    std::string error_reason;
    FILE* f;
    std::vector<std::string> sections;
    std::map<std::string, std::vector<std::string>> keys;
    std::map<std::pair<std::string, std::string>, std::string> values;
    std::set<std::pair<std::string, std::string>> value_unused;

    void Open(std::string filename)
    {
        Open_File_Safely(&f, filename.c_str(), "r");
        char buffer[CHAR_LENGTH_MAX], buffer2[CHAR_LENGTH_MAX];
        while (1)
        {
            if (skip_space_lines(buffer, f) == NULL)
            {
                break;
            }
            if (sscanf(buffer, "[[[%s]]]", buffer2) != 1)
            {
                error_reason = "Fail to read a new section '[[[ SECTION ]]]'";
            }
            std::string section = string_strip(buffer2);
            sections.push_back(section);
            while (1)
            {
                if (skip_space_lines(buffer, f) == NULL)
                {
                    error_reason = string_format(
                        "Fail to read the end of the section '[[ end ]]' for "
                        "%SECTION%",
                        {{"SECTION", section}});
                    break;
                }
                if (sscanf(buffer, "[[%s]]", buffer2) != 1)
                {
                    error_reason = string_format(
                        "Fail to read a new key '[[ KEYWORD ]]' for %SECTION%",
                        {{"SECTION", section}});
                }
                std::string key = string_strip(buffer2);
                if (key == "end")
                {
                    break;
                }
                keys[section].push_back(key);
                auto pos = ftell(f);
                std::string value;
                while (skip_space_lines(buffer, f) != NULL)
                {
                    strcpy(buffer2, string_strip(buffer).c_str());
                    if (sscanf(buffer2, "[[%s]]", buffer) == 1)
                    {
                        fseek(f, pos, SEEK_SET);
                        break;
                    }
                    value += buffer2;
                    pos = ftell(f);
                }
                if (value.empty())
                {
                    error_reason = string_format(
                        "Fail to read the value of [[[ %SECTION% ]]] [[ %KEY% "
                        "]]",
                        {{"KEY", key}, {"SECTION", section}});
                }
                values[{section, key}] = value;
                value_unused.insert({section, key});
            }
        }
    }

    bool Section_Exist(std::string section) { return keys.count(section); }

    bool Key_Exist(std::string section, std::string key)
    {
        return values.count({section, key});
    }

    std::string Get_Value(std::string section, std::string key)
    {
        value_unused.erase({section, key});
        return values[{section, key}];
    }

    void Close() { fclose(f); }
};
