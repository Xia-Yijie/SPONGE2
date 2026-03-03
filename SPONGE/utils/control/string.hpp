#pragma once

// 判断两个字符串是否相等（无视大小写）
inline bool is_str_equal(const char* a_str, const char* b_str,
                         int case_sensitive = 0)
{
    int i = 0;
    char a;
    char b;
    while (true)
    {
        if (a_str[i] == 0 && b_str[i] == 0)
        {
            return 1;
        }
        else if (a_str[i] == 0 || b_str[i] == 0)
        {
            return 0;
        }
        else
        {
            a = a_str[i];
            b = b_str[i];
            if (!case_sensitive)
            {
                if (a >= 65 && a <= 90)
                {
                    a = a - 65 + 97;
                }
                if (b >= 65 && b <= 90)
                {
                    b = b - 65 + 97;
                }
            }
            if (a != b)
            {
                return 0;
            }
        }
        i = i + 1;
    }
}

// 判断字符串是否是int
inline bool is_str_int(const char* str)
{
    bool hasNum = false;
    for (int index = 0; *str != '\0'; str++, index++)
    {
        switch (*str)
        {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                hasNum = true;
                break;
            case '-':
            case '+':
                if (index != 0)
                {
                    return false;
                }
                break;
            default:
                return false;
        }
    }
    return hasNum;
}

// 判断字符串是否是float
inline bool is_str_float(const char* str)
{
    bool isE = false, isPoint = false, numBefore = false, numBehind = false,
         hasNum = false;
    for (int index = 0; *str != '\0'; str++, index++)
    {
        switch (*str)
        {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                hasNum = true;
                if (isE)
                {
                    numBehind = true;
                }
                else
                {
                    numBefore = true;
                }
                break;
            case '+':
            case '-':
                if (index != 0)
                {
                    return false;
                }
                break;
            case 'e':
            case 'E':
                if (isE || !numBefore)
                {
                    return false;
                }
                else
                {
                    isPoint = true;
                    index = -1;
                    isE = true;
                }
                break;
            case '.':
                if (isPoint)
                {
                    return false;
                }
                else
                {
                    isPoint = true;
                }
                break;
            default:
                return false;
        }
    }
    if (!numBefore)
    {
        return false;
    }
    else if (isE && !numBehind)
    {
        return false;
    }
    return hasNum;
}

// 字符串去掉前后空格
inline std::string string_strip(std::string string)
{
    string.erase(string.find_last_not_of("\n") + 1);
    string.erase(0, string.find_first_not_of(" "));
    string.erase(string.find_last_not_of(" ") + 1);
    return string;
}

// 字符串分割
inline std::vector<std::string> string_split(std::string string,
                                             std::string separators)
{
    std::vector<std::string> result;
    if (string.size() == 0) return result;
    size_t last_pos = string.find_first_not_of(separators, 0);
    size_t pos = string.find_first_of(separators, last_pos);
    while (pos != string.npos)
    {
        result.push_back(string.substr(last_pos, pos - last_pos));
        last_pos = string.find_first_not_of(separators, pos);
        pos = string.find_first_of(separators, last_pos);
    }
    result.push_back(string.substr(last_pos, pos - last_pos));
    return result;
}

// 字符串替换
inline std::string string_replace(std::string string, std::string old,
                                  std::string new_)
{
    std::string result = string;
    size_t pos = result.find(old, 0);
    while (pos != result.npos)
    {
        result.replace(pos, old.length(), new_);
        pos += new_.length();
        pos = result.find(old, pos);
    }
    return result;
}

// 字符串格式化输出
inline std::string string_format(std::string string,
                                 std::map<std::string, std::string> dict)
{
    std::string result = string;
    for (auto& pair : dict)
    {
        result = string_replace(result, "%" + pair.first + "%", pair.second);
    }
    return result;
}

// 字符串拼接
inline std::string string_join(
    std::string pattern, std::string separator,
    std::vector<std::vector<std::string>> string_vectors)
{
    std::string result;
    int n = string_vectors.size();
    int l = string_vectors[0].size();
    for (int i = 0; i < l; i++)
    {
        std::string one_result = pattern;
        for (int j = 0; j < n; j++)
        {
            one_result =
                string_replace(one_result, "%" + std::to_string(j) + "%",
                               string_vectors[j][i]);
        }
        one_result = string_replace(one_result, "%INDEX%", std::to_string(i));
        result += one_result;
        if (i != l - 1) result += separator;
    }
    return result;
}
