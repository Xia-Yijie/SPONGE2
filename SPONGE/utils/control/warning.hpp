#pragma once

inline void CONTROLLER::Warn(const char* warning)
{
    if (warn_of_initialization)
    {
        warnings.push_back(warning);
    }
    else
    {
        printf("Warning: %s\n", warning);
    }
}

inline void CONTROLLER::Deprecated(const char* deprecated_command,
                                   const char* recommanded_command,
                                   const char* version, const char* reason)
{
    if (this->Command_Exist(deprecated_command))
    {
        std::string value = this->Command(deprecated_command);
        std::string hint =
            "The command '%DEPRECATED_COMMAND% = %VALUE%' has been deprecated "
            "since version %VERSION%.\n";
        hint += "Use '%RECOMMANDED_COMMAND%' instead.\n";
        hint += "Reason:\n\t%REASON%";
        hint =
            string_format(hint, {{"RECOMMANDED_COMMAND", recommanded_command}});
        hint = string_format(hint, {{"REASON", reason}});
        hint =
            string_format(hint, {{"DEPRECATED_COMMAND", deprecated_command}});
        hint = string_format(hint, {{"VERSION", version}});
        hint = string_format(hint, {{"VALUE", value}});
        this->Warn(hint.c_str());
    }
}
