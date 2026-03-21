#pragma once
#include "basis_common.hpp"

extern std::map<std::string, std::vector<ShellData>> BASIS_DEF2_SVP;
extern void Initialize_Basis_Def2_SVP();

extern std::map<std::string, std::vector<ShellData>> BASIS_DEF2_TZVP;
extern void Initialize_Basis_Def2Tzvp();

extern std::map<std::string, std::vector<ShellData>> BASIS_DEF2_TZVPP;
extern void Initialize_Basis_Def2Tzvpp();

extern std::map<std::string, std::vector<ShellData>> BASIS_DEF2_QZVP;
extern void Initialize_Basis_Def2Qzvp();
