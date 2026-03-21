#pragma once
#include "basis_common.hpp"

extern std::map<std::string, std::vector<ShellData>> BASIS_STO_3G;
extern void Initialize_Basis_Sto3g();

extern std::map<std::string, std::vector<ShellData>> BASIS_3_21G;
extern void Initialize_Basis_321g();

extern std::map<std::string, std::vector<ShellData>> BASIS_631G;
extern void Initialize_Basis_631G();

extern std::map<std::string, std::vector<ShellData>> BASIS_6_31GSTAR;
extern void Initialize_Basis_631gstar();

extern std::map<std::string, std::vector<ShellData>> BASIS_6_31GSTARSTAR;
extern void Initialize_Basis_631gstarstar();

extern std::map<std::string, std::vector<ShellData>> BASIS_6_311G;
extern void Initialize_Basis_6311g();

extern std::map<std::string, std::vector<ShellData>> BASIS_6_311GSTAR;
extern void Initialize_Basis_6311gstar();

extern std::map<std::string, std::vector<ShellData>> BASIS_6_311GSTARSTAR;
extern void Initialize_Basis_6311gstarstar();
