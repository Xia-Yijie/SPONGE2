#pragma once

#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace Xponge
{
namespace Amber
{

struct AtomParameter
{
    double mass = 0.0;
    std::string lj_type;
};

struct ProperTerm
{
    double k = 0.0;
    double phase = 0.0;
    int periodicity = 0;
};

struct GaffParameters
{
    std::map<std::string, AtomParameter> atom;
    std::map<std::pair<std::string, std::string>, std::pair<double, double>>
        bond;
    std::map<std::tuple<std::string, std::string, std::string>,
             std::pair<double, double>>
        angle;
    std::map<std::tuple<std::string, std::string, std::string, std::string>,
             std::vector<ProperTerm>>
        proper;
    std::map<std::tuple<std::string, std::string, std::string, std::string>,
             ProperTerm>
        improper;
    std::map<std::string, std::pair<double, double>> lj;
};

struct CmapParameter
{
    int resolution = 24;
    std::vector<double> parameters;
};

struct FrcmodXpongeData
{
    std::vector<std::string> sections;
    std::map<std::string, CmapParameter> cmap;
};

GaffParameters Load_Gaff_Parameters(const std::string& dat_path,
                                    const std::string& frcmod_path = "");

FrcmodXpongeData Load_Frcmod_As_Xponge_Data(const std::string& filename);

std::pair<std::string, std::string> Canonical2(const std::string& a,
                                               const std::string& b);
std::tuple<std::string, std::string, std::string> Canonical3(
    const std::string& a,
    const std::string& b,
    const std::string& c);

}  // namespace Amber
}  // namespace Xponge
