#pragma once

#include <errno.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#ifdef _WIN32
#ifndef F_OK
#define F_OK 0
#endif
#define access _access
#define chdir _chdir
#define getcwd _getcwd
#endif

inline std::string Get_Current_Working_Directory()
{
    char* buffer = NULL;
    buffer = getcwd(NULL, 0);
    if (buffer == NULL) return "Unknown";
    std::string path = buffer;
    free(buffer);
    return path;
}

inline bool Path_Exists(const std::string& path)
{
    return access(path.c_str(), F_OK) == 0;
}

inline int Set_Current_Working_Directory(const std::string& path)
{
    return chdir(path.c_str());
}

inline bool Is_Absolute_Path(const std::string& path)
{
    if (path.empty()) return false;
    if (path[0] == '/') return true;
    if (path.size() >= 2 && path[0] == '\\' && path[1] == '\\') return true;
    return path.size() >= 3 && path[1] == ':' &&
           (path[2] == '/' || path[2] == '\\');
}

inline std::string Parent_Path(const std::string& path)
{
    const std::size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return "";
    if (slash == 0) return path.substr(0, 1);
    if (slash == 2 && path[1] == ':') return path.substr(0, 3);
    return path.substr(0, slash);
}

inline std::string Join_Path(const std::string& dir, const std::string& name)
{
    if (dir.empty() || Is_Absolute_Path(name)) return name;
    const char last = dir[dir.size() - 1];
    if (last == '/' || last == '\\') return dir + name;
    return dir + "/" + name;
}

inline std::string Path_Extension(const std::string& path)
{
    const std::size_t slash = path.find_last_of("/\\");
    const std::size_t name_start =
        (slash == std::string::npos) ? 0 : slash + 1;
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || dot <= name_start) return "";
    return path.substr(dot);
}

inline std::string Get_SPONGE_Directory()
{
    char path[CHAR_LENGTH_MAX] = {0};
#if defined(__APPLE__)
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0)
    {
        std::string buffer(size, '\0');
        if (_NSGetExecutablePath(buffer.data(), &size) == 0)
        {
            return buffer.c_str();
        }
        return Get_Current_Working_Directory();
    }
    return path;
#else
    int l = readlink("/proc/self/exe", path, CHAR_LENGTH_MAX - 1);
    if (l > 0 && l < CHAR_LENGTH_MAX)
    {
        path[l] = 0;
        return path;
    }
    return Get_Current_Working_Directory();
#endif
}

inline std::string Get_Wall_Time()
{
    time_t timep;
    time(&timep);
    return asctime(localtime(&timep));
}

#ifdef USE_GPU
inline std::string Get_Device_Runtime_Arch_Name(const deviceProp& prop)
{
#ifdef USE_HIP
    if (prop.gcnArchName[0] != '\0')
    {
        return prop.gcnArchName;
    }
    return "unknown";
#else
    return "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
#endif
}
#endif

#ifdef USE_CUDA
static __global__ void device_get_built_arch(int* answer)
{
#ifdef __CUDA_ARCH__
    *answer = __CUDA_ARCH__ / 10;
#else
    *answer = 0;
#endif
}

// 获取当前GPU的架构
inline int Get_Built_Arch()
{
    int answer, *d_answer;
    Device_Malloc_Safely((void**)&d_answer, sizeof(int));
    device_get_built_arch<<<1, 1>>>(d_answer);
    deviceMemcpy(&answer, d_answer, sizeof(int), deviceMemcpyDeviceToHost);
    deviceFree(d_answer);
    d_answer = NULL;
    return answer;
}
#endif  // USE_CUDA
