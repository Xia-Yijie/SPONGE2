/*
nvrtc for CUDA backend,
directly compiling for CPU backend
*/
#pragma once

#include "../../common.h"
#include "../../control.h"

#ifdef USE_CUDA
struct JIT_Function
{
   private:
    CUfunction function;

   public:
    std::string error_reason;

    void Compile(std::string source)
    {
        std::string common_h =
#include "jit.h"
            const char* headers[1] = {common_h.c_str()};
        const char* header_names[1] = {"common.h"};
        nvrtcProgram prog;
        nvrtcCreateProgram(&prog, source.c_str(), NULL, 1, headers,
                           header_names);

        std::string arch = "-arch sm_";
        deviceProp prop;

        getDeviceProperties(&prop, 0);
        int runtime_arch_bin = prop.major * 10 + prop.minor;
        arch += std::to_string(runtime_arch_bin);
        const char* opts[] = {"--use_fast_math", arch.c_str()};
        if (nvrtcCompileProgram(prog, 1, opts) != NVRTC_SUCCESS)
        {
            size_t logSize;
            nvrtcGetProgramLogSize(prog, &logSize);
            char* log_ = new char[logSize];
            nvrtcGetProgramLog(prog, log_);
            error_reason = log_;
            delete log_;
            return;
        }
        size_t pos1 = source.find("extern");
        size_t pos2 =
            source.find(string_format("%q%C%q%", {{"q", {'"'}}}), pos1);
        if (pos2 == source.npos)
        {
            error_reason =
                R"(extern "C" should be placed in front of the function name)";
            return;
        }
        pos1 = source.find_first_of("(", pos2);
        pos2 = source.find_last_of(" ", pos1);
        std::string name = string_strip(source.substr(pos2, pos1 - pos2));
        if (name == "__launch_bounds__")
        {
            pos1 = source.find_first_of("(", pos1 + 1);
            pos2 = source.find_last_of(" ", pos1);
            name = string_strip(source.substr(pos2, pos1 - pos2));
        }
        size_t ptxSize;
        nvrtcGetPTXSize(prog, &ptxSize);
        char* ptx = new char[ptxSize];
        nvrtcGetPTX(prog, ptx);
        CUmodule module;
        if (cuModuleLoadDataEx(&module, ptx, 0, 0, 0) != CUDA_SUCCESS)
        {
            error_reason = string_format(
                "Fail to load the module from PTX for %f%", {{"f", name}});
            return;
        }
        if (cuModuleGetFunction(&function, module, name.c_str()) !=
            CUDA_SUCCESS)
        {
            error_reason = string_format(
                "Fail to get the name from the module for %f%", {{"f", name}});
            return;
        }
        delete ptx;

        return;
    }

    void operator()(dim3 blocks, dim3 threads, cudaStream_t stream,
                    unsigned int shared_memory_size, std::vector<void*> args)
    {
        CUresult result = cuLaunchKernel(
            function, blocks.x, blocks.y, blocks.z, threads.x, threads.y,
            threads.z, shared_memory_size, stream, &args[0], NULL);
        if (result != CUDA_SUCCESS)
        {
            const char* name;
            const char* string;
            cuGetErrorName(result, &name);
            cuGetErrorString(result, &string);
            error_reason = string_format("Kernel Launch Error %NAME%: %STRING%",
                                         {{"NAME", name}, {"STRING", string}});
            printf("Kernel Launch Error %s: %s\n", name, string);
        }
    }
};
#else

struct JIT_Function
{
   private:
    std::string temp_string;
    void (*function)(void** args);

   public:
    std::string error_reason;
    void Compile(std::string source)
    {
        size_t pos1 = source.find("extern");
        size_t pos2 =
            source.find(string_format("%q%C%q%", {{"q", {'"'}}}), pos1);
        if (pos2 == source.npos)
        {
            error_reason =
                R"(extern "C" should be placed in front of the function name)";
            return;
        }
        pos1 = source.find_first_of("(", pos2);
        pos2 = source.find_last_of(" ", pos1);
        std::string func_name = string_strip(source.substr(pos2, pos1 - pos2));
        if (func_name == "__launch_bounds__")
        {
            pos1 = source.find_first_of("(", pos1 + 1);
            pos2 = source.find_last_of(" ", pos1);
            func_name = string_strip(source.substr(pos2, pos1 - pos2));
        }
        pos1 = source.find_first_of("(", pos2);
        pos2 = source.find_first_of(")", pos1);
        std::vector<std::string> args =
            string_split(source.substr(pos1 + 1, pos2 - pos1 - 1), ",");
        source = source.replace(pos1 + 1, pos2 - pos1 - 1, "void** args");
        pos2 = source.find_first_of("{", pos1) + 1;
        auto is_identifier_char = [](char c)
        { return (c == '_') || std::isalnum(static_cast<unsigned char>(c)); };
        for (int i = args.size() - 1; i >= 0; i--)
        {
            std::string new_arg = string_strip(args[i]);
            if (new_arg.empty())
            {
                error_reason = "Fail to parse argument list for JIT wrapper.";
                return;
            }
            size_t var_end = new_arg.find_last_not_of(" \t");
            if (var_end == std::string::npos)
            {
                error_reason = "Fail to parse argument list for JIT wrapper.";
                return;
            }
            size_t var_start = var_end;
            while (var_start > 0 && is_identifier_char(new_arg[var_start]))
            {
                var_start--;
            }
            if (!is_identifier_char(new_arg[var_start]))
            {
                var_start++;
            }
            if (var_start > var_end)
            {
                error_reason = "Fail to parse argument list for JIT wrapper.";
                return;
            }
            std::string type_name = string_strip(new_arg.substr(0, var_start));
            std::string var_name = string_strip(
                new_arg.substr(var_start, var_end - var_start + 1));
            if (type_name.empty() || var_name.empty())
            {
                error_reason = "Fail to parse argument list for JIT wrapper.";
                return;
            }
            std::string stmt = type_name + " " + var_name + " = *((" +
                               type_name + "*)(args[" + std::to_string(i) +
                               "]));";
            source.insert(pos2, stmt);
        }
        int fd = -1;
#ifdef _WIN32
        char temp_dir[MAX_PATH] = {0};
        char temp_name[MAX_PATH] = {0};
        if (GetTempPathA(MAX_PATH, temp_dir) == 0 ||
            GetTempFileNameA(temp_dir, "SPJ", 0, temp_name) == 0)
        {
            error_reason = "Fail to create a temporary file";
            return;
        }
        temp_string = temp_name;
        fd = _open(temp_string.c_str(), _O_RDWR | _O_BINARY | _O_TRUNC);
#else
        char temp_name[CHAR_LENGTH_MAX] = "/tmp/SPONGE_jittemp_XXXXXX";
        fd = mkstemp(temp_name);
#endif
        if (fd < 0)
        {
            error_reason = "Fail to create a temporary file";
            return;
        }
#ifdef _WIN32
        if (_write(fd, source.c_str(),
                   static_cast<unsigned int>(source.size())) !=
            static_cast<int>(source.size()))
#else
        if (write(fd, source.c_str(), source.size()) != source.size())
#endif
        {
            error_reason = "Fail to write the source to the temporary file";
            return;
        }
#ifdef _WIN32
        _close(fd);
#else
        temp_string = temp_name;
#endif
        auto quote_path = [](const fs::path& path)
        {
            std::string value = path.string();
            if (value.find_first_of(" \t\"'") != std::string::npos)
            {
                value = "\"" + value + "\"";
            }
            return value;
        };

        // 1) 读取 compile_commands.json，复用主程序的编译选项
        const fs::path bin_path = Get_SPONGE_Directory();
        const fs::path bin_dir = bin_path.parent_path();
        fs::path project_root = bin_dir.parent_path();
        fs::path cmake_cm_file;
        std::vector<fs::path> cmake_cm_candidates = {
            bin_dir / "compile_commands.json",
            project_root / "build" / "compile_commands.json",
            project_root / "build-cpu" / "compile_commands.json",
            project_root / "build-debug" / "compile_commands.json",
            project_root / "build-asan" / "compile_commands.json",
            project_root / "build-ubsan" / "compile_commands.json"};
        for (const auto& candidate : cmake_cm_candidates)
        {
            if (fs::exists(candidate))
            {
                cmake_cm_file = candidate;
                break;
            }
        }
        if (cmake_cm_file.empty())
        {
            error_reason = "找不到 compile_commands.json。尝试路径: ";
            for (size_t i = 0; i < cmake_cm_candidates.size(); i++)
            {
                if (i != 0) error_reason += ", ";
                error_reason += cmake_cm_candidates[i].string();
            }
            return;
        }
        std::ifstream f_cm(cmake_cm_file);
        if (!f_cm.is_open())
        {
            error_reason = "无法读取 compile_commands.json: ";
            error_reason += cmake_cm_file.string();
            return;
        }
        std::string line;
        std::string configure;
        while (std::getline(f_cm, line))
        {
            auto pos = line.find("\"command\":");
            if (pos == std::string::npos) continue;
            pos = line.find('"', pos + 9);
            if (pos == std::string::npos) continue;
            pos += 1;
            auto end = line.find("-o", pos);
            if (end == std::string::npos) continue;
            configure = line.substr(pos, end - pos);
            break;
        }
        f_cm.close();
        if (configure.empty())
        {
            error_reason = "无法从 compile_commands.json 中解析编译命令";
            return;
        }
        while (!configure.empty() &&
               (configure.back() == '\n' || configure.back() == '\r'))
        {
            configure.pop_back();
        }

        // 2) 定位 common.h 所在目录，补充 -I 参数
        fs::path include_dir;
        std::vector<fs::path> include_candidates = {
            project_root / "SPONGE", project_root, bin_path.parent_path()};
        for (const auto& candidate : include_candidates)
        {
            if (!candidate.empty() && fs::exists(candidate / "common.h"))
            {
                include_dir = candidate;
                break;
            }
        }
        if (include_dir.empty())
        {
            error_reason = "无法定位 common.h，项目根目录: ";
            error_reason += project_root.string();
            return;
        }

        configure += " -fvisibility=default -fPIC -shared -o " + temp_string +
                     ".so -x c++ " + temp_string + " -I" +
                     quote_path(include_dir) + " > " + temp_string +
                     ".log 2>&1";
        if (system(configure.c_str()))
        {
            error_reason = "Fail to compile the source file: \n";
            error_reason += configure;
            return;
        }
#ifdef _WIN32
        if (_unlink(temp_string.c_str()) != 0)
#else
        if (unlink(temp_string.c_str()) != 0)
#endif
        {
            error_reason = "Fail to unlink the temporary file: ";
            error_reason += temp_string;
            return;
        }
#ifdef _WIN32
        std::string lib_name = temp_string + ".dll";
        HMODULE handle = dlopen(lib_name.c_str(), 0);
        if (handle == NULL)
        {
            error_reason =
                "Fail to load dynamic library, win32 error code: " +
                std::to_string(static_cast<unsigned long>(GetLastError()));
            return;
        }
        function = reinterpret_cast<void (*)(void**)>(
            dlsym(handle, func_name.c_str()));
        if (function == NULL)
        {
            error_reason =
                "Fail to find symbol, win32 error code: " +
                std::to_string(static_cast<unsigned long>(GetLastError()));
            return;
        }
#else
        std::string lib_name = temp_string + ".so";
        dlerror();
        HMODULE handle = dlopen(lib_name.c_str(), RTLD_NOW);
        if (const char* err = dlerror())
        {
            error_reason = err;
            return;
        }
        dlerror();
        function = (void (*)(void**))dlsym(handle, func_name.c_str());
        if (const char* err = dlerror())
        {
            error_reason = err;
            return;
        }
#endif
    }

    void operator()(dim3 blocks, dim3 threads, deviceStream_t stream,
                    unsigned int shared_memory_size,
                    std::initializer_list<const void*> args)
    {
        std::vector<void*> temp;
        temp.reserve(args.size());
        for (const void* ptr : args)
        {
            temp.push_back(const_cast<void*>(ptr));
        }
        function(&temp[0]);
    }

    // 重构以直接读入 std::vector<void*>
    void operator()(dim3 blocks, dim3 threads, deviceStream_t stream,
                    unsigned int shared_memory_size, std::vector<void*> args)
    {
        function(&args[0]);
    }
};
#endif
