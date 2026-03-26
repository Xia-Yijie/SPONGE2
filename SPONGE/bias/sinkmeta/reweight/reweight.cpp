#include "../meta.h"
#include "../util.h"
using sinkmeta::split_sentence;

#ifdef USE_GPU
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
float exp_added(float a, const float b)
{
    return fmaxf(a, b) + logf(1.0 + expf(-fabsf(a - b)));
}
using TimePoint = std::chrono::time_point<std::chrono::system_clock>;

std::string GetTime(TimePoint& local_time)
{
    local_time = std::chrono::system_clock::now();
    time_t now_time = std::chrono::system_clock::to_time_t(local_time);
    std::string time_str(asctime(localtime(&now_time)));
    return time_str.substr(0, time_str.find('\n'));
}

std::string GetDuration(const TimePoint& late_time, const TimePoint& early_time,
                   float& duration)
{
    // Some constants.
    const auto elapsed = late_time - early_time;
    const size_t milliseconds = static_cast<size_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
    const size_t seconds = milliseconds / 1000000L;
    const size_t microseconds = milliseconds % 1000000L;
    const size_t Second2Day = 86400L;
    const size_t Second2Hour = 3600L;
    const size_t Second2Minute = 60L;
    const size_t day = seconds / Second2Day;
    const size_t hour = seconds % Second2Day / Second2Hour;
    const size_t min = seconds % Second2Day % Second2Hour / Second2Minute;
    const size_t second = seconds % Second2Day % Second2Hour % Second2Minute;
    // Calculate duration in second.
    const int BufferSize = 2048;
    char buffer[BufferSize];
    sprintf(buffer,
            "%lu days %lu hours %lu minutes %lu seconds %.1f milliseconds",
            static_cast<unsigned long>(day), static_cast<unsigned long>(hour),
            static_cast<unsigned long>(min), static_cast<unsigned long>(second),
            microseconds * 0.001);
    duration = milliseconds * 0.000001;  // From millisecond to second.
    return std::string(buffer);
}
struct Zdiff
{
    const float f1, f2;

    Zdiff(float f1, float f2) : f1(f1), f2(f2) {}

    __host__ __device__ float operator()(const float& x) const
    {
        return expf(f1 * x - f2);
    }
};
struct ExpMinusMax
{
    const float maxVal;

    ExpMinusMax(float maxVal) : maxVal(maxVal) {}

    __host__ __device__ float operator()(const float& x) const
    {
        return expf(x - maxVal);
    }
};
#ifdef USE_GPU
using DeviceFloatVector = thrust::device_vector<float>;
using DeviceDoubleVector = thrust::device_vector<double>;

static __global__ void Update_Grid_With_Hill(
    int total_size, int ndim,
    const int* num_points, const float* lower, const float* spacing,
    const float* hill_centers, const float* hill_inv_w,
    const float* hill_periods,
    float factor, int update_force,
    float* potential, float* force)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    float dx[8], df[8];
    int flat = idx;
    for (int d = 0; d < ndim; ++d)
    {
        int i = flat % num_points[d];
        flat /= num_points[d];
        float coord = lower[d] + (i + 0.5f) * spacing[d];
        float diff = coord - hill_centers[d];
        if (hill_periods[d] > 0.0f)
        {
            diff -= roundf(diff / hill_periods[d]) * hill_periods[d];
        }
        float x = diff * hill_inv_w[d];
        dx[d] = expf(-0.5f * x * x);
        df[d] = -x * hill_inv_w[d] * dx[d];
    }

    float pot = 1.0f;
    for (int d = 0; d < ndim; ++d)
        pot *= dx[d];
    potential[idx] += factor * pot;

    if (update_force)
    {
        for (int d = 0; d < ndim; ++d)
        {
            float tder = 1.0f;
            for (int j = 0; j < ndim; ++j)
                tder *= (j == d) ? df[j] : dx[j];
            force[idx * ndim + d] += factor * tder;
        }
    }
}

static __global__ void Reduce_Max_Kernel(
    int n, const float* data, float* block_max)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float val = -1e30f;
    if (idx < n) val = data[idx];
    if (idx + blockDim.x < n) val = fmaxf(val, data[idx + blockDim.x]);
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) block_max[blockIdx.x] = sdata[0];
}
#else
using DeviceFloatVector = std::vector<float>;
using DeviceDoubleVector = std::vector<double>;

static void Update_Grid_With_Hill(
    int total_size, int ndim,
    const int* num_points, const float* lower, const float* spacing,
    const float* hill_centers, const float* hill_inv_w,
    const float* hill_periods,
    float factor, int update_force,
    float* potential, float* force)
{
    for (int idx = 0; idx < total_size; ++idx)
    {
        float dx[8], df[8];
        int flat = idx;
        for (int d = 0; d < ndim; ++d)
        {
            int i = flat % num_points[d];
            flat /= num_points[d];
            float coord = lower[d] + (i + 0.5f) * spacing[d];
            float diff = coord - hill_centers[d];
            if (hill_periods[d] > 0.0f)
            {
                diff -= roundf(diff / hill_periods[d]) * hill_periods[d];
            }
            float x = diff * hill_inv_w[d];
            dx[d] = expf(-0.5f * x * x);
            df[d] = -x * hill_inv_w[d] * dx[d];
        }

        float pot = 1.0f;
        for (int d = 0; d < ndim; ++d)
            pot *= dx[d];
        potential[idx] += factor * pot;

        if (update_force)
        {
            for (int d = 0; d < ndim; ++d)
            {
                float tder = 1.0f;
                for (int j = 0; j < ndim; ++j)
                    tder *= (j == d) ? df[j] : dx[j];
                force[idx * ndim + d] += factor * tder;
            }
        }
    }
}

static void Reduce_Max_Kernel(
    int n, const float* data, float* block_max)
{
    float val = -1e30f;
    for (int i = 0; i < n; ++i)
    {
        val = fmaxf(val, data[i]);
    }
    block_max[0] = val;
}
#endif

float PartitionFunction(const float factor, float& i_max,
                        const DeviceFloatVector& values)
{
    if (values.empty() || factor < 0.0000001)
    {
        return 0.0;
    }
#ifdef USE_GPU
    i_max = *thrust::max_element(values.begin(), values.end());
    float maxVal = factor * i_max;

    float sum = thrust::transform_reduce(values.begin(), values.end(),
                                         Zdiff(factor, maxVal), 0.0,
                                         thrust::plus<float>());
#else
    i_max = *std::max_element(values.begin(), values.end());
    float maxVal = factor * i_max;
    float sum = 0.0f;
    for (const auto& x : values)
    {
        sum += std::exp(factor * x - maxVal);
    }
#endif
    return maxVal + logf(sum);
}
float logSumExp(const std::vector<float>& values)
{
    if (values.empty()) return -std::numeric_limits<float>::infinity();

#ifdef USE_GPU
    DeviceDoubleVector d_values(values.begin(), values.end());
    float maxVal = *thrust::max_element(d_values.begin(), d_values.end());

    float sum = thrust::transform_reduce(d_values.begin(), d_values.end(),
                                         ExpMinusMax(maxVal), 0.0,
                                         thrust::plus<float>());
#else
    float maxVal = *std::max_element(values.begin(), values.end());
    float sum = 0.0f;
    for (const auto& v : values)
    {
        sum += std::exp(v - maxVal);
    }
#endif
    return maxVal + logf(sum);
}

void hilllog(const std::string fn, const std::vector<float>& hillcenter,
             const std::vector<float>& hillheight)
{
    if (!fn.empty())
    {
        std::ofstream hillsout;
        hillsout.open(fn.c_str(), std::fstream::app);
        hillsout.precision(8);
        for (auto& gauss : hillcenter)
        {
            hillsout << gauss << "\t";
        }
        for (auto& hh : hillheight)
        {
            hillsout << hh << "\t";
        }
        hillsout << std::endl;
        hillsout.close();
    }
}
void showProgressBar(int progress, int total, int barWidth = 70)
{
    float percentage = static_cast<float>(progress) / total;
    int pos = barWidth * percentage;

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(percentage * 100.0) << " %\r";
    std::cout.flush();
}
bool META::Read_Edge_File(const char* file_name, std::vector<float>& potential)
{
    FILE* temp_file = NULL;
    int grid_size = 0;
    bool readsuccess = true;
    int total = mgrid->total_size;
    std::vector<Gdata> force_from_file;
    printf("Reading %d grid of edge effect\n", total);
    temp_file = fopen(file_name, "r+");
    if (temp_file != NULL)
    {
        fseek(temp_file, 0, SEEK_END);

        if (ftell(temp_file) == 0)
        {
            printf("Edge file %s is empty\n", file_name);
        }
        else
        {
            Open_File_Safely(&temp_file, file_name, "r");
            char temp_char[256] = " ";  // empty but not nullptr
            int scanf_ret = 0;
            char* grid_val = temp_char;
            while (grid_val != NULL)
            {
                grid_val = fgets(temp_char, 256, temp_file);  // grid line
                std::vector<std::string> words;
                int nwords = split_sentence(temp_char, words);
                Gdata force(ndim, 0.);
                potential.push_back(
                    logf(std::stof(words[ndim])));  /// log sum exp!!!!!
                if (nwords == 1 + ndim * 2)
                {
                    for (int i = 0; i < ndim; ++i)
                    {
                        force[i] = std::stof(words[1 + ndim + i]);
                    }
                }
                else
                {
                    printf(
                        "Error reading Edge file %s, line %d of %d:\n Format "
                        "should have %d while only %d have been read\n",
                        file_name, grid_size, total, 1 + 2 * ndim, nwords);
                    return false;
                }
                ++grid_size;
                force_from_file.push_back(force);
            }
        }
        fclose(temp_file);
    }
    if (grid_size - 1 != total)
    {
        printf("Error reading Edge file %s, line %d of %d\n", file_name,
               grid_size, total);
        return false;
    }
    mgrid->normal_lse = potential;
    sum_max = *std::max_element(potential.begin(), potential.end());
    if (scatter_size < total && do_negative)
    {
        for (int idx = 0; idx < mgrid->total_size; ++idx)
        {
            for (int d = 0; d < ndim; ++d)
            {
                mgrid->normal_force[idx * ndim + d] = force_from_file[idx][d];
            }
        }
    }
    return readsuccess;
}
// Load hills from output file.
int META::Load_Hills(const std::string& fn)
{
    std::ifstream hillsin(fn.c_str(), std::ios::in);
    if (!hillsin.is_open())
    {
        printf("Warning, No record of hills\n");
        return 0;
    }
    const std::string file_content((std::istreambuf_iterator<char>(hillsin)),
                                   std::istreambuf_iterator<char>());
    hillsin.close();

    const int& cvsize = ndim;
    std::istringstream iss(file_content);
    std::string tstr;
    std::vector<std::string> words;
    int num_hills = 0;
    while (std::getline(iss, tstr, '\n'))
    {
        Axis values;
        split_sentence(tstr, words);
        if (words.size() < cvsize + 1)
        {
            printf("The format of Hills file \"%s\" near \"%s\" is wrong.",
                   fn.c_str(), tstr.c_str());
        }
        for (int i = 0; i < cvsize; ++i)
        {
            float center = std::stof(words[i]);
            values.push_back(center);
        }
        float theight = std::stof(words[cvsize]);
        if (do_negative || use_scatter)
        {
            float p_max = std::stof(words[cvsize + 1]);
            int p_id = std::stoi(words[cvsize + 2]);
            if (p_id < scatter_size)
            {
                float Phi_s = expf(mgrid->normal_lse[mgrid->Get_Flat_Index(values)]);
                float vshift =
                    (p_max + dip * CONSTANT_kB * temperature) *
                    expf(-mgrid->normal_lse[mgrid->Get_Flat_Index(
                        mscatter->Get_Coordinate(p_id))]);
                vsink.push_back(Phi_s * vshift);
            }
            else
            {
                printf("Error reading sink projecting id: %d\n", p_id);
                return -1;
            }
        }
        Hill newhill = Hill(values, sigmas, periods, theight);
        hills.push_back(newhill);
        ++num_hills;
    }
    return num_hills;
}
float META::Calc_Hill(const Axis& values, const int i)
{
    float potential = 0;
    for (int j = 0; j < i; ++j)
    {
        Hill& hill = hills[j];
        const Gdata& tder = hill.Calc_Hill(values);
        potential += hill.potential * hill.height;
    }
    return potential;
}
float META::Sum_Hills(int history_freq)
{
    if (history_freq == 0)
    {
        return 0.;
    }
    TimePoint start_time, end_time;
    float duration;
    GetTime(start_time);
    int nhills = Load_Hills("myhill.log");
    FILE* temp_file = NULL;
    printf("\r\nLoad hills file successfully, now calculate RCT!!!\n");
    Open_File_Safely(&temp_file, "history.log", "w");
    // first loop: history
    float old_potential;
    minus_beta_f_plus_v =
        1. / (welltemp_factor - 1.) / CONSTANT_kB / temperature;  /// 300K
    minus_beta_f = welltemp_factor * minus_beta_f_plus_v;
    float total_gputime = 0.;
    for (int i = 0; i < nhills; ++i)
    {
        showProgressBar(i, nhills);
        Hill& hill = hills[i];
        Axis values;
        for (auto& gauss : hill.gsf)
        {
            values.push_back(gauss.GetCenter());
        }
        old_potential = Calc_Hill(values, i);
        if (history_freq != 0 && (i % history_freq == 0))
        {
            mgrid->potential.assign(mgrid->total_size, 0.0f);
            TimePoint tstart, tend;
            float gputime;
            GetTime(tstart);
            // RCT calculation
            DeviceFloatVector d;
            if (use_scatter)
            {
                for (int iter = 0; iter < scatter_size; ++iter)
                {
                    mscatter->potential[iter] =
                        Calc_Hill(mscatter->Get_Coordinate(iter), i);
                }
#ifdef USE_GPU
                d = DeviceFloatVector(mscatter->potential.begin(),
                                      mscatter->potential.end());
#else
                d = mscatter->potential;
#endif
            }
            else  // use grid
            {
                for (int idx = 0; idx < mgrid->total_size; ++idx)
                {
                    mgrid->potential[idx] = Calc_Hill(mgrid->Get_Coordinates(idx), i);
                }
#ifdef USE_GPU
                d = DeviceFloatVector(mgrid->potential.begin(),
                                      mgrid->potential.end());
#else
                d = mgrid->potential;
#endif
            }
            GetTime(tend);
            GetDuration(tend, tstart, gputime);
            total_gputime += gputime;
            float Z_0 = PartitionFunction(minus_beta_f, potential_max, d);
            float Z_V = PartitionFunction(minus_beta_f_plus_v, potential_max, d);
            rct = CONSTANT_kB * temperature * (Z_0 - Z_V);
            float rbias = old_potential - rct;
            fprintf(temp_file, "%f\t%f\t%f\t%f\n", old_potential, rbias, rct,
                    vsink[i]);
        }
    }
    fclose(temp_file);
    GetTime(end_time);
    GetDuration(end_time, start_time, duration);
    int hours = floor(duration / 3600);
    float nohour = duration - 3600 * hours;
    int mins = floor(nohour / 60);
    float seconds = nohour - 60 * mins;
    printf(
        "The RBIAS & RCT calculation cost %f of %f seconds: %d hour %d min %f "
        "second\n",
        total_gputime / duration, duration, hours, mins, seconds);
    return old_potential;
}
void META::Edge_Effect(const int dim, const int scatter_size)
{
    std::vector<float> potential_from_file;
    const char* file_name = edge_file_name;

    int total = mgrid->total_size;
    if (scatter_size == total)
    {
        float normalization = 1.0;
        float sqrtpi = sqrtf(CONSTANT_Pi * 2);
        for (int i = 0; i < ndim; i++)
        {
            normalization /= cv_deltas[i] * sigmas[i] / sqrtpi;
        }
        mgrid->normal_lse.assign(mgrid->total_size, log(normalization));
    }
    if (!Read_Edge_File(file_name, potential_from_file))
    {
        int it_progress = 0;
        printf("Calculation the %d grid of edge effect\n", total);
        FILE* temp_file = NULL;
        Open_File_Safely(&temp_file, file_name, "w+");
        // default 1-dimensional scatter, maybe slow for 3D-mask!
        Axis esigmas;
        float adjust_factor = 1.0;
        for (int i = 0; i < ndim; ++i)
        {
            esigmas.push_back(sigmas[i] * adjust_factor);
        }
        for (int gidx = 0; gidx < mgrid->total_size; ++gidx)
        {
            showProgressBar(++it_progress, total);
            const Axis values = mgrid->Get_Coordinates(gidx);
            double sum_hills = 0.;
            std::vector<float> prefactor;
            if (catheter)
            {
                float R = sqrtf(sigma_s * sigma_s -
                                2.0 * delta_sigma[mscatter->Get_Index(values)]);
                for (int i = 0; i < ndim; ++i)
                {
                    esigmas[i] = R;
                }
            }
            Hill hill = Hill(values, esigmas, periods, 1.0);
            std::vector<int> indices;
            if (do_cutoff)
            {
                indices = mscatter->Get_Neighbor(values, cutoff);
            }
            else
            {
                indices = std::vector<int>(scatter_size);
                std::iota(indices.begin(), indices.end(), 0);
            }
            for (auto index : indices)
            {
                const Axis& neighbor = mscatter->Get_Coordinate(index);
                float pregauss = 0.;
                for (int i = 0; i < ndim; ++i)
                {
                    float diff = (values[i] - neighbor[i]);
                    if (periods[i] != 0.0)
                    {
                        diff -= roundf(diff / periods[i]) * periods[i];
                    }
                    float distance = diff * esigmas[i];
                    pregauss -= 0.5 * distance * distance;
                }
                if (do_negative)
                {
                    const Gdata& tder = hill.Calc_Hill(neighbor);
                    float hill_potential = hill.potential;
                    float* nf_data = &mgrid->normal_force[gidx * ndim];
                    if (catheter)
                    {
                        float* v = &mscatter->rotate_v[index * ndim];
                        float s = Project_To_Path(
                            Gdata(v, v + ndim), values, neighbor);
                        float dss = delta_sigma[index] * s * s;
                        pregauss -= dss;
                        float s_shrink = expf(-dss);
                        hill_potential *= s_shrink;
                        for (int i = 0; i < ndim; ++i)
                        {
                            float dx = values[i] - neighbor[i];
                            if (periods[i] != 0.0)
                            {
                                dx -= roundf(dx / periods[i]) * periods[i];
                            }
                            for (int j = 0; j < ndim; ++j)
                            {
                                float partial =
                                    2 * delta_sigma[index] * v[i] * v[j] * dx;
                                nf_data[i] += partial * hill_potential;
                            }
                            nf_data[i] += tder[i] * s_shrink;
                        }
                    }
                    else
                    {
                        for (int i = 0; i < ndim; ++i)
                        {
                            nf_data[i] += tder[i];
                        }
                    }
                }
                prefactor.push_back(pregauss);
            }

            float logsumhills = logSumExp(prefactor);
            sum_max = fmaxf(logsumhills, sum_max);
            std::vector<float> sum_potential(1 + ndim, expf(logsumhills));
            mgrid->normal_lse[gidx] = logsumhills;
            if (do_negative)
            {
                float* nf_data = &mgrid->normal_force[gidx * ndim];
                for (int i = 0; i < ndim; ++i)
                {
                    sum_potential[i + 1] = nf_data[i];
                }
            }
            for (auto& v : values)
            {
                fprintf(temp_file, "%f\t", v);
            }
            for (auto& s : sum_potential)
            {
                fprintf(temp_file, "%f\t", s);
            }
            fprintf(temp_file, "\n");
        }
        fclose(temp_file);
    }
    if (dim == 1)
    {
        Pick_Scatter("lnbias.dat");
    }
}

void META::Pick_Scatter(const std::string fn)
{
    std::ofstream hillsout;
    hillsout.open(fn.c_str(), std::fstream::out);
    hillsout.precision(8);
    for (int index = 0; index < scatter_size; ++index)
    {
        const Axis& neighbor = mscatter->Get_Coordinate(index);
        float lnbias = mgrid->normal_lse[mgrid->Get_Flat_Index(neighbor)];
        hillsout << index << "\t" << lnbias << "\t" << exp(lnbias) << std::endl;
    }
    hillsout.close();
}
float META::Normalization(const Axis& values, float factor, bool do_normalise)
{
    if (do_normalise)
    {
        if (usegrid)
        {
            return factor * expf(-mgrid->normal_lse[0]);
        }
        if (convmeta)
        {
            return factor *
                   expf(-mgrid->normal_lse[mgrid->Get_Flat_Index(
                       mscatter->Get_Coordinate(max_index))]);
        }
        else
        {
            return factor * expf(-mgrid->normal_lse[mgrid->Get_Flat_Index(values)]);
        }
    }
    else
    {
        return factor;
    }
}
void META::Get_Height(const Axis& values)
{
    Estimate(values, true, false);
    height = height_0;
    if (temperature < 0.00001 || welltemp_factor > 60000)
    {
        return;  // avoid /0 = nan
    }
    if (is_welltemp == 1)
    {
        // height_welltemp
        height = height_0 * expf(-potential_backup / (welltemp_factor - 1) /
                                 CONSTANT_kB / temperature);
    }
}

float META::Calc_V_Shift(const Axis& values)
{
    if (!do_negative)
    {
        return 0.;
    }
    int nidx = mgrid->Get_Flat_Index(values);
    if (convmeta)
    {
        return new_max * expf(mgrid->normal_lse[nidx]);
    }
    else  // GRW
    {
        return new_max * (mgrid->normal_lse[nidx] - sum_max) *
               expf(mgrid->normal_lse[nidx]);
    }
}
void META::Get_Reweighting_Bias(float temp)
{
    if (temperature < 0.00001)
    {
        return;  // avoid /0 = nan
    }
    float beta = 1.0 / CONSTANT_kB / temperature;
    minus_beta_f_plus_v = beta / (welltemp_factor - 1.);
    minus_beta_f = welltemp_factor * minus_beta_f_plus_v;
    bias = potential_local;
    rbias = potential_backup;
    float Z_0_sink = 0.;
    float Z_V_sink = 0.;
    if (mscatter != nullptr)
    {
        for (int iter = 0; iter < scatter_size; ++iter)
        {
            const Axis& coor = mscatter->Get_Coordinate(iter);
            Estimate(coor, true, false);
            Z_0_sink = exp_added(Z_0_sink, minus_beta_f * potential_backup);
            Z_V_sink = exp_added(Z_V_sink, minus_beta_f_plus_v * potential_backup +
                                               beta * Calc_V_Shift(coor));
            if (potential_backup > potential_max)
            {
                max_index = iter;
                potential_max = potential_backup;
            }
        }
    }
    else if (mgrid != nullptr)
    {
        if (subhill)
        {
            for (int idx = 0; idx < mgrid->total_size; ++idx)
            {
                Estimate(mgrid->Get_Coordinates(idx), true, false);
                potential_max = max(potential_max, potential_backup);
            }
        }
        else
        {
            Launch_Device_Kernel(
                Reduce_Max_Kernel,
                reduce_num_blocks, 256,
                sizeof(float) * 256, NULL,
                mgrid->total_size, mgrid->d_potential, d_reduce_buf);
            deviceMemcpy(h_reduce_buf, d_reduce_buf,
                         sizeof(float) * reduce_num_blocks,
                         deviceMemcpyDeviceToHost);
            for (int i = 0; i < reduce_num_blocks; ++i)
            {
                potential_max = fmaxf(potential_max, h_reduce_buf[i]);
            }
        }
    }
    rct = CONSTANT_kB * temperature * (Z_0_sink - Z_V_sink);
    rbias -= rct + temp;
}

void META::Add_Potential(float temp, int steps)
{
    if (!is_initialized)
    {
        return;
    }
    if (potential_update_interval <= 0)
    {
        return;
    }
    if (steps % potential_update_interval == 0)
    {
        Axis values;
        for (int i = 0; i < cvs.size(); ++i)
        {
            values.push_back(cvs[i]->value);
        }
        Get_Height(values);
        float vshift;
        if (use_scatter)
        {
            Axis projected_values =
                mscatter->Get_Coordinate(mscatter->Get_Index(values));
            vshift = Calc_V_Shift(projected_values);
        }
        else
        {
            vshift = Calc_V_Shift(values);
        }
        Get_Reweighting_Bias(vshift);
        if (catheter)
        {
            float R = sqrtf(sigma_s * sigma_s -
                            2.0 * delta_sigma[mscatter->Get_Index(values)]);
            for (int i = 0; i < ndim; ++i)
            {
                sigmas[i] = R;
            }
        }
        Hill hill = Hill(values, sigmas, periods, height);
        hills.push_back(hill);
        Axis hillinfo;
        hillinfo.push_back(height);
        if (do_negative)
        {
            hillinfo.push_back(potential_max);
            hillinfo.push_back(max_index);
        }
        if (mscatter != nullptr)
        {
            hillinfo.push_back(mscatter->Get_Index(values));
        }
        hilllog("myhill.log", values, hillinfo);
        exit_tag = 0.0;
        if (!kde && subhill)
        {
            const Gdata& tder = hill.Calc_Hill(values);
            if (mgrid != nullptr)
            {
                mgrid->potential[mgrid->Get_Flat_Index(values)] +=
                    height * hill.potential;
            }
            else if (mscatter != nullptr)
            {
                mscatter->potential[mscatter->Get_Index(values)] +=
                    height * hill.potential;
            }
            return;
        }
        float factor = Normalization(values, height,
                                     kde);  // height with normalized factor
        std::vector<int> indices;
        if (use_scatter)
        {
            if (do_cutoff)
            {
                indices = mscatter->Get_Neighbor(values, cutoff);
            }
            else
            {
                indices = std::vector<int>(scatter_size);
                std::iota(indices.begin(), indices.end(), 0);
            }
            for (auto index : indices)
            {
                const Axis& coord = mscatter->Get_Coordinate(index);
                float* data = &mscatter->force[index * ndim];
                const Gdata& tder = hill.Calc_Hill(coord);
                if (catheter == 3)
                {
                    float* v = &mscatter->rotate_v[index * ndim];
                    float s = Project_To_Path(
                        Gdata(v, v + ndim), coord, values);
                    float dss = delta_sigma[index] * s * s;
                    for (int i = 0; i < ndim; ++i)
                    {
                        for (int j = 0; j < ndim; ++j)
                        {
                            data[i] += factor * 2 * (values[i] - coord[i]) *
                                       delta_sigma[index] * v[i] * v[j] *
                                       hill.potential * expf(-dss);
                        }
                        data[i] += factor * tder[i];
                    }
                }
                else if (catheter == 2)
                {
                    Axis values2, coord2;
                    Cartesian_To_Path(values, values2);
                    Cartesian_To_Path(coord, coord2);
                    Hill hill2 = Hill(values2, sigmas, periods, height);
                    Gdata tder2 = hill2.Calc_Hill(coord2);
                    float* R = &mscatter->rotate_matrix[index * ndim * ndim];
                    float* v = &mscatter->rotate_v[index * ndim];
                    float s = Project_To_Path(
                        Gdata(v, v + ndim), coord, values);
                    float dss = delta_sigma[index] * s * s;
                    for (int i = 0; i < ndim; ++i)
                    {
                        float delta1 = 0.0;
                        float delta2 =
                            tder[i] + 2 * delta_sigma[index] * s * v[i] *
                                          hill.potential * expf(-dss);
                        float delta3 = tder[i];
                        float dx = values[i] - coord[i];
                        for (int j = 0; j < ndim; ++j)
                        {
                            float R_ij = R[i + j * ndim];
                            delta1 += R_ij * tder2[j];
                            delta3 += 2 * delta_sigma[index] * v[i] * v[j] *
                                      dx * hill.potential * expf(-dss);
                        }
                        data[i] += factor * delta3;
                    }
                }
                else
                {
                    for (size_t i = 0; i < ndim; ++i)
                    {
                        data[i] += factor * tder[i];
                    }
                }
                float potential_temp = factor * hill.potential;
                if (mscatter != nullptr)
                {
                    mscatter->potential[index] += potential_temp;
                }
            }
        }
        // Update grid potential and force with hill on device
        if (mgrid != nullptr)
        {
            float h_centers[8], h_inv_w[8], h_periods[8];
            for (int d = 0; d < ndim; ++d)
            {
                h_centers[d] = hill.gsf[d].GetCenter();
                h_inv_w[d] = hill.gsf[d].GetWidth();
                h_periods[d] = periods[d];
            }
            deviceMemcpy(d_hill_centers, h_centers, sizeof(float) * ndim,
                         deviceMemcpyHostToDevice);
            deviceMemcpy(d_hill_inv_w, h_inv_w, sizeof(float) * ndim,
                         deviceMemcpyHostToDevice);
            deviceMemcpy(d_hill_periods, h_periods, sizeof(float) * ndim,
                         deviceMemcpyHostToDevice);
            int update_force = (!subhill && !mgrid->force.empty()) ? 1 : 0;
            Launch_Device_Kernel(
                Update_Grid_With_Hill,
                (mgrid->total_size + 255) / 256, 256, 0, NULL,
                mgrid->total_size, ndim,
                mgrid->d_num_points, mgrid->d_lower, mgrid->d_spacing,
                d_hill_centers, d_hill_inv_w, d_hill_periods,
                factor, update_force,
                mgrid->d_potential, mgrid->d_force);
            mgrid->Sync_To_Host();
        }
        if (mscatter != nullptr) mscatter->Sync_To_Device();
    }
}
