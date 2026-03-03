#ifndef BASIC_BACKEND_H
#define BASIC_BACKEND_H
#define GPU_ARCH_NAME "CUDA"

#include <cuda.h>
#include <cuda_runtime.h>

#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include "nvrtc.h"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
static __device__ __forceinline__ double atomicAdd(double* address, double val)
{
    auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(
            address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

#define Philox4_32_10_t curandStatePhilox4_32_10_t
#define device_rand_init curand_init
#define device_get_4_normal_distributed_random_numbers(rand_float4,   \
                                                       rand_state, i) \
    rand_float4[i] = curand_normal4(rand_state + i)

#define DEVICE_INIT_SUCCESS CUDA_SUCCESS
#define DEVICE_MALLOC_SUCCESS cudaSuccess

#define deviceInit cuInit
#define deviceGetDeviceCount cudaGetDeviceCount
#define deviceProp cudaDeviceProp
#define getDeviceProperties cudaGetDeviceProperties
#define setWorkingDevice cudaSetDevice

#define deviceMalloc cudaMalloc
#define deviceMemcpy cudaMemcpy
#define deviceMemcpyAsync cudaMemcpyAsync
#define deviceMemcpyKind cudaMemcpyKind
#define deviceMemcpyHostToHost cudaMemcpyHostToHost
#define deviceMemcpyHostToDevice cudaMemcpyHostToDevice
#define deviceMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define deviceMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define deviceMemcpyDefault cudaMemcpyDefault
#define deviceMemset(PTR, VAL, SIZE) cudaMemsetAsync(PTR, VAL, SIZE, NULL)
#define deviceFree cudaFree
#define deviceMalloc cudaMalloc

#define deviceError_t cudaError_t
#define deviceGetErrorName cudaGetErrorName
#define deviceGetErrorString cudaGetErrorString
#define deviceErrorLaunchOutOfResources cudaErrorLaunchOutOfResources
#define deviceErrorInvalidValue cudaErrorInvalidValue
#define deviceErrorInvalidConfiguration cudaErrorInvalidConfiguration
#define deviceGetLastError cudaGetLastError

#define deviceStream_t cudaStream_t
#define deviceStreamCreate cudaStreamCreate
#define deviceStreamDestroy cudaStreamDestroy
#define deviceStreamSynchronize cudaStreamSynchronize

#define Launch_Device_Kernel(kernel, grid, block, sm_memory, stream, ...) \
    kernel<<<grid, block, sm_memory, stream>>>(__VA_ARGS__)

#define FULL_MASK 0xffffffff
#define hostDeviceSynchronize cudaDeviceSynchronize
#endif  // BASIC_BACKEND_H

#ifndef FFT_BACKEND_H
#define FFT_BACKEND_H

#include <cufftw.h>
#define FFT_LIBRARY_NAME "cuFFT"
#define FFT_COMPLEX cufftComplex
#define REAL(c) c.x
#define IMAGINARY(c) c.y
#define FFT_HANDLE cufftHandle
#define FFT_SUCCESS CUFFT_SUCCESS
#define FFT_RESULT cufftResult

#define FFT_TYPE cufftType
#define FFT_R2C CUFFT_R2C
#define FFT_C2R CUFFT_C2R
#define FFT_SIZE_t int

#define deviceFFTPlanMany cufftPlanMany
#define deviceFFTExecR2C cufftExecR2C
#define deviceFFTExecC2R cufftExecC2R
#define deviceFFTDestroy cufftDestroy

#endif  // FFT_BACKEND_H

#ifndef BLAS_BACKEND_H
#define BLAS_BACKEND_H

#include <cublas_v2.h>
#define BLAS_LIBRARY_NAME "cuBLAS"
#define BLAS_HANDLE cublasHandle_t
#define BLAS_SUCCESS CUBLAS_STATUS_SUCCESS

#define deviceBlasCreate cublasCreate
#define deviceBlasDestroy cublasDestroy
#define deviceBlasSgeam cublasSgeam
#define deviceBlasSgemm cublasSgemm

#endif  // BLAS_BACKEND_H

#ifndef SOLVER_BACKEND_H
#define SOLVER_BACKEND_H

#include <cusolverDn.h>
#define SOLVER_LIBRARY_NAME "cuSolver"
#define SOLVER_HANDLE cusolverDnHandle_t
#define SOLVER_SUCCESS CUSOLVER_STATUS_SUCCESS

#define deviceSolverCreate cusolverDnCreate
#define deviceSolverDestroy cusolverDnDestroy

#endif  // SOLVER_BACKEND_H
