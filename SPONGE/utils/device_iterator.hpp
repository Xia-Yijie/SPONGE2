#pragma once

// -----------------线程并行迭代器定义 ------------------

#define STRINGIFY(x) #x
#define PRAGMA(x) _Pragma(#x)

// ------------------FOR 循环----------------------------
#ifdef GPU_ARCH_NAME
#define SIMPLE_DEVICE_FOR(i, N)                    \
    int i = blockDim.x * blockIdx.x + threadIdx.x; \
    if (i < N)
#else
#define SIMPLE_DEVICE_FOR(i, N) \
    PRAGMA(omp parallel for)    \
    for (int i = 0; i < N; i++)
#endif

// ------------------线程求和----------------------------------

__device__ __forceinline__ void Warp_Sum_To(int* y, int& x, int delta)
{
#ifdef GPU_ARCH_NAME
    for (delta >>= 1; delta > 0; delta >>= 1)
    {
        x += __shfl_down_sync(FULL_MASK, x, delta);
    }
    if (threadIdx.x == 0)
#endif
    {
        atomicAdd(y, x);
    }
}
__device__ __forceinline__ void Warp_Sum_To(float* y, float& x, int delta)
{
#ifdef GPU_ARCH_NAME
    for (delta >>= 1; delta > 0; delta >>= 1)
    {
        x += __shfl_down_sync(FULL_MASK, x, delta);
    }
    if (threadIdx.x == 0)
#endif
    {
        atomicAdd(y, x);
    }
}
__device__ __forceinline__ void Warp_Sum_To(VECTOR* y, VECTOR& x, int delta)
{
#ifdef GPU_ARCH_NAME
    for (delta >>= 1; delta > 0; delta >>= 1)
    {
        x.x += __shfl_down_sync(FULL_MASK, x.x, delta);
        x.y += __shfl_down_sync(FULL_MASK, x.y, delta);
        x.z += __shfl_down_sync(FULL_MASK, x.z, delta);
    }
    if (threadIdx.x == 0)
#endif
    {
        atomicAdd(y, x);
    }
}
__device__ __forceinline__ void Warp_Sum_To(LTMatrix3* y, LTMatrix3& x,
                                            int delta)
{
#ifdef GPU_ARCH_NAME
    for (delta >>= 1; delta > 0; delta >>= 1)
    {
        x.a11 += __shfl_down_sync(FULL_MASK, x.a11, delta);
        x.a21 += __shfl_down_sync(FULL_MASK, x.a21, delta);
        x.a22 += __shfl_down_sync(FULL_MASK, x.a22, delta);
        x.a31 += __shfl_down_sync(FULL_MASK, x.a31, delta);
        x.a32 += __shfl_down_sync(FULL_MASK, x.a32, delta);
        x.a33 += __shfl_down_sync(FULL_MASK, x.a33, delta);
    }
    if (threadIdx.x == 0)
#endif
    {
        atomicAdd(y, x);
    }
}
