#ifndef SPONGE_LANE_GROUP_SCALAR_H
#define SPONGE_LANE_GROUP_SCALAR_H

#include "lane_group.h"

struct LaneMask
{
    unsigned int bits;

    __host__ __device__ __forceinline__ LaneMask() : bits(0) {}
    __host__ __device__ __forceinline__ explicit LaneMask(unsigned int value)
        : bits(value)
    {
    }
};

struct LaneGroup
{
    __host__ __device__ __forceinline__ static int Width()
    {
        return 1;
    }

    __host__ __device__ __forceinline__ static int Lane_Id()
    {
        return 0;
    }

    __host__ __device__ __forceinline__ static LaneMask Active_Mask()
    {
        return LaneMask(1);
    }

    __host__ __device__ __forceinline__ static LaneMask Ballot(bool predicate)
    {
        return LaneMask(predicate ? 1U : 0U);
    }

    __host__ __device__ __forceinline__ static bool Any(bool predicate)
    {
        return predicate;
    }

    __host__ __device__ __forceinline__ static bool All(bool predicate)
    {
        return predicate;
    }

    __host__ __device__ __forceinline__ static int Count(LaneMask mask)
    {
        return LaneGroup_PopCount(mask.bits);
    }

    __host__ __device__ __forceinline__ static int First_Lane(LaneMask mask)
    {
        return mask.bits == 0 ? -1 : 0;
    }

    __host__ __device__ __forceinline__ static LaneMask Lower_Lane_Mask()
    {
        return LaneMask(0);
    }

    __host__ __device__ __forceinline__ static int Prefix_Count(LaneMask mask)
    {
        return Count(LaneMask(mask.bits & Lower_Lane_Mask().bits));
    }

    template <typename T>
    __host__ __device__ __forceinline__ static T Broadcast(T value,
                                                           int src_lane)
    {
        (void)src_lane;
        return value;
    }

    template <typename T>
    __host__ __device__ __forceinline__ static T Shuffle_Down(T value,
                                                              int delta)
    {
        (void)delta;
        return value;
    }

    template <typename T>
    __host__ __device__ __forceinline__ static T Reduce_Sum(T value)
    {
        return value;
    }

    template <typename T>
    __host__ __device__ __forceinline__ static T Reduce_Sum(T value, int width)
    {
        (void)width;
        return value;
    }
};

#endif  // SPONGE_LANE_GROUP_SCALAR_H
