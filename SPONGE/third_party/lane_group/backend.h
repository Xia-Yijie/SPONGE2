#ifndef SPONGE_LANE_GROUP_BACKEND_H
#define SPONGE_LANE_GROUP_BACKEND_H

#if defined(SPONGE_LANE_GROUP_CUDA) || defined(USE_CUDA)
#include "cuda.h"
#elif defined(SPONGE_LANE_GROUP_HIP) || defined(USE_HIP)
#include "hip.h"
#elif defined(SPONGE_LANE_GROUP_SVE2) || defined(SPONGE_LANE_GROUP_SVE) || \
    defined(__ARM_FEATURE_SVE)
#include "sve.h"
#elif defined(SPONGE_LANE_GROUP_AVX512) || defined(__AVX512F__)
#include "avx512.h"
#elif defined(SPONGE_LANE_GROUP_AVX2) || defined(__AVX2__)
#include "avx2.h"
#elif defined(SPONGE_LANE_GROUP_AVX) || defined(__AVX__)
#include "avx.h"
#elif defined(SPONGE_LANE_GROUP_SSE42) || defined(__SSE4_2__)
#include "sse42.h"
#elif defined(SPONGE_LANE_GROUP_NEON) || defined(__ARM_NEON)
#include "neon.h"
#else
#include "scalar.h"
#endif

#endif  // SPONGE_LANE_GROUP_BACKEND_H
