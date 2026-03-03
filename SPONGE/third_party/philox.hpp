/*
Xia Yijie modified from https://github.com/DiamonDinoia/philox
*/
/*
MIT License

Copyright (c) 2024 Marco Barbone

Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef PHILOX_HPP
#define PHILOX_HPP

#if defined(_MSC_VER)
#define PHILOX_INLINE __forceinline
#else
#define PHILOX_INLINE inline __attribute__((always_inline))
#endif

#include <cmath>
#include <cstdint>
#include <iostream>

class philox
{
   protected:
    using uint64_t = std::uint64_t;
    using uint32_t = std::uint32_t;
    using uint16_t = std::uint16_t;
    using uint8_t = std::uint8_t;
};

class Philox4x32_10 : protected philox
{
   public:
    PHILOX_INLINE explicit Philox4x32_10(
        philox::uint64_t seed_value = PHILOX4x32_DEFAULT_SEED,
        philox::uint64_t subsequence = 0, philox::uint64_t offset = 0) noexcept
    {
        seed(seed_value, subsequence, offset);
    }

    PHILOX_INLINE Philox4x32_10(const Philox4x32_10& other) noexcept
    {
        m_counter = other.m_counter;
        m_result = other.m_result;
        m_key = other.m_key;
        m_substate = other.m_substate;
    };

    PHILOX_INLINE Philox4x32_10(Philox4x32_10&& other) noexcept
    {
        m_counter = other.m_counter;
        m_result = other.m_result;
        m_key = other.m_key;
        m_substate = other.m_substate;
    };

    PHILOX_INLINE Philox4x32_10& operator=(const Philox4x32_10& other) noexcept
    {
        m_counter = other.m_counter;
        m_result = other.m_result;
        m_key = other.m_key;
        m_substate = other.m_substate;
        return *this;
    };

    PHILOX_INLINE Philox4x32_10& operator=(Philox4x32_10&& other) noexcept
    {
        m_counter = other.m_counter;
        m_result = other.m_result;
        m_key = other.m_key;
        m_substate = other.m_substate;
        return *this;
    };

    PHILOX_INLINE void seed(philox::uint64_t seed_value,
                            const philox::uint64_t subsequence,
                            const philox::uint64_t offset)
    {
        m_key.m64i = seed_value;
        restart(subsequence, offset);
    }

    PHILOX_INLINE void restart(const philox::uint64_t subsequence,
                               const philox::uint64_t offset) noexcept
    {
        m_counter = {0, 0, 0, 0};
        m_result = {0, 0, 0, 0};
        m_substate = 0;
        discard_subsequence_impl(subsequence);
        discard_impl(offset);
        m_result = ten_rounds(m_counter, m_key);
    }

    PHILOX_INLINE void discard(philox::uint64_t offset) noexcept
    {
        discard_impl(offset);
        m_result = ten_rounds(m_counter, m_key);
    }

    PHILOX_INLINE void discard_subsequence(
        philox::uint64_t subsequence) noexcept
    {
        discard_subsequence_impl(subsequence);
        m_result = ten_rounds(m_counter, m_key);
    }

    PHILOX_INLINE philox::uint32_t next_32() noexcept
    {
        const auto ret = m_result.data[m_substate++];
        if (m_substate == 4)
        {
            discard_state();
            m_result = ten_rounds(m_counter, m_key);
            m_substate = 0;
        }
        return ret;
    }

    PHILOX_INLINE philox::uint64_t next_64() noexcept
    {
        uint2 xy = {next_32(), next_32()};
        return xy.m64i;
    }

    PHILOX_INLINE float uniform() noexcept
    {
        constexpr float max_value = (float)((uint32_t)-1);
        float f = next_32() / max_value;
        return f;
    }

    PHILOX_INLINE void normal(float* x, int i) noexcept
    {
        float u1 = uniform();
        float u2 = uniform();
        float u3 = uniform();
        float u4 = uniform();
        u1 = u1 ? sqrtf(-2 * logf(u1)) : 0;
        u2 = 2.0f * 3.141592654f * u2;
        u3 = u3 ? sqrtf(-2 * logf(u3)) : 0;
        u4 = 2.0f * 3.141592654f * u4;
        x[4 * i] = u1 * cosf(u2);
        x[4 * i + 1] = u1 * sinf(u2);
        x[4 * i + 2] = u3 * cosf(u4);
        x[4 * i + 3] = u3 * sinf(u4);
    }

   protected:
    static constexpr philox::uint64_t PHILOX4x32_DEFAULT_SEED{
        0xdeadbeefdeadbeefULL};
    static constexpr philox::uint32_t PHILOX_M4x32_0{0xD2511F53U};
    static constexpr philox::uint32_t PHILOX_M4x32_1{0xCD9E8D57U};
    static constexpr philox::uint32_t PHILOX_W32_0{0x9E3779B9U};
    static constexpr philox::uint32_t PHILOX_W32_1{0xBB67AE85U};

    union uint4
    {
        struct
        {
            philox::uint32_t x;
            philox::uint32_t y;
            philox::uint32_t z;
            philox::uint32_t w;
        };
        philox::uint32_t data[4];
    };

    union uint2
    {
        struct
        {
            philox::uint32_t x;
            philox::uint32_t y;
        };
        philox::uint64_t m64i;
    };

    union fuint
    {
        unsigned int u;
        float f;
    };

    uint4 m_counter{};
    uint4 m_result{};
    uint2 m_key{};
    philox::uint8_t m_substate{};

    PHILOX_INLINE void discard_subsequence_impl(
        philox::uint64_t subsequence) noexcept
    {
        uint2 hl;
        hl.m64i = subsequence;
        const auto temp = m_counter.z;
        m_counter.z += hl.x;
        m_counter.w += hl.y + (m_counter.z < temp);
    }

    PHILOX_INLINE void discard_state() noexcept { bump_counter(); }

    PHILOX_INLINE void discard_state(philox::uint64_t offset) noexcept
    {
        uint2 hl;
        hl.m64i = offset;
        const auto temp = m_counter;
        m_counter.x += hl.x;
        m_counter.y += hl.y + (m_counter.x < temp.x);
        m_counter.z += (m_counter.y < temp.y);
        m_counter.w += (m_counter.z < temp.z);
    }

    PHILOX_INLINE void discard_impl(philox::uint64_t offset) noexcept
    {
        // Adjust offset for subset
        m_substate += offset & 3;
        philox::uint64_t counter_offset = offset >> 2;
        counter_offset += m_substate < 4 ? 0 : 1;
        m_substate += m_substate < 4 ? 0 : -4;
        // Discard states
        discard_state(counter_offset);
    }

    PHILOX_INLINE void bump_counter() noexcept
    {
        if (++m_counter.x) return;
        if (++m_counter.y) return;
        if (++m_counter.z) return;
        ++m_counter.w;
    }

    PHILOX_INLINE static uint2 mulhilo32(philox::uint32_t x,
                                         philox::uint32_t y) noexcept
    {
        uint2 xy;
        xy.m64i = static_cast<philox::uint64_t>(x) * y;
        return {xy.x, xy.y};
    }

    PHILOX_INLINE static uint2 bumpkey(uint2 key) noexcept
    {
        key.x += PHILOX_W32_0;
        key.y += PHILOX_W32_1;
        return key;
    }

    PHILOX_INLINE static uint4 single_round(const uint4& counter,
                                            const uint2& key) noexcept
    {
        // Source: Random123
        const auto hl0 = mulhilo32(PHILOX_M4x32_0, counter.x);
        const auto hl1 = mulhilo32(PHILOX_M4x32_1, counter.z);
        return {hl1.y ^ counter.y ^ key.x, hl1.x, hl0.y ^ counter.w ^ key.y,
                hl0.x};
    }

    PHILOX_INLINE static uint4 ten_rounds(uint4 counter, uint2 key) noexcept
    {
        counter = single_round(counter, key);
        key = bumpkey(key);  // 1
        counter = single_round(counter, key);
        key = bumpkey(key);  // 2
        counter = single_round(counter, key);
        key = bumpkey(key);  // 3
        counter = single_round(counter, key);
        key = bumpkey(key);  // 4
        counter = single_round(counter, key);
        key = bumpkey(key);  // 5
        counter = single_round(counter, key);
        key = bumpkey(key);  // 6
        counter = single_round(counter, key);
        key = bumpkey(key);  // 7
        counter = single_round(counter, key);
        key = bumpkey(key);  // 8
        counter = single_round(counter, key);
        key = bumpkey(key);                 // 9
        return single_round(counter, key);  // 10
    }
};

#endif
