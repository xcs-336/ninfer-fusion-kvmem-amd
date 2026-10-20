#pragma once
#include <hip/hip_runtime.h>

namespace ninfer::ops {
// Native signed INT8 WMMA. Preserve the existing m16n8k32 packed-register
// interface using two m16n16k16 steps; unused output columns carry zero.
// All 32 lanes execute the complete function. Unsaturated INT32 arithmetic
// wraps on overflow, matching the original mma.sync without .satfinite.
__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3,
                                       unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                       unsigned b0, unsigned b1) {
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#if !defined(__gfx1201__)
#error "The INT8 fragment route requires gfx1201."
#endif
    if (__builtin_amdgcn_wavefrontsize() != 32) __builtin_trap();
    using Int2 = int __attribute__((ext_vector_type(2)));
    using Int8 = int __attribute__((ext_vector_type(8)));
    const unsigned lane = __builtin_amdgcn_mbcnt_lo(~0u, 0u);
    const unsigned row = lane & 15u, group = lane >> 4;
    Int8 accumulator{};
#pragma unroll
    for (unsigned reg = 0; reg < 8; ++reg) {
        const unsigned source = 4u * reg + row / 2u;
        const int x0 = __shfl(c0, source, 32), x1 = __shfl(c1, source, 32);
        const int x2 = __shfl(c2, source, 32), x3 = __shfl(c3, source, 32);
        const int lower = (row & 1u) ? x1 : x0, upper = (row & 1u) ? x3 : x2;
        accumulator[reg] = row < 8 ? (group ? upper : lower) : 0;
    }
#pragma unroll
    for (unsigned slice = 0; slice < 2; ++slice) {
        Int2 left{}, right{};
#pragma unroll
        for (unsigned word = 0; word < 2; ++word) {
            const unsigned source = 4u * (row & 7u) + group + 2u * word;
            const unsigned low = __shfl(slice ? a2 : a0, source, 32);
            const unsigned high = __shfl(slice ? a3 : a1, source, 32);
            const unsigned packed_b = __shfl(slice ? b1 : b0, source, 32);
            left[word] = __builtin_bit_cast(int, row < 8 ? low : high);
            right[word] = __builtin_bit_cast(int, row < 8 ? packed_b : 0u);
        }
        accumulator = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            true, left, true, right, accumulator, false);
    }
    const unsigned col = 2u * (lane & 3u);
    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#pragma unroll
    for (unsigned reg = 0; reg < 8; ++reg) {
        const int x0 = __shfl(accumulator[reg], col, 32);
        const int x1 = __shfl(accumulator[reg], col + 1u, 32);
        const int x2 = __shfl(accumulator[reg], col + 16u, 32);
        const int x3 = __shfl(accumulator[reg], col + 17u, 32);
        if ((lane >> 2) == reg) { d0 = x0; d1 = x1; d2 = x2; d3 = x3; }
    }
    c0 = d0; c1 = d1; c2 = d2; c3 = d3;
#endif
}
} // namespace ninfer::ops
