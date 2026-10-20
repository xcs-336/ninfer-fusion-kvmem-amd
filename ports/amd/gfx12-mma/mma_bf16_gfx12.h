#pragma once

#include <hip/hip_runtime.h>

// Experimental native gfx1201 implementation. Not installed in the Engine.
// All 32 lanes must execute every shuffle and WMMA together, without divergence.
// Input/output packed-register layout matches mma.m16n8k16.row.col.f32.bf16.bf16.f32.
namespace ninfer::amd_candidate {

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3,
                                      unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                      unsigned b0, unsigned b1) {
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#if !defined(__gfx1201__)
#error "This candidate requires compilation targeting gfx1201."
#endif
    // ROCm 10 clang does not define __AMDGCN_WAVEFRONT_SIZE. The build forces
    // -mno-wavefrontsize64; retain a device check rather than inventing a macro.
    if (__builtin_amdgcn_wavefrontsize() != 32) { __builtin_trap(); }
#if !__has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12)
#error "Compiler lacks the gfx12 BF16 wave32 WMMA builtin."
#endif
    using Short2 = short __attribute__((ext_vector_type(2)));
    using Short8 = short __attribute__((ext_vector_type(8)));
    using Float8 = float __attribute__((ext_vector_type(8)));
    const unsigned lane = __builtin_amdgcn_mbcnt_lo(~0u, 0u);
    const unsigned row = lane & 15u;
    const unsigned half_wave = lane >> 4;
    Short8 wa{}, wb{};
    Float8 wc{};

#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        const unsigned source_a = 4u * (row & 7u) + 2u * half_wave + (word & 1u);
        // Shuffle uniform source registers BEFORE selecting a row. Selecting a0/a1
        // before the shuffle would use the source lane's row, not this lane's row.
        const unsigned low = __shfl(word < 2 ? a0 : a2, source_a, 32);
        const unsigned high = __shfl(word < 2 ? a1 : a3, source_a, 32);
        const unsigned ap = row < 8 ? low : high;
        const unsigned source_b = 4u * (row & 7u) + 2u * half_wave + (word & 1u);
        const unsigned gathered_b = __shfl(word < 2 ? b0 : b1, source_b, 32);
        const unsigned bp = row < 8 ? gathered_b : 0u;
        const Short2 av = __builtin_bit_cast(Short2, ap);
        const Short2 bv = __builtin_bit_cast(Short2, bp);
        wa[2 * word] = av[0];
        wa[2 * word + 1] = av[1];
        wb[2 * word] = bv[0];
        wb[2 * word + 1] = bv[1];
    }

#pragma unroll
    for (unsigned reg = 0; reg < 8; ++reg) {
        const unsigned source_c = 4u * reg + row / 2u;
        const float v0 = __shfl(c0, source_c, 32);
        const float v1 = __shfl(c1, source_c, 32);
        const float v2 = __shfl(c2, source_c, 32);
        const float v3 = __shfl(c3, source_c, 32);
        const float lower = (row & 1u) ? v1 : v0;
        const float upper = (row & 1u) ? v3 : v2;
        wc[reg] = row < 8 ? (half_wave ? upper : lower) : 0.0f;
    }

    const Float8 wd = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(wa, wb, wc);
    const unsigned col = 2u * (lane & 3u);
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#pragma unroll
    for (unsigned reg = 0; reg < 8; ++reg) {
        // All lanes execute all four shuffles; only then choose the destination row.
        const float v0 = __shfl(wd[reg], col, 32);
        const float v1 = __shfl(wd[reg], col + 1u, 32);
        const float v2 = __shfl(wd[reg], 16u + col, 32);
        const float v3 = __shfl(wd[reg], 17u + col, 32);
        if ((lane >> 2) == reg) {
            d0 = v0;
            d1 = v1;
            d2 = v2;
            d3 = v3;
        }
    }
    c0 = d0;
    c1 = d1;
    c2 = d2;
    c3 = d3;
#endif
}

} // namespace ninfer::amd_candidate
