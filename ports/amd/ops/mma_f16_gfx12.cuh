#pragma once

#include <hip/hip_runtime.h>

// Native gfx1201 implementation of NInfer's m16n8k16.row.col FP16 fragments.
// All 32 lanes must participate. The 8 otherwise unused columns of the native
// 16x16 instruction are zero padded; no input conversion changes the FP16 bits.
// Builtin definitions: https://clang.llvm.org/docs/AMDGPUBuiltinReference.html
namespace ninfer::ops {
namespace f16_gfx12_detail {
using Half2 = _Float16 __attribute__((ext_vector_type(2)));
using Half8 = _Float16 __attribute__((ext_vector_type(8)));
using UShort8 = unsigned short __attribute__((ext_vector_type(8)));
using Float8 = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ void operands(Half8& a, Half8& b, unsigned a0,
    unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = lane & 15u, half_wave = lane >> 4;
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        const unsigned source = 4u * (row & 7u) + 2u * half_wave + (word & 1u);
        // Every source register must be shuffled before selecting the row;
        // selecting in the source lane would select that lane's wrong row.
        const unsigned low = __shfl(word < 2 ? a0 : a2, source, 32);
        const unsigned high = __shfl(word < 2 ? a1 : a3, source, 32);
        const unsigned bp = __shfl(word < 2 ? b0 : b1, source, 32);
        const Half2 av = __builtin_bit_cast(Half2, row < 8 ? low : high);
        const Half2 bv = __builtin_bit_cast(Half2, row < 8 ? bp : 0u);
        a[2 * word] = av[0]; a[2 * word + 1] = av[1];
        b[2 * word] = bv[0]; b[2 * word + 1] = bv[1];
    }
}
} // namespace f16_gfx12_detail

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#if !defined(__gfx1201__) || !__has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
#error "NInfer FP16 MMA requires gfx1201 and the gfx12 wave32 FP16 WMMA builtin."
#endif
    if (__builtin_amdgcn_wavefrontsize() != 32) __builtin_trap();
    using namespace f16_gfx12_detail;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = lane & 15u, half_wave = lane >> 4;
    Half8 a{}, b{};
    operands(a, b, a0, a1, a2, a3, b0, b1);
    Float8 c{};
#pragma unroll
    for (unsigned reg = 0; reg < 8; ++reg) {
        const unsigned source = 4u * reg + row / 2u;
        const float v0 = __shfl(c0, source, 32), v1 = __shfl(c1, source, 32);
        const float v2 = __shfl(c2, source, 32), v3 = __shfl(c3, source, 32);
        const float lower = (row & 1u) ? v1 : v0;
        const float upper = (row & 1u) ? v3 : v2;
        c[reg] = row < 8 ? (half_wave ? upper : lower) : 0.0f;
    }
    const Float8 d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    const unsigned col = 2u * (lane & 3u);
#pragma unroll
    for (unsigned reg = 0; reg < 8; ++reg) {
        const float v0 = __shfl(d[reg], col, 32), v1 = __shfl(d[reg], col + 1u, 32);
        const float v2 = __shfl(d[reg], 16u + col, 32), v3 = __shfl(d[reg], 17u + col, 32);
        if ((lane >> 2) == reg) { c0 = v0; c1 = v1; c2 = v2; c3 = v3; }
    }
#endif
}

// d0 contains FP16 c0,c1 and d1 contains c2,c3, low half first. The native
// instruction consumes and returns FP16 accumulators at every call; it must
// not silently retain an FP32 accumulator across the caller's MMA sequence.
__device__ __forceinline__ void mma_f16_f16acc(unsigned& d0, unsigned& d1,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#if !defined(__gfx1201__) || !__has_builtin(__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12)
#error "NInfer FP16 accumulation requires gfx1201 and native gfx12 wave32 FP16 WMMA."
#endif
    if (__builtin_amdgcn_wavefrontsize() != 32) __builtin_trap();
    using namespace f16_gfx12_detail;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = lane & 15u, half_wave = lane >> 4;
    Half8 a{}, b{}, c{};
    operands(a, b, a0, a1, a2, a3, b0, b1);
#pragma unroll
    for (unsigned reg = 0; reg < 8; ++reg) {
        const unsigned source = 4u * reg + row / 2u;
        const unsigned low = __shfl(d0, source, 32), high = __shfl(d1, source, 32);
        const unsigned packed = half_wave ? high : low;
        const unsigned short bits = row < 8 ?
            static_cast<unsigned short>(packed >> (16u * (row & 1u))) : 0;
        c[reg] = __builtin_bit_cast(_Float16, bits);
    }
    const Half8 d = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(a, b, c);
    // gfx12 packs rows 2*r and 2*r+1 in the low/high halves of accumulator
    // VGPR r: Half8 element g therefore denotes output row g within a group.
    // Bit-cast the complete vector before indexing. ROCm 10 clang 23 compiled
    // bit_cast<unsigned short>(d[g]) with every component reading element 0;
    // the whole-vector bit-cast preserves all eight extractelement indices.
    const UShort8 output_bits = __builtin_bit_cast(UShort8, d);
    const unsigned col = 2u * (lane & 3u);
#pragma unroll
    for (unsigned reg = 0; reg < 8; ++reg) {
        const unsigned bits = output_bits[reg];
        const unsigned v0 = __shfl(bits, col, 32), v1 = __shfl(bits, col + 1u, 32);
        const unsigned v2 = __shfl(bits, 16u + col, 32), v3 = __shfl(bits, 17u + col, 32);
        if ((lane >> 2) == reg) { d0 = v0 | (v1 << 16); d1 = v2 | (v3 << 16); }
    }
#endif
}
} // namespace ninfer::ops
