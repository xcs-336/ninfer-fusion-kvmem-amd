#pragma once
#include <hip/hip_runtime.h>

namespace ninfer::ops {
namespace nvfp4_gfx12_detail {
// The stored NVFP4 contract admits UE4M3 bytes 0..126: zero, positive
// subnormals and positive finite normals. MSB=1 and NaN=127 are not admitted.
// A finite E2M1 code times such a scale is exactly representable in FP16:
// nonzero magnitude lies in [2^-10,2688], with at most seven significant bits.
__device__ __forceinline__ _Float16 decode_scaled(unsigned code, unsigned scale) {
    const unsigned magnitude = code & 7u;
    unsigned bits = magnitude == 0 ? 0u : magnitude == 1 ? 0x3f000000u :
        ((126u + (magnitude >> 1)) << 23) | ((magnitude & 1u) << 22);
    bits |= (code & 8u) << 28;
    const float value = __builtin_bit_cast(float, bits);
    const unsigned exponent = scale >> 3, mantissa = scale & 7u;
    const float factor = scale >= 127u ? __builtin_bit_cast(float, 0x7fc00000u) :
        exponent ? __builtin_bit_cast(float, ((exponent + 120u) << 23) | (mantissa << 20)) :
        float(mantissa) * 0x1p-9f;
    const _Float16 rounded = static_cast<_Float16>(value * factor);
    // Clang contracts multiply-plus-half-conversion into v_fma_mixlo_f16
    // with an added +0. That instruction changes negative zero to +0.
    // UE4M3 scales are nonnegative, so restore the exact E2M1 sign in the
    // stored half bits, including code8 and every negative code at scale0.
    const unsigned short half_bits = static_cast<unsigned short>(
        (__builtin_bit_cast(unsigned short, rounded) & 0x7fffu) | ((code & 8u) << 12));
    return __builtin_bit_cast(_Float16, half_bits);
}
} // namespace nvfp4_gfx12_detail

// m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3, scale_vec::4X,
// byte-id-a=thread-id-a=byte-id-b=thread-id-b=0, matching NInfer's caller.
// PTX defines D=(A*scale_A)*(B*scale_B)+C, with four K16 scales per row/column.
// The selectors use A quad lanes0/1 for rows0..7/8..15 and B lane0 for
// columns0..7; all four little-endian scale bytes are consumed in K order.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
// gfx1201 has no floating-point FP4 WMMA: decode only the current register
// fragments and execute four native FP16-input/FP32-accumulator WMMA steps.
// No allocation, global scratch, weight conversion pass or extra storage.
// Every lane of the wave32 wave must participate throughout this function.
__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1,
    unsigned sfa, unsigned sfb) {
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#if !defined(__gfx1201__) || !__has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
#error "NInfer NVFP4 register decomposition requires native gfx1201 wave32 FP16 WMMA."
#endif
    if (__builtin_amdgcn_wavefrontsize() != 32) __builtin_trap();
    using Half8 = _Float16 __attribute__((ext_vector_type(8)));
    using Float8 = float __attribute__((ext_vector_type(8)));
    const unsigned lane = threadIdx.x & 31u;
    const unsigned coordinate = lane & 15u, group = lane >> 4;
    Float8 accumulator{};
#pragma unroll
    for (unsigned row = 0; row < 8; ++row) {
        const unsigned source = 4u * row + coordinate / 2u;
        const float x0 = __shfl(c0, source, 32), x1 = __shfl(c1, source, 32);
        const float x2 = __shfl(c2, source, 32), x3 = __shfl(c3, source, 32);
        const float lower = (coordinate & 1u) ? x1 : x0;
        const float upper = (coordinate & 1u) ? x3 : x2;
        accumulator[row] = coordinate < 8 ? (group ? upper : lower) : 0.0f;
    }
    const unsigned a_scale_word = __shfl(sfa, 4u * (coordinate & 7u) + (coordinate >> 3), 32);
    const unsigned b_scale_word = __shfl(sfb, 4u * (coordinate & 7u), 32);
#pragma unroll
    for (unsigned slice = 0; slice < 4; ++slice) {
        const unsigned as = (a_scale_word >> (8u * slice)) & 255u;
        const unsigned bs = coordinate < 8 ? ((b_scale_word >> (8u * slice)) & 255u) : 0u;
        Half8 left{}, right{};
#pragma unroll
        for (unsigned quartet = 0; quartet < 2; ++quartet) {
            const unsigned source = 4u * (coordinate & 7u) + 2u * (slice & 1u) + quartet;
            const unsigned low = __shfl(slice < 2 ? a0 : a2, source, 32);
            const unsigned high = __shfl(slice < 2 ? a1 : a3, source, 32);
            const unsigned raw_b = __shfl(slice < 2 ? b0 : b1, source, 32);
            const unsigned ap = coordinate < 8 ? low : high;
            const unsigned bp = coordinate < 8 ? raw_b : 0u;
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const unsigned shift = 4u * (4u * group + i);
                left[4 * quartet + i] = nvfp4_gfx12_detail::decode_scaled((ap >> shift) & 15u, as);
                right[4 * quartet + i] = nvfp4_gfx12_detail::decode_scaled((bp >> shift) & 15u, bs);
            }
        }
        accumulator = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(left, right, accumulator);
    }
    const unsigned column = 2u * (lane & 3u);
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#pragma unroll
    for (unsigned row = 0; row < 8; ++row) {
        const float x0 = __shfl(accumulator[row], column, 32);
        const float x1 = __shfl(accumulator[row], column + 1u, 32);
        const float x2 = __shfl(accumulator[row], column + 16u, 32);
        const float x3 = __shfl(accumulator[row], column + 17u, 32);
        if ((lane >> 2) == row) { d0 = x0; d1 = x1; d2 = x2; d3 = x3; }
    }
    c0 = d0; c1 = d1; c2 = d2; c3 = d3;
#endif
}
} // namespace ninfer::ops
