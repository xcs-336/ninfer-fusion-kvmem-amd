#pragma once
#include <hip/hip_runtime.h>

namespace ninfer::ops {
// Native OCP FP8 E4M3 (signed zero, finite maximum 448; not E4M3-FNUZ).
// Preserve NInfer's m16n8k32 packed-register interface using two gfx12
// m16n16k16 WMMA steps, zero padding the unused eight output columns.
// All 32 lanes must execute every shuffle and both instructions together.
// https://clang.llvm.org/docs/AMDGPUBuiltinReference.html
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#if !defined(__gfx1201__) || !__has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12)
#error "NInfer OCP FP8 MMA requires gfx1201 and native gfx12 wave32 FP8 WMMA."
#endif
    if (__builtin_amdgcn_wavefrontsize() != 32) __builtin_trap();
    using Int2 = int __attribute__((ext_vector_type(2)));
    using Float8 = float __attribute__((ext_vector_type(8)));
    const unsigned lane = threadIdx.x & 31u;
    const unsigned column = lane & 15u, row_group = lane >> 4;
    Float8 accumulator{};
#pragma unroll
    for (unsigned row = 0; row < 8; ++row) {
        const unsigned source = 4u * row + column / 2u;
        const float x0 = __shfl(c0, source, 32), x1 = __shfl(c1, source, 32);
        const float x2 = __shfl(c2, source, 32), x3 = __shfl(c3, source, 32);
        const float lower = (column & 1u) ? x1 : x0;
        const float upper = (column & 1u) ? x3 : x2;
        accumulator[row] = column < 8 ? (row_group ? upper : lower) : 0.0f;
    }
#pragma unroll
    for (unsigned slice = 0; slice < 2; ++slice) {
        Int2 left{}, right{};
#pragma unroll
        for (unsigned word = 0; word < 2; ++word) {
            // Native word w owns K=4*group+8*w .. +3 in this 16-wide
            // slice. A PTX lane owns four adjacent bytes of each 16-wide
            // half, so source lane component = group + 2*w.
            const unsigned source = 4u * (column & 7u) + row_group + 2u * word;
            const unsigned low = __shfl(slice ? a2 : a0, source, 32);
            const unsigned high = __shfl(slice ? a3 : a1, source, 32);
            const unsigned packed_b = __shfl(slice ? b1 : b0, source, 32);
            left[word] = __builtin_bit_cast(int, column < 8 ? low : high);
            right[word] = __builtin_bit_cast(int, column < 8 ? packed_b : 0u);
        }
        accumulator = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
            left, right, accumulator);
    }
    const unsigned column_pair = 2u * (lane & 3u);
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#pragma unroll
    for (unsigned row = 0; row < 8; ++row) {
        const float x0 = __shfl(accumulator[row], column_pair, 32);
        const float x1 = __shfl(accumulator[row], column_pair + 1u, 32);
        const float x2 = __shfl(accumulator[row], column_pair + 16u, 32);
        const float x3 = __shfl(accumulator[row], column_pair + 17u, 32);
        if ((lane >> 2) == row) { d0 = x0; d1 = x1; d2 = x2; d3 = x3; }
    }
    c0 = d0; c1 = d1; c2 = d2; c3 = d3;
#endif
}
} // namespace ninfer::ops
