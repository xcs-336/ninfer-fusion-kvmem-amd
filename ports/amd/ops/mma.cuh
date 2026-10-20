#pragma once
#include "ops/common/mma_nvfp4_gfx12.cuh"
#include "ops/common/ldmatrix.cuh"
#include "ops/common/mma_bf16_gfx12.h"
#include "ops/common/mma_f16_gfx12.cuh"
#include "ops/common/mma_s8_gfx12.cuh"
#include "ops/common/mma_fp8_gfx12.cuh"

namespace ninfer::ops {
// Existing production BF16 tile kernels keep their packed-register interface;
// the arithmetic is executed by native gfx1201 WMMA. Other MMA formats require
// their own native implementation and qualification before their routes build.
using ::ninfer::amd_candidate::mma_bf16;

// RDNA4 has no TF32 matrix instruction. Retain the m16n8k8 fragment interface
// used by GDN, but evaluate its mathematical operation with native FP32 SIMD
// FMA and wave32 shuffles. There is no TF32 truncation or BF16 downcast here;
// observable state/storage casts remain in the owning Op. This is a correctness
// route, not a claim of NVIDIA Tensor Core throughput or bitwise TF32 identity.
__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned group = lane >> 2;
    const unsigned column = 2u * (lane & 3u);
#pragma unroll
    for (unsigned k = 0; k < 8; ++k) {
        // Shuffle each uniform register before selecting the row/column value.
        const unsigned lo = __shfl(k < 4 ? a0 : a2, 4u * group + (k & 3u), 32);
        const unsigned hi = __shfl(k < 4 ? a1 : a3, 4u * group + (k & 3u), 32);
        const unsigned bx = __shfl(k < 4 ? b0 : b1, 4u * column + (k & 3u), 32);
        const unsigned by = __shfl(k < 4 ? b0 : b1, 4u * (column + 1u) + (k & 3u), 32);
        const float al = __builtin_bit_cast(float, lo);
        const float ah = __builtin_bit_cast(float, hi);
        const float bl = __builtin_bit_cast(float, bx);
        const float bh = __builtin_bit_cast(float, by);
        c0 = fmaf(al, bl, c0);
        c1 = fmaf(al, bh, c1);
        c2 = fmaf(ah, bl, c2);
        c3 = fmaf(ah, bh, c3);
    }
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3,
                                         float a0, float a1, float a2, float a3,
                                         float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __builtin_bit_cast(unsigned, a0),
                  __builtin_bit_cast(unsigned, a1), __builtin_bit_cast(unsigned, a2),
                  __builtin_bit_cast(unsigned, a3), __builtin_bit_cast(unsigned, b0),
                  __builtin_bit_cast(unsigned, b1));
}
} // namespace ninfer::ops
