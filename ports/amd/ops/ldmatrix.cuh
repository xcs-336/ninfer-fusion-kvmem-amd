#pragma once
#include "memory.cuh"

namespace ninfer::ops {
template <int Matrix, bool Transpose>
__device__ __forceinline__ unsigned ldmatrix_word(unsigned address) {
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
    const unsigned lane = __builtin_amdgcn_mbcnt_lo(~0u, 0u);
    const unsigned row = lane / 4;
    const unsigned column = 2 * (lane % 4);
    using LdsHalf = const unsigned short __attribute__((address_space(3)));
    if constexpr (Transpose) {
        const unsigned lo_address = __shfl(address, Matrix * 8 + column, 32);
        const unsigned hi_address = __shfl(address, Matrix * 8 + column + 1, 32);
        const auto* lo = reinterpret_cast<LdsHalf*>(static_cast<std::uintptr_t>(lo_address));
        const auto* hi = reinterpret_cast<LdsHalf*>(static_cast<std::uintptr_t>(hi_address));
        return static_cast<unsigned>(lo[row]) | (static_cast<unsigned>(hi[row]) << 16);
    } else {
        const unsigned source = __shfl(address, Matrix * 8 + row, 32);
        const auto* p = reinterpret_cast<LdsHalf*>(static_cast<std::uintptr_t>(source));
        return static_cast<unsigned>(p[column]) | (static_cast<unsigned>(p[column + 1]) << 16);
    }
#else
    return 0;
#endif
}
__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned address) {
    r0 = ldmatrix_word<0, false>(address); r1 = ldmatrix_word<1, false>(address);
}
__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2,
                                         unsigned& r3, unsigned address) {
    ldmatrix_x2(r0, r1, address);
    r2 = ldmatrix_word<2, false>(address); r3 = ldmatrix_word<3, false>(address);
}
__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned address) {
    r0 = ldmatrix_word<0, true>(address); r1 = ldmatrix_word<1, true>(address);
}
__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                           unsigned& r3, unsigned address) {
    ldmatrix_x2_t(r0, r1, address);
    r2 = ldmatrix_word<2, true>(address); r3 = ldmatrix_word<3, true>(address);
}
} // namespace ninfer::ops
