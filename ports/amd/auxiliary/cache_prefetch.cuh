#pragma once

#include <hip/hip_runtime.h>

namespace ninfer::amd {

// A native scalar data-prefetch replaces PTX's explicit NVIDIA L2 request.
// gfx1201 supplies s_prefetch_data. Its length encodes 1--32 chunks of 128B,
// so lane zero coalesces the original wave's contiguous 32 requests and clamps
// the tail to the caller-owned span. AMD targets the scalar/constant cache,
// rather than promising NVIDIA L2 residency. Hardware may treat the request as
// a no-op when MODE.SCALAR_PREFETCH_EN is zero; no cache-performance claim is
// made without measurement. Tensor values and the Op's workspace are unchanged.
__device__ __forceinline__ void prefetch_for_read(const void* address,
                                                unsigned long long remaining_bytes) {
    if ((threadIdx.x & 31U) == 0U) {
        const unsigned chunks = remaining_bytes >= 4096ULL
                                    ? 32U
                                    : static_cast<unsigned>((remaining_bytes - 1ULL) / 128ULL + 1ULL);
        __builtin_amdgcn_s_prefetch_data(address, chunks - 1U);
    }
}

} // namespace ninfer::amd
