#pragma once

#include <hip/hip_runtime.h>

namespace ninfer::amd {

// gfx1201 has a whole-workgroup split barrier, but its hardware does not have
// gfx12.5 named subset barriers. A producer group therefore rendezvous through
// native LDS atomics. Every participating wave must be complete and converged;
// the first N waves of one CTA are the only supported participants. This helper
// owns no allocation and must be initialized before a whole-CTA __syncthreads.
struct PartialWaveBarrier {
    unsigned arrivals;
    unsigned generation;
};

__device__ __forceinline__ void initialize(PartialWaveBarrier& barrier) {
    if (threadIdx.x == 0) {
        barrier.arrivals = 0;
        barrier.generation = 0;
    }
    __syncthreads();
}

__device__ __forceinline__ void arrive_and_wait(PartialWaveBarrier& barrier,
                                               unsigned participating_threads) {
    // Publish each lane's prior LDS stores before the leader advertises arrival.
    __threadfence_block();
    __builtin_amdgcn_wave_barrier();
    if ((threadIdx.x & 31) == 0) {
        const unsigned generation = atomicAdd(&barrier.generation, 0U);
        const unsigned arrival = atomicAdd(&barrier.arrivals, 1U);
        if (arrival + 1 == participating_threads / 32) {
            atomicExch(&barrier.arrivals, 0U);
            __threadfence_block();
            atomicAdd(&barrier.generation, 1U);
        } else {
            while (atomicAdd(&barrier.generation, 0U) == generation) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __builtin_amdgcn_wave_barrier();
    __threadfence_block();
}

} // namespace ninfer::amd
