#include "partial_barrier.cuh"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void check(hipError_t status) {
    if (status != hipSuccess) { throw std::runtime_error(hipGetErrorString(status)); }
}

template <int ProducerWaves, int Threads>
__global__ void exercise(unsigned* failures, unsigned* result) {
    __shared__ ninfer::amd::PartialWaveBarrier barrier;
    __shared__ unsigned slots[ProducerWaves * 32];
    ninfer::amd::initialize(barrier);
    const unsigned tid = threadIdx.x;
    unsigned sum = 0;
    for (unsigned epoch = 1; epoch <= 257; ++epoch) {
        if (tid < ProducerWaves * 32) {
            slots[tid] = epoch * 1024 + tid;
            ninfer::amd::arrive_and_wait(barrier, ProducerWaves * 32);
            // All producer waves read data published by a different wave and
            // lane, with a different schedule at each iteration.
            const unsigned other = (tid + 33 + epoch) % (ProducerWaves * 32);
            if (slots[other] != epoch * 1024 + other) { atomicAdd(failures, 1U); }
            sum += slots[other];
            ninfer::amd::arrive_and_wait(barrier, ProducerWaves * 32);
        } else {
            // Disjoint consumer waves do independent work until the enclosing
            // workgroup barrier. A mistaken __syncthreads subset would hang.
            sum += epoch * 7 + tid;
        }
        __syncthreads();
    }
    result[blockIdx.x * Threads + tid] = sum;
}

template <int ProducerWaves, int Threads>
unsigned run() {
    constexpr unsigned Blocks = 7;
    unsigned* failures = nullptr;
    unsigned* output = nullptr;
    check(hipMalloc(&failures, sizeof(unsigned)));
    check(hipMalloc(&output, Blocks * Threads * sizeof(unsigned)));
    check(hipMemset(failures, 0, sizeof(unsigned)));
    exercise<ProducerWaves, Threads><<<Blocks, Threads>>>(failures, output);
    check(hipGetLastError());
    check(hipDeviceSynchronize());
    unsigned count = 0;
    check(hipMemcpy(&count, failures, sizeof(unsigned), hipMemcpyDeviceToHost));
    std::vector<unsigned> values(Blocks * Threads);
    check(hipMemcpy(values.data(), output, values.size() * sizeof(unsigned), hipMemcpyDeviceToHost));
    for (unsigned block = 0; block < Blocks; ++block) {
        for (unsigned tid = 0; tid < Threads; ++tid) {
            unsigned expected = 0;
            for (unsigned epoch = 1; epoch <= 257; ++epoch) {
                const unsigned other = (tid + 33 + epoch) % (ProducerWaves * 32);
                expected += tid < ProducerWaves * 32 ? epoch * 1024 + other : epoch * 7 + tid;
            }
            if (values[block * Threads + tid] != expected) { ++count; }
        }
    }
    check(hipFree(output));
    check(hipFree(failures));
    return count;
}
} // namespace

int main() {
    try {
        const auto failures = run<2, 512>() + run<4, 512>() + run<6, 256>() + run<8, 384>();
        std::printf("native producer-only LDS barrier: %s (%u errors)\n",
                    failures == 0 ? "PASS" : "FAIL", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
