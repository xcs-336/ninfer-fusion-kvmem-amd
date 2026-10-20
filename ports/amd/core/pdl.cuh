#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>
#include <utility>

// Native HIP has no equivalent of the CUDA programmatic grid-dependency launch
// protocol in this SDK. Stream-ordered launches establish the same producer /
// consumer data dependencies, including during graph capture, without overlap.
namespace ninfer::pdl {
struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem_bytes = 0;
    hipStream_t stream = nullptr;
};
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline hipError_t launch_dependent(const LaunchConfig& launch,
                                                void (*kernel)(KernelArgs...), CallArgs&&... args) {
    kernel<<<launch.grid, launch.block, launch.dynamic_smem_bytes, launch.stream>>>(
        std::forward<CallArgs>(args)...);
    return hipGetLastError();
}
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline hipError_t launch_consumer(const LaunchConfig& launch,
                                               void (*kernel)(KernelArgs...), CallArgs&&... args) {
    return launch_dependent(launch, kernel, std::forward<CallArgs>(args)...);
}
[[nodiscard]] inline bool graph_dependent(hipStream_t stream) noexcept {
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream, &status) != hipSuccess) {
        (void)hipGetLastError();
        return false;
    }
    return status == hipStreamCaptureStatusActive;
}
// These device calls belong to the launch protocol, not the numerical kernel.
// Both eager and graph launches are already serialized before kernel entry.
__device__ __forceinline__ void trigger_dependents() {}
__device__ __forceinline__ void wait_for_dependencies() {}
__device__ __forceinline__ void enter() {}
__device__ __forceinline__ void enter_streaming() {}
} // namespace ninfer::pdl
