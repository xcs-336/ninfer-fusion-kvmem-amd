// Isolate the native HIP VMM memory-copy and memset paths from NInfer policy.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

void check(hipError_t error, const char* call) {
    if (error != hipSuccess) {
        throw std::runtime_error(std::string(call) + ": " + hipGetErrorName(error));
    }
}
#define H(call) check((call), #call)
__global__ void fill_bytes(unsigned char* out, std::size_t bytes, unsigned char value) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes;
         i += blockDim.x * gridDim.x) { out[i] = value; }
}
__global__ void observe(const unsigned char* home, const unsigned char* overlay,
                        std::size_t granule, unsigned char* out) {
    if (threadIdx.x == 0) {
        out[0] = home[0]; out[1] = home[2 * granule];
        out[2] = overlay[0]; out[3] = overlay[granule];
    }
}
int main(int argc, char** argv) {
    try {
        H(hipSetDevice(0));
        unsigned char* device_out = nullptr;
        H(hipMalloc(&device_out, 4));
        if (argc > 2 && std::string(argv[2]) == "warm") {
            fill_bytes<<<1, 32>>>(device_out, 4, 0);
            H(hipGetLastError());
            H(hipDeviceSynchronize());
            std::cerr << "kernel module warmed before VMM\n";
        }
        hipMemAllocationProp prop{};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = 0;
        std::size_t g = 0;
        H(hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum));
        if (argc > 1) { g = std::max(g, static_cast<std::size_t>(std::stoull(argv[1]))); }
        void* home_raw = nullptr;
        void* overlay_raw = nullptr;
        H(hipMemAddressReserve(&home_raw, 4 * g, g, nullptr, 0));
        H(hipMemAddressReserve(&overlay_raw, 2 * g, g, nullptr, 0));
        auto* home = static_cast<unsigned char*>(home_raw);
        auto* overlay = static_cast<unsigned char*>(overlay_raw);
        std::array<hipMemGenericAllocationHandle_t, 4> handles{};
        hipMemAccessDesc access{};
        access.location = prop.location;
        access.flags = hipMemAccessFlagsProtReadWrite;
        for (std::size_t i = 0; i < handles.size(); ++i) {
            H(hipMemCreate(&handles[i], g, &prop, 0));
            H(hipMemMap(home + i * g, g, 0, handles[i], 0));
            H(hipMemSetAccess(home + i * g, g, &access, 1));
        }
        std::cerr << "VMM mapped granule=" << g << "; initial kernel fill\n";
        fill_bytes<<<8, 128>>>(home, 4 * g, 0x5a);
        H(hipGetLastError());
        H(hipDeviceSynchronize());
        std::cerr << "initial fill complete; remap\n";
        for (std::size_t rank = 0; rank < 2; ++rank) {
            const auto i = 1 + 2 * rank;
            H(hipMemUnmap(home + i * g, g));
            H(hipMemMap(overlay + rank * g, g, 0, handles[i], 0));
            H(hipMemSetAccess(overlay + rank * g, g, &access, 1));
        }
        int failures = 0;
        std::cerr << "remap complete\n";
        for (bool native_kernel : {false, true}) {
            fill_bytes<<<8, 128>>>(home, g, 0x5a);
            fill_bytes<<<8, 128>>>(home + 2 * g, g, 0x5a);
            H(hipGetLastError());
            H(hipDeviceSynchronize());
            std::cerr << "resident restore complete; " << (native_kernel ? "kernel fill" : "runtime fill") << '\n';
            if (native_kernel) {
                fill_bytes<<<8, 128>>>(overlay, 2 * g, 0x27);
                H(hipGetLastError());
            } else { H(hipMemset(overlay, 0x27, 2 * g)); }
            observe<<<1, 32>>>(home, overlay, g, device_out);
            H(hipGetLastError());
            H(hipDeviceSynchronize());
            std::cerr << "GPU observation complete; runtime copies\n";
            std::array<unsigned char, 4> gpu{}, runtime{};
            H(hipMemcpy(gpu.data(), device_out, 4, hipMemcpyDeviceToHost));
            H(hipMemcpy(&runtime[0], home, 1, hipMemcpyDeviceToHost));
            H(hipMemcpy(&runtime[1], home + 2 * g, 1, hipMemcpyDeviceToHost));
            H(hipMemcpy(&runtime[2], overlay, 1, hipMemcpyDeviceToHost));
            H(hipMemcpy(&runtime[3], overlay + g, 1, hipMemcpyDeviceToHost));
            const std::array<unsigned char, 4> expected{0x5a, 0x5a, 0x27, 0x27};
            std::cout << (native_kernel ? "kernel-fill" : "hipMemset-fill") << " granule=" << g;
            std::cout << " gpu-observed=";
            for (auto x : gpu) { std::cout << static_cast<unsigned>(x) << ','; }
            std::cout << " hipMemcpy-observed=";
            for (auto x : runtime) { std::cout << static_cast<unsigned>(x) << ','; }
            std::cout << '\n';
            if (gpu != expected || runtime != expected) { ++failures; }
        }
        H(hipFree(device_out));
        H(hipMemUnmap(home, g));
        H(hipMemUnmap(home + 2 * g, g));
        for (std::size_t rank = 0; rank < 2; ++rank) { H(hipMemUnmap(overlay + rank * g, g)); }
        for (auto handle : handles) { H(hipMemRelease(handle)); }
        H(hipMemAddressFree(home, 4 * g));
        H(hipMemAddressFree(overlay, 2 * g));
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
