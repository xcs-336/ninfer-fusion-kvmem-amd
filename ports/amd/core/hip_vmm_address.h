#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>

namespace ninfer::amd {
using DeviceAddress = std::byte*;
inline hipError_t reserve(DeviceAddress* out, std::size_t bytes, std::size_t alignment,
                          DeviceAddress hint, unsigned long long flags) {
    void* address = nullptr;
    const auto status = hipMemAddressReserve(&address, bytes, alignment, hint, flags);
    if (status == hipSuccess) { *out = static_cast<std::byte*>(address); }
    return status;
}
} // namespace ninfer::amd
