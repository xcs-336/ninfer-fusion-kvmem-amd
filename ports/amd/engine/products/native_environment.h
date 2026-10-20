#pragma once

#include <hip/hip_version.h>
#include <string>

namespace ninfer::amd {

// HIP documents major*10^7 + minor*10^5 + patch for these native APIs.
// Its values have no CUDA-version interpretation.
inline std::string hip_version_string(int encoded) {
    if (encoded <= 0) { return {}; }
    return std::to_string(encoded / 10000000) + "." +
        std::to_string((encoded % 10000000) / 100000) + "." +
        std::to_string(encoded % 100000);
}

inline std::string hip_compile_version_string() {
    return hip_version_string(HIP_VERSION);
}

} // namespace ninfer::amd
