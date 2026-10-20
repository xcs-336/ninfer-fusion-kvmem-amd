#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <vector>

namespace ninfer::amd {

// The Flash-Next owner retains artifact and pinned staging source storage until
// the copy stream completes. Its only CUDA attribute is Stream source access.
// The installed HIP batch API does not support attribute arrays; its normal
// stream ordering and the same caller-owned source lifetime provide this case.
// HIP additionally requires mutable pointer metadata and exposes a failure
// index. Neither changes the pointed-to source bytes or permits an API failure
// to fall back silently to another execution mechanism.
inline hipError_t copy_batch_on_stream(void** destinations, const void** sources,
                                     std::size_t* sizes, std::size_t count,
                                     hipStream_t stream) {
    if (count == 0) { return hipSuccess; }
    std::vector<void*> source_metadata;
    source_metadata.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        source_metadata.push_back(const_cast<void*>(sources[index]));
    }
    std::size_t failure_index = count;
    return hipMemcpyBatchAsync(destinations, source_metadata.data(), sizes, count,
                               nullptr, nullptr, 0, &failure_index, stream);
}

} // namespace ninfer::amd
