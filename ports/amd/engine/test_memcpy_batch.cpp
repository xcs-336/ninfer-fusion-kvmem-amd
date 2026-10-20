#include "memcpy_batch.h"

#include "core/arena.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(hipError_t status, const char* operation) {
    if (status != hipSuccess) { throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status)); }
}
struct Stream {
    hipStream_t value{};
    Stream() { check(hipStreamCreate(&value), "create stream"); }
    ~Stream() { if (value != nullptr) { (void)hipStreamDestroy(value); } }
};
struct Pinned {
    std::uint8_t* bytes{};
    explicit Pinned(std::size_t count) { check(hipHostMalloc(reinterpret_cast<void**>(&bytes), count), "allocate staging"); }
    ~Pinned() { if (bytes != nullptr) { (void)hipHostFree(bytes); } }
};

int run(std::size_t count) {
    Stream stream;
    if (count == 0) {
        check(ninfer::amd::copy_batch_on_stream(nullptr, nullptr, nullptr, 0, stream.value), "empty batch");
        return 0;
    }
    constexpr std::size_t Guard = 32, PinnedStride = 4096;
    constexpr std::size_t Sizes[] = {1, 31, 257, 4096, 513, 128};
    Pinned staging(count * PinnedStride);
    std::vector<std::vector<std::uint8_t>> pageable(count);
    std::vector<const void*> sources(count);
    std::vector<std::size_t> sizes(count), offsets(count);
    std::size_t total = Guard;
    for (std::size_t index = 0; index < count; ++index) {
        sizes[index] = Sizes[index % 6];
        offsets[index] = total;
        total += sizes[index] + Guard;
        pageable[index].resize(sizes[index]);
        sources[index] = index % 2 == 0 ? pageable[index].data() : staging.bytes + index * PinnedStride;
    }
    std::vector<std::uint8_t> expected(total, 0xA5), actual(total);
    ninfer::DeviceBuffer device(total);
    device.copy_from_host(expected.data(), expected.size());
    std::vector<void*> destinations(count);
    for (std::size_t index = 0; index < count; ++index) {
        destinations[index] = static_cast<std::uint8_t*>(device.p) + offsets[index];
    }
    for (unsigned phase = 0; phase < 2; ++phase) {
        for (std::size_t index = 0; index < count; ++index) {
            auto* source = const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(sources[index]));
            for (std::size_t byte = 0; byte < sizes[index]; ++byte) {
                source[byte] = static_cast<std::uint8_t>(17 * index + 131 * byte + 79 * phase);
            }
            std::memcpy(expected.data() + offsets[index], source, sizes[index]);
        }
        check(ninfer::amd::copy_batch_on_stream(destinations.data(), sources.data(), sizes.data(), count, stream.value),
              "native HIP batch copy");
        check(hipStreamSynchronize(stream.value), "batch stream completion");
        device.copy_to_host(actual.data(), actual.size());
        if (actual != expected) {
            const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
            std::fprintf(stderr, "native batch count=%zu phase=%u exact/guard mismatch at %zu\n", count, phase,
                         static_cast<std::size_t>(mismatch.first - actual.begin()));
            return 1;
        }
        for (std::size_t index = 0; index < count; ++index) {
            if (std::memcmp(sources[index], expected.data() + offsets[index], sizes[index]) != 0) {
                std::fprintf(stderr, "native batch modified source %zu\n", index);
                return 1;
            }
        }
    }
    return 0;
}
} // namespace

int main() {
    try {
        int failures = 0;
        for (const std::size_t count : {0U, 1U, 6U, 32U}) { failures += run(count); }
        std::printf("native HIP batch copy: %s (pageable/pinned sources, two updates, exact bytes and guards)\n",
                    failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "native HIP batch copy: %s\n", error.what());
        return 1;
    }
}
