// Additional lifetime qualification for the native HIP direct-suffix weight pool. The unchanged
// upstream weight-pool test remains the full-arena byte-exact oracle. This test exercises captured
// home reads, per-extent captured workspace writes, changing loan addresses and transaction moves.
#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/evictable_weight_pool.h"

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

__global__ void read_weight_markers(const std::byte* home, std::uint32_t* output,
                                    std::size_t chunk) {
    const unsigned index = threadIdx.x;
    if (index < 3) {
        output[index] = *reinterpret_cast<const std::uint32_t*>(home + (index + 3) * chunk);
    }
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

} // namespace

int main() {
    int count = 0;
    const hipError_t availability = hipGetDeviceCount(&count);
    if (availability == hipErrorNoDevice || availability == hipErrorInsufficientDriver ||
        (availability == hipSuccess && count == 0)) {
        std::cout << "SKIP: no usable HIP device\n";
        return 77;
    }
    try {
        CUDA_CHECK(availability);
        ninfer::DeviceContext device(0);
        constexpr std::size_t chunk = ninfer::EvictableWeightPool::kChunkBytes;
        constexpr std::size_t bytes = 6 * chunk - chunk / 2;
        ninfer::EvictableWeightPool pool(device, {.arena_bytes = bytes,
                                                .evictable_tail_bytes = 3 * chunk});
        const ninfer::DeviceSpan arena = pool.arena();
        auto* const home = static_cast<std::byte*>(arena.data);
        CUDA_CHECK(hipMemsetAsync(home, 0x61, bytes, device.stream));
        std::array<std::uint32_t, 3> expected{0x01234567U, 0x89ABCDEFU, 0x76543210U};
        for (std::size_t i = 0; i < expected.size(); ++i) {
            CUDA_CHECK(hipMemcpyAsync(home + (i + 3) * chunk, &expected[i], sizeof(expected[i]),
                                      hipMemcpyHostToDevice, device.stream));
        }
        pool.capture_window_mirror(chunk + 1, device.stream);
        require(pool.mirror_bytes() == chunk + chunk / 2, "unaligned mirror excludes padding");

        ninfer::DeviceArena output_arena(256);
        const auto output = output_arena.alloc_bytes(sizeof(expected));
        ninfer::DecodeGraphDefinition home_definition;
        ninfer::DecodeGraphExecutable home_graph;
        home_definition.capture(device.stream, [&] {
            read_weight_markers<<<1, 32, 0, device.stream>>>(
                home, static_cast<std::uint32_t*>(output.data), chunk);
            CUDA_CHECK(hipGetLastError());
        });
        home_graph.instantiate(home_definition);

        const std::size_t requests[] = {chunk + 1, 2 * chunk, 1, chunk};
        void* two_chunk_base = nullptr;
        void* one_chunk_base = nullptr;
        for (std::size_t cycle = 0; cycle < std::size(requests); ++cycle) {
            if (cycle == 2) {
                // A one-chunk loan must preserve writes both before and inside the maximum window.
                expected[0] = 0xBAD00003U;
                expected[1] = 0xBAD00004U;
                for (std::size_t i = 0; i < 2; ++i) {
                    CUDA_CHECK(hipMemcpyAsync(home + (i + 3) * chunk, &expected[i],
                                              sizeof(expected[i]), hipMemcpyHostToDevice,
                                              device.stream));
                }
            }
            auto loan = pool.evict(requests[cycle], device.stream);
            const std::size_t extent = cycle < 2 ? 2 * chunk : chunk;
            require(loan.leased().bytes == extent, "loan rounds exactly to requested chunks");
            require(loan.leased().data == home + 6 * chunk - extent,
                    "every loan borrows the actual arena suffix");
            if (extent == 2 * chunk) {
                if (two_chunk_base != nullptr) {
                    require(loan.leased().data == two_chunk_base,
                            "same rounded extent retains the same workspace pointer");
                }
                two_chunk_base = loan.leased().data;
            } else {
                if (one_chunk_base != nullptr) {
                    require(loan.leased().data == one_chunk_base,
                            "one-chunk extent retains the same workspace pointer");
                }
                one_chunk_base = loan.leased().data;
                require(one_chunk_base != two_chunk_base,
                        "different rounded extents intentionally have different pointers");
            }
            ninfer::DecodeGraphDefinition workspace_definition;
            ninfer::DecodeGraphExecutable workspace_graph;
            workspace_definition.capture(device.stream, [&] {
                CUDA_CHECK(hipMemsetAsync(loan.leased().data, 0xA5, extent, device.stream));
            });
            workspace_graph.instantiate(workspace_definition);
            workspace_graph.launch(device.stream);
            auto moved = std::move(loan);
            require(!loan.open() && moved.open(), "move construction transfers the transaction");
            ninfer::EvictableWeightPool::Transaction assigned;
            assigned = std::move(moved);
            require(!moved.open() && assigned.open(), "move assignment transfers the transaction");
            assigned.close();
            assigned.close();
            require(!pool.transaction_open() && !pool.poisoned(), "restore closes without poison");
            require(pool.arena().data == home, "weight home addresses remain fixed");

            home_graph.launch(device.stream);
            std::array<std::uint32_t, 3> observed{};
            CUDA_CHECK(hipMemcpyAsync(observed.data(), output.data, sizeof(observed),
                                      hipMemcpyDeviceToHost, device.stream));
            CUDA_CHECK(hipStreamSynchronize(device.stream));
            require(observed == expected, "captured home reads see exact restored/untouched words");
        }
        std::cout << "PASS: native HIP weight suffixes, exact untouched markers, transaction moves, "
                     "4 captured workspace writes and stable captured home reads\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native HIP memory graph qualification failed: " << error.what() << '\n';
        return 1;
    }
}
