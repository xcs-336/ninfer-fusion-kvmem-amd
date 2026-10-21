// NEW private host/oracle utility copy from frozen six-case fixture. Original file is untouched.
// No private launch wrapper is included. Original FP64 oracle/guard utilities are preserved.
// SOURCE ONLY candidate, not built/qualified. Attention bodies are compiled in
// the new public-policy library; the utility's only GPU kernel is a raw borrower.
// This does not enable a production Op or Engine gate.
#include "amd-b1/public_bf16.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "amd-memory/kv_segmented_pool_owner.h"
#include "core/host_kv_arena.h"
#include "op_check.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::amd::memory;
constexpr std::size_t GuardBytes = 256;
constexpr std::uint8_t Guard = 0xa5;
constexpr int Groups = 16, LogicalPages = 8, LogicalTokens = LogicalPages * 64, D = 256;
constexpr float Scale = 0.0625f;

void need(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
void check(hipError_t value, const char* where) {
    if (value != hipSuccess) throw std::runtime_error(std::string(where) + ": " + hipGetErrorString(value));
}

class Stream {
public:
    Stream() { check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking), "create test stream"); }
    ~Stream() { if (value) { (void)hipStreamSynchronize(value); (void)hipStreamDestroy(value); } }
    Stream(const Stream&) = delete;
    hipStream_t value = nullptr;
};

class Buffer {
public:
    explicit Buffer(std::size_t bytes) : bytes(bytes) {
        need(bytes != 0 && bytes <= std::numeric_limits<std::size_t>::max() - 2 * GuardBytes,
             "test buffer length overflow");
        check(hipMalloc(&allocation, bytes + 2 * GuardBytes), "test guarded allocation");
        data = static_cast<std::uint8_t*>(allocation) + GuardBytes;
        check(hipMemset(allocation, Guard, bytes + 2 * GuardBytes), "initialize guards");
        check(hipStreamSynchronize(nullptr), "finish default-stream guard initialization");
    }
    ~Buffer() { if (allocation) { (void)hipDeviceSynchronize(); (void)hipFree(allocation); } }
    Buffer(const Buffer&) = delete;
    void upload(const void* host, hipStream_t stream) {
        check(hipMemcpyAsync(data, host, bytes, hipMemcpyHostToDevice, stream), "upload test bytes");
    }
    template <typename T> std::vector<T> read(hipStream_t stream) const {
        need(bytes % sizeof(T) == 0, "test read element size");
        std::vector<T> out(bytes / sizeof(T));
        check(hipMemcpyAsync(out.data(), data, bytes, hipMemcpyDeviceToHost, stream), "read test bytes");
        check(hipStreamSynchronize(stream), "finish test read");
        return out;
    }
    void guards(hipStream_t stream) const {
        std::array<std::uint8_t, GuardBytes> prefix{}, suffix{};
        check(hipMemcpyAsync(prefix.data(), allocation, GuardBytes, hipMemcpyDeviceToHost, stream), "prefix guard");
        check(hipMemcpyAsync(suffix.data(), static_cast<const std::uint8_t*>(data) + bytes,
                             GuardBytes, hipMemcpyDeviceToHost, stream), "suffix guard");
        check(hipStreamSynchronize(stream), "finish guard read");
        need(std::all_of(prefix.begin(), prefix.end(), [](auto x) { return x == Guard; }) &&
             std::all_of(suffix.begin(), suffix.end(), [](auto x) { return x == Guard; }), "guard was modified");
    }
    void* allocation = nullptr;
    void* data = nullptr;
    std::size_t bytes = 0;
};

class Graph {
public:
    ~Graph() {
        (void)hipDeviceSynchronize();
        if (exec) (void)hipGraphExecDestroy(exec);
        if (graph) (void)hipGraphDestroy(graph);
    }
    Graph() = default;
    Graph(const Graph&) = delete;
    template <typename Callable> void capture(hipStream_t stream, Callable&& body) {
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), "begin real math capture");
        body();
        check(hipStreamEndCapture(stream, &graph), "end real math capture");
        check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate real math graph");
    }
    void replay(hipStream_t stream) {
        check(hipGraphLaunch(exec, stream), "replay same real math graph");
        check(hipStreamSynchronize(stream), "finish same graph replay");
    }
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
};

std::uint16_t bf16(float value) {
    // Independent CPU BF16 RNE; patterned values happen to be exactly representable.
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}
double fp64(std::uint16_t value) {
    return static_cast<double>(std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16));
}
std::vector<std::uint16_t> pattern(std::size_t count, unsigned seed, int denominator) {
    std::vector<std::uint16_t> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto mixed = static_cast<std::uint32_t>((i * 37u + (i / 17u) * 13u + seed * 23u) % 97u);
        out[i] = bf16(static_cast<float>(static_cast<int>(mixed) - 48) / denominator);
    }
    return out;
}
std::size_t logical_index(int heads, int h, int token, int d) {
    return static_cast<std::size_t>(d) + D * (h + static_cast<std::size_t>(heads) * token);
}

std::vector<double> oracle(const std::vector<std::uint16_t>& q,
                          const std::vector<std::uint16_t>& keys,
                          const std::vector<std::uint16_t>& values,
                          int kv_heads, int base, int width) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int q_heads = kv_heads == 4 ? 24 : 16;
    const int group = q_heads / kv_heads;
    std::vector<double> out(static_cast<std::size_t>(D) * q_heads * width);
    std::vector<double> scores(base + width);
    for (int t = 0; t < width; ++t) for (int h = 0; h < q_heads; ++h) {
        const int kv = h / group, end = base + t + 1;
        double maximum = -std::numeric_limits<double>::infinity();
        for (int key = 0; key < end; ++key) {
            double dot = 0;
            for (int d = 0; d < D; ++d)
                dot += fp64(q[logical_index(q_heads, h, t, d)]) * fp64(keys[logical_index(kv_heads, kv, key, d)]);
            scores[key] = dot * static_cast<double>(Scale);
            maximum = std::max(maximum, scores[key]);
        }
        double denominator = 0;
        for (int key = 0; key < end; ++key) { scores[key] = std::exp(scores[key] - maximum); denominator += scores[key]; }
        for (int d = 0; d < D; ++d) {
            double numerator = 0;
            for (int key = 0; key < end; ++key)
                numerator += scores[key] * fp64(values[logical_index(kv_heads, kv, key, d)]);
            out[logical_index(q_heads, h, t, d)] = numerator / denominator;
        }
    }
    return out;
}

__global__ void poison_borrowed_span(std::uint8_t* bytes, std::uint64_t size) {
    for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) bytes[i] = 0xd3;
}


} // namespace
