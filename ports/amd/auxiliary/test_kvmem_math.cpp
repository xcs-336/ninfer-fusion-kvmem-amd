#include "ops/kvmem/kvmem_retrieve_launch.h"
#include "ops/kvmem/mean_k_index_launch.h"
#include "ops/kvmem/raw_k_shadow_launch.h"
#include "ops/op_tester.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {
void check(hipError_t status) { cuda_check(status, "native KVMem test"); }
struct Stream {
    hipStream_t value{};
    Stream() { check(hipStreamCreate(&value)); }
    ~Stream() { if (value != nullptr) { (void)hipStreamDestroy(value); } }
};

void captured(hipStream_t stream, const std::function<void()>& launch, int replays) {
    hipGraph_t definition{};
    hipGraphExec_t executable{};
    check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    launch();
    check(hipStreamEndCapture(stream, &definition));
    check(hipGraphInstantiateWithFlags(&executable, definition, 0));
    for (int replay = 0; replay < replays; ++replay) { check(hipGraphLaunch(executable, stream)); }
    check(hipStreamSynchronize(stream));
    check(hipGraphExecDestroy(executable));
    check(hipGraphDestroy(definition));
}

template <class T>
int readonly(const std::string& label, const DeviceBuffer& device, const std::vector<T>& host) {
    const auto actual = from_device<T>(device, host.size());
    if (std::memcmp(actual.data(), host.data(), host.size() * sizeof(T)) == 0) { return 0; }
    std::fprintf(stderr, "%s input bits changed\n", label.c_str());
    return 1;
}

int raw_copy() {
    int failures = 0;
    Stream stream;
    for (const std::size_t count : {0U, 1U, 31U, 32U, 129U, 513U, 65536U}) {
        std::vector<std::uint16_t> source(std::max<std::size_t>(count, 1));
        for (std::size_t i = 0; i < source.size(); ++i) { source[i] = static_cast<std::uint16_t>(i); }
        std::vector<std::uint16_t> expected(count + 64, 0xA55A);
        std::copy_n(source.begin(), count, expected.begin() + 32);
        std::vector<std::uint16_t> initial(count + 64, 0xA55A);
        auto input = to_device(source), output = to_device(initial);
        const auto launch = [&] {
            ops::detail::launch_raw_k_shadow_copy(source.empty() ? nullptr : input.p,
                static_cast<std::uint16_t*>(output.p) + 32, static_cast<std::int64_t>(count), stream.value);
        };
        if (count == 513) { captured(stream.value, launch, 2); }
        else { launch(); check(hipStreamSynchronize(stream.value)); }
        const std::string label = "KVMem raw BF16 bit-copy/guard n=" + std::to_string(count);
        failures += verify_exact(label.c_str(),
                                 from_device<std::uint16_t>(output, expected.size()), expected);
        failures += readonly("KVMem raw source", input, source);
    }
    return failures;
}

int mean_accumulate() {
    // Separate source/destination layers, padded row and layer strides, a head
    // dimension crossing a wave boundary, and a partial tail continued twice.
    constexpr int SourceLayers = 2, DestinationLayers = 4, LayerBegin = 1;
    constexpr int Heads = 3, Dim = 37, BlockTokens = 4, FullBlocks = 2, Tail = 3;
    constexpr int Width = FullBlocks * BlockTokens + Tail;
    constexpr int KVSize = Heads * Dim + 5, SourceStride = Width * KVSize + 13;
    constexpr int Blocks = 7, FirstBlock = 2;
    constexpr int SumElements = Blocks * Heads * Dim;
    constexpr int IndexStride = SumElements + Blocks + 11;
    std::vector<std::uint16_t> source(SourceLayers * SourceStride, 0x7FC1);
    for (int layer = 0; layer < SourceLayers; ++layer) {
        for (int token = 0; token < Width; ++token) {
            for (int channel = 0; channel < Heads * Dim; ++channel) {
                const float value = static_cast<float>((layer * 37 + token * 11 + channel * 3) % 257 - 128) / 16.0F;
                source[layer * SourceStride + token * KVSize + channel] = f32_to_bf16(value);
            }
        }
    }
    std::vector<float> initial(DestinationLayers * IndexStride, 0.0F), expected = initial;
    for (int layer = 0; layer < DestinationLayers; ++layer) {
        std::fill(initial.begin() + layer * IndexStride + SumElements + Blocks,
                  initial.begin() + (layer + 1) * IndexStride, 15.75F);
    }
    expected = initial;
    for (int layer = 0; layer < SourceLayers; ++layer) {
        for (int block = 0; block < FullBlocks + 1; ++block) {
            const int fill = block < FullBlocks ? BlockTokens : Tail;
            const auto base = (layer + LayerBegin) * IndexStride;
            expected[base + SumElements + FirstBlock + block] = static_cast<float>(2 * fill);
            for (int channel = 0; channel < Heads * Dim; ++channel) {
                double sum = 0.0;
                for (int token = 0; token < fill; ++token) {
                    sum += bf16_to_f32(source[layer * SourceStride +
                        (block * BlockTokens + token) * KVSize + channel]);
                }
                expected[base + (FirstBlock + block) * Heads * Dim + channel] = static_cast<float>(2.0 * sum);
            }
        }
    }
    auto input = to_device(source), output = to_device(initial);
    Stream stream;
    captured(stream.value, [&] {
        check(ops::detail::launch_mean_k_accumulate(input.p, output.p, SourceLayers, LayerBegin,
            FirstBlock, FullBlocks, Tail, Heads, Dim, KVSize, BlockTokens, SourceStride,
            Blocks, IndexStride, stream.value));
    }, 2);
    return verify_exact("KVMem mean-K sum/count/layer/tail exact", from_device<float>(output, expected.size()), expected)
         + readonly("KVMem mean-K represented input", input, source);
}

std::uint16_t exact_half(float value) {
    // All fixture values are exact multiples of 1/32 in [-1,1], so conversion
    // has no rounding ambiguity. Decode/packing is independent of HIP types.
    std::uint32_t bits{};
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7FFFFFFFU) == 0) { return static_cast<std::uint16_t>(bits >> 16); }
    const int exponent = static_cast<int>((bits >> 23) & 255) - 127 + 15;
    return static_cast<std::uint16_t>(((bits >> 16) & 0x8000) | (exponent << 10) | ((bits >> 13) & 1023));
}

int retrieve(ops::KvMemRetrieveDtype dtype, int variant) {
    ops::KvMemRetrieveConfig config;
    config.dtype = dtype;
    config.n_layers = variant == 0 || variant == 3 ? 1 : 2;
    config.n_query_tokens = variant == 0 ? 1 : variant == 1 ? 3 : 2;
    config.n_heads = variant == 0 ? 4 : variant == 1 ? 6 : variant == 2 ? 24 : 8;
    config.n_kv_heads = variant == 2 ? 4 : 2;
    config.head_dim = variant == 0 || variant == 2 ? 256 : variant == 1 ? 37 : 64;
    config.n_blocks = variant == 0 ? 1 : variant == 1 ? 17 : variant == 2 ? 257 : 31;
    config.q_token_begin = variant == 0 ? 0 : 2;
    config.q_layer_stride = config.q_token_begin + config.n_query_tokens + 3;
    config.kbar_layer_stride = config.n_blocks + 3;
    config.n_layers_total = variant == 2 ? 3 : config.n_layers;
    config.mask_mode = variant == 0 ? ops::KvMemRetrieveMaskMode::Never
                        : variant == 1 ? ops::KvMemRetrieveMaskMode::Official
                                       : ops::KvMemRetrieveMaskMode::Always;
    config.budget_blocks = variant == 1 ? 7 : 0;
    config.sink_blocks = variant == 1 ? 2 : variant == 2 ? 3 : config.n_blocks;
    config.recent_blocks = variant == 1 ? 3 : variant == 2 ? 5 : config.n_blocks;
    const bool per_layer = variant == 2, graph = variant == 2;
    const int replays = graph ? 2 : 1;
    const bool has_counts = variant != 0;
    const std::size_t q_layer_elements = static_cast<std::size_t>(config.q_layer_stride) * config.n_heads * config.head_dim;
    const std::size_t k_layer_elements = static_cast<std::size_t>(config.kbar_layer_stride) * config.n_kv_heads * config.head_dim;
    const std::size_t q_bytes = dtype == ops::KvMemRetrieveDtype::F32 ? 4 : 2;
    std::vector<std::uint8_t> query(config.n_layers * q_layer_elements * q_bytes, 0xFF);
    std::vector<double> q_reference(config.n_layers * q_layer_elements, 0.0);
    std::vector<float> keys(config.n_layers * k_layer_elements, std::numeric_limits<float>::quiet_NaN());
    std::vector<int> counts(config.n_blocks);
    for (int block = 0; block < config.n_blocks; ++block) { counts[block] = 1 + (block * 5) % 7; }
    for (int layer = 0; layer < config.n_layers; ++layer) {
        for (int token = 0; token < config.n_query_tokens; ++token) {
            for (int head = 0; head < config.n_heads; ++head) {
                for (int dim = 0; dim < config.head_dim; ++dim) {
                    const auto index = layer * q_layer_elements +
                        ((config.q_token_begin + token) * config.n_heads + head) * config.head_dim + dim;
                    const float value = static_cast<float>((layer * 23 + token * 7 + head * 3 + dim * 11) % 65 - 32) / 32.0F;
                    q_reference[index] = value;
                    if (q_bytes == 4) { std::memcpy(query.data() + index * 4, &value, 4); }
                    else {
                        const std::uint16_t code = dtype == ops::KvMemRetrieveDtype::BF16 ? f32_to_bf16(value) : exact_half(value);
                        std::memcpy(query.data() + index * 2, &code, 2);
                    }
                }
            }
        }
        for (int block = 0; block < config.n_blocks; ++block) {
            for (int head = 0; head < config.n_kv_heads; ++head) {
                for (int dim = 0; dim < config.head_dim; ++dim) {
                    const auto index = layer * k_layer_elements + (block * config.n_kv_heads + head) * config.head_dim + dim;
                    const float mean = static_cast<float>((layer * 17 + block * 13 + head * 19 + dim * 5) % 65 - 32) / 32.0F;
                    keys[index] = mean * (has_counts ? counts[block] : 1);
                }
            }
        }
    }
    // Derive the included set directly from the documented mask formula.
    int begin = 0, end = config.n_blocks;
    const bool competing = config.mask_mode == ops::KvMemRetrieveMaskMode::Always ||
        (config.mask_mode == ops::KvMemRetrieveMaskMode::Official && config.budget_blocks > 0 && config.n_blocks > config.budget_blocks);
    const int sink = std::min(config.sink_blocks, config.n_blocks), recent = std::min(config.recent_blocks, config.n_blocks);
    if (competing && sink + recent < config.n_blocks) { begin = sink; end -= recent; }
    std::vector<double> expected(config.n_blocks, 0.25);
    const double scale = 1.0 / std::sqrt(static_cast<double>(config.head_dim));
    for (int layer = 0; layer < config.n_layers; ++layer) {
        for (int token = 0; token < config.n_query_tokens; ++token) {
            for (int head = 0; head < config.n_heads; ++head) {
                const int kv = head / (config.n_heads / config.n_kv_heads);
                const auto qi = layer * q_layer_elements + ((config.q_token_begin + token) * config.n_heads + head) * config.head_dim;
                std::vector<double> scores(end - begin);
                double maximum = -std::numeric_limits<double>::infinity();
                for (int block = begin; block < end; ++block) {
                    const auto ki = layer * k_layer_elements + (block * config.n_kv_heads + kv) * config.head_dim;
                    double dot = 0.0;
                    for (int dim = 0; dim < config.head_dim; ++dim) { dot += q_reference[qi + dim] * keys[ki + dim]; }
                    scores[block - begin] = scale * dot / (has_counts ? counts[block] : 1);
                    maximum = std::max(maximum, scores[block - begin]);
                }
                double denominator = 0.0;
                for (auto& score : scores) { score = std::exp(score - maximum); denominator += score; }
                for (int block = begin; block < end; ++block) {
                    expected[block] += replays * scores[block - begin] / denominator /
                        (static_cast<double>(config.n_layers_total) * config.n_heads);
                }
            }
        }
    }
    auto q_device = to_device(query), k_device = to_device(keys), count_device = to_device(counts);
    std::vector<float> output(config.n_blocks + 32, -1234.5F);
    std::fill(output.begin() + 16, output.end() - 16, 0.25F);
    auto score_device = to_device(output);
    auto* score = static_cast<float*>(score_device.p) + 16;
    Stream stream;
    const auto launch = [&] {
        if (per_layer) {
            auto call = config;
            call.n_layers = 1;
            for (int layer = 0; layer < config.n_layers; ++layer) {
                check(ops::kvmem_retrieve_scores(call, score,
                    static_cast<std::uint8_t*>(q_device.p) + layer * q_layer_elements * q_bytes,
                    static_cast<float*>(k_device.p) + layer * k_layer_elements,
                    has_counts ? static_cast<int*>(count_device.p) : nullptr, stream.value));
            }
        } else {
            check(ops::kvmem_retrieve_scores(config, score, q_device.p, static_cast<float*>(k_device.p),
                has_counts ? static_cast<int*>(count_device.p) : nullptr, stream.value));
        }
    };
    if (graph) { captured(stream.value, launch, replays); }
    else { launch(); check(hipStreamSynchronize(stream.value)); }
    const auto all_actual = from_device<float>(score_device, output.size());
    std::vector<double> actual(all_actual.begin() + 16, all_actual.end() - 16);
    const std::string label = "KVMem retrieve dtype=" + std::to_string(static_cast<int>(dtype)) + " case=" + std::to_string(variant);
    int failures = verify_pointwise(label, actual, expected, PointwiseCriterion{2e-6, 2e-5});
    const double expected_mass = static_cast<double>(config.n_query_tokens) * config.n_layers / config.n_layers_total * replays;
    const double actual_mass = std::accumulate(actual.begin(), actual.end(), 0.0) - 0.25 * config.n_blocks;
    if (std::abs(actual_mass - expected_mass) > 2e-5 * std::max(1.0, expected_mass)) { ++failures; }
    if (!std::equal(all_actual.begin(), all_actual.begin() + 16, output.begin()) ||
        !std::equal(all_actual.end() - 16, all_actual.end(), output.end() - 16)) { ++failures; }
    failures += readonly(label + " query", q_device, query) + readonly(label + " index", k_device, keys) +
                readonly(label + " counts", count_device, counts);
    return failures;
}
} // namespace

int main() {
    try {
        int failures = raw_copy() + mean_accumulate();
        for (const auto type : {ops::KvMemRetrieveDtype::BF16, ops::KvMemRetrieveDtype::F16,
                               ops::KvMemRetrieveDtype::F32}) {
            for (int variant = 0; variant < 4; ++variant) { failures += retrieve(type, variant); }
        }
        std::printf("KVMem original raw-copy/mean-K/retrieval oracle: %s\n", failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "KVMem oracle: %s\n", error.what());
        return 1;
    }
}
