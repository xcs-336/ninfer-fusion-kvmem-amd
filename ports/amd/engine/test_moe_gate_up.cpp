#include "moe_gate_up.h"
#include "ops/op_tester.h"

#include <hip/hip_runtime.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace flash = ninfer::models::qwen3_8_flash_next::detail;

namespace {
constexpr int K = 2560, Pairs = 640, Rows = 2 * Pairs, Experts = 4;
constexpr int Groups = K / 16, CodeStride = Rows * K / 2, ScaleStride = Rows * Groups;
constexpr std::size_t Guard = 32;
constexpr std::uint16_t Canary = 0x5A5A;
// Same BF16 reduction criterion used for the directly qualified attention
// contraction. Inputs here begin at the represented NVFP4/FP8 boundary; their
// generation from BF16 is deliberately outside this phase's mathematical Op.
constexpr ReductionCriterion Criterion{2.8e-3, 1e-3, kBf16GrossRelativeFloor};

unsigned hash(unsigned value) {
    value ^= value >> 16;
    value *= 0x7FEB352DU;
    value ^= value >> 15;
    value *= 0x846CA68BU;
    return value ^ (value >> 16);
}
double fp4(unsigned bits) {
    constexpr double values[] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (bits & 8U) != 0 ? -values[bits & 7U] : values[bits & 7U];
}
double fp8(unsigned bits) {
    const int exponent = (bits >> 3) & 15, fraction = bits & 7;
    const double magnitude = exponent == 0 ? std::ldexp(double(fraction), -9)
        : std::ldexp(1.0 + double(fraction) / 8.0, exponent - 7);
    return (bits & 128U) != 0 ? -magnitude : magnitude;
}
std::size_t scale_offset(int row, int group) {
    // Independent decode of the published K16M128x4 stored plane: a 512-byte
    // tile holds 128 rows and four groups; rows 32 apart share a 16-byte line.
    return (std::size_t(row / 128) * (Groups / 4) + group / 4) * 512
        + (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
}
struct Stream {
    hipStream_t value{};
    Stream() { cuda_check(hipStreamCreate(&value), "create stream"); }
    ~Stream() { if (value != nullptr) { (void)hipStreamDestroy(value); } }
};
template <class T>
int readonly(const char* label, const DeviceBuffer& device, const std::vector<T>& original) {
    const auto actual = from_device<T>(device, original.size());
    if (std::memcmp(actual.data(), original.data(), original.size() * sizeof(T)) == 0) { return 0; }
    std::fprintf(stderr, "%s immutable input changed\n", label);
    return 1;
}

struct Bank {
    std::vector<std::uint8_t> codes, scales;
    std::vector<float> divisors, decoded;
    DeviceBuffer device_codes, device_scales, device_divisors;
    Bank() : codes(Experts * CodeStride), scales(Experts * ScaleStride),
             divisors{1.0F, 2.0F, 4.0F, 8.0F}, decoded(std::size_t(Experts) * Rows * K) {
        for (std::size_t index = 0; index < codes.size(); ++index) { codes[index] = static_cast<std::uint8_t>(hash(unsigned(index) + 117)); }
        for (std::size_t index = 0; index < scales.size(); ++index) {
            scales[index] = static_cast<std::uint8_t>(8 + hash(unsigned(index) + 919) % 32);
        }
        for (int expert = 0; expert < Experts; ++expert) {
            for (int row = 0; row < Rows; ++row) {
                for (int column = 0; column < K; ++column) {
                    const auto packed = codes[expert * CodeStride + row * (K / 2) + column / 2];
                    const double value = fp4((packed >> (4 * (column & 1))) & 15U)
                        * fp8(scales[expert * ScaleStride + scale_offset(row, column / 16)]);
                    decoded[(std::size_t(expert) * Rows + row) * K + column] = static_cast<float>(value);
                }
            }
        }
        device_codes = to_device(codes);
        device_scales = to_device(scales);
        device_divisors = to_device(divisors);
    }
    flash::MoeWeights weights() const {
        flash::MoeWeights result{};
        result.expert_gate_up = {static_cast<const std::byte*>(device_codes.p),
            static_cast<const std::byte*>(device_scales.p), static_cast<const float*>(device_divisors.p),
            Experts, Rows, K, CodeStride, ScaleStride, false};
        return result;
    }
};

int run(const Bank& bank, int tokens) {
    std::vector<std::uint8_t> codes(tokens * K / 2), scales(tokens * Groups);
    std::vector<double> input(tokens * K);
    for (std::size_t index = 0; index < codes.size(); ++index) { codes[index] = static_cast<std::uint8_t>(hash(unsigned(index) + 771)); }
    for (std::size_t index = 0; index < scales.size(); ++index) { scales[index] = static_cast<std::uint8_t>(40 + hash(unsigned(index) + 139) % 16); }
    for (int token = 0; token < tokens; ++token) {
        for (int column = 0; column < K; ++column) {
            input[token * K + column] = fp4((codes[token * (K / 2) + column / 2] >> (4 * (column & 1))) & 15U)
                * fp8(scales[token * Groups + column / 16]);
        }
    }
    std::vector<int> counts(512, 0), offsets(513, 0), active(512, -1), total{3};
    counts[0] = tokens;
    counts[2] = std::max(1, tokens - 2);
    counts[3] = std::max(1, tokens / 2 + 1);
    active[0] = 3; active[1] = 0; active[2] = 2;
    for (int expert = 0; expert < 512; ++expert) { offsets[expert + 1] = offsets[expert] + counts[expert]; }
    std::vector<int> token_map(offsets.back()), path_map(offsets.back());
    const auto payload = std::size_t(tokens) * 11 * Pairs;
    std::vector<bool> written(payload, false);
    std::vector<double> expected;
    std::vector<std::size_t> output_indices;
    for (int expert : {0, 2, 3}) {
        const int path = expert == 0 ? 0 : expert == 2 ? 3 : 9;
        for (int item = 0; item < counts[expert]; ++item) {
            const int token = (item * 5 + expert) % tokens;
            token_map[offsets[expert] + item] = token;
            path_map[offsets[expert] + item] = path;
            for (int pair = 0; pair < Pairs; ++pair) {
                const float* gate_row = bank.decoded.data() + (std::size_t(expert) * Rows + pair) * K;
                const float* up_row = gate_row + Pairs * K;
                double gate = 0.0, up = 0.0;
                // Naive FP64 dot products across the entire represented K.
                // No production codec, MMA, tile order or scale loader is used.
                for (int column = 0; column < K; ++column) {
                    gate += input[token * K + column] * gate_row[column];
                    up += input[token * K + column] * up_row[column];
                }
                gate /= bank.divisors[expert];
                up /= bank.divisors[expert];
                const double silu = gate >= 0.0 ? gate / (1.0 + std::exp(-gate))
                    : gate * std::exp(gate) / (1.0 + std::exp(gate));
                const auto index = (std::size_t(token) * 11 + path) * Pairs + pair;
                if (written[index]) { throw std::runtime_error("oracle fixture has duplicate token/path ownership"); }
                written[index] = true;
                output_indices.push_back(index);
                expected.push_back(silu * up);
            }
        }
    }
    auto d_codes = to_device(codes), d_scales = to_device(scales), d_counts = to_device(counts);
    auto d_offsets = to_device(offsets), d_active = to_device(active), d_total = to_device(total);
    auto d_tokens = to_device(token_map), d_paths = to_device(path_map);
    std::vector<std::uint16_t> initial(payload + 2 * Guard, Canary);
    auto d_output = to_device(initial);
    flash::FlashNextMoeWorkspace workspace{};
    workspace.act_codes = Tensor(d_codes.p, DType::U8, {K / 2, tokens});
    workspace.act_scales = Tensor(d_scales.p, DType::U8, {Groups, tokens});
    workspace.expert_counts = Tensor(d_counts.p, DType::I32, {512});
    workspace.expert_offsets = Tensor(d_offsets.p, DType::I32, {513});
    workspace.active_expert_ids = Tensor(d_active.p, DType::I32, {512});
    workspace.active_count = Tensor(d_total.p, DType::I32, {1});
    workspace.grouped_tokens = Tensor(d_tokens.p, DType::I32, {int(token_map.size())});
    workspace.grouped_paths = Tensor(d_paths.p, DType::I32, {int(path_map.size())});
    workspace.activations = Tensor(static_cast<std::uint16_t*>(d_output.p) + Guard,
                                  DType::BF16, {Pairs, 11, tokens});
    const auto weights = bank.weights();
    Stream stream;
    int failures = 0;
    for (const auto schedule : {flash::NativeGateUpSchedule::Legacy, flash::NativeGateUpSchedule::Async,
                                flash::NativeGateUpSchedule::V2}) {
        d_output.copy_from_host(initial.data(), initial.size() * sizeof(std::uint16_t));
        // This is the same actual host phase called by the full production MoE
        // launcher, not a test kernel or a lower-precision reference route.
        flash::native_flash_next_moe_gate_up_launch(schedule, weights, workspace, stream.value);
        cuda_check(hipStreamSynchronize(stream.value), "production gate/up completion");
        const auto result = from_device<std::uint16_t>(d_output, initial.size());
        std::vector<double> actual;
        actual.reserve(expected.size());
        for (const auto index : output_indices) { actual.push_back(bf16_to_f32(result[Guard + index])); }
        const std::string label = "Flash-Next packed NVFP4 gate/up T=" + std::to_string(tokens) +
            " schedule=" + std::to_string(int(schedule));
        failures += verify_reduction(label, actual, expected, Criterion);
        for (std::size_t index = 0; index < result.size(); ++index) {
            const bool live = index >= Guard && index < Guard + payload && written[index - Guard];
            if (!live && result[index] != Canary) {
                std::fprintf(stderr, "%s guard/unselected path changed at %zu\n", label.c_str(), index);
                ++failures;
                break;
            }
        }
    }
    failures += readonly("activation codes", d_codes, codes) + readonly("activation scales", d_scales, scales)
        + readonly("expert counts", d_counts, counts) + readonly("expert offsets", d_offsets, offsets)
        + readonly("active expert IDs", d_active, active) + readonly("active count", d_total, total)
        + readonly("grouped tokens", d_tokens, token_map) + readonly("grouped paths", d_paths, path_map);
    return failures;
}
} // namespace

int main() {
    try {
        Bank bank;
        int failures = 0;
        for (const int tokens : {1, 9, 17}) { failures += run(bank, tokens); }
        failures += readonly("stored weight codes", bank.device_codes, bank.codes)
            + readonly("stored weight scales", bank.device_scales, bank.scales)
            + readonly("expert divisors", bank.device_divisors, bank.divisors);
        std::printf("Flash-Next production gate/up packed FP64 oracle: %s (3 schedules, experts, tails, exact guards)\n",
                    failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Flash-Next gate/up oracle: %s\n", error.what());
        return 1;
    }
}
