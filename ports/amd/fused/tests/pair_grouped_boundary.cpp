// Independent, bounded qualification of the actual C256 pair candidate. This
// complements the unchanged full dense public-Op suite; it never replaces it.
#include "core/weight.h"
#include "ninfer/ops/weight_input.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear_pair/q8/q8_pair_plan.h"
#include "ops/op_tester.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
constexpr int K = 2048, N = 1024, ParentRows = 6144, FirstRow = 4096, MaximumT = 256;
constexpr int Nonzeros = 16;
constexpr std::size_t CodeBytes = std::size_t(ParentRows) * K;
constexpr std::size_t ScaleBytes = std::size_t(ParentRows) * (K / 32) * 2;

std::uint16_t bf16_rne(double value) {
    // Every pre-output sum is an integer multiple of 2^-10 with magnitude <256:
    // FP64 and FP32 both represent it exactly. Only BF16 output rounding remains.
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(static_cast<float>(value));
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

double stored_weight(const std::vector<std::uint8_t>& payload, int row, int k) {
    const unsigned raw = payload[std::size_t(row) * K + k];
    const int code = raw < 128 ? static_cast<int>(raw) : static_cast<int>(raw) - 256;
    const std::size_t offset = CodeBytes + (std::size_t(row) * (K / 32) + k / 32) * 2;
    const unsigned half = unsigned(payload[offset]) | (unsigned(payload[offset + 1]) << 8);
    // Test scales are positive binary16 powers of two (zero significand).
    if ((half & 0x83ffU) != 0 || (half >> 10) == 0 || (half >> 10) >= 31)
        throw std::runtime_error("invalid independent dyadic test scale");
    return std::ldexp(static_cast<double>(code), static_cast<int>(half >> 10) - 15);
}

int verify_unchanged(const test::GuardedDeviceBuffer& device,
                     std::span<const std::uint8_t> expected, const char* label) {
    std::vector<std::uint8_t> actual(expected.size());
    device.copy_to_host(actual.data(), actual.size());
    if (std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) return 0;
    std::cerr << label << " input bytes changed\n";
    return 1;
}
}

int main() {
    using namespace ninfer;
    if (test::cuda_unavailable()) return 77;
    try {
        std::vector<std::uint8_t> payload(CodeBytes + ScaleBytes);
        for (int row = 0; row < ParentRows; ++row) {
            for (int k = 0; k < K; ++k) {
                const int code = (row * 19 + k * 29 + (k / 32) * 7) % 255 - 127;
                payload[std::size_t(row) * K + k] = static_cast<std::uint8_t>(code);
            }
            for (int group = 0; group < K / 32; ++group) {
                const unsigned half = unsigned(7 + ((row + group) & 3)) << 10; // 2^-8..2^-5
                const std::size_t offset = CodeBytes + (std::size_t(row) * (K / 32) + group) * 2;
                payload[offset] = static_cast<std::uint8_t>(half);
                payload[offset + 1] = static_cast<std::uint8_t>(half >> 8);
            }
        }
        std::vector<std::uint16_t> activation(std::size_t(K) * MaximumT, 0);
        std::array<std::array<int, Nonzeros>, MaximumT> positions{};
        std::array<std::array<double, Nonzeros>, MaximumT> values{};
        for (int token = 0; token < MaximumT; ++token)
            for (int item = 0; item < Nonzeros; ++item) {
                const int k = (token * 17 + item * 127) % K;
                const double value = std::ldexp(((token >> item) & 1) ? -1.0 : 1.0,
                                               (token + item) % 5 - 2);
                positions[token][item] = k;
                values[token][item] = value;
                activation[std::size_t(token) * K + k] = bf16_rne(value);
            }
        test::GuardedDeviceBuffer dx(activation.size() * sizeof(std::uint16_t));
        test::GuardedDeviceBuffer dw(payload.size());
        dx.copy_from_host(activation.data(), dx.bytes());
        dw.copy_from_host(payload.data(), payload.size());
        const std::array<std::uint64_t, 2> parent_shape{ParentRows, K};
        const WeightParent parent{weight_geometry(QType::Q8_G32_FP16, QuantLayout::RowSplit,
                                                  parent_shape),
                                  static_cast<const std::byte*>(dw.data())};
        const auto region = [&](std::uint64_t row) {
            return WeightView{{N, K}, {{&parent, row * K, (row + N) * K}}};
        };
        const Weight first_weight = ops::prepare_linear_weight({region(FirstRow)}).weight;
        const Weight second_weight = ops::prepare_linear_weight({region(FirstRow + N)}).weight;
        int failures = 0;
        std::uint64_t checked = 0;
        using Route = ops::detail::LinearRouteTable;
        for (const Route route : {Route::Legacy, Route::Unified}) {
            ops::detail::force_linear_route_table(route);
            for (const int tokens : {129, 255, 256}) {
                test::GuardedDeviceBuffer first(std::size_t(N) * tokens * 2);
                test::GuardedDeviceBuffer second(std::size_t(N) * tokens * 2);
                Tensor x(dx.data(), DType::BF16, {K, tokens});
                Tensor a(first.data(), DType::BF16, {N, tokens});
                Tensor b(second.data(), DType::BF16, {N, tokens});
                for (int repeat = 0; repeat < 2; ++repeat) {
                    first.fill(0xff);
                    second.fill(0xff);
                    ops::detail::q8_pair_execute_schedule(
                        ops::detail::Q8PairScheduleId::DualSplitKMediumC256,
                        x, first_weight, second_weight, a, b, nullptr);
                    test::cuda_synchronize();
                    for (int output = 0; output < 2; ++output) {
                        const auto& buffer = output == 0 ? first : second;
                        std::vector<std::uint16_t> actual(std::size_t(N) * tokens);
                        buffer.copy_to_host(actual.data(), actual.size() * 2);
                        for (int token = 0; token < tokens; ++token)
                            for (int row = 0; row < N; ++row) {
                                double sum = 0;
                                for (int item = 0; item < Nonzeros; ++item)
                                    sum += stored_weight(payload, FirstRow + output * N + row,
                                                         positions[token][item]) * values[token][item];
                                if (static_cast<double>(static_cast<float>(sum)) != sum)
                                    throw std::runtime_error("dyadic FP64 oracle was not exactly FP32 representable");
                                const auto expected = bf16_rne(sum);
                                const auto got = actual[std::size_t(token) * N + row];
                                if (got != expected) {
                                    if (failures < 12)
                                        std::cerr << "route=" << int(route) << " T=" << tokens
                                            << " repeat=" << repeat << " output=" << output
                                            << " token=" << token << " row=" << row
                                            << " bits=" << got << " expected=" << expected << '\n';
                                    ++failures;
                                }
                                ++checked;
                            }
                    }
                    failures += first.verify_guards("pair C256 first");
                    failures += second.verify_guards("pair C256 second");
                }
            }
        }
        ops::detail::force_linear_route_table(std::nullopt);
        failures += dx.verify_guards("pair C256 input");
        failures += dw.verify_guards("pair C256 weights");
        failures += verify_unchanged(dx,
            {reinterpret_cast<const std::uint8_t*>(activation.data()), activation.size() * 2}, "activation");
        failures += verify_unchanged(dw, payload, "weights");
        std::cout << (failures == 0 ? "PASS" : "FAIL") << " pair C256 exact FP64 boundary outputs="
                  << checked << " failures=" << failures << '\n';
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        ops::detail::force_linear_route_table(std::nullopt);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
