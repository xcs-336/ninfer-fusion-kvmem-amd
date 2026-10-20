// Qualify the actual production GEMM body, not a test-owned matrix kernel.
#include "ops/linear/bf16/bf16_gemm_mma.cuh"
#include "ops/op_tester.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::ops;
using namespace ninfer::ops::detail;
template <int N, int K> struct Geometry {
    static constexpr int kOutputRows = N;
    static constexpr int kInputRows = K;
};
using Serial = Bf16MmaSchedule<32, 16, 64, 16, 16, 2, 1, Cache::cg, Cache::ca,
    Bf16MmaFragmentPipeline::Serial, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Plain>;
using PingPong = Bf16MmaSchedule<32, 16, 64, 16, 16, 2, 1, Cache::cg, Cache::ca,
    Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::RowFast, Bf16MmaSwizzle::Xor64>;

template <class G, class S>
void launch(const DeviceBuffer& x, const DeviceBuffer& w, void* out, int tokens, hipStream_t stream) {
    const dim3 grid((G::kOutputRows / S::kBlockRows) * ((tokens + S::kBlockCols - 1) / S::kBlockCols));
    const Bf16MmaContiguousOutput destination{static_cast<__hip_bfloat16*>(out), G::kOutputRows};
    const auto* input = static_cast<const __hip_bfloat16*>(x.p);
    const auto* weight = static_cast<const __hip_bfloat16*>(w.p);
    if (tokens % S::kBlockCols == 0) {
        bf16_gemm_mma_kernel<G, S, true><<<grid, S::kThreads, S::kSharedBytes, stream>>>(input, weight, destination, tokens);
    } else {
        bf16_gemm_mma_kernel<G, S, false><<<grid, S::kThreads, S::kSharedBytes, stream>>>(input, weight, destination, tokens);
    }
    CUDA_CHECK(hipGetLastError());
}
template <class G>
std::vector<double> oracle(const std::vector<float>& x, const std::vector<float>& weights,
                           int tokens, const std::vector<int>& rows) {
    std::vector<double> result(static_cast<std::size_t>(tokens) * rows.size());
    for (int t = 0; t < tokens; ++t) {
        for (std::size_t r = 0; r < rows.size(); ++r) {
            double sum = 0;
            const float* w = weights.data() + static_cast<std::size_t>(rows[r]) * G::kInputRows;
            const float* a = x.data() + static_cast<std::size_t>(t) * G::kInputRows;
            for (int k = 0; k < G::kInputRows; ++k) { sum += static_cast<double>(a[k]) * w[k]; }
            result[static_cast<std::size_t>(t) * rows.size() + r] = sum;
        }
    }
    return result;
}
template <class G, class S>
int qualify(const DeviceBuffer& x, const DeviceBuffer& w, GuardedDeviceBuffer& output, int tokens,
             const std::vector<int>& rows, const std::vector<double>& reference, DeviceContext& device,
             bool graph, const char* schedule) {
    output.fill(0xff);
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    if (graph) {
        definition.capture(device.stream, [&] { launch<G, S>(x, w, output.data(), tokens, device.stream); });
        executable.instantiate(definition);
        executable.launch(device.stream);
    } else { launch<G, S>(x, w, output.data(), tokens, device.stream); }
    CUDA_CHECK(hipStreamSynchronize(device.stream));
    const auto actual_all = from_device_bf16(output.data(), static_cast<std::size_t>(tokens) * G::kOutputRows);
    for (double value : actual_all) {
        if (!std::isfinite(value)) { throw std::runtime_error("unwritten or nonfinite production GEMM output"); }
    }
    std::vector<double> actual(reference.size());
    for (int t = 0; t < tokens; ++t) {
        for (std::size_t r = 0; r < rows.size(); ++r) {
            actual[static_cast<std::size_t>(t) * rows.size() + r] = actual_all[static_cast<std::size_t>(t) * G::kOutputRows + rows[r]];
        }
    }
    // Same A16 criterion as tests/ops/linear/linear_test_common.cpp. The FP64
    // reference is evaluated before the observable final BF16 storage cast.
    const ReductionCriterion criterion{1.0 / 256, 1.0 / 256, 2.0 / 256};
    const auto label = std::string("production BF16 GEMM N=") + std::to_string(G::kOutputRows) +
        " K=" + std::to_string(G::kInputRows) + " T=" + std::to_string(tokens) + " " + schedule + (graph ? " graph" : " eager");
    return verify_reduction(label, actual, reference, criterion) + output.verify_guards(label);
}
template <class G>
int run(const std::vector<int>& counts, bool both_schedules) {
    std::vector<float> weights(static_cast<std::size_t>(G::kOutputRows) * G::kInputRows);
    fill_uniform(weights, 0x789123U, -2, 2);
    round_to_bf16(weights);
    auto device_weight = to_device_bf16(weights);
    std::vector<int> rows;
    if constexpr (G::kOutputRows <= 256) {
        for (int i = 0; i < G::kOutputRows; ++i) { rows.push_back(i); }
    } else {
        for (int i = 0; i < 16; ++i) { rows.push_back(i * (G::kOutputRows - 1) / 15); }
    }
    DeviceContext device;
    int failures = 0;
    for (int tokens : counts) {
        std::vector<float> input(static_cast<std::size_t>(tokens) * G::kInputRows);
        fill_uniform(input, 0x83521U + tokens, -2, 2);
        round_to_bf16(input);
        auto device_input = to_device_bf16(input);
        GuardedDeviceBuffer output(static_cast<std::size_t>(tokens) * G::kOutputRows * sizeof(std::uint16_t));
        const auto reference = oracle<G>(input, weights, tokens, rows);
        failures += qualify<G, PingPong>(device_input, device_weight, output, tokens, rows, reference, device, false, "pingpong-xor");
        if (both_schedules) {
            failures += qualify<G, Serial>(device_input, device_weight, output, tokens, rows, reference, device, false, "serial-plain");
        }
        if (tokens == 17 || tokens == 32) {
            failures += qualify<G, PingPong>(device_input, device_weight, output, tokens, rows, reference, device, true, "pingpong-xor");
        }
        const auto after = from_device_bf16(device_input, input.size());
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (after[i] != input[i]) { throw std::runtime_error("production GEMM modified activation input"); }
        }
    }
    const auto after = from_device_bf16(device_weight, weights.size());
    for (std::size_t i = 0; i < weights.size(); ++i) {
        if (after[i] != weights[i]) { throw std::runtime_error("production GEMM modified weight input"); }
    }
    return failures;
}
int main() {
    try {
        int failures = run<Geometry<256, 5120>>({1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 64, 65, 129}, true);
        failures += run<Geometry<5120, 6144>>({1, 17, 32}, false);
        std::cout << (failures ? "FAIL" : "PASS") << " production BF16 GEMM; full N256 output oracle, 16 sampled rows for N5120; all outputs checked finite\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
