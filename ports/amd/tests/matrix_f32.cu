// Exercise the production fragment adapter against a row-major FP64 oracle.
#include "ops/common/mma.cuh"
#include "ops/op_tester.h"
#include "core/device.h"
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
struct MatrixCase { float a[16 * 8], b[8 * 8], c[16 * 8]; };

__global__ void fragment_product(const MatrixCase* cases, float* result, int count,
                                  int repetitions, bool bits) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned index = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (index >= count) return; // Whole waves leave together.
    const MatrixCase& sample = cases[index];
    const unsigned row = lane / 4, k = lane % 4, col = 2 * k;
    float c0 = sample.c[row * 8 + col], c1 = sample.c[row * 8 + col + 1];
    float c2 = sample.c[(row + 8) * 8 + col], c3 = sample.c[(row + 8) * 8 + col + 1];
    const float a0 = sample.a[row * 8 + k], a1 = sample.a[(row + 8) * 8 + k];
    const float a2 = sample.a[row * 8 + k + 4], a3 = sample.a[(row + 8) * 8 + k + 4];
    const float b0 = sample.b[k * 8 + row], b1 = sample.b[(k + 4) * 8 + row];
    for (int repeat = 0; repeat < repetitions; ++repeat) {
        if (bits) {
            ops::mma_tf32_bits(c0, c1, c2, c3, __builtin_bit_cast(unsigned, a0),
                __builtin_bit_cast(unsigned, a1), __builtin_bit_cast(unsigned, a2),
                __builtin_bit_cast(unsigned, a3), __builtin_bit_cast(unsigned, b0),
                __builtin_bit_cast(unsigned, b1));
        } else { ops::mma_tf32(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1); }
    }
    float* output = result + index * 128;
    output[row * 8 + col] = c0; output[row * 8 + col + 1] = c1;
    output[(row + 8) * 8 + col] = c2; output[(row + 8) * 8 + col + 1] = c3;
}

int main() {
    try {
        constexpr int exact_count = 16 * 8 * 8;
        std::vector<MatrixCase> cases(exact_count + 64);
        for (int index = 0; index < exact_count; ++index) {
            const int k = index % 8, col = (index / 8) % 8, row = index / 64;
            cases[index].a[row * 8 + k] = (index & 1) ? -2 : 2;
            cases[index].b[k * 8 + col] = (index & 2) ? -3 : 3;
            for (int i = 0; i < 128; ++i) cases[index].c[i] = float((i % 13) - 6);
        }
        std::mt19937 random(0x594832U);
        for (int index = exact_count; index < int(cases.size()); ++index) {
            auto finite = [&]() {
                // Deliberately retain low FP32 mantissa bits absent in TF32/BF16.
                const unsigned value = (random() & 0x807fffffu) |
                    (unsigned(115 + random() % 25) << 23);
                return std::bit_cast<float>(value);
            };
            for (float& value : cases[index].a) value = finite();
            for (float& value : cases[index].b) value = finite();
            for (float& value : cases[index].c) value = finite();
        }
        const std::size_t bytes = cases.size() * sizeof(MatrixCase);
        DeviceBuffer input(bytes);
        CUDA_CHECK(hipMemcpy(input.p, cases.data(), bytes, hipMemcpyHostToDevice));
        int failures = 0;
        for (int threads : {32, 128}) for (int repetitions : {1, 3}) for (bool bits : {false, true}) {
            GuardedDeviceBuffer output(cases.size() * 128 * sizeof(float));
            output.fill(0xff);
            const int per_block = threads / 32;
            fragment_product<<<(cases.size() + per_block - 1) / per_block, threads>>>(
                static_cast<const MatrixCase*>(input.p), static_cast<float*>(output.data()),
                int(cases.size()), repetitions, bits);
            CUDA_CHECK(hipGetLastError());
            CUDA_CHECK(hipDeviceSynchronize());
            std::vector<float> actual(cases.size() * 128);
            CUDA_CHECK(hipMemcpy(actual.data(), output.data(), actual.size() * sizeof(float), hipMemcpyDeviceToHost));
            std::vector<double> numerical_actual, reference;
            for (std::size_t index = 0; index < cases.size(); ++index) {
                const auto& sample = cases[index];
                for (int row = 0; row < 16; ++row) for (int col = 0; col < 8; ++col) {
                    double expected = sample.c[row * 8 + col];
                    for (int repeat = 0; repeat < repetitions; ++repeat)
                        for (int k = 0; k < 8; ++k)
                            expected += double(sample.a[row * 8 + k]) * double(sample.b[k * 8 + col]);
                    const float got = actual[index * 128 + row * 8 + col];
                    if (index < exact_count) {
                        if (double(got) != expected) ++failures;
                    } else { numerical_actual.push_back(got); reference.push_back(expected); }
                }
            }
            failures += verify_reduction("native FP32 fragment FMA", numerical_actual, reference,
                                          ReductionCriterion{2e-6, 1e-6, 4e-6});
            failures += output.verify_guards("FP32 fragment FMA");
        }
        std::vector<MatrixCase> after(cases.size());
        CUDA_CHECK(hipMemcpy(after.data(), input.p, bytes, hipMemcpyDeviceToHost));
        if (std::memcmp(after.data(), cases.data(), bytes) != 0) ++failures;
        std::cout << (failures ? "FAIL" : "PASS") << " native FP32 fragment route: "
                  << cases.size() << " matrices, eight launch/repetition/API combinations, "
                  << cases.size() * 128 * 8 << " outputs; full FP32 operands, no TF32 throughput claim\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
