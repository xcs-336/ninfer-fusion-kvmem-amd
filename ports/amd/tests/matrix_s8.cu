#include "ops/common/mma_s8_gfx12.cuh"
#include "ops/op_tester.h"
#include "core/device.h"
#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
struct MatrixCase { std::int8_t a[16 * 32], b[32 * 8]; std::int32_t c[16 * 8]; };
__device__ unsigned pack_a(const std::int8_t* p, int row, int k) {
    unsigned result = 0;
    for (int byte = 0; byte < 4; ++byte)
        result |= unsigned(static_cast<unsigned char>(p[row * 32 + k + byte])) << (8 * byte);
    return result;
}
__device__ unsigned pack_b(const std::int8_t* p, int k, int col) {
    unsigned result = 0;
    for (int byte = 0; byte < 4; ++byte)
        result |= unsigned(static_cast<unsigned char>(p[(k + byte) * 8 + col])) << (8 * byte);
    return result;
}
__global__ void products(const MatrixCase* cases, std::int32_t* result, int count, int repetitions) {
    const int lane = threadIdx.x & 31;
    const int index = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (index >= count) return;
    const auto& sample = cases[index];
    const int row = lane / 4, k = 4 * (lane % 4), col = 2 * (lane % 4);
    int c0 = sample.c[row * 8 + col], c1 = sample.c[row * 8 + col + 1];
    int c2 = sample.c[(row + 8) * 8 + col], c3 = sample.c[(row + 8) * 8 + col + 1];
    const unsigned a0 = pack_a(sample.a, row, k), a1 = pack_a(sample.a, row + 8, k);
    const unsigned a2 = pack_a(sample.a, row, k + 16), a3 = pack_a(sample.a, row + 8, k + 16);
    const unsigned b0 = pack_b(sample.b, k, row), b1 = pack_b(sample.b, k + 16, row);
    for (int repeat = 0; repeat < repetitions; ++repeat)
        ops::mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    auto* output = result + index * 128;
    output[row * 8 + col] = c0; output[row * 8 + col + 1] = c1;
    output[(row + 8) * 8 + col] = c2; output[(row + 8) * 8 + col + 1] = c3;
}
int main() {
    try {
        constexpr int basis_count = 16 * 8 * 32;
        std::vector<MatrixCase> cases(basis_count + 80);
        for (int index = 0; index < basis_count; ++index) {
            const int k = index % 32, col = (index / 32) % 8, row = index / 256;
            cases[index].a[row * 32 + k] = (index & 1) ? -128 : 127;
            cases[index].b[k * 8 + col] = (index & 2) ? -127 : 126;
            for (int i = 0; i < 128; ++i) cases[index].c[i] = i * 41 - 97;
        }
        std::mt19937 random(0x319874U);
        for (int index = basis_count; index < int(cases.size()); ++index) {
            for (auto& value : cases[index].a) value = std::bit_cast<std::int8_t>(std::uint8_t(random()));
            for (auto& value : cases[index].b) value = std::bit_cast<std::int8_t>(std::uint8_t(random()));
            for (int i = 0; i < 128; ++i) {
                if (index >= basis_count + 64) {
                    cases[index].c[i] = (i & 1) ? std::numeric_limits<int>::min() + i
                                               : std::numeric_limits<int>::max() - i;
                } else cases[index].c[i] = int(random() % 65537) - 32768;
            }
        }
        const auto bytes = cases.size() * sizeof(MatrixCase);
        DeviceBuffer input(bytes);
        CUDA_CHECK(hipMemcpy(input.p, cases.data(), bytes, hipMemcpyHostToDevice));
        std::size_t failures = 0, checked = 0;
        for (int threads : {32, 128}) for (int repetitions : {1, 3}) {
            GuardedDeviceBuffer output(cases.size() * 128 * sizeof(std::int32_t));
            output.fill(0x7f);
            const int per_block = threads / 32;
            products<<<(cases.size() + per_block - 1) / per_block, threads>>>(
                static_cast<const MatrixCase*>(input.p), static_cast<std::int32_t*>(output.data()),
                int(cases.size()), repetitions);
            CUDA_CHECK(hipGetLastError());
            CUDA_CHECK(hipDeviceSynchronize());
            std::vector<std::int32_t> actual(cases.size() * 128);
            CUDA_CHECK(hipMemcpy(actual.data(), output.data(), actual.size() * 4, hipMemcpyDeviceToHost));
            for (std::size_t index = 0; index < cases.size(); ++index)
                for (int row = 0; row < 16; ++row) for (int col = 0; col < 8; ++col) {
                    std::int64_t expected = cases[index].c[row * 8 + col];
                    for (int k = 0; k < 32; ++k)
                        expected += std::int64_t(repetitions) * cases[index].a[row * 32 + k] * cases[index].b[k * 8 + col];
                    const auto wrapped = std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(expected));
                    if (actual[index * 128 + row * 8 + col] != wrapped) {
                        if (failures < 6) std::cerr << "mismatch case=" << index << " row=" << row
                            << " col=" << col << " expected=" << wrapped << " got=" << actual[index * 128 + row * 8 + col] << '\n';
                        ++failures;
                    }
                    ++checked;
                }
            failures += output.verify_guards("native INT8 matrix");
        }
        std::vector<MatrixCase> after(cases.size());
        CUDA_CHECK(hipMemcpy(after.data(), input.p, bytes, hipMemcpyDeviceToHost));
        if (std::memcmp(after.data(), cases.data(), bytes)) ++failures;
        std::cout << (failures ? "FAIL" : "PASS") << " signed INT8 WMMA: " << cases.size()
                  << " matrices, four configurations, " << checked << " exact outputs; wrap-overflow included\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
