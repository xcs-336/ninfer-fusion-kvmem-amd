// Independent FP64 qualification of native OCP E4M3 m16n8k32 fragments.
// Compile with AMD clang -x hip, gfx1201, wave32. GPU execution requires the
// shared benchmark lock; --host-oracle-only performs no HIP API call.
#include "../ops/mma_fp8_gfx12.cuh"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Byte = std::uint8_t;
struct MatrixCase { Byte a[512]{}, b[256]{}; float c[128]{}; };
constexpr std::size_t kGuard = 256;
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
struct Buffer {
    void* base = nullptr;
    explicit Buffer(std::size_t bytes) { check(hipMalloc(&base, bytes)); }
    ~Buffer() { if (base) (void)hipFree(base); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
double decode(Byte bits) {
    const unsigned exponent = (bits >> 3) & 15u, mantissa = bits & 7u;
    if (exponent == 15 && mantissa == 7) return std::numeric_limits<double>::quiet_NaN();
    const double value = exponent ? std::ldexp(double(8 + mantissa), int(exponent) - 10)
                                  : std::ldexp(double(mantissa), -9);
    return (bits & 128) ? -value : value;
}
std::vector<Byte> finite_encodings() {
    std::vector<Byte> result;
    for (unsigned bits = 0; bits < 256; ++bits)
        if (std::isfinite(decode(Byte(bits)))) result.push_back(Byte(bits));
    return result;
}
__device__ unsigned pack_a(const Byte* p, unsigned row, unsigned k) {
    unsigned packed = 0;
#pragma unroll
    for (unsigned byte = 0; byte < 4; ++byte)
        packed |= unsigned(p[row * 32 + k + byte]) << (8 * byte);
    return packed;
}
__device__ unsigned pack_b(const Byte* p, unsigned k, unsigned column) {
    unsigned packed = 0;
#pragma unroll
    for (unsigned byte = 0; byte < 4; ++byte)
        packed |= unsigned(p[(k + byte) * 8 + column]) << (8 * byte);
    return packed;
}
__global__ void products(const MatrixCase* input, float* output, unsigned count, int repetitions) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned index = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (index >= count) return; // The branch is uniform within each wave.
    const auto& s = input[index];
    const unsigned row = lane / 4, k = 4 * (lane % 4), column = 2 * (lane % 4);
    float c0 = s.c[row * 8 + column], c1 = s.c[row * 8 + column + 1];
    float c2 = s.c[(row + 8) * 8 + column], c3 = s.c[(row + 8) * 8 + column + 1];
    const unsigned a0 = pack_a(s.a, row, k), a1 = pack_a(s.a, row + 8, k);
    const unsigned a2 = pack_a(s.a, row, k + 16), a3 = pack_a(s.a, row + 8, k + 16);
    const unsigned b0 = pack_b(s.b, k, row), b1 = pack_b(s.b, k + 16, row);
    for (int rep = 0; rep < repetitions; ++rep)
        ninfer::ops::mma_fp8_e4m3(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    float* destination = output + index * 128;
    destination[row * 8 + column] = c0; destination[row * 8 + column + 1] = c1;
    destination[(row + 8) * 8 + column] = c2; destination[(row + 8) * 8 + column + 1] = c3;
}
int verify(const std::vector<MatrixCase>& cases, const std::vector<float>& actual,
           unsigned exact_count, int repetitions) {
    int failures = 0;
    double error_squared = 0, reference_squared = 0, max_ratio = 0;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& s = cases[i];
        for (unsigned row = 0; row < 16; ++row) for (unsigned column = 0; column < 8; ++column) {
            double expected = s.c[row * 8 + column], sum_abs = std::fabs(expected);
            for (int rep = 0; rep < repetitions; ++rep)
                for (unsigned k = 0; k < 32; ++k) {
                    const double term = decode(s.a[row * 32 + k]) * decode(s.b[k * 8 + column]);
                    expected += term; sum_abs += std::fabs(term);
                }
            const double got = actual[i * 128 + row * 8 + column];
            const double n = 32.0 * repetitions + 1.0, unit = std::ldexp(1.0, -24);
            const double bound = n * unit / (1.0 - n * unit) * sum_abs;
            const double error = std::fabs(got - expected);
            // Exact single-product checks cover every finite OCP E4M3 input
            // pair. Both signed zeros are represented; their output sign is
            // not promised by PTX MMA's unspecified accumulation rounding.
            const bool pass = std::isfinite(got) &&
                (i < exact_count ? got == expected : error <= bound);
            if (i >= exact_count) {
                if (bound) max_ratio = std::max(max_ratio, error / bound);
                error_squared += error * error; reference_squared += expected * expected;
            }
            if (!pass && failures++ < 8)
                std::cerr << "case=" << i << " row=" << row << " col=" << column
                          << " actual=" << got << " expected=" << expected << " bound=" << bound << '\n';
        }
    }
    const double relative_l2 = std::sqrt(error_squared / std::max(reference_squared, 1e-100));
    if (!std::isfinite(relative_l2) || relative_l2 > 2e-6) ++failures;
    std::cout << "OCP FP8 E4M3 WMMA: repetitions=" << repetitions << " relative_l2=" << relative_l2
              << " max_error/bound=" << max_ratio << " failures=" << failures << '\n';
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool host_only = argc == 2 && std::string(argv[1]) == "--host-oracle-only";
        if (argc > 2 || (argc == 2 && !host_only))
            throw std::runtime_error("usage: matrix_fp8 [--host-oracle-only]");
        const auto encodings = finite_encodings();
        if (encodings.size() != 254 || decode(1) != std::ldexp(1.0, -9) ||
            decode(7) != std::ldexp(7.0, -9) || decode(8) != std::ldexp(1.0, -6) ||
            decode(0x38) != 1 || decode(0x40) != 2 || decode(0x44) != 3 ||
            decode(0x7e) != 448 || decode(0xfe) != -448 ||
            !std::signbit(decode(0x80)) || !std::isnan(decode(0x7f)) || !std::isnan(decode(0xff)))
            throw std::runtime_error("independent OCP E4M3 oracle boundary check failed");
        if (host_only) {
            std::cout << "PASS CPU-only OCP E4M3 oracle: 254 finite encodings, signed zeros, subnormals, maximum448, two NaNs\n";
            return 0;
        }
        constexpr unsigned basis_count = 16 * 8 * 32;
        constexpr unsigned a_groups = 16, b_groups = 32;
        constexpr unsigned exact_count = basis_count + a_groups * b_groups + 2;
        std::vector<MatrixCase> cases(exact_count + 96);
        for (unsigned i = 0; i < basis_count; ++i) {
            const unsigned k = i % 32, column = (i / 32) % 8, row = i / 256;
            cases[i].a[row * 32 + k] = (i & 1) ? 0xc0 : 0x40; // +/-2
            cases[i].b[k * 8 + column] = (i & 2) ? 0xc4 : 0x44; // +/-3
            for (unsigned j = 0; j < 128; ++j) cases[i].c[j] = float(int((j * 17) % 31) - 15);
        }
        // Cartesian coverage of all 254*254 finite input encoding pairs.
        // One nonzero K term makes every represented product and up to three
        // accumulations exactly representable in FP32, even for subnormals.
        for (unsigned ag = 0; ag < a_groups; ++ag) for (unsigned bg = 0; bg < b_groups; ++bg) {
            const unsigned tile = ag * b_groups + bg, k = (tile * 11) % 32;
            auto& s = cases[basis_count + tile];
            for (unsigned row = 0; row < 16; ++row)
                if (ag * 16 + row < encodings.size()) s.a[row * 32 + k] = encodings[ag * 16 + row];
            for (unsigned column = 0; column < 8; ++column)
                if (bg * 8 + column < encodings.size()) s.b[k * 8 + column] = encodings[bg * 8 + column];
        }
        for (unsigned tile = exact_count - 2; tile < exact_count; ++tile)
            for (unsigned j = 0; j < 128; ++j) {
                const std::uint32_t bits = (j & 1 ? 0x80000000u : 0) |
                    (tile == exact_count - 2 ? 1u + j : 0x00800000u + j);
                cases[tile].c[j] = std::bit_cast<float>(bits);
            }
        std::mt19937 random(0x8e4a301u);
        for (unsigned i = exact_count; i < cases.size(); ++i) {
            auto finite = [&]() { return encodings[random() % encodings.size()]; };
            for (Byte& v : cases[i].a) v = finite();
            for (Byte& v : cases[i].b) v = finite();
            for (float& v : cases[i].c)
                v = std::bit_cast<float>((random() & 0x807fffffu) | ((100 + random() % 46) << 23));
        }
        const std::size_t input_bytes = cases.size() * sizeof(MatrixCase);
        Buffer input(input_bytes + 2 * kGuard);
        check(hipMemset(input.base, 0xa7, input_bytes + 2 * kGuard));
        auto* device_input = reinterpret_cast<MatrixCase*>(static_cast<unsigned char*>(input.base) + kGuard);
        check(hipMemcpy(device_input, cases.data(), input_bytes, hipMemcpyHostToDevice));
        int failures = 0;
        for (int threads : {32, 128}) for (int repetitions : {1, 3}) {
            const std::size_t bytes = cases.size() * 128 * sizeof(float);
            Buffer output(bytes + 2 * kGuard);
            check(hipMemset(output.base, 0xa7, bytes + 2 * kGuard));
            auto* data = reinterpret_cast<float*>(static_cast<unsigned char*>(output.base) + kGuard);
            const int per_block = threads / 32;
            products<<<(cases.size() + per_block - 1) / per_block, threads>>>(
                device_input, data, unsigned(cases.size()), repetitions);
            check(hipGetLastError()); check(hipDeviceSynchronize());
            std::vector<float> actual(cases.size() * 128);
            check(hipMemcpy(actual.data(), data, bytes, hipMemcpyDeviceToHost));
            failures += verify(cases, actual, exact_count, repetitions);
            std::vector<unsigned char> guards(bytes + 2 * kGuard);
            check(hipMemcpy(guards.data(), output.base, guards.size(), hipMemcpyDeviceToHost));
            for (std::size_t i = 0; i < kGuard; ++i)
                if (guards[i] != 0xa7 || guards[kGuard + bytes + i] != 0xa7) ++failures;
        }
        std::vector<unsigned char> after(input_bytes + 2 * kGuard);
        check(hipMemcpy(after.data(), input.base, after.size(), hipMemcpyDeviceToHost));
        if (std::memcmp(after.data() + kGuard, cases.data(), input_bytes)) ++failures;
        for (std::size_t i = 0; i < kGuard; ++i)
            if (after[i] != 0xa7 || after[kGuard + input_bytes + i] != 0xa7) ++failures;
        std::cout << (failures ? "FAIL" : "PASS") << " native gfx12 OCP FP8 fragments: " << cases.size()
                  << " matrices, four configurations, " << cases.size() * 128 * 4
                  << " outputs; all64516 finite FP8 pairs, FP64 dense oracle, FP32 subnormal accumulators, guards/unchanged inputs\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
