// Independent row-major FP64 qualification of the production FP16 fragment
// adapter. Compile with AMD clang -x hip, gfx1201 and wave32. This executable
// does not benchmark throughput and must be run under the shared GPU lock.
#include "../ops/mma_f16_gfx12.cuh"
#include <algorithm>
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
using U16 = std::uint16_t;
struct MatrixCase { U16 a[256]{}, b[128]{}, c[128]{}; };
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

// IEEE binary16 decoding and ties-to-even encoding use only host FP64 and
// integer operations, independently of HIP half types and the WMMA adapter.
double decode(U16 bits) {
    const unsigned e = (bits >> 10) & 31u, m = bits & 1023u;
    double value;
    if (e == 31) value = m ? std::numeric_limits<double>::quiet_NaN() :
                            std::numeric_limits<double>::infinity();
    else value = e ? std::ldexp(1024.0 + m, int(e) - 25) : std::ldexp(double(m), -24);
    return (bits & 0x8000u) ? -value : value;
}
unsigned ties_even(double x) {
    const double lower = std::floor(x), delta = x - lower;
    const unsigned n = static_cast<unsigned>(lower);
    return n + unsigned(delta > 0.5 || (delta == 0.5 && (n & 1u)));
}
U16 encode(double value) {
    const U16 sign = std::signbit(value) ? 0x8000u : 0;
    value = std::fabs(value);
    if (std::isnan(value)) return sign | 0x7e00u;
    if (std::isinf(value)) return sign | 0x7c00u;
    if (value == 0) return sign;
    if (value < std::ldexp(1.0, -14)) return sign | U16(ties_even(std::ldexp(value, 24)));
    int exponent;
    std::frexp(value, &exponent);
    --exponent;
    if (exponent > 15) return sign | 0x7c00u;
    unsigned significand = ties_even(std::ldexp(value, 10 - exponent));
    if (significand == 2048) { significand = 1024; ++exponent; }
    if (exponent > 15) return sign | 0x7c00u;
    return sign | U16((exponent + 15) << 10) | U16(significand - 1024);
}

__device__ unsigned pack(U16 x, U16 y) { return unsigned(x) | (unsigned(y) << 16); }
__global__ void fragment_product(const MatrixCase* input, void* output, unsigned count,
                                 int repetitions, bool half_accumulation) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned index = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (index >= count) return; // Whole waves leave together.
    const MatrixCase& s = input[index];
    const unsigned row = lane / 4, k = 2 * (lane % 4), col = k;
    const unsigned a0 = pack(s.a[row * 16 + k], s.a[row * 16 + k + 1]);
    const unsigned a1 = pack(s.a[(row + 8) * 16 + k], s.a[(row + 8) * 16 + k + 1]);
    const unsigned a2 = pack(s.a[row * 16 + k + 8], s.a[row * 16 + k + 9]);
    const unsigned a3 = pack(s.a[(row + 8) * 16 + k + 8], s.a[(row + 8) * 16 + k + 9]);
    const unsigned b0 = pack(s.b[k * 8 + row], s.b[(k + 1) * 8 + row]);
    const unsigned b1 = pack(s.b[(k + 8) * 8 + row], s.b[(k + 9) * 8 + row]);
    if (half_accumulation) {
        unsigned d0 = pack(s.c[row * 8 + col], s.c[row * 8 + col + 1]);
        unsigned d1 = pack(s.c[(row + 8) * 8 + col], s.c[(row + 8) * 8 + col + 1]);
        for (int i = 0; i < repetitions; ++i)
            ninfer::ops::mma_f16_f16acc(d0, d1, a0, a1, a2, a3, b0, b1);
        U16* result = static_cast<U16*>(output) + index * 128;
        result[row * 8 + col] = U16(d0); result[row * 8 + col + 1] = U16(d0 >> 16);
        result[(row + 8) * 8 + col] = U16(d1); result[(row + 8) * 8 + col + 1] = U16(d1 >> 16);
    } else {
        const auto half_to_float = [](U16 bits) { return float(__builtin_bit_cast(_Float16, bits)); };
        float c0 = half_to_float(s.c[row * 8 + col]);
        float c1 = half_to_float(s.c[row * 8 + col + 1]);
        float c2 = half_to_float(s.c[(row + 8) * 8 + col]);
        float c3 = half_to_float(s.c[(row + 8) * 8 + col + 1]);
        for (int i = 0; i < repetitions; ++i)
            ninfer::ops::mma_f16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
        float* result = static_cast<float*>(output) + index * 128;
        result[row * 8 + col] = c0; result[row * 8 + col + 1] = c1;
        result[(row + 8) * 8 + col] = c2; result[(row + 8) * 8 + col + 1] = c3;
    }
}

int verify(const std::vector<MatrixCase>& cases, const std::vector<unsigned char>& output,
           unsigned exact_count, int repetitions, bool half_accumulation) {
    int failures = 0;
    double max_ratio = 0, error_squared = 0, reference_squared = 0;
    std::size_t special_outputs = 0;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& s = cases[i];
        for (unsigned row = 0; row < 16; ++row) for (unsigned col = 0; col < 8; ++col) {
            double product = 0, sum_abs = 0;
            for (unsigned k = 0; k < 16; ++k) {
                const double term = decode(s.a[row * 16 + k]) * decode(s.b[k * 8 + col]);
                product += term; sum_abs += std::fabs(term);
            }
            double expected = decode(s.c[row * 8 + col]);
            for (int rep = 0; rep < repetitions; ++rep) {
                expected += product;
                if (half_accumulation) expected = decode(encode(expected));
            }
            const auto offset = i * 128 + row * 8 + col;
            double actual;
            if (half_accumulation) { U16 bits; std::memcpy(&bits, output.data() + offset * 2, 2); actual = decode(bits); }
            else { float value; std::memcpy(&value, output.data() + offset * 4, 4); actual = value; }
            bool pass = false;
            double bound = 0;
            if (!std::isfinite(expected)) {
                pass = std::isinf(expected) && std::isinf(actual) &&
                       (std::signbit(expected) == std::signbit(actual));
                ++special_outputs;
            } else if (i < exact_count) {
                // Integer basis and single-product boundaries are exactly
                // representable after the semantic FP16 output cast. Zero's
                // sign is not part of PTX MMA's documented rounding contract.
                pass = actual == expected;
            } else {
                // FP32 sum: conservative gamma(16*repetitions+1) bound. FP16
                // native accumulation permits half precision internally; use
                // its gamma bound plus a normalised L2 gate over dense cases.
                const double unit = half_accumulation ? std::ldexp(1.0, -11) : std::ldexp(1.0, -24);
                const double n = 16.0 * repetitions + 1.0;
                bound = n * unit / (1.0 - n * unit) *
                    (repetitions * sum_abs + std::fabs(decode(s.c[row * 8 + col]))) +
                    (half_accumulation ? repetitions * std::ldexp(1.0, -24) : std::ldexp(1.0, -120));
                const double error = std::fabs(actual - expected);
                pass = std::isfinite(actual) && error <= bound;
                if (bound) max_ratio = std::max(max_ratio, error / bound);
                error_squared += error * error;
                reference_squared += expected * expected;
            }
            if (!pass && failures++ < 8)
                std::cerr << "matrix=" << i << " row=" << row << " col=" << col
                          << " actual=" << actual << " expected=" << expected << " bound=" << bound << '\n';
        }
    }
    const double relative_l2 = std::sqrt(error_squared / std::max(reference_squared, 1e-100));
    const double l2_limit = half_accumulation ? 1.0 / 1024.0 : 2e-6;
    if (!std::isfinite(relative_l2) || relative_l2 > l2_limit) ++failures;
    std::cout << "FP16 inputs, " << (half_accumulation ? "FP16" : "FP32") << " accumulation: "
              << "repetitions=" << repetitions << " relative_l2=" << relative_l2
              << " max_error/bound=" << max_ratio << " overflow_outputs=" << special_outputs
              << " failures=" << failures << '\n';
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool host_only = argc == 2 && std::string(argv[1]) == "--host-oracle-only";
        if (argc > 2 || (argc == 2 && !host_only))
            throw std::runtime_error("usage: matrix_f16 [--host-oracle-only]");
        // This option performs no HIP API call and permits CPU-only checks
        // while another conversation owns the shared GPU lock.
        for (unsigned bits = 0; bits < 65536; ++bits)
            if ((bits & 0x7c00u) != 0x7c00u && encode(decode(U16(bits))) != bits)
                throw std::runtime_error("independent binary16 oracle roundtrip failed");
        if (encode(std::ldexp(1.0, -25)) != 0 ||
            encode(std::ldexp(3.0, -25)) != 2 ||
            encode(std::ldexp(1023.5, -24)) != 0x0400 ||
            encode(65504) != 0x7bff || encode(65519) != 0x7bff ||
            encode(65520) != 0x7c00 || encode(-65520) != 0xfc00)
            throw std::runtime_error("independent binary16 oracle ties/overflow failed");
        if (host_only) {
            std::cout << "PASS CPU-only FP16 oracle: 63488 finite encodings, signed zeros, subnormal ties and overflow\n";
            return 0;
        }
        constexpr unsigned basis_count = 16 * 8 * 16;
        constexpr U16 boundary_values[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff,
            0x0400, 0x8400, 0x3555, 0xb555, 0x3bff, 0xbbff, 0x3c00, 0xbc00,
            0x3c01, 0xbc01, 0x7bff, 0xfbff};
        constexpr unsigned boundary_count = sizeof(boundary_values) / sizeof(boundary_values[0]);
        constexpr unsigned exact_count = basis_count + boundary_count * 2;
        std::vector<MatrixCase> cases(exact_count + 96);
        for (unsigned i = 0; i < basis_count; ++i) {
            const unsigned k = i % 16, col = (i / 16) % 8, row = i / 128;
            cases[i].a[row * 16 + k] = encode((i & 1) ? -2 : 2);
            cases[i].b[k * 8 + col] = encode((i & 2) ? -3 : 3);
            for (unsigned j = 0; j < 128; ++j) cases[i].c[j] = encode(int(j % 13) - 6);
        }
        for (unsigned i = 0; i < boundary_count * 2; ++i) {
            auto& sample = cases[basis_count + i];
            const unsigned k = (i * 7) & 15;
            for (unsigned row = 0; row < 16; ++row)
                sample.a[row * 16 + k] = boundary_values[i % boundary_count];
            for (unsigned col = 0; col < 8; ++col)
                sample.b[k * 8 + col] = encode(i < boundary_count ? 1 : 2);
        }
        std::mt19937 random(0xf169012u);
        for (unsigned i = exact_count; i < cases.size(); ++i) {
            auto finite = [&]() { return U16((random() & 0x83ffu) | ((8 + random() % 11) << 10)); };
            for (U16& v : cases[i].a) v = finite();
            for (U16& v : cases[i].b) v = finite();
            for (U16& v : cases[i].c) v = finite();
        }
        const std::size_t input_bytes = cases.size() * sizeof(MatrixCase);
        Buffer input(input_bytes + 2 * kGuard);
        check(hipMemset(input.base, 0xa7, input_bytes + 2 * kGuard));
        auto* device_input = reinterpret_cast<MatrixCase*>(static_cast<unsigned char*>(input.base) + kGuard);
        check(hipMemcpy(device_input, cases.data(), input_bytes, hipMemcpyHostToDevice));
        int failures = 0;
        for (int threads : {32, 128}) for (int repetitions : {1, 3})
            for (bool half_accumulation : {false, true}) {
                const std::size_t bytes = cases.size() * 128 * (half_accumulation ? 2 : 4);
                Buffer output(bytes + 2 * kGuard);
                check(hipMemset(output.base, 0xa7, bytes + 2 * kGuard));
                void* data = static_cast<unsigned char*>(output.base) + kGuard;
                const int per_block = threads / 32;
                fragment_product<<<(cases.size() + per_block - 1) / per_block, threads>>>(
                    device_input, data, unsigned(cases.size()), repetitions, half_accumulation);
                check(hipGetLastError()); check(hipDeviceSynchronize());
                std::vector<unsigned char> actual(bytes);
                check(hipMemcpy(actual.data(), data, bytes, hipMemcpyDeviceToHost));
                failures += verify(cases, actual, exact_count, repetitions, half_accumulation);
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
        std::cout << (failures ? "FAIL" : "PASS") << " native gfx12 FP16 fragments: "
                  << cases.size() << " matrices, 8 launch/repetition/accumulator combinations, "
                  << cases.size() * 128 * 8 << " outputs; exact basis/boundaries, FP64 dense oracle, input/guard checks\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
