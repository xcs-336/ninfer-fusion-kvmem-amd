// Native gfx1201 NVFP4 register decomposition: independent packed-code FP64
// oracle, exact E2M1*UE4M3 decode and complete m16n8k64/scale selector checks.
// Run GPU mode only under the shared lock; --host-oracle-only calls no HIP API.
#include "../ops/mma_nvfp4_gfx12.cuh"
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
// Row-major A and column-major B, two E2M1 nibbles per byte, low nibble first.
// Scale bytes are row-major [16,4] and column-major [4,8], respectively.
struct MatrixCase { Byte a[512]{}, b[256]{}, sa[64]{}, sb[32]{}; float c[128]{}; };
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
double code_value(unsigned code) {
    constexpr double values[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    return code & 8 ? -values[code & 7] : values[code & 7];
}
double scale_value(unsigned code) {
    if (code >= 127) throw std::runtime_error("scale outside finite unsigned E4M3 contract");
    const unsigned exponent = code >> 3, mantissa = code & 7;
    return exponent ? std::ldexp(8.0 + mantissa, int(exponent) - 10) : std::ldexp(double(mantissa), -9);
}
void set_code(Byte* plane, unsigned index, unsigned code) {
    const unsigned shift = 4 * (index & 1);
    plane[index / 2] = Byte((plane[index / 2] & ~(15u << shift)) | (code << shift));
}
unsigned get_code(const Byte* plane, unsigned index) { return (plane[index / 2] >> (4 * (index & 1))) & 15; }
__device__ unsigned load_word(const Byte* plane) {
    return unsigned(plane[0]) | (unsigned(plane[1]) << 8) |
           (unsigned(plane[2]) << 16) | (unsigned(plane[3]) << 24);
}
__global__ void decode_operands(float* output) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 16 * 127) output[index] = float(ninfer::ops::nvfp4_gfx12_detail::decode_scaled(index & 15u, index >> 4));
}
__global__ void products(const MatrixCase* input, float* output, unsigned count, int repetitions) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned index = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (index >= count) return;
    const auto& s = input[index];
    const unsigned row = lane >> 2, k = 8 * (lane & 3), column = 2 * (lane & 3);
    float c0 = s.c[row * 8 + column], c1 = s.c[row * 8 + column + 1];
    float c2 = s.c[(row + 8) * 8 + column], c3 = s.c[(row + 8) * 8 + column + 1];
    const unsigned a0 = load_word(s.a + row * 32 + k / 2);
    const unsigned a1 = load_word(s.a + (row + 8) * 32 + k / 2);
    const unsigned a2 = load_word(s.a + row * 32 + k / 2 + 16);
    const unsigned a3 = load_word(s.a + (row + 8) * 32 + k / 2 + 16);
    const unsigned b0 = load_word(s.b + row * 32 + k / 2);
    const unsigned b1 = load_word(s.b + row * 32 + k / 2 + 16);
    const unsigned a_scale_row = 8 * (lane & 1) + row;
    // Original selector0 uses only quad lanes0/1 for A and lane0 for B.
    // Poison ignored registers with bytes outside the UE4M3 admitted domain;
    // reading arbitrary copies instead of the specified lanes must fail.
    const unsigned sfa = (lane & 2) ? 0xffffffffu : load_word(s.sa + a_scale_row * 4);
    const unsigned sfb = (lane & 3) ? 0xffffffffu : load_word(s.sb + row * 4);
    for (int rep = 0; rep < repetitions; ++rep)
        ninfer::ops::mma_nvfp4_e4m3(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, sfa, sfb);
    float* destination = output + index * 128;
    destination[row * 8 + column] = c0; destination[row * 8 + column + 1] = c1;
    destination[(row + 8) * 8 + column] = c2; destination[(row + 8) * 8 + column + 1] = c3;
}
int guards(const Buffer& buffer, std::size_t bytes, const char* label) {
    std::vector<Byte> after(bytes + 2 * kGuard);
    check(hipMemcpy(after.data(), buffer.base, after.size(), hipMemcpyDeviceToHost));
    int failures = 0;
    for (std::size_t i = 0; i < kGuard; ++i)
        if (after[i] != 0xa7 || after[kGuard + bytes + i] != 0xa7) {
            if (failures < 8) std::cerr << label << " guard byte=" << i << " before=" << unsigned(after[i])
                << " after=" << unsigned(after[kGuard + bytes + i]) << " expected=167\n";
            ++failures;
        }
    return failures;
}
int verify(const std::vector<MatrixCase>& cases, const std::vector<float>& actual,
           unsigned exact_count, int repetitions) {
    int failures = 0;
    double squared_error = 0, squared_reference = 0, max_ratio = 0;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& s = cases[i];
        double represented_a[1024], represented_b[512];
        for (unsigned row = 0; row < 16; ++row) for (unsigned k = 0; k < 64; ++k)
            represented_a[row * 64 + k] = code_value(get_code(s.a, row * 64 + k)) * scale_value(s.sa[row * 4 + k / 16]);
        for (unsigned col = 0; col < 8; ++col) for (unsigned k = 0; k < 64; ++k)
            represented_b[col * 64 + k] = code_value(get_code(s.b, col * 64 + k)) * scale_value(s.sb[col * 4 + k / 16]);
        for (unsigned row = 0; row < 16; ++row) for (unsigned column = 0; column < 8; ++column) {
            double expected = s.c[row * 8 + column], sum_abs = std::fabs(expected);
            for (int rep = 0; rep < repetitions; ++rep)
                for (unsigned k = 0; k < 64; ++k) {
                    const double a = represented_a[row * 64 + k];
                    const double b = represented_b[column * 64 + k];
                    const double term = a * b;
                    expected += term; sum_abs += std::fabs(term);
                }
            const double got = actual[i * 128 + row * 8 + column];
            const double n = 64.0 * repetitions + 1.0, unit = std::ldexp(1.0, -24);
            const double bound = n * unit / (1 - n * unit) * sum_abs;
            const double error = std::fabs(got - expected);
            const bool pass = std::isfinite(got) && (i < exact_count ? got == expected : error <= bound);
            if (i >= exact_count) {
                if (bound) max_ratio = std::max(max_ratio, error / bound);
                squared_error += error * error; squared_reference += expected * expected;
            }
            if (!pass && failures++ < 8)
                std::cerr << "case=" << i << " row=" << row << " col=" << column
                          << " actual=" << got << " expected=" << expected << " bound=" << bound << '\n';
        }
    }
    const double relative_l2 = std::sqrt(squared_error / std::max(squared_reference, 1e-100));
    if (!std::isfinite(relative_l2) || relative_l2 > 2e-6) ++failures;
    std::cout << "NVFP4 via FP16 WMMA: repetitions=" << repetitions << " relative_l2=" << relative_l2
              << " max_error/bound=" << max_ratio << " failures=" << failures << '\n';
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool host_only = argc == 2 && std::string(argv[1]) == "--host-oracle-only";
        if (argc > 2 || (argc == 2 && !host_only))
            throw std::runtime_error("usage: matrix_nvfp4 [--host-oracle-only]");
        // Exhaustively prove the proposed operand cast loses no information,
        // using independent host arithmetic rather than the device decoder.
        for (unsigned scale = 0; scale < 127; ++scale) for (unsigned code = 0; code < 16; ++code) {
            const double value = std::fabs(code_value(code) * scale_value(scale));
            if (value == 0) continue;
            int exponent; std::frexp(value, &exponent); --exponent;
            const double mantissa = std::ldexp(value, 10 - exponent);
            if (exponent < -14 || exponent > 15 || std::floor(mantissa) != mantissa)
                throw std::runtime_error("finite NVFP4 operand is not exactly FP16");
        }
        if (scale_value(1) != std::ldexp(1.0, -9) || scale_value(126) != 448 ||
            code_value(1) != 0.5 || code_value(7) != 6 || !std::signbit(code_value(8)))
            throw std::runtime_error("independent NVFP4 oracle boundary check failed");
        if (host_only) {
            std::cout << "PASS CPU-only NVFP4 oracle: all2032 admitted scaled operands exactly FP16, positive finite scales0..126\n";
            return 0;
        }
        int failures = 0;
        {
            constexpr std::size_t bytes = 2032 * sizeof(float);
            Buffer output(bytes + 2 * kGuard);
            check(hipMemset(output.base, 0xa7, bytes + 2 * kGuard));
            auto* data = reinterpret_cast<float*>(static_cast<Byte*>(output.base) + kGuard);
            decode_operands<<<8, 256>>>(data); check(hipGetLastError()); check(hipDeviceSynchronize());
            std::vector<float> actual(2032); check(hipMemcpy(actual.data(), data, bytes, hipMemcpyDeviceToHost));
            for (unsigned i = 0; i < 2032; ++i) {
                const float expected = float(code_value(i & 15) * scale_value(i >> 4));
                if (std::bit_cast<std::uint32_t>(expected) != std::bit_cast<std::uint32_t>(actual[i])) {
                    if (failures < 12) std::cerr << "decoded operand code=" << (i & 15) << " scale=" << (i >> 4)
                        << " expected=" << expected << " actual=" << actual[i] << " expected_bits="
                        << std::bit_cast<std::uint32_t>(expected) << " actual_bits=" << std::bit_cast<std::uint32_t>(actual[i]) << '\n';
                    ++failures;
                }
            }
            std::cout << "NVFP4 exact scaled operand decode: failures=" << failures << '\n';
            failures += guards(output, bytes, "decoded operands");
        }
        constexpr unsigned basis_count = 16 * 8 * 64;
        constexpr unsigned scale_pair_count = 8 * 16;
        constexpr unsigned exact_count = basis_count + scale_pair_count + 4;
        std::vector<MatrixCase> cases(exact_count + 96);
        for (auto& s : cases) {
            for (unsigned row = 0; row < 16; ++row) for (unsigned block = 0; block < 4; ++block)
                s.sa[row * 4 + block] = Byte(0x20 + 8 * ((row + block) % 5));
            for (unsigned col = 0; col < 8; ++col) for (unsigned block = 0; block < 4; ++block)
                s.sb[col * 4 + block] = Byte(0x28 + 8 * ((2 * col + block) % 5));
        }
        for (unsigned i = 0; i < basis_count; ++i) {
            const unsigned k = i % 64, column = (i / 64) % 8, row = i / 512;
            set_code(cases[i].a, row * 64 + k, i & 1 ? 12 : 4); // +/-2
            set_code(cases[i].b, column * 64 + k, i & 2 ? 13 : 5); // +/-3
            for (unsigned j = 0; j < 128; ++j) cases[i].c[j] = float(int(j % 13) - 6);
        }
        // All127^2 admitted scale pairs with a single E2M1=6 product.
        for (unsigned ag = 0; ag < 8; ++ag) for (unsigned bg = 0; bg < 16; ++bg) {
            const unsigned tile = ag * 16 + bg, k = (tile * 13) % 64;
            auto& s = cases[basis_count + tile];
            for (unsigned row = 0; row < 16; ++row) {
                set_code(s.a, row * 64 + k, 7);
                s.sa[row * 4 + k / 16] = Byte(ag * 16 + row < 127 ? ag * 16 + row : 0);
            }
            for (unsigned column = 0; column < 8; ++column) {
                set_code(s.b, column * 64 + k, 7);
                s.sb[column * 4 + k / 16] = Byte(bg * 8 + column < 127 ? bg * 8 + column : 0);
            }
        }
        // All16^2 E2M1 code pairs, including signed zero and maximum magnitude.
        for (unsigned tile = 0; tile < 2; ++tile) {
            auto& s = cases[basis_count + scale_pair_count + tile];
            const unsigned k = 17 + 32 * tile;
            for (unsigned row = 0; row < 16; ++row) set_code(s.a, row * 64 + k, row);
            for (unsigned column = 0; column < 8; ++column) set_code(s.b, column * 64 + k, column + 8 * tile);
        }
        for (unsigned tile = exact_count - 2; tile < exact_count; ++tile)
            for (unsigned j = 0; j < 128; ++j)
                cases[tile].c[j] = std::bit_cast<float>((j & 1 ? 0x80000000u : 0) |
                    (tile == exact_count - 2 ? 1u + j : 0x00800000u + j));
        std::mt19937 random(0xe2e4a301u);
        for (unsigned i = exact_count; i < cases.size(); ++i) {
            for (Byte& byte : cases[i].a) byte = Byte(random());
            for (Byte& byte : cases[i].b) byte = Byte(random());
            for (Byte& scale : cases[i].sa) scale = Byte(random() % 127);
            for (Byte& scale : cases[i].sb) scale = Byte(random() % 127);
            for (float& c : cases[i].c)
                c = std::bit_cast<float>((random() & 0x807fffffu) | ((110 + random() % 31) << 23));
        }
        const std::size_t input_bytes = cases.size() * sizeof(MatrixCase);
        Buffer input(input_bytes + 2 * kGuard);
        check(hipMemset(input.base, 0xa7, input_bytes + 2 * kGuard));
        auto* device_input = reinterpret_cast<MatrixCase*>(static_cast<Byte*>(input.base) + kGuard);
        check(hipMemcpy(device_input, cases.data(), input_bytes, hipMemcpyHostToDevice));
        for (int threads : {32, 128}) for (int repetitions : {1, 3}) {
            const std::size_t bytes = cases.size() * 128 * sizeof(float);
            Buffer output(bytes + 2 * kGuard);
            check(hipMemset(output.base, 0xa7, bytes + 2 * kGuard));
            auto* data = reinterpret_cast<float*>(static_cast<Byte*>(output.base) + kGuard);
            const int per_block = threads / 32;
            products<<<(cases.size() + per_block - 1) / per_block, threads>>>(
                device_input, data, unsigned(cases.size()), repetitions);
            check(hipGetLastError()); check(hipDeviceSynchronize());
            std::vector<float> actual(cases.size() * 128);
            check(hipMemcpy(actual.data(), data, bytes, hipMemcpyDeviceToHost));
            failures += verify(cases, actual, exact_count, repetitions);
            failures += guards(output, bytes, "matrix output");
        }
        std::vector<Byte> after(input_bytes);
        check(hipMemcpy(after.data(), device_input, input_bytes, hipMemcpyDeviceToHost));
        if (std::memcmp(after.data(), cases.data(), input_bytes)) {
            std::cerr << "NVFP4 input bytes changed\n"; ++failures;
        }
        failures += guards(input, input_bytes, "matrix inputs");
        std::cout << (failures ? "FAIL" : "PASS") << " native NVFP4 register decomposition: " << cases.size()
                  << " matrices, four configurations, " << cases.size() * 128 * 4
                  << " matrix outputs plus2032 exact operand decodes; FP64 oracle, poisoned ignored scale lanes, guards/unchanged inputs\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
