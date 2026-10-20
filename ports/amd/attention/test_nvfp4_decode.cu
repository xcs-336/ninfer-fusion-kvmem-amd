// Qualify the actual persistent-KV codec, independently of attention and MMA.
#include "ops/common/mma.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
constexpr int Cases = 256 * 127;
struct alignas(16) Result {
    int4 half8, half16_lo, half16_hi, bf16_8;
    unsigned sdk_pair;
};

void check(hipError_t error) {
    if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
}

__global__ void decode(Result* results) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= Cases) return;
    const unsigned pair = index % 256;
    const unsigned scale = index / 256;
    alignas(16) std::uint8_t codes[8];
    for (int i = 0; i < 8; ++i)
        codes[i] = ((pair + 3 * i) & 15u) | ((((pair >> 4) + 5 * i) & 15u) << 4);
    const auto half16 = ninfer::ops::kv_cache_nvfp4_dequant_f16x16(codes, scale);
    __hip_fp4x2_e2m1 encoded;
    encoded.__x = codes[0];
    const __half2 sdk = static_cast<__half2>(encoded);
    results[index] = {ninfer::ops::kv_cache_nvfp4_dequant_f16x8(codes, scale),
        half16.lo, half16.hi, ninfer::ops::kv_cache_nvfp4_dequant_bf16x8(codes, scale),
        *reinterpret_cast<const unsigned*>(&sdk)};
}

double code_value(unsigned code) {
    constexpr double magnitudes[] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return std::copysign(magnitudes[code & 7u], code & 8u ? -1.0 : 1.0);
}

double scale_value(unsigned code) {
    const unsigned exponent = code / 8, mantissa = code % 8;
    return exponent ? std::ldexp(1.0 + double(mantissa) / 8.0, int(exponent) - 7)
                    : std::ldexp(double(mantissa), -9);
}

std::uint16_t represented_bits(double value, bool bf16) {
    const unsigned sign = std::signbit(value) ? 0x8000u : 0u;
    if (value == 0.0) return std::uint16_t(sign);
    int exponent = 0;
    const double fraction = std::frexp(std::fabs(value), &exponent);
    const unsigned precision = bf16 ? 8 : 11;
    const unsigned leading = 1u << (precision - 1);
    unsigned significand = unsigned(std::nearbyint(std::ldexp(fraction, precision)));
    if (significand == 2 * leading) { significand = leading; ++exponent; }
    // Every nonzero E2M1 * admitted E4M3 value lies in the normal FP16 range.
    return std::uint16_t(sign | (unsigned(exponent + (bf16 ? 126 : 14)) << (precision - 1)) |
                         (significand - leading));
}

unsigned verify(const int4& packed, unsigned pair, unsigned scale, int first, bool bf16,
                const char* name, std::uint64_t& checked, unsigned& printed) {
    std::uint16_t actual[8];
    std::memcpy(actual, &packed, sizeof(actual));
    unsigned failures = 0;
    for (int j = 0; j < 8; ++j) {
        const unsigned byte = (first + j) / 2;
        const unsigned code = ((first + j) & 1) ? (((pair >> 4) + 5 * byte) & 15u)
                                              : ((pair + 3 * byte) & 15u);
        const auto expected = represented_bits(code_value(code) * scale_value(scale), bf16);
        ++checked;
        if (actual[j] != expected) {
            ++failures;
            if (printed++ < 16) std::fprintf(stderr,
                "%s pair=%02x scale=%02x lane=%d code=%x actual=%04x expected=%04x\n",
                name, pair, scale, first + j, code, actual[j], expected);
        }
    }
    return failures;
}
} // namespace

int main() {
    try {
        Result* device = nullptr;
        check(hipMalloc(&device, Cases * sizeof(Result)));
        decode<<<(Cases + 127) / 128, 128>>>(device);
        check(hipGetLastError());
        std::vector<Result> results(Cases);
        check(hipMemcpy(results.data(), device, results.size() * sizeof(Result), hipMemcpyDeviceToHost));
        check(hipFree(device));
        unsigned failures = 0, printed = 0, sdk_negative_zero_defects = 0;
        std::uint64_t checked = 0;
        for (unsigned scale = 0; scale < 127; ++scale) for (unsigned pair = 0; pair < 256; ++pair) {
            const auto& result = results[scale * 256 + pair];
            failures += verify(result.half8, pair, scale, 0, false, "FP16x8", checked, printed);
            failures += verify(result.half16_lo, pair, scale, 0, false, "FP16x16lo", checked, printed);
            failures += verify(result.half16_hi, pair, scale, 8, false, "FP16x16hi", checked, printed);
            failures += verify(result.bf16_8, pair, scale, 0, true, "BF16x8", checked, printed);
            if ((pair & 15u) == 8u && (result.sdk_pair & 0xffffu) != 0x8000u) ++sdk_negative_zero_defects;
            if ((pair >> 4) == 8u && (result.sdk_pair >> 16) != 0x8000u) ++sdk_negative_zero_defects;
        }
        std::printf("Actual KV NVFP4 codec: %s (%llu exact halves, all256 packed pairs/all127 scales, "
                    "%u errors, SDK direct-half negative-zero defects=%u)\n",
                    failures == 0 ? "PASS" : "FAIL", static_cast<unsigned long long>(checked),
                    failures, sdk_negative_zero_defects);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "actual KV NVFP4 codec: %s\n", error.what());
        return 1;
    }
}
