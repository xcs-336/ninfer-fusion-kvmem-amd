#include "../core/hip_intrinsics.h"
#include <hip/hip_fp8.h>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

void check(hipError_t error, const char* call) {
    if (error != hipSuccess) { throw std::runtime_error(std::string(call) + ": " + hipGetErrorName(error)); }
}
#define H(call) check((call), #call)
__global__ void decode_fp8(float* output) {
    const unsigned i = threadIdx.x;
    __hip_fp8_e4m3 a;
    __hip_fp8_e5m2 b;
    a.__x = static_cast<unsigned char>(i);
    b.__x = static_cast<unsigned char>(i);
    output[i] = static_cast<float>(a);
    output[256 + i] = static_cast<float>(b);
}
__global__ void round_bf16(const std::uint32_t* input, std::uint16_t* output, std::size_t count) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        output[i] = __builtin_bit_cast(std::uint16_t, __float2bfloat16(__builtin_bit_cast(float, input[i])));
    }
}
double fp8_reference(unsigned bits, bool e5m2) {
    const unsigned mantissa_bits = e5m2 ? 2 : 3;
    const unsigned exponent_mask = e5m2 ? 31 : 15;
    const int bias = e5m2 ? 15 : 7;
    const unsigned mantissa = bits & ((1u << mantissa_bits) - 1);
    const unsigned exponent = (bits >> mantissa_bits) & exponent_mask;
    double value;
    if (e5m2 && exponent == 31) {
        value = mantissa ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
    } else if (!e5m2 && exponent == 15 && mantissa == 7) {
        value = std::numeric_limits<double>::quiet_NaN();
    } else if (exponent == 0) {
        value = std::ldexp(static_cast<double>(mantissa), 1 - bias - static_cast<int>(mantissa_bits));
    } else {
        value = std::ldexp(1.0 + static_cast<double>(mantissa) / (1u << mantissa_bits),
                           static_cast<int>(exponent) - bias);
    }
    return bits & 128 ? -value : value;
}
int main() {
    try {
        H(hipSetDevice(0));
        hipDeviceProp_t prop{};
        H(hipGetDeviceProperties(&prop, 0));
        if (std::string(prop.gcnArchName).find("gfx1201") != 0) { return 77; }
        float* device_fp8 = nullptr;
        H(hipMalloc(&device_fp8, 512 * sizeof(float)));
        decode_fp8<<<1, 256>>>(device_fp8);
        H(hipGetLastError());
        std::vector<float> decoded(512);
        H(hipMemcpy(decoded.data(), device_fp8, 512 * sizeof(float), hipMemcpyDeviceToHost));
        for (unsigned i = 0; i < 512; ++i) {
            const double reference = fp8_reference(i % 256, i >= 256);
            if (std::isnan(reference)) {
                if (!std::isnan(decoded[i])) { throw std::runtime_error("FP8 NaN decode mismatch"); }
            } else if (std::bit_cast<std::uint32_t>(decoded[i]) !=
                       std::bit_cast<std::uint32_t>(static_cast<float>(reference))) {
                throw std::runtime_error("OCP FP8 decode mismatch at encoding=" + std::to_string(i));
            }
        }
        H(hipFree(device_fp8));
        std::vector<std::uint32_t> input;
        for (std::uint32_t high = 0; high < 65536; ++high) {
            if ((high & 0x7f80u) == 0x7f80u) { continue; }
            for (std::uint32_t low : {0u, 0x7fffu, 0x8000u, 0x8001u, 0xffffu}) {
                input.push_back((high << 16) | low);
            }
        }
        std::uint32_t* device_input = nullptr;
        std::uint16_t* device_output = nullptr;
        H(hipMalloc(&device_input, input.size() * 4));
        H(hipMalloc(&device_output, input.size() * 2));
        H(hipMemcpy(device_input, input.data(), input.size() * 4, hipMemcpyHostToDevice));
        round_bf16<<<static_cast<unsigned>((input.size() + 255) / 256), 256>>>(device_input, device_output, input.size());
        H(hipGetLastError());
        std::vector<std::uint16_t> rounded(input.size());
        H(hipMemcpy(rounded.data(), device_output, rounded.size() * 2, hipMemcpyDeviceToHost));
        for (std::size_t i = 0; i < input.size(); ++i) {
            const auto reference = static_cast<std::uint16_t>((input[i] + 0x7fffu + ((input[i] >> 16) & 1u)) >> 16);
            if (rounded[i] != reference) { throw std::runtime_error("BF16 RNE mismatch at input bits=" + std::to_string(input[i])); }
        }
        H(hipFree(device_input)); H(hipFree(device_output));
        std::cout << "PASS OCP FP8 exact decode:512 encodings; BF16 finite RNE boundary patterns:" << input.size() << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
