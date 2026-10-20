// Standalone native-AMD qualification candidate. Compile with hipcc, never nvcc.
#include "mma_bf16_gfx12.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(hipError_t status, const char* call) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(call) + ": " + hipGetErrorString(status));
    }
}
#define HIP_CHECK(call) check((call), #call)

struct Input {
    std::uint16_t a[16 * 16]{};
    std::uint16_t b[16 * 8]{};
    float c[16 * 8]{};
};

std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}
double decode(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
}

__device__ unsigned pair(std::uint16_t lo, std::uint16_t hi) {
    return static_cast<unsigned>(lo) | (static_cast<unsigned>(hi) << 16);
}

__global__ void evaluate(const Input* inputs, float* outputs, int count, int repeats) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned tile = tid / 32;
    if (tile >= static_cast<unsigned>(count)) { return; } // Uniform for the whole wave.
    const unsigned lane = tid % 32;
    const unsigned g = lane / 4, t = lane % 4;
    const Input& v = inputs[tile];
    // Test-owned packing follows NVIDIA's published fragment coordinates, not the
    // candidate's gather formulas. Row-major storage is just a fixture encoding.
    const unsigned a0 = pair(v.a[g * 16 + 2 * t], v.a[g * 16 + 2 * t + 1]);
    const unsigned a1 = pair(v.a[(g + 8) * 16 + 2 * t], v.a[(g + 8) * 16 + 2 * t + 1]);
    const unsigned a2 = pair(v.a[g * 16 + 2 * t + 8], v.a[g * 16 + 2 * t + 9]);
    const unsigned a3 = pair(v.a[(g + 8) * 16 + 2 * t + 8], v.a[(g + 8) * 16 + 2 * t + 9]);
    const unsigned b0 = pair(v.b[(2 * t) * 8 + g], v.b[(2 * t + 1) * 8 + g]);
    const unsigned b1 = pair(v.b[(2 * t + 8) * 8 + g], v.b[(2 * t + 9) * 8 + g]);
    float c0 = v.c[g * 8 + 2 * t], c1 = v.c[g * 8 + 2 * t + 1];
    float c2 = v.c[(g + 8) * 8 + 2 * t], c3 = v.c[(g + 8) * 8 + 2 * t + 1];
    for (int i = 0; i < repeats; ++i) {
        ninfer::amd_candidate::mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    }
    float* out = outputs + tile * 130 + 1;
    out[g * 8 + 2 * t] = c0;
    out[g * 8 + 2 * t + 1] = c1;
    out[(g + 8) * 8 + 2 * t] = c2;
    out[(g + 8) * 8 + 2 * t + 1] = c3;
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) { HIP_CHECK(hipMalloc(&data, bytes)); }
    ~DeviceBuffer() { if (data) { (void)hipFree(data); } }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* data = nullptr;
};

class Stream {
public:
    Stream() { HIP_CHECK(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
    ~Stream() { (void)hipStreamDestroy(value); }
    hipStream_t value{};
};

struct Fixtures {
    std::vector<Input> values;
    std::size_t exact_cases = 0;
};

Fixtures fixtures() {
    Fixtures result;
    const auto with_c = [] {
        Input v;
        for (int i = 0; i < 128; ++i) { v.c[i] = static_cast<float>(i - 64) / 4.0f; }
        return v;
    };
    // Unique nonzero accumulators reveal missing or transposed C transport.
    result.values.push_back(with_c());
    // Every output row, column and K coordinate contributes once in isolation.
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 16; ++k) {
                Input v = with_c();
                v.a[i * 16 + k] = bf16(1.0f);
                v.b[k * 8 + j] = bf16(2.0f);
                result.values.push_back(v);
            }
        }
    }
    Input dense = with_c();
    for (int i = 0; i < 256; ++i) { dense.a[i] = bf16(static_cast<float>(i % 17 - 8)); }
    for (int i = 0; i < 128; ++i) { dense.b[i] = bf16(static_cast<float>(i % 19 - 9)); }
    result.values.push_back(dense);
    result.exact_cases = result.values.size();
    std::uint32_t state = 0x35e7a901u;
    const auto next = [&state] {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    };
    // Includes cancellation, signs and operand exponent variation, on normal BF16.
    for (int tile = 0; tile < 48; ++tile) {
        Input v = with_c();
        for (auto& x : v.a) {
            const int mantissa = static_cast<int>(next() % 511) - 255;
            const int exponent = static_cast<int>(next() % 17) - 8;
            x = bf16(std::ldexp(static_cast<float>(mantissa) / 128.0f, exponent));
        }
        for (auto& x : v.b) {
            const int mantissa = static_cast<int>(next() % 511) - 255;
            const int exponent = static_cast<int>(next() % 17) - 8;
            x = bf16(std::ldexp(static_cast<float>(mantissa) / 128.0f, exponent));
        }
        result.values.push_back(v);
    }
    return result;
}

void qualify(const Fixtures& cases, const std::vector<float>& actual, int repeats, int threads) {
    double max_error = 0, max_scaled = 0, max_relative_l2 = 0;
    std::size_t worst = 0;
    constexpr double unit = std::numeric_limits<float>::epsilon();
    for (std::size_t tile = 0; tile < cases.values.size(); ++tile) {
        if (actual[tile * 130] != -123456.0f || actual[tile * 130 + 129] != -123456.0f) {
            throw std::runtime_error("output guard was overwritten");
        }
        const auto& v = cases.values[tile];
        double error2 = 0, reference2 = 0;
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 8; ++j) {
                double sum = 0, absolute_sum = 0;
                for (int k = 0; k < 16; ++k) {
                    const double product = decode(v.a[i * 16 + k]) * decode(v.b[k * 8 + j]);
                    sum += product;
                    absolute_sum += std::abs(product);
                }
                const double expected = v.c[i * 8 + j] + repeats * sum;
                const double observed = actual[tile * 130 + 1 + i * 8 + j];
                const double error = std::abs(observed - expected);
                const double bound = 32 * unit * (std::abs(v.c[i * 8 + j]) +
                                                  repeats * absolute_sum) + 1e-30;
                if (!std::isfinite(observed) || !std::isfinite(expected) || error > bound ||
                    (tile < cases.exact_cases && error != 0)) {
                    throw std::runtime_error("numeric mismatch at tile=" + std::to_string(tile) +
                                             " row=" + std::to_string(i) + " col=" +
                                             std::to_string(j) + " observed=" +
                                             std::to_string(observed) + " expected=" +
                                             std::to_string(expected) + " bound=" +
                                             std::to_string(bound));
                }
                if (error > max_error) { max_error = error; worst = tile; }
                max_scaled = std::max(max_scaled, error / bound);
                error2 += error * error;
                reference2 += expected * expected;
            }
        }
        const double relative_l2 = reference2 > 0 ? std::sqrt(error2 / reference2) : std::sqrt(error2);
        if (relative_l2 > 1e-5) {
            throw std::runtime_error("relative L2 failed at tile=" + std::to_string(tile) +
                                     " observed=" + std::to_string(relative_l2) + " limit=1e-5");
        }
        max_relative_l2 = std::max(max_relative_l2, relative_l2);
    }
    std::cout << "{\"scope\":\"BF16 MMA primitive\",\"status\":\"PASS\",\"cases\":"
              << cases.values.size() << ",\"exact_cases\":" << cases.exact_cases
              << ",\"block_threads\":" << threads << ",\"repeats\":" << repeats
              << ",\"max_absolute_error\":" << max_error << ",\"max_error_to_bound\":"
              << max_scaled << ",\"max_per_tile_relative_l2\":" << max_relative_l2
              << ",\"worst_absolute_error_tile\":" << worst << "}\n";
}
} // namespace

int main() {
    try {
        int count = 0;
        const auto found = hipGetDeviceCount(&count);
        if (found != hipSuccess || count == 0) {
            std::cerr << "NOT_RUN: no usable HIP device: " << hipGetErrorString(found) << '\n';
            return 77;
        }
        HIP_CHECK(hipSetDevice(0));
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, 0));
        if (std::string(prop.gcnArchName).find("gfx1201") != 0 || prop.warpSize != 32) {
            std::cerr << "NOT_RUN: expected gfx1201 wave32; found " << prop.gcnArchName << '\n';
            return 77;
        }
        int runtime = 0, driver = 0;
        HIP_CHECK(hipRuntimeGetVersion(&runtime));
        HIP_CHECK(hipDriverGetVersion(&driver));
        std::cout << std::setprecision(12);
        std::cout << "{\"device\":\"" << prop.name << "\",\"arch\":\"" << prop.gcnArchName
                  << "\",\"wave_size\":" << prop.warpSize << ",\"total_vram_bytes\":"
                  << prop.totalGlobalMem << ",\"hip_runtime\":" << runtime
                  << ",\"hip_driver_api_version\":" << driver << "}\n";
        Stream stream;
        const auto cases = fixtures();
        const auto input_bytes = cases.values.size() * sizeof(Input);
        std::vector<float> actual(cases.values.size() * 130, -123456.0f);
        const auto output_bytes = actual.size() * sizeof(float);
        std::size_t free_before = 0, total = 0;
        HIP_CHECK(hipMemGetInfo(&free_before, &total));
        std::size_t minimum_free_sample = free_before;
        DeviceBuffer inputs(input_bytes), outputs(output_bytes);
        auto sample_memory = [&] {
            std::size_t available = 0;
            HIP_CHECK(hipMemGetInfo(&available, &total));
            minimum_free_sample = std::min(minimum_free_sample, available);
        };
        sample_memory();
        HIP_CHECK(hipMemcpyAsync(inputs.data, cases.values.data(), input_bytes,
                                hipMemcpyHostToDevice, stream.value));
        for (int threads : {32, 128}) {
            for (int repeats : {1, 3}) {
                std::fill(actual.begin(), actual.end(), -123456.0f);
                HIP_CHECK(hipMemcpyAsync(outputs.data, actual.data(), output_bytes,
                                        hipMemcpyHostToDevice, stream.value));
                const auto blocks = static_cast<unsigned>((cases.values.size() * 32 + threads - 1) / threads);
                evaluate<<<blocks, threads, 0, stream.value>>>(static_cast<const Input*>(inputs.data),
                    static_cast<float*>(outputs.data), static_cast<int>(cases.values.size()), repeats);
                HIP_CHECK(hipGetLastError());
                HIP_CHECK(hipMemcpyAsync(actual.data(), outputs.data, output_bytes,
                                        hipMemcpyDeviceToHost, stream.value));
                HIP_CHECK(hipStreamSynchronize(stream.value));
                sample_memory();
                qualify(cases, actual, repeats, threads);
            }
        }
        std::vector<Input> preserved(cases.values.size());
        HIP_CHECK(hipMemcpyAsync(preserved.data(), inputs.data, input_bytes,
                                hipMemcpyDeviceToHost, stream.value));
        HIP_CHECK(hipStreamSynchronize(stream.value));
        if (std::memcmp(preserved.data(), cases.values.data(), input_bytes) != 0) {
            throw std::runtime_error("read-only inputs changed");
        }
        std::cout << "{\"owned_buffer_peak_bytes\":" << input_bytes + output_bytes
                  << ",\"sampled_device_used_high_water_bytes\":" << total - minimum_free_sample
                  << ",\"sampled_device_used_delta_bytes\":" << free_before - minimum_free_sample
                  << ",\"memory_scope\":\"two owned buffers exact; device use sampled, includes other processes\","
                     "\"engine_peak_vram\":\"NOT_MEASURED\",\"inputs_preserved\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
