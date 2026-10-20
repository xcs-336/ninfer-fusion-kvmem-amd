#include "../ops/ldmatrix.cuh"
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void check(hipError_t status, const char* call) {
    if (status != hipSuccess) { throw std::runtime_error(std::string(call) + ": " + hipGetErrorName(status)); }
}
#define H(call) check((call), #call)
template <int Bytes>
__global__ void copies(const unsigned char* source, unsigned char* output, int valid) {
    __shared__ unsigned char shared[32 * Bytes];
    auto* destination = shared + threadIdx.x * Bytes;
    ninfer::ops::cp_async_zfill<Bytes>(destination, source + threadIdx.x * Bytes, valid);
    ninfer::ops::cp_commit();
    ninfer::ops::cp_wait<0>();
    __syncthreads();
    // A different lane observes each copy, so missing visibility is observable.
    for (int i = 0; i < Bytes; ++i) {
        const auto index = ((threadIdx.x + 7) % 32) * Bytes + i;
        output[index] = shared[index];
    }
}
template <bool Transposed, int Matrices>
__global__ void matrix_load(unsigned* output, unsigned padding) {
    __shared__ unsigned short shared[4][8][16];
    const unsigned lane = threadIdx.x;
    for (unsigned i = lane; i < 4 * 8 * 16; i += 32) {
        reinterpret_cast<unsigned short*>(shared)[i] = static_cast<unsigned short>(1000 + i * 7);
    }
    __syncthreads();
    const unsigned matrix = lane / 8, row = lane % 8;
    // Different row addresses and offsets defeat a contiguous-tile-only implementation.
    const auto address = ninfer::ops::smem_addr(&shared[matrix][row][padding]);
    unsigned r[4]{};
    if constexpr (Matrices == 4 && Transposed) {
        ninfer::ops::ldmatrix_x4_t(r[0], r[1], r[2], r[3], address);
    } else if constexpr (Matrices == 4) {
        ninfer::ops::ldmatrix_x4(r[0], r[1], r[2], r[3], address);
    } else if constexpr (Transposed) {
        ninfer::ops::ldmatrix_x2_t(r[0], r[1], address);
    } else { ninfer::ops::ldmatrix_x2(r[0], r[1], address); }
    for (int m = 0; m < Matrices; ++m) { output[lane * 4 + m] = r[m]; }
}
int main() {
    try {
        H(hipSetDevice(0));
        hipDeviceProp_t prop{};
        H(hipGetDeviceProperties(&prop, 0));
        if (prop.warpSize != 32 || std::string(prop.gcnArchName).find("gfx1201") != 0) { return 77; }
        unsigned* device = nullptr;
        H(hipMalloc(&device, 32 * 16));
        std::array<unsigned, 128> observed{};
        std::size_t checked = 0;
        for (unsigned padding : {0u, 4u, 8u}) {
            for (bool transpose : {false, true}) {
                for (int matrices : {2, 4}) {
                    if (transpose && matrices == 4) { matrix_load<true, 4><<<1, 32>>>(device, padding); }
                    else if (transpose) { matrix_load<true, 2><<<1, 32>>>(device, padding); }
                    else if (matrices == 4) { matrix_load<false, 4><<<1, 32>>>(device, padding); }
                    else { matrix_load<false, 2><<<1, 32>>>(device, padding); }
                    H(hipGetLastError());
                    H(hipMemcpy(observed.data(), device, sizeof(observed), hipMemcpyDeviceToHost));
                    for (unsigned lane = 0; lane < 32; ++lane) {
                        for (int matrix = 0; matrix < matrices; ++matrix) {
                            unsigned expected = 0;
                            for (unsigned half = 0; half < 2; ++half) {
                                const unsigned row = transpose ? 2 * (lane % 4) + half : lane / 4;
                                const unsigned col = transpose ? lane / 4 : 2 * (lane % 4) + half;
                                const unsigned value = 1000 + (matrix * 128 + row * 16 + col + padding) * 7;
                                expected |= value << (16 * half);
                            }
                            if (observed[lane * 4 + matrix] != expected) {
                                throw std::runtime_error("LDS fragment coordinate mismatch");
                            }
                            ++checked;
                        }
                    }
                }
            }
        }
        unsigned char* source = nullptr;
        H(hipMalloc(&source, 32 * 16));
        std::array<unsigned char, 32 * 16> input{}, output{};
        for (std::size_t i = 0; i < input.size(); ++i) { input[i] = static_cast<unsigned char>(i * 13 + 9); }
        H(hipMemcpy(source, input.data(), input.size(), hipMemcpyHostToDevice));
        for (int bytes : {4, 8, 16}) {
            for (int valid = 0; valid <= bytes; ++valid) {
                if (bytes == 4) { copies<4><<<1,32>>>(source, reinterpret_cast<unsigned char*>(device), valid); }
                else if (bytes == 8) { copies<8><<<1,32>>>(source, reinterpret_cast<unsigned char*>(device), valid); }
                else { copies<16><<<1,32>>>(source, reinterpret_cast<unsigned char*>(device), valid); }
                H(hipGetLastError());
                H(hipMemcpy(output.data(), device, 32 * bytes, hipMemcpyDeviceToHost));
                for (int i = 0; i < 32 * bytes; ++i) {
                    const unsigned char expected = i % bytes < valid ? input[i] : 0;
                    if (output[i] != expected) { throw std::runtime_error("zero-fill or LDS visibility mismatch"); }
                    ++checked;
                }
            }
        }
        H(hipFree(source)); H(hipFree(device));
        std::cout << "PASS native LDS fragments and zero-filled copies; exact elements=" << checked << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
