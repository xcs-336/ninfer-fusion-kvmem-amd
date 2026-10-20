// Host-only native qualification. Include the ACTUAL generated production header.
// No GPU/runtime/model/old W4A8 phase-II call. No global AVX/VNNI compilation flags.
// Default: encoder, row-output rounding, exact integer rows across admitted ISA tiers.
// --forward: also full H2560/I640 sparse expert against an independent FP64 nonlinear oracle.
#include "models/qwen3_8_flash_next/impl/cpu_expert_nvfp4.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace ce = ninfer::models::qwen3_8_flash_next::detail::cpu_expert;
namespace p = ce::ported;

namespace {
struct Verdict {
    std::uint64_t checks = 0, failures = 0, encoded_values = 0, integer_outputs = 0;
    std::uint64_t rounding_outputs = 0, forward_outputs = 0, matrix_calls = 0;
    void check(bool ok, std::string_view label) {
        ++checks;
        if (!ok) {
            ++failures;
            if (failures <= 24) std::cerr << "FAIL " << label << '\n';
        }
    }
};

// Independent numerical format references: no production decoder/encoder helpers.
double bf16_value(std::uint16_t word) {
    const unsigned e = (word >> 7) & 255u, fraction = word & 127u;
    if (e == 255) {
        return fraction ? std::numeric_limits<double>::quiet_NaN()
                        : std::copysign(std::numeric_limits<double>::infinity(),
                                        (word & 0x8000u) ? -1.0 : 1.0);
    }
    const double magnitude = e ? std::ldexp(double(128u + fraction), int(e) - 134)
                               : std::ldexp(double(fraction), -133);
    return std::copysign(magnitude, (word & 0x8000u) ? -1.0 : 1.0);
}

std::uint16_t bf16_round(double value) {
    if (std::isnan(value)) return 0x7fc0u;
    const unsigned sign = std::signbit(value) ? 0x8000u : 0u;
    const double a = std::fabs(value);
    if (std::isinf(a)) return std::uint16_t(sign | 0x7f80u);
    if (a == 0) return std::uint16_t(sign);
    if (a < std::ldexp(1.0, -126)) {
        const unsigned m = unsigned(std::nearbyint(std::ldexp(a, 133)));
        return std::uint16_t(sign | m);  // m==128 carries into minimum normal.
    }
    int exponent = 0;
    const double f = std::frexp(a, &exponent);
    unsigned m = unsigned(std::nearbyint(std::ldexp(f, 8)));
    if (m == 256) { m = 128; ++exponent; }
    const int field = exponent + 126;
    if (field >= 255) return std::uint16_t(sign | 0x7f80u);
    return std::uint16_t(sign | (unsigned(field) << 7) | (m - 128));
}

int exponent_reference(const std::uint16_t* words, int count) {
    double largest = 0;
    for (int i = 0; i < count; ++i) {
        const double value = bf16_value(words[i]);
        if (!std::isfinite(value)) return -1;
        largest = std::max(largest, std::fabs(value));
    }
    return largest == 0 ? 1 : std::max(1, std::ilogb(largest) + 127);
}

std::int32_t x_reference(std::uint16_t word, int emax) {
    return std::int32_t(std::nearbyint(std::ldexp(bf16_value(word), 147 - emax)));
}

int doubled_e2m1(unsigned code) {
    constexpr int magnitudes[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    return (code & 8u) ? -magnitudes[code & 7u] : magnitudes[code & 7u];
}

int q9_reference(unsigned code) {
    // Production A16 scale contract is nonnegative finite E4M3FN: 0x00..0x7e.
    if (code > 126) throw std::invalid_argument("inadmissible A16 fixture scale");
    static const std::array<int, 127> table = [] {
        std::array<int, 127> out{};
        for (unsigned word = 0; word < out.size(); ++word) {
            const unsigned e = word / 8, m = word % 8;
            const double value = e ? std::ldexp(1.0 + double(m) / 8.0, int(e) - 7)
                                   : std::ldexp(double(m), -9);
            out[word] = int(std::nearbyint(value * 512.0));
        }
        return out;
    }();
    return table[code];
}

std::uint16_t row_output_reference(std::int64_t sum, int exponent, float multiplier) {
    if (exponent < 0) return 0x7fc0u;
    // The declared contract rounds int64 to FP32, rescales by one exact power,
    // then rounds one FP32 multiply. No production split-exponent/bit codec.
    const float integer_rounded = static_cast<float>(sum);
    const float scaled = std::scalbn(integer_rounded, exponent - 157);
    const float product = scaled * multiplier;
    return bf16_round(double(product));
}

void check_block(Verdict& v, const p::A16Block& block, const std::uint16_t* words, int exponent) {
    int hi_sum = 0;
    for (int j = 0; j < 16; ++j) {
        const int x = exponent < 0 ? 0 : x_reference(words[j], exponent);
        const int hi = int(std::floor(double(x) / 65536.0));
        const unsigned low = unsigned(x - 65536 * hi);
        const bool ok = block.lo[j] == low % 256 && block.mid[j] == low / 256 && block.hi[j] == hi;
        if (!ok && v.failures < 24) std::cerr << "block word=" << words[j] << " emax=" << exponent
            << " lane=" << j << " X=" << x << " actual_lo_mid_hi=" << unsigned(block.lo[j]) << ','
            << unsigned(block.mid[j]) << ',' << int(block.hi[j]) << '\n';
        v.check(ok, "independent numerical BF16->A16 block");
        hi_sum += hi;
        ++v.encoded_values;
    }
    v.check(block.hi_sum == hi_sum, "A16 hi_sum");
}

void encoding_tests(Verdict& v) {
    // Every BF16 word, with valid emax offsets straddling RNE and zero cutoffs.
    constexpr int gaps[] = {0, 1, 13, 14, 20, 21, 22};
    std::array<std::uint16_t, 16> words{};
    for (unsigned bits = 0; bits < 65536; ++bits) {
        words.fill(std::uint16_t(bits));
        const auto saved = words;
        const int minimum = exponent_reference(words.data(), 16);
        v.check(p::a16_column_exponent(words.data(), 16) == minimum, "column exponent all BF16 words");
        if (minimum < 0) {
            check_block(v, p::a16_block(words.data(), -1), words.data(), -1);
        } else {
            for (int gap : gaps) {
                if (minimum + gap > 254) continue;
                const int exponent = minimum + gap;
                const int got = p::a16_value(std::uint16_t(bits), exponent);
                const int expected = x_reference(std::uint16_t(bits), exponent);
                if (got != expected && v.failures < 24) std::cerr << "encode word=" << bits << " emax="
                    << exponent << " got=" << got << " expected=" << expected << '\n';
                v.check(got == expected, "a16_value exhaustive numerical RNE");
                check_block(v, p::a16_block(words.data(), exponent), words.data(), exponent);
            }
        }
        v.check(words == saved, "encoder input preservation");
    }
    // Mixed signs/exponents test a shared column emax and nonfinite anywhere.
    for (int e = 1; e <= 254; ++e) {
        for (int j = 0; j < 16; ++j) {
            const int field = std::max(0, e - j * 2);
            words[j] = std::uint16_t((j & 1 ? 0x8000 : 0) | (field << 7) | ((j * 17) & 127));
        }
        const int exponent = exponent_reference(words.data(), 16);
        v.check(p::a16_column_exponent(words.data(), 16) == exponent, "mixed-column exponent");
        check_block(v, p::a16_block(words.data(), exponent), words.data(), exponent);
    }
    constexpr std::int64_t sums[] = {0, 1, -1, 127, -127, 16777215, 16777216, 16777217,
        -16777217, (std::int64_t{1} << 54) - 1, -(std::int64_t{1} << 54) + 1,
        std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::int64_t>::min()};
    constexpr int exponents[] = {-1, 1, 2, 32, 126, 127, 128, 150, 200, 254};
    constexpr float multipliers[] = {0.25f, 0.4f, 1.0f, 2.0f, 7.25f};
    for (auto sum : sums) for (int exponent : exponents) for (float multiplier : multipliers) {
        const auto got = p::a16_row_output(sum, exponent, multiplier);
        const auto expected = row_output_reference(sum, exponent, multiplier);
        if (got != expected && v.failures < 24) std::cerr << "row_output S=" << sum << " emax=" << exponent
            << " multiplier=" << multiplier << " got=" << got << " expected=" << expected << '\n';
        v.check(got == expected, "A16 row output independent numeric BF16 RNE");
        ++v.rounding_outputs;
    }
}

enum class Pattern { Dense, SparseGateUp, SparseDown };
struct Matrix {
    static constexpr std::size_t guard = 64;
    int rows, columns, seed;
    Pattern pattern;
    std::vector<std::uint8_t> codes, scales, saved_codes, saved_scales;
    std::array<bool, 16> code_coverage{};
    std::array<bool, 127> scale_coverage{};
    std::array<bool, 16 * 127> pair_coverage{};
    Matrix(int n, int k, int s, Pattern kind = Pattern::Dense)
        : rows(n), columns(k), seed(s), pattern(kind),
          codes(guard * 2 + std::size_t(n) * k / 2, 0xa7),
          scales(guard * 2 + std::size_t(n) * k / 16, 0xb9) {
        if (n % 128 || k % 64) throw std::invalid_argument("NVFP4 fixture shape");
        for (int row = 0; row < rows; ++row) {
            for (int column = 0; column < columns; column += 2) {
                const auto lo = code(row, column), hi = code(row, column + 1);
                codes[guard + std::size_t(row) * (columns / 2) + column / 2] = std::uint8_t(lo | (hi << 4));
                code_coverage[lo] = code_coverage[hi] = true;
                pair_coverage[16 * scale(row, column / 16) + lo] = true;
                pair_coverage[16 * scale(row, column / 16) + hi] = true;
            }
        }
        // Inverse storage-order enumeration, not the production scale-index formula.
        std::size_t position = guard;
        for (int mt = 0; mt < rows / 128; ++mt)
            for (int kt = 0; kt < columns / 64; ++kt)
                for (int lane = 0; lane < 32; ++lane)
                    for (int quarter = 0; quarter < 4; ++quarter)
                        for (int group = 0; group < 4; ++group) {
                            const unsigned value = scale(128 * mt + 32 * quarter + lane, 4 * kt + group);
                            scales[position++] = std::uint8_t(value);
                            scale_coverage[value] = true;
                        }
        if (position != scales.size() - guard) throw std::logic_error("scale fixture extent");
        saved_codes = codes;
        saved_scales = scales;
    }
    unsigned code(int row, int column) const {
        if (pattern == Pattern::SparseGateUp) {
            const int half = rows / 2, i = row % half;
            if (column != i + (row >= half ? 1 : 0)) return 0;
            return row < half ? 2u : (i % 3 == 0 ? 9u : 1u); // g:1, u:+/-0.5.
        }
        if (pattern == Pattern::SparseDown) {
            if (column != (row * 13) % columns) return 0;
            return row & 1 ? 10u : 2u; // +/-1.
        }
        return unsigned(column + 3 * row + 5 * (column / 16) + 7 * (row / 128) + seed) & 15u;
    }
    unsigned scale(int row, int group) const {
        return pattern == Pattern::Dense ? unsigned(row + 29 * group + seed) % 127u : 0x38u;
    }
    const std::uint8_t* code_data() const { return codes.data() + guard; }
    const std::uint8_t* scale_data() const { return scales.data() + guard; }
    void preserve(Verdict& v) const {
        v.check(codes == saved_codes && scales == saved_scales, "matrix inputs and prefix/suffix guards");
    }
};

struct Activations {
    int columns, ncols, blocks;
    std::vector<std::uint16_t> words, saved_words;
    std::vector<p::A16Block> encoded;
    std::vector<unsigned char> saved_encoded;
    std::array<const p::A16Block*, 8> pointers{};
    std::array<int, 8> exponents{};
    Activations(int k, int t, Verdict& v, bool forward = false, int basis = -1)
        : columns(k), ncols(t), blocks(k / 16), words(std::size_t(k) * t),
          encoded(std::size_t(blocks) * t) {
        constexpr std::uint16_t regular[] = {0, 0x8000, 0x3f80, 0xbf80, 0x3f00, 0xbf00,
            0x3f81, 0xbf81, 0x3881, 0xb881, 0x3883, 0xb883, 0x3500, 0xb500, 0x3501, 0xb501,
            0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080};
        constexpr std::uint16_t sparse[] = {0x3f80, 0xbf80, 0x3f00, 0xbf00, 0x3e80, 0xbe80,
                                          0, 0x8000, 0x3f40, 0xbf40, 0x3e00, 0xbe00};
        for (int c = 0; c < t; ++c) {
            auto* input = words.data() + std::size_t(c) * k;
            for (int j = 0; j < k; ++j) {
                if (basis >= 0) input[j] = j == basis + c ? (c & 1 ? 0xbf80 : 0x3f80) : 0;
                else if (forward) input[j] = sparse[(j + c * 5) % std::size(sparse)];
                else if (c % 4 == 1) input[j] = j & 1 ? 0x8000 : 0;
                else if (c % 4 == 2) input[j] = std::uint16_t((j & 1 ? 0x8000 : 0) | (1 + j % 127));
                else if (c % 4 == 3) input[j] = std::uint16_t((j & 1 ? 0x8000 : 0) | (200 << 7) | (j % 128));
                else input[j] = regular[(j + c * 7) % std::size(regular)];
            }
            exponents[c] = exponent_reference(input, k);
            v.check(p::a16_column_exponent(input, k) == exponents[c], "production-shaped column exponent");
            auto* output = encoded.data() + std::size_t(c) * blocks;
            p::encode_a16_blocks(input, exponents[c], 0, blocks, output);
            for (int b = 0; b < blocks; ++b) check_block(v, output[b], input + b * 16, exponents[c]);
            pointers[c] = output;
        }
        saved_words = words;
        saved_encoded.resize(encoded.size() * sizeof(p::A16Block));
        std::memcpy(saved_encoded.data(), encoded.data(), saved_encoded.size());
    }
    void preserve(Verdict& v) const {
        v.check(words == saved_words && std::memcmp(encoded.data(), saved_encoded.data(), saved_encoded.size()) == 0,
                "BF16 inputs and encoded blocks preserved");
    }
};

const char* tier_name(p::CpuIsa tier) {
    switch (tier) {
    case p::CpuIsa::kScalar: return "scalar";
    case p::CpuIsa::kAvx2: return "avx2";
    case p::CpuIsa::kAvxVnni: return "avx-vnni";
    }
    return "invalid";
}

std::vector<p::CpuIsa> admitted_tiers(Verdict& v) {
    std::vector<p::CpuIsa> out;
    for (auto tier : {p::CpuIsa::kScalar, p::CpuIsa::kAvx2, p::CpuIsa::kAvxVnni}) {
        if (p::isa_supported(tier)) {
            out.push_back(tier);
            std::cout << "TIER_ADMITTED " << tier_name(tier) << '\n';
        } else std::cout << "TIER_UNAVAILABLE_NOT_QUALIFIED " << tier_name(tier) << '\n';
    }
    v.check(!out.empty() && out.front() == p::CpuIsa::kScalar, "scalar always admitted");
    v.check(p::best_isa() == out.back(), "best_isa is the highest actual admitted tier");
    return out;
}

void integer_case(Verdict& v, const Matrix& m, bool gate_up, const Activations& a,
                  int begin, int end, const std::vector<p::CpuIsa>& tiers, bool red_arms = false) {
    const std::size_t count = std::size_t(end - begin) * 16 * a.ncols;
    std::vector<std::int64_t> expected(count);
    std::vector<std::int32_t> represented_x(a.words.size());
    for (int c = 0; c < a.ncols; ++c) for (int k = 0; k < m.columns; ++k)
        represented_x[std::size_t(c) * m.columns + k] =
            x_reference(a.words[std::size_t(c) * m.columns + k], a.exponents[c]);
    for (int virtual_row = begin * 16; virtual_row < end * 16; ++virtual_row) {
        // Expected logical gate/up output ordering, independent of production accessors.
        const int row = gate_up ? (virtual_row % 2 ? m.rows / 2 + virtual_row / 2 : virtual_row / 2)
                                : virtual_row;
        for (int c = 0; c < a.ncols; ++c) {
            std::int64_t sum = 0;
            const auto* input = represented_x.data() + std::size_t(c) * m.columns;
            for (int group = 0; group < m.columns / 16; ++group) {
                std::int64_t dot = 0;
                for (int j = 0; j < 16; ++j) {
                    const int k = group * 16 + j;
                    dot += std::int64_t(doubled_e2m1(m.code(row, k))) * input[k];
                }
                sum += dot * q9_reference(m.scale(row, group));
            }
            expected[std::size_t(virtual_row - begin * 16) * a.ncols + c] = sum;
        }
    }
    constexpr std::int64_t sentinel = 0x6a731ce25b0948d1LL;
    constexpr std::size_t guard = 16;
    for (auto tier : tiers) for (int prefetch : {0, 3}) {
        std::vector<std::int64_t> output(count + 2 * guard, sentinel);
        p::rg_row_sums_ours(tier, gate_up, m.rows / 2, m.code_data(), m.columns / 2,
            m.scale_data(), m.columns / 64, m.columns / 16, begin, end, a.pointers.data(),
            a.ncols, output.data() + guard, 0, prefetch);
        ++v.matrix_calls;
        for (std::size_t i = 0; i < count; ++i) {
            const bool ok = output[guard + i] == expected[i];
            if (!ok && v.failures < 24) std::cerr << "context tier=" << tier_name(tier) << " N=" << m.rows
                << " K=" << m.columns << " T=" << a.ncols << " gu=" << gate_up << " rg=" << begin
                << ':' << end << " index=" << i << " got=" << output[guard + i] << " expected=" << expected[i] << '\n';
            v.check(ok, "independent exact integer rowsum");
            ++v.integer_outputs;
        }
        v.check(std::all_of(output.begin(), output.begin() + guard, [](auto x) { return x == sentinel; }) &&
                std::all_of(output.end() - guard, output.end(), [](auto x) { return x == sentinel; }), "integer output guards");
        m.preserve(v);
        a.preserve(v);
        if (red_arms && prefetch == 0) {
            for (int sabotage : {1, 2, 3}) {
                std::fill(output.begin(), output.end(), sentinel);
                p::rg_row_sums_ours(tier, gate_up, m.rows / 2, m.code_data(), m.columns / 2,
                    m.scale_data(), m.columns / 64, m.columns / 16, begin, end, a.pointers.data(),
                    a.ncols, output.data() + guard, sabotage, 0);
                v.check(!std::equal(expected.begin(), expected.end(), output.begin() + guard), "existing sabotage must fail independent oracle");
                v.check(std::all_of(output.begin(), output.begin() + guard, [](auto x) { return x == sentinel; }) &&
                        std::all_of(output.end() - guard, output.end(), [](auto x) { return x == sentinel; }), "red-arm output guards");
                m.preserve(v);
                a.preserve(v);
            }
        }
    }
}

void matrix_tests(Verdict& v, const std::vector<p::CpuIsa>& tiers) {
    for (int k : {64, 128}) {
        Matrix m(256, k, k / 64);
        v.check(std::all_of(m.code_coverage.begin(), m.code_coverage.end(), [](bool x) { return x; }), "all16 weight codes in fixture");
        v.check(std::all_of(m.scale_coverage.begin(), m.scale_coverage.end(), [](bool x) { return x; }), "all127 admitted scale bytes in fixture");
        v.check(std::all_of(m.pair_coverage.begin(), m.pair_coverage.end(), [](bool x) { return x; }), "all2032 code/scale pairs in fixture");
        for (int t = 1; t <= 8; ++t) {
            Activations a(k, t, v);
            integer_case(v, m, false, a, 0, 16, tiers, t == 3);
            integer_case(v, m, true, a, 1, 4, tiers);
            integer_case(v, m, true, a, 7, 10, tiers);
            integer_case(v, m, false, a, 15, 16, tiers);
        }
        if (k == 64) for (int basis : {0, 8}) {
            // Sixteen one-hot columns isolate every code x every admitted scale:
            // each of the first127 rows has one scale in group0 and all16 codes.
            Activations a(k, 8, v, false, basis);
            integer_case(v, m, false, a, 0, 16, tiers);
        }
    }
    // Actual production H2560/I640 matrices, full row ranges at T1 and T8.
    for (bool gu : {true, false}) {
        Matrix m(gu ? 1280 : 2560, gu ? 2560 : 640, gu ? 11 : 23);
        for (int t : {1, 8}) {
            Activations a(m.columns, t, v);
            integer_case(v, m, gu, a, 0, m.rows / 16, tiers);
        }
    }
}

void sparse_forward(Verdict& v, const std::vector<p::CpuIsa>& tiers) {
    Matrix gu(1280, 2560, 0, Pattern::SparseGateUp), down(2560, 640, 0, Pattern::SparseDown);
    constexpr float gu_divisor = 2.0f, down_divisor = 4.0f;
    for (int t : {1, 2, 8}) {
        Activations a(2560, t, v, true);
        std::vector<std::uint16_t> expected(std::size_t(t) * 2560);
        for (int c = 0; c < t; ++c) {
            std::array<std::uint16_t, 640> hidden{};
            const auto* input = a.words.data() + std::size_t(c) * 2560;
            for (int i = 0; i < 640; ++i) {
                double gate_dot = 0.0;
                gate_dot += bf16_value(input[i]) / 2.0;
                const double gate = bf16_value(bf16_round(gate_dot));
                const double up_coefficient = i % 3 == 0 ? -0.25 : 0.25;
                double up_dot = 0.0;
                up_dot += bf16_value(input[i + 1]) * up_coefficient;
                const double up = bf16_value(bf16_round(up_dot));
                hidden[i] = bf16_round((gate / (1.0 + std::exp(-gate))) * up);
            }
            for (int row = 0; row < 2560; ++row) {
                double down_dot = 0.0;
                down_dot += bf16_value(hidden[(row * 13) % 640]) * (row & 1 ? -0.25 : 0.25);
                expected[std::size_t(c) * 2560 + row] = bf16_round(down_dot);
            }
        }
        std::array<const std::uint16_t*, 8> xp{};
        for (int c = 0; c < t; ++c) xp[c] = a.words.data() + std::size_t(c) * 2560;
        for (auto tier : tiers) {
            constexpr std::size_t guard = 32;
            std::vector<std::uint16_t> output(std::size_t(t) * 2560 + 2 * guard, 0x5aad);
            std::array<std::uint16_t*, 8> yp{};
            for (int c = 0; c < t; ++c) yp[c] = output.data() + guard + std::size_t(c) * 2560;
            const bool ran = p::ported_forward_a16(tier, gu.code_data(), 1280, gu.scale_data(), gu_divisor,
                down.code_data(), 320, down.scale_data(), down_divisor, t, xp.data(), yp.data(), 0, 0);
            v.check(ran, "real whole-forward production call");
            for (std::size_t i = 0; i < expected.size(); ++i) {
                if (output[guard + i] != expected[i] && v.failures < 24) std::cerr << "forward tier=" << tier_name(tier)
                    << " T=" << t << " index=" << i << " got=" << output[guard + i] << " expected=" << expected[i] << '\n';
                v.check(output[guard + i] == expected[i], "sparse whole-forward independent FP64 nonlinear/BF16 oracle");
                ++v.forward_outputs;
            }
            v.check(std::all_of(output.begin(), output.begin() + guard, [](auto x) { return x == 0x5aad; }) &&
                    std::all_of(output.end() - guard, output.end(), [](auto x) { return x == 0x5aad; }), "forward output guards");
            gu.preserve(v); down.preserve(v); a.preserve(v);
            for (int invalid : {0, 9}) v.check(!p::ported_forward_a16(tier, gu.code_data(), 1280, gu.scale_data(), gu_divisor,
                down.code_data(), 320, down.scale_data(), down_divisor, invalid, xp.data(), yp.data(), 0, 0), "forward ncols rejects original invalid contract");
            v.check(!p::ported_forward_a16(tier, nullptr, 1280, gu.scale_data(), gu_divisor,
                down.code_data(), 320, down.scale_data(), down_divisor, t, xp.data(), yp.data(), 0, 0), "forward rejects null code face");
            for (float bad : {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN()}) {
                v.check(!p::ported_forward_a16(tier, gu.code_data(), 1280, gu.scale_data(), bad,
                    down.code_data(), 320, down.scale_data(), down_divisor, t, xp.data(), yp.data(), 0, 0), "forward rejects nonpositive/NaN divisor");
            }
            gu.preserve(v); down.preserve(v); a.preserve(v);
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    bool forward = false;
    if (argc == 2 && std::string_view(argv[1]) == "--forward") forward = true;
    else if (argc != 1) { std::cerr << "usage: ninfer_amd_cpu_expert_a16_test [--forward]\n"; return 2; }
    try {
        if (std::fegetround() != FE_TONEAREST || !std::numeric_limits<float>::is_iec559)
            throw std::runtime_error("qualification requires IEEE float/FE_TONEAREST; no rounding-mode override");
        volatile float minimum_normal = std::numeric_limits<float>::min();
        const float subnormal = minimum_normal * 0.5f;
        if (subnormal == 0 || subnormal + subnormal != minimum_normal)
            throw std::runtime_error("qualification requires gradual FP32 subnormals; no FTZ/DAZ override");
        Verdict verdict;
        const auto tiers = admitted_tiers(verdict);
        encoding_tests(verdict);
        matrix_tests(verdict, tiers);
        if (forward) sparse_forward(verdict, tiers);
        std::cout << "CPU_EXPERT_A16 scope=available_tiers_only independent_numeric_encoder=1 independent_integer_rows=1"
                  << " isolated_code_scale_pairs=2032"
                  << " encoder_values=" << verdict.encoded_values << " integer_outputs=" << verdict.integer_outputs
                  << " row_round_outputs=" << verdict.rounding_outputs << " matrix_calls=" << verdict.matrix_calls
                  << " sparse_fp64_forward=" << forward << " forward_outputs=" << verdict.forward_outputs
                  << " checks=" << verdict.checks << " failures=" << verdict.failures
                  << " verdict=" << (verdict.failures == 0 ? "PASS" : "FAIL") << '\n';
        std::cout << "SCOPE dense nonlinear/whole-model quality, throughput, service concurrency and unavailable tiers NOT_QUALIFIED\n";
        return verdict.failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL qualification exception: " << error.what() << '\n';
        return 1;
    }
}
