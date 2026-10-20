"""Local Clang CPU ISA targets and truthful CPU/OS dispatch; math unchanged."""


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Expected one CPU-expert source anchor: ' + before[:100])
    return source.replace(before, after)


CAPABILITIES = r'''
// Clang requires explicit instruction features on the vector functions even
// with the MSVC ABI. Baseline dispatch/scalar code retains its original target.
#if defined(__clang__) && (defined(__x86_64__) || defined(_M_X64))
#define NINFER_AMD_CPU_TARGET(features) __attribute__((target(features)))
#else
#define NINFER_AMD_CPU_TARGET(features)
#endif

struct NativeCpuCapabilities {
    bool ssse3 = false;
    bool avx2 = false;
    bool avx_vnni = false;
};

#if defined(_MSC_VER)
NINFER_AMD_CPU_TARGET("xsave") inline std::uint64_t native_cpu_xcr0() noexcept {
    return _xgetbv(0);
}
#endif

inline const NativeCpuCapabilities& native_cpu_capabilities() noexcept {
    static const NativeCpuCapabilities result = [] {
        NativeCpuCapabilities out;
#if defined(_MSC_VER)
        int leaf[4]{};
        __cpuidex(leaf, 0, 0);
        const unsigned maximum = static_cast<unsigned>(leaf[0]);
        if (maximum < 1) return out;
        __cpuidex(leaf, 1, 0);
        const unsigned ecx = static_cast<unsigned>(leaf[2]);
        out.ssse3 = (ecx & (1u << 9)) != 0;
        constexpr unsigned state = (1u << 26) | (1u << 27) | (1u << 28);
        if ((ecx & state) != state) return out;
        // Never execute XGETBV before the CPU advertises XSAVE/OSXSAVE/AVX.
        if ((native_cpu_xcr0() & 6u) != 6u || maximum < 7) return out;
        __cpuidex(leaf, 7, 0);
        const unsigned subleaves = static_cast<unsigned>(leaf[0]);
        out.avx2 = out.ssse3 && (static_cast<unsigned>(leaf[1]) & (1u << 5)) != 0;
        if (out.avx2 && subleaves >= 1) {
            __cpuidex(leaf, 7, 1);
            out.avx_vnni = (static_cast<unsigned>(leaf[0]) & (1u << 4)) != 0;
        }
#endif
        return out;
    }();
    return result;
}

'''


def transform(source):
    namespace = 'namespace ninfer::models::qwen3_8_flash_next::detail::cpu_expert {\n'
    source = once(source, namespace, namespace + CAPABILITIES)

    # Keep generic diagnostic/bias entry points callable without higher ISA.
    # The scalar arms operate on the same represented signed nibble integers.
    source = once(source, 'inline __m128i expand_codes_i8(',
                  'NINFER_AMD_CPU_TARGET("ssse3") inline __m128i expand_codes_i8_ssse3(')
    anchor = '// One 16-k group: exact sum of a_u8 x w_i8 (int32; |P_g| <= 48960).'
    source = once(source, anchor, r'''inline __m128i expand_codes_i8(const std::uint8_t* codes8) noexcept {
    if (native_cpu_capabilities().ssse3) return expand_codes_i8_ssse3(codes8);
    alignas(16) std::int8_t values[16];
    for (int j = 0; j < 8; ++j) {
        values[2*j] = kE2m1X2Lut[codes8[j] & 15u];
        values[2*j+1] = kE2m1X2Lut[codes8[j] >> 4];
    }
    return _mm_load_si128(reinterpret_cast<const __m128i*>(values));
}

''' + anchor)
    source = once(source, 'inline std::int32_t dot_group_i32(',
        'NINFER_AMD_CPU_TARGET("avx2,avxvnni") inline std::int32_t dot_group_i32_avxvnni(')
    anchor = '// E4M3 byte -> q9 = value x 512, exact for every finite e4m3'
    source = once(source, anchor, r'''inline std::int32_t dot_group_i32(const std::uint8_t* act16,
                                                   const std::uint8_t* codes8) noexcept {
    if (native_cpu_capabilities().avx_vnni) return dot_group_i32_avxvnni(act16, codes8);
    std::int32_t result = 0;
    for (int j = 0; j < 8; ++j) {
        result += static_cast<std::int32_t>(act16[2*j]) * kE2m1X2Lut[codes8[j] & 15u];
        result += static_cast<std::int32_t>(act16[2*j+1]) * kE2m1X2Lut[codes8[j] >> 4];
    }
    return result;
}

''' + anchor)
    source = once(source, 'inline std::int32_t row_sum_i32_simd(',
        'NINFER_AMD_CPU_TARGET("ssse3") inline std::int32_t row_sum_i32_ssse3(')
    anchor = '// 256-entry q9 table, filled FROM e4m3_to_q9'
    source = once(source, anchor, r'''inline std::int32_t row_sum_i32_simd(const std::uint8_t* codes8) noexcept {
    if (native_cpu_capabilities().ssse3) return row_sum_i32_ssse3(codes8);
    std::int32_t result = 0;
    for (int j = 0; j < 8; ++j)
        result += kE2m1X2Lut[codes8[j] & 15u] + kE2m1X2Lut[codes8[j] >> 4];
    return result;
}

''' + anchor)

    start = source.index('[[nodiscard]] inline IsaFlags isa_flags() noexcept {')
    end = source.index('\n}\n', start) + 2
    source = source[:start] + '''[[nodiscard]] inline IsaFlags isa_flags() noexcept {
#if defined(_MSC_VER)
    const auto& caps = native_cpu_capabilities();
    IsaFlags result;
    result.avx2 = caps.avx2 ? 1 : 0;
    result.avx_vnni = caps.avx_vnni ? 1 : 0;
    return result;
#else
    return IsaFlags{};
#endif
}''' + source[end:]
    source = once(source, '    return f.avx_vnni == 1;',
                  '    return f.avx2 == 1 && f.avx_vnni == 1;')

    for signature, features in (
            ('inline __m256i dot4_avx2(', 'avx2'),
            ('inline __m256i dot4_avxvnni(', 'avx2,avxvnni'),
            ('inline __m128i dot4_avx2_128(', 'ssse3'),
            ('inline std::int32_t hsum4_epi32(', 'ssse3'),
            ('inline __m256i scales8_avx2_impl(', 'avx2'),
            ('inline void rs_ours_avxvnni(', 'avx2,avxvnni')):
        source = once(source, signature, f'NINFER_AMD_CPU_TARGET("{features}") ' + signature)
    for invocation, features in (
            ('INFERNIX_AVX2_ROW_SUMS_ORIG(rs_orig_avx2, dot4_avx2)', 'avx2'),
            ('INFERNIX_AVX2_ROW_SUMS_ORIG(rs_orig_avxvnni, dot4_avxvnni)', 'avx2,avxvnni'),
            ('INFERNIX_ROW_SUMS_OURS(rs_ours_avx2, dot4_avx2_128)', 'avx2')):
        source = once(source, invocation, f'NINFER_AMD_CPU_TARGET("{features}")\n' + invocation)
    source = once(source, '    const auto join = [](__m128i lo, __m128i hi) {',
                  '    const auto join = [](__m128i lo, __m128i hi) NINFER_AMD_CPU_TARGET("avx2") {')
    return source + '\n#undef NINFER_AMD_CPU_TARGET\n'
