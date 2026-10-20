"""Adapt the original attention source view to gfx1201's physical limits.

The original storage codecs, append kernels, online softmax, WMMA consumers,
masking and independent host oracles remain the production implementations.
Only private tile geometry, target route policy and execution synchronization
change here.
"""
import argparse
from pathlib import Path
import re


parser = argparse.ArgumentParser()
parser.add_argument('--generated', required=True, type=Path)
args = parser.parse_args()
generated = args.generated.resolve()
owned = Path(__file__).resolve().parent
marker = '// Native gfx1201 attention physical-policy overlay.\n'


def edit(relative, transform):
    path = generated / relative
    source = path.read_text(encoding='utf-8')
    if source.startswith(marker):
        return
    result = marker + transform(source)
    path.write_text(result, encoding='utf-8', newline='\n')


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f'Expected one source anchor: {before[:120]!r}')
    return source.replace(before, after)


prefix = 'src/ops/softmax_attention/dense/causal_cache/'


def bf16_geometry(source):
    # Each warp owns sixteen rows. Two wave32 groups retain exactly the same
    # production formula with Br=Bc=32, and Q/K/V fit in 49,152 bytes of LDS.
    source = replace_once(source, 'kCausalPromptBr        = 64', 'kCausalPromptBr        = 32')
    source = replace_once(source, 'kCausalPromptBc        = 64', 'kCausalPromptBc        = 32')
    source = replace_once(source, 'kCausalPromptThreads   = 128', 'kCausalPromptThreads   = 64')
    return source


edit(prefix + 'prompt_common.cuh', bf16_geometry)
def bf16_kernel(source):
    source = replace_once(source, 'static_assert(Threads == 128);', 'static_assert(Threads == 64);')
    # Bc no longer equals a physical page; the next half-page stays in its
    # current physical allocation. The staging helper already masks k0.
    return replace_once(source, 'block_table[kb + 1]',
                        'block_table[((kb + 1) * Bc) >> kPagedKVPageShift]')


edit(prefix + 'prompt_bf16.cuh', bf16_kernel)


def int8_geometry(source):
    source = replace_once(source, 'kCausalPromptI8Br         = 64', 'kCausalPromptI8Br         = 32')
    source = replace_once(source, 'kCausalPromptI8Bc         = 64', 'kCausalPromptI8Bc         = 32')
    source = replace_once(source, 'static_assert(kCausalPromptI8DConsumers == 4);',
                          'static_assert(kCausalPromptI8DConsumers == 8);')
    source = replace_once(source, 'static_assert(PVNtPerWarp == 8);',
                          'static_assert(PVNtPerWarp == 4);')
    source = replace_once(source, 'static_assert(kCausalPromptI8SmemBytes == 93184);',
                          'static_assert(kCausalPromptI8SmemBytes == 44544);\n'
                          'static_assert(kCausalPromptI8SmemBytes <= 65536);')
    # The original Bc64 key tile starts at page offset zero. With Bc32 every
    # second tile starts at offset32; only global cache addresses receive the
    # absolute in-page offset, while LDS continues to use its local key_l.
    source, count = re.subn(
        r'(\bkv_cache_(?:int8_quant_scale_index|int4_value_scale_index|packed_key_chunk_index|'
        r'int4_value_code_index|int8_quant_code_index)<[^;]*?,\s*)key_l(\))',
        r'\1(key & kPagedKVPageMask)\2', source)
    if count != 6:
        raise ValueError(f'Expected six INT8 prompt cache address updates, found {count}')
    return source


edit(prefix + 'prompt_i8.cuh', int8_geometry)


def subset_barrier(source, participants):
    source = '#include "ops/softmax_attention/amd_partial_barrier.cuh"\n' + source
    # All participating kernels declare their dynamic arena before any
    # producer-only branch. Initialize the separate fixed-size LDS object with
    # a whole-CTA rendezvous; it is never overlaid on live numerical storage.
    pattern = r'(    extern __shared__[^\n]+\n)'
    replacement = (r'\1    __shared__ ninfer::amd::PartialWaveBarrier producer_barrier;\n'
                   '    ninfer::amd::initialize(producer_barrier);\n')
    source, count = re.subn(pattern, replacement, source, count=1)
    if count != 1:
        raise ValueError('Expected one dynamic arena declaration')
    source, count = re.subn(r'asm volatile\("bar\.sync 1, 128;" ::: "memory"\);',
                            f'ninfer::amd::arrive_and_wait(producer_barrier, {participants});', source)
    if count != 2:
        raise ValueError('Expected two producer-only prompt barriers')
    return source


for family in ('Fp8', 'K8V4', 'Nvfp4'):
    def quantized_geometry(source, family=family):
        # FP8's original Br32/Bc32 footprint is 68,608 bytes, above gfx1201's
        # 64 KiB. One 16-row producer tile, split across two waves, and sixteen
        # dimension consumers retain the original contractions and storage.
        source = replace_once(source, f'kCausalPrompt{family}Br            = 32',
                              f'kCausalPrompt{family}Br            = 16')
        source = replace_once(source, f'static_assert(kCausalPrompt{family}DConsumers == 8);',
                              f'static_assert(kCausalPrompt{family}DConsumers == 16);\n'
                              f'static_assert(kCausalPrompt{family}SmemBytes <= 65536);')
        source = replace_once(source, 'static_assert(PVNtPerWarp == 4);',
                              'static_assert(PVNtPerWarp == 2);')
        return subset_barrier(source, 'ProducerWarps * 32')
    edit(prefix + f'prompt_{family.lower()}.cuh', quantized_geometry)


def small_int8_barrier(source):
    source = '#include "ops/softmax_attention/amd_partial_barrier.cuh"\n' + source
    anchor = '    extern __shared__ __align__(16) std::int8_t dynamic_r_s[];'
    source = replace_once(source, anchor,
                          anchor + '\n    __shared__ ninfer::amd::PartialWaveBarrier producer_barrier;\n'
                                   '    ninfer::amd::initialize(producer_barrier);')
    return replace_once(source,
                        'asm volatile("bar.sync 1, %0;\\n" ::"r"(ProducerThreads) : "memory");',
                        'ninfer::amd::arrive_and_wait(producer_barrier, ProducerThreads);')


edit(prefix + 'small_t_i8.cuh', small_int8_barrier)


def small_int8_launch(source):
    # An inherited Bc64 dynamic arena alone consumes all 64 KiB, before Q/P,
    # scales and the producer barrier. Both tile sizes are original supported
    # schedules; evaluate every window with Bc32 on this target. Split bounds
    # are computed inside the actual kernel from Bc, so their union is intact.
    source = replace_once(
        source,
        'auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena,',
        'auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int RequestedKeyBlock, bool DynamicArena,')
    source = replace_once(
        source,
        'int QkSplit = 1, bool EarlyFetch = false>() {',
        'int QkSplit = 1, bool EarlyFetch = false>() {\n'
        '        static_assert(RequestedKeyBlock == 32 || RequestedKeyBlock == 64);\n'
        '        constexpr int KeyBlock = 32; // gfx1201 private LDS schedule\n')
    start = source.index('    // A device profile names the tier per window:')
    end = source.index('\n}\n\n} // namespace small_t_i8', start)
    return (source[:start] + '''    // A new execution target starts with one resource-valid schedule,
    // not a route table measured on NVIDIA hardware. This is the original
    // production kernel for every storage family, width, mask and append mode;
    // four consumer waves per row tile give each lane eight FP32 PV outputs.
    // Both PV accumulation profiles above remain explicitly selectable.
    (void)implementation_window;
    constexpr int RowTiles = (TokenTile * Geometry::GroupSize + 15) / 16;
    launch.template operator()<4 * RowTiles, 1, 32, false>();
    CUDA_CHECK(hipGetLastError());''' + source[end:])


edit(prefix + 'small_t_i8_launch.cuh', small_int8_launch)


def small_int8_occupancy(source):
    start = source.index('int i8_partial_ctas_per_sm(')
    end = source.index('\n}\n', start) + len('\n}\n')
    return (source[:start] + '''int i8_partial_ctas_per_sm(std::int32_t tokens,
                           std::int32_t window, KvCacheStorage storage) {
    (void)tokens;
    (void)window;
    (void)storage;
    // Native initial schedule: one resource-heavy CTA per CU. This count
    // also feeds the capacity/active-split policy, so it follows the launcher.
    return 1;
}
''' + source[end:])


edit(prefix + 'small_t.cu', small_int8_occupancy)


def prompt_launch(source):
    source = replace_once(source,
                          '#include "ops/softmax_attention/dense/causal_cache/prompt_i8_fast.cuh"',
                          '// gfx1201 selects the complete native FP32-PV prompt route below.')
    source = re.sub(r'^static_assert\(kPromptWaveRows[^\n]*CausalPromptI8FastShape[^\n]*\n', '',
                    source, flags=re.M)
    start = source.index('// Both fast INT8 variants run one CTA per SM')
    end = source.index('template <typename Geometry, typename CacheView, typename Metadata>', start)
    source = source[:start] + source[end:]
    start = source.index('    fast = causal_attention_prompt_fast_kernel(cache.storage, fast);')
    end = source.index('    const Tensor& cache_k = cache.k_pages;', start)
    source = (source[:start] +
              '    // The NVIDIA fast schedule requires 84--100 KiB LDS. The hint\n'
              '    // is execution policy, so preserve the same represented inputs\n'
              '    // and full FP32-PV mathematical route within gfx1201\'s 64 KiB.\n'
              '    (void)fast;\n' + source[end:])
    start = source.index('bool causal_attention_prompt_fast_kernel(KvCacheStorage storage, bool requested) {')
    end = source.index('\n}\n', start) + len('\n}\n')
    source = (source[:start] +
              'bool causal_attention_prompt_fast_kernel(KvCacheStorage storage, bool requested) {\n'
              '    (void)storage;\n    (void)requested;\n    return false;\n}\n' + source[end:])
    return source


edit(prefix + 'prompt.cu', prompt_launch)


def exact_e8_root_arithmetic(source):
    # E8 root bytes are a public exact codec boundary. In the installed HIP
    # compiler, CUDA-spelled _rn wrappers default to ordinary expressions and
    # native sqrt unless OCML_BASIC_ROUNDED_OPERATIONS is defined before the
    # implicit runtime header. Ordinary multiply/subtract can contract into an
    # FMA, changing residual-axis tie decisions. The SDK declares sqrt/div RTE
    # functions that its shipped ocml.bc does not actually define, so do not
    # use those un-linkable declarations. The shipped sqrt_f32 lowers to exact
    # llvm.sqrt.f32, and ordinary division is correctly rounded under HIP's
    # default HIPCorrectlyRoundedDivSqrt mode (explicit in the build policy).
    # Neither operation below has afn/arcp/contract relaxation in the IR.
    for operation in ('add', 'sub', 'mul'):
        before = f'__f{operation}_rn('
        if before not in source:
            raise ValueError(f'Expected E8 root arithmetic intrinsic: {before}')
        source = source.replace(before, f'__ocml_{operation}_rte_f32(')
    source = replace_once(source, '__fsqrt_rn(', '__ocml_sqrt_f32(')
    source = replace_once(source, '__fdiv_rn(', 'kv_cache_e8_root_divide_rn(')
    anchor = '__device__ __forceinline__ KVCacheE8RootBlock kv_cache_e8_root_encode('
    source = replace_once(source, anchor,
        '''// Correctly rounded division is part of the stored E8 root code.
__device__ __forceinline__ float kv_cache_e8_root_divide_rn(float numerator, float denominator) {
#pragma clang fp contract(off)
    return numerator / denominator;
}

__device__ __forceinline__ float kv_cache_e8_root_reciprocal_rn(float value) {
    return kv_cache_e8_root_divide_rn(1.0f, value);
}

''' + anchor)
    if source.count('__frcp_rn(') != 1:
        raise ValueError('Expected one E8 root unit-direction reciprocal')
    return source.replace('__frcp_rn(', 'kv_cache_e8_root_reciprocal_rn(')


edit('src/ops/kv_cache/e8_root_codec.cuh', exact_e8_root_arithmetic)

# A CUDA register-allocation ceiling is a schedule directive, not a semantic
# value. Let AMD's compiler allocate native VGPRs; do not copy an NVIDIA count.
for path in (generated / 'src/ops/softmax_attention').rglob('*'):
    if path.suffix not in ('.cu', '.cuh'):
        continue
    source = path.read_text(encoding='utf-8')
    transformed = re.sub(r'__maxnreg__\([^)]*\)\s*', '', source)
    if transformed != source:
        path.write_text(transformed, encoding='utf-8', newline='\n')

destination = generated / 'src/ops/softmax_attention/amd_partial_barrier.cuh'
destination.write_text((owned / 'partial_barrier.cuh').read_text(encoding='utf-8'),
                       encoding='utf-8', newline='\n')
print('Prepared complete attention storage families with gfx1201 LDS geometry and subset barriers')
