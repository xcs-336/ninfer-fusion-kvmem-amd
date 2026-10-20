"""Read the actual public Engine/component/model manifests for the HIP build.

This file owns no replacement model or Engine. Every listed compile owner uses
the original source-view TU and the original exported package header hierarchy.
"""
import argparse
import json
from pathlib import Path
import re
import sys

# The selected embedded Python 3.11 runtime does not add the script directory
# to sys.path automatically.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import moe_resources
import cpu_expert


parser = argparse.ArgumentParser()
parser.add_argument('--engine', required=True, type=Path)
parser.add_argument('--source-view', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
engine = args.engine.resolve(strict=True)
view = args.source_view.resolve(strict=True)
output = args.output.resolve()
manifest = {'components': {}, 'tests': {}, 'inputs': [], 'changes': []}


def write(relative, content):
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.read_text(encoding='utf-8') != content:
        path.write_text(content, encoding='utf-8', newline='\n')


def commands(relative, name):
    path = engine / relative
    manifest['inputs'].append(relative)
    text = path.read_text(encoding='utf-8-sig')
    # These explicit source manifests have no nested executable expressions.
    # Drop line comments before parsing commands so commented historical paths
    # cannot accidentally become active sources.
    text = re.sub(r'(?m)#.*$', '', text)
    # Preserve the original finite test matrix (e.g. all five ngram real
    # lifecycle/archive/thinking/stop/concurrent suites).
    def expand_loop(match):
        variable, values, body = match.groups()
        items = values.split()
        if any('$' in item or item in ('IN', 'RANGE') for item in items):
            # Other components' CTest-only variable loops are irrelevant here;
            # leave them intact. Any unresolved active source still fails in
            # source_path instead of being silently omitted.
            return match[0]
        return '\n'.join(body.replace('${' + variable + '}', item) for item in items)
    text = re.sub(r'foreach\(\s*(\w+)\s+([^)]*)\)(.*?)endforeach\(\)',
                  expand_loop, text, flags=re.S)
    for body in re.findall(r'\b' + re.escape(name) + r'\((.*?)\)', text, re.S):
        yield re.findall(r'"([^"\n]*)"|([^\s()]+)', body)


def tokens(command):
    return [quoted or unquoted for quoted, unquoted in command]


def source_path(token, owner):
    directory = (engine / owner).parent
    token = token.replace('${CMAKE_CURRENT_LIST_DIR}', directory.as_posix())
    token = token.replace('${CMAKE_CURRENT_SOURCE_DIR}', directory.as_posix())
    token = token.replace('${PROJECT_SOURCE_DIR}', engine.as_posix())
    if '$' in token:
        raise ValueError(f'Unresolved original source expression: {token}')
    original = Path(token) if Path(token).is_absolute() else directory / token
    original = original.resolve(strict=True)
    relative = original.relative_to(engine).as_posix()
    if not relative.startswith('third_party/'):
        if not (view / relative).is_file():
            raise FileNotFoundError(f'Missing original HIP source-view TU: {relative}')
        if view != output:
            write(relative, (view / relative).read_text(encoding='utf-8'))
    return relative


def add_sources(component, owner, values):
    paths = manifest['components'].setdefault(component, [])
    for token in values:
        if not re.search(r'\.(?:cpp|cc|cu|c)$', token):
            continue
        path = source_path(token, owner)
        if path in paths:
            raise ValueError(f'Duplicate original TU in {component}: {path}')
        paths.append(path)


def library_sources(owner, target, component):
    for command in commands(owner, 'add_library'):
        values = tokens(command)
        if values[0] == target:
            add_sources(component, owner, values[1:])


library_sources('src/text/CMakeLists.txt', 'ninfer_text', 'text')
for command in commands('src/text/CMakeLists.txt', 'target_sources'):
    values = tokens(command)
    if values[0] == 'ninfer_jinja':
        add_sources('jinja_bridge', 'src/text/CMakeLists.txt', values[1:])
library_sources('src/calibration/CMakeLists.txt', 'ninfer_calibration', 'calibration')
library_sources('src/runtime/CMakeLists.txt', 'ninfer_engine', 'engine')
library_sources('src/models/CMakeLists.txt', 'ninfer_model_loading', 'model_loading')

model_manifest = 'src/models/CMakeLists.txt'
for command in commands(model_manifest, 'include'):
    include = tokens(command)[0]
    if not include.endswith('_sources.cmake') and not include.endswith('/sources.cmake'):
        continue
    relative = (Path('src/models') / include).as_posix()
    for source_command in commands(relative, 'target_sources'):
        values = tokens(source_command)
        component = {'ninfer_model_loading': 'model_loading',
                     'ninfer_model_runtime': 'model_runtime'}[values[0]]
        # One original program source uses the src/models/ relative spelling,
        # while all the remaining entries use CMAKE_CURRENT_LIST_DIR.
        normalized = []
        for value in values[1:]:
            if value.startswith('qwen3_5/'):
                value = '${PROJECT_SOURCE_DIR}/src/models/' + value
            normalized.append(value)
        add_sources(component, relative, normalized)

# The original media owner always compiles FFmpeg decode, with an explicitly
# selectable native PNG TU. Keep that original option in Engine.cmake.
add_sources('media', 'src/media/CMakeLists.txt', ['decode/decode.cpp'])
add_sources('media_native_png', 'src/media/CMakeLists.txt', ['decode/png_decode.cpp'])
manifest['inputs'].append('src/media/CMakeLists.txt')


def add_tests(owner, selected=None):
    for command in commands(owner, 'ninfer_add_test'):
        values = tokens(command)
        name = values[0]
        if selected is not None and name not in selected:
            continue
        record = {'sources': [], 'libraries': [], 'arguments': [],
                  'needs_source_dir': 'NEEDS_SOURCE_DIR' in values,
                  'skip_return_code': None}
        field = None
        for value in values[1:]:
            if value in ('NEEDS_SOURCE_DIR', 'STANDALONE'):
                continue
            if value in ('SOURCES', 'LIBRARIES', 'TEST_ARGS'):
                field = {'SOURCES': 'sources', 'LIBRARIES': 'libraries',
                         'TEST_ARGS': 'arguments'}[value]
                continue
            if field is None:
                raise ValueError(f'Unknown original test argument: {value}')
            record[field].append(source_path(value, owner) if field == 'sources' else value)
        if not record['sources']:
            raise ValueError(f'Original test has no sources: {name}')
        manifest['tests'][name] = record
    for command in commands(owner, 'set_tests_properties'):
        values = tokens(command)
        if 'SKIP_RETURN_CODE' not in values:
            continue
        code = int(values[values.index('SKIP_RETURN_CODE') + 1])
        for name in values[:values.index('PROPERTIES')]:
            if name in manifest['tests']:
                manifest['tests'][name]['skip_return_code'] = code


add_tests('tests/models/qwen3_5/tests.cmake')
add_tests('tests/cmake/RuntimeTests.cmake',
          {'ninfer_context_cache_defaults_test', 'ninfer_sampling_defaults_test'})
add_tests('tests/cmake/CoreTests.cmake', {'ninfer_jinja_test', 'ninfer_structured_output_test'})
add_tests('tests/cmake/ProductTests.cmake', {'ninfer_media_decode_test'})


def edit(relative, transform):
    source = (view / relative).read_text(encoding='utf-8')
    marker = '// Native AMD Engine capability overlay.\n'
    if not source.startswith(marker):
        write(relative, marker + transform(source))


def integer_capability(source, expected):
    before = '#if defined(NINFER_SM8X_COMPAT)'
    if source.count(before) != expected:
        raise ValueError(f'Expected {expected} model INT8 capability gates')
    return source.replace(before, before + ' || defined(NINFER_HIP)')


edit('src/models/qwen3_5/execution/parameters.cpp', lambda source: integer_capability(source, 3))
edit('tests/models/qwen3_5/test_mlp_a8_decode_wiring.cpp', lambda source: integer_capability(source, 5))
manifest['changes'].append({
    'mechanism': 'SM8X-only compile gate around model INT8 policy preparation',
    'replacement': 'explicit NINFER_HIP capability for the actual native INT8 production routes',
    'guarantee': 'original shapes, phase policies, hints, workspace checks and all oracle assertions retained',
})


def native_batch_copy(source):
    before = '#if CUDART_VERSION >= 12080'
    if source.count(before) != 2:
        raise ValueError('Expected two original Flash-Next batch-copy capability gates')
    source = source.replace(before, '#if defined(NINFER_HIP)')
    before = '''            hipMemcpyAttributes batch_attr{};
            batch_attr.srcAccessOrder     = hipMemcpySrcAccessOrderStream;
            std::size_t batch_attr_idx[1] = {0};
            CUDA_CHECK(hipMemcpyBatchAsync(batch_dst.data(), batch_src.data(), batch_sizes.data(),
                                            batch_count, &batch_attr, batch_attr_idx, 1, stream));'''
    if source.count(before) != 1:
        raise ValueError('Expected one original Flash-Next Stream-ordered copy batch')
    source = source.replace(before,
        '''            // HIP's actual API has no attribute-array support. The sources
            // remain caller-owned until this stream completes, as in the
            // original Stream-order attribute contract.
            CUDA_CHECK(ninfer::amd::copy_batch_on_stream(batch_dst.data(), batch_src.data(),
                                                        batch_sizes.data(), batch_count, stream));''')
    # The installed runtime exports its own wait-value API. Its five-argument
    # signature has a native mask; an actual runtime failure keeps the original
    # caller's blocking-await behavior instead of querying a CUDA version.
    source, count = re.subn(r'CuStreamWaitValue32Fn cu_serve_stream_wait_value32\(\) \{.*?\n\}',
        '''CuStreamWaitValue32Fn cu_serve_stream_wait_value32() {
    static_assert(kCuStreamWaitValueEq == hipStreamWaitValueEq);
    return +[](void* stream, std::uintptr_t address, std::uint32_t value, unsigned flags) -> int {
        return static_cast<int>(hipStreamWaitValue32(reinterpret_cast<hipStream_t>(stream),
            reinterpret_cast<void*>(address), value, flags, 0xFFFFFFFFU));
    };
}''', source, flags=re.S)
    if count != 1:
        raise ValueError('Expected one original CPU-serve driver-entry resolver')
    return '#include "models/amd_memcpy_batch.h"\n' + source


edit('src/models/qwen3_8_flash_next/impl/expert_staging.cpp', native_batch_copy)
write('src/models/amd_memcpy_batch.h', (Path(__file__).parent / 'memcpy_batch.h').read_text(encoding='utf-8'))
manifest['changes'].append({
    'mechanism': 'CUDA-version-gated optional Flash-Next batch DMA',
    'replacement': 'real HIP batch API with native failure-index argument and mutable pointer metadata',
    'guarantee': 'same source/destination byte spans and stream; original per-copy default preserved; native errors remain fatal',
    'qualification': 'native runtime batch-copy and model staging execution remain unqualified until actual tests run',
})


def qsa_public_composition(source):
    function = source.index('void flash_next_qsa_attention_prefill_launch(')
    start = source.index('    const bool is_fp8 = (cache.key_pages.dtype == DType::FP8_E4M3FN);', function)
    end = source.index('    const int elements    = kQueryHeads * kHeadDim * tokens;', start)
    # This is already the original launcher's default use_mma=true route. The
    # same complete selected-block Op now owns all native schedules. Keep every
    # query preparation/cache-store/gate instruction before/after this span.
    # Chunking preserves the original private caller's whole positive token
    # domain without imposing the public Op's per-launch token cap on a model.
    native = '''    // gfx1201 composes the complete public selected-block Op. The
    // private NVIDIA 80,064-byte LDS schedule has no native qualification.
    // use_mma is a private execution hint; represented operands and every
    // observable BF16/FP8 cast remain those of the model contract.
    (void)use_mma;
    constexpr int MaxQueryTokensPerLaunch = 262144;
    for (int begin = 0; begin < tokens;) {
        const int count = std::min(MaxQueryTokensPerLaunch, tokens - begin);
        const Tensor query = scratch.query.slice(2, begin, count);
        const Tensor positions = token_indices.slice(0, begin, count);
        const Tensor selections = selected_blocks.slice(1, begin, count);
        const Tensor counts = selected_counts.slice(0, begin, count);
        Tensor attended = scratch.attended.slice(2, begin, count);
        ops::selected_block_attention(query, positions, table_row, selections, counts,
            {cache.key_pages, cache.value_pages, cache.block_tables}, attended, stream);
        begin += count;
    }
'''
    return '#include <algorithm>\n' + source[:start] + native + source[end:]


edit('src/models/qwen3_8_flash_next/impl/qsa_attention_kernels.cu', qsa_public_composition)
manifest['changes'].append({
    'mechanism': 'private Flash-Next QSA prefill schedules, including NVIDIA 80064-byte LDS schedule',
    'replacement': 'composition of the original public selected_block_attention production Op',
    'guarantee': 'same D256/H24/KV2 scale1/16, page map, selections, causal tail, storage, BF16 output and surrounding model kernels',
    'token_domain': 'caller-owned views split only at the public per-launch cap; no model token-domain narrowing',
    'qualification': 'public selected-block Op has independent FP64/readonly/guard qualification; unused private MMA schedule and whole model integration are not thereby qualified',
})

edit('src/models/qwen3_8_flash_next/impl/moe_kernels.cu', moe_resources.transform)
write('src/models/amd_moe_gate_up.h', (Path(__file__).parent / 'moe_gate_up.h').read_text(encoding='utf-8'))
manifest['changes'].append(moe_resources.report())


def native_staging_slices(source):
    before = '#if defined(NINFER_SM8X_COMPAT) && !defined(NINFER_SM120_NVFP4)'
    if source.count(before) != 1:
        raise ValueError('Expected one original Flash-Next bounded staging slice gate')
    source = source.replace(before, '''#if defined(NINFER_HIP) || (defined(NINFER_SM8X_COMPAT) && !defined(NINFER_SM120_NVFP4))
#if defined(NINFER_HIP)
    // The staging planes and per-call metadata retain their original finite
    // capacity. Resident NVFP4 banks keep the complete native MMA token domain.
    const bool needs_bounded_staging = weights.experts_gguf || weights.expert_gate_up.on_host;
#else
    constexpr bool needs_bounded_staging = true;
#endif''')
    before = '    if (tokens > slice_width) {'
    if source.count(before) != 1:
        raise ValueError('Expected one original finite token-slice reentry')
    return source.replace(before, '    if (needs_bounded_staging && tokens > slice_width) {')


edit('src/models/qwen3_8_flash_next/impl/moe.cpp', native_staging_slices)
manifest['changes'].append({
    'mechanism': 'SM8X-only finite host-tiered expert staging token reentry',
    'replacement': 'explicit native HIP staging admission only for actual on_host NVFP4 or GGUF expert banks',
    'guarantee': 'original slice_width, capacities, ordered tensor views, guards and GGUF wide-arm refusal retained; resident NVFP4 keeps the full native MMA token domain',
    'qualification': 'actual optional staging execution and full model remain pending',
})


def native_router_score_loads(source):
    for before, after in [('__ldcg(token_scores + id)', 'token_scores[id]'),
                          ('__ldcg(token_scores + kExperts)', 'token_scores[kExperts]')]:
        if source.count(before) != 1:
            raise ValueError('Expected one original readonly router score load: ' + before)
        source = source.replace(before, after)
    before = '''    // Selection: warp t handles token t. The scores are read with __ldcg (L2) so that no L1 line
    // from before the fence is consulted. No block-level barrier follows, so warps may leave.'''
    after = '''    // Selection: wave t handles token t. Native ordinary global loads follow
    // the original device-scope publication fences and atomic last-arrival ticket.
    // HIP memory ordering supplies visibility; no NVIDIA L2-hint equivalence is
    // assumed. No block-level barrier follows, so waves may leave.'''
    if source.count(before) != 1:
        raise ValueError('Expected one original router score cache-hint explanation')
    return source.replace(before, after)


edit('src/models/qwen3_8_flash_next/impl/moe_route.cu', native_router_score_loads)
manifest['changes'].append({
    'mechanism': 'two NVIDIA cache-global float loads in fused expert routing',
    'replacement': 'ordinary native global loads after the original device-scope fences and atomic last-arrival publication',
    'guarantee': 'same FP32 score bits, ticket/barrier ordering, stable rank/tie rule, softmax and sigmoid; native cache hierarchy is not NVIDIA L2 residency',
    'qualification': 'actual router and whole-model execution remain pending',
})


def native_indexer_topk(source):
    source = source.replace('#include <hipcub/device/device_topk.hpp>',
                            '#include <rocprim/device/device_topk.hpp>')
    source, includes = re.subn(r'#include <cuda/__(?:execution|stream)/[^>]+>\n', '', source)
    if includes != 4:
        raise ValueError('Expected four CUDA-only TopK execution/stream headers')
    start = source.index('auto topk_env(')
    end = source.index('std::size_t topk_temp_bytes(', start)
    source = source[:start] + '''// Native TopK keeps the original exact uint64 total-order keys.
// The original 512-item bitonic pass below still establishes final ordering.
hipError_t topk_max_pairs(void* temp, std::size_t& temp_bytes, const std::uint64_t* keys_in,
                           std::uint64_t* keys_out, const std::int32_t* values_in,
                           std::int32_t* values_out, int num_items, int k, hipStream_t stream) {
    return rocprim::topk_pairs<rocprim::default_config, true>(temp, temp_bytes,
        keys_in, keys_out, values_in, values_out, num_items, k,
        rocprim::identity_decomposer{}, stream, false);
}

''' + source[end:]
    return source


edit('src/models/qwen3_8_flash_next/impl/cpu_expert_nvfp4.h', cpu_expert.transform)
manifest['changes'].append({
    'mechanism': 'original MSVC-only CPU expert compilation assumed global intrinsic features',
    'replacement': 'Clang local SSSE3/AVX2/AVXVNNI targets plus CPUID/max-leaf/OSXSAVE/XCR0 dispatch and exact scalar diagnostic fallbacks',
    'guarantee': 'all original scalar/AVX2/AVXVNNI A16 math and dispatch tiers retained; no global ISA flag or AVX512 claim; runtime qualification pending',
})


def native_ple_scale_bits(source):
    before = '''    const __half_raw hr{scale_bits};
    const float scale = __half2float(__half(hr));'''
    if source.count(before) != 1:
        raise ValueError('Expected one original packed PLE FP16 scale bit conversion')
    return source.replace(before,
        '    const float scale = __half2float(__ushort_as_half(scale_bits));')


edit('src/models/qwen3_8_flash_next/impl/ple_decode_kernels.cu', native_ple_scale_bits)
manifest['changes'].append({
    'mechanism': 'CUDA uint16 aggregate initialization of __half_raw has a different HIP member type',
    'replacement': 'native __ushort_as_half bit reinterpretation before the original FP16-to-FP32 scale conversion',
    'guarantee': 'identical stored FP16 bits, scale groups, Q4 codes and BF16 output arithmetic; no quantization format change',
})

edit('src/models/qwen3_8_flash_next/impl/qsa_indexer_kernels.cu', native_indexer_topk)
manifest['changes'].append({
    'mechanism': 'CUB DeviceTopK plus CUDA execution environment',
    'replacement': 'real rocPRIM topk_pairs descending with the same K-output capacity and queried native workspace',
    'guarantee': 'original score/id packed total-order bits, final bitonic sort, tie policy, padding and published counts unchanged',
    'qualification': 'native runtime TopK and full QSA indexer oracle still pending',
})

manifest['inputs'] = list(dict.fromkeys(manifest['inputs']))
for name, paths in manifest['components'].items():
    if not paths:
        raise ValueError(f'Original component has no TUs: {name}')
write('engine-manifest.json', json.dumps(manifest, indent=2) + '\n')
print('Public Engine original source owners: ' + ', '.join(
    f'{name}={len(paths)}' for name, paths in manifest['components'].items()))
print(f'Original Engine/frontend/media tests: {len(manifest["tests"])}')
