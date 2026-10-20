"""Generate a HIP source view and explicit qualified overrides; retain the CUDA tree.

API spelling is taken from the installed AMD HIPIFY table. Semantic changes are
explicit below. This does not translate PTX or claim the Engine is implemented.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument('--sdk', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
port = Path(__file__).resolve().parent
engine = port.parents[1] / 'src-tree/fusion-engine-src'
out = args.output.resolve()
hipify = args.sdk / 'libexec/hipify/hipify-perl'
table = hipify.read_text(encoding='utf-8')
pairs = dict(re.findall(r'\bk\("([A-Za-z_][A-Za-z_0-9]*)", "([A-Za-z_][A-Za-z_0-9]*)",', table))
pairs.pop('__CUDA_ARCH__', None)
pairs['__CUDACC__'] = '__HIPCC__'
pairs['cudaHostAlloc'] = 'hipHostMalloc'
pairs['cudaFuncSetAttribute'] = 'ninfer_hip_func_set_attribute'
pairs['__vsub4'] = 'ninfer_hip_vsub4'
pairs['__float2bfloat16_rn'] = '__float2bfloat16'
pairs['__floats2bfloat162_rn'] = 'ninfer_hip_floats2bfloat162_rn'
pairs['__float2bfloat162_rn'] = 'ninfer_hip_float2bfloat162_rn'
# NVIDIA artifacts store OCP FP8, not AMD's older FNUZ representation. gfx1201
# supports the OCP types natively; never change the exponent bias or -0 encoding.
for suffix in ('e4m3', 'e5m2'):
    for width in ('', 'x2', 'x4'):
        pairs[f'__nv_fp8{width}_{suffix}'] = f'__hip_fp8{width}_{suffix}'
pairs['__NV_E4M3'] = '__HIP_E4M3'
pairs['__NV_E5M2'] = '__HIP_E5M2'
for width in ('', 'x2', 'x4'):
    pairs[f'__nv_fp4{width}_e2m1'] = f'__hip_fp4{width}_e2m1'
pairs['__NV_E2M1'] = '__HIP_E2M1'
pairs['cub'] = 'hipcub'
for symbol in ('__shfl_sync', '__shfl_xor_sync', '__shfl_down_sync', '__shfl_up_sync',
               '__ballot_sync', '__all_sync', '__any_sync', '__syncwarp',
               '__reduce_max_sync', '__reduce_min_sync', '__reduce_add_sync',
               '__reduce_and_sync', '__reduce_or_sync', '__reduce_xor_sync'):
    pairs[symbol] = 'ninfer_hip_' + symbol.removeprefix('__')
headers = {'cuda.h': 'hip/hip_runtime.h', 'cuda_runtime.h': 'hip/hip_runtime.h',
           'cuda_runtime_api.h': 'hip/hip_runtime_api.h',
           'cuda_profiler_api.h': 'hip/hip_runtime_api.h', 'cuda_bf16.h': 'hip/hip_bf16.h',
           'cuda_fp16.h': 'hip/hip_fp16.h', 'cuda_fp8.h': 'hip/hip_fp8.h',
           'cuda_fp4.h': 'hip/hip_fp4.h',
           'math_constants.h': 'hip/hip_math_constants.h', 'cublas_v2.h': 'hipblas/hipblas.h'}
word = re.compile(r'\b[A-Za-z_][A-Za-z_0-9]*\b')

def write(relative, text):
    destination = out / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or destination.read_text(encoding='utf-8') != text:
        destination.write_text(text, encoding='utf-8', newline='\n')

sources = [path for folder in ('src', 'include', 'tests', 'apps', 'bench')
           for path in (engine / folder).rglob('*')]
for source in sources:
    if not source.is_file() or source.suffix not in ('.h', '.hpp', '.cuh', '.cpp', '.cu'):
        continue
    relative = source.relative_to(engine)
    text = source.read_text(encoding='utf-8-sig')
    for original, replacement in headers.items():
        text = text.replace(f'<{original}>', f'<{replacement}>')
    text = re.sub(r'<cub/([^>]+)\.cuh>', r'<hipcub/\1.hpp>', text)
    text = word.sub(lambda m: pairs.get(m[0], m[0]), text)
    if relative.as_posix() == 'src/core/evictable_kv_pool.h':
        anchor = '    [[nodiscard]] static bool supported(const DeviceContext& device);'
        if text.count(anchor) != 1:
            raise ValueError('expected one raw KV driver capability declaration')
        text = text.replace(anchor,
            '    // Raw driver VMM availability is insufficient for safe concurrent KV loans.\n'
            '    // This Windows/gfx1201 backend failed resident-byte isolation after\n'
            '    // noncontiguous remapping. Keep raw diagnostics available, but do not\n'
            '    // admit this lending contract into the public Engine.\n'
            '    [[nodiscard]] static constexpr bool noncontiguous_lending_qualified() noexcept {\n'
            '        return false;\n'
            '    }\n\n' + anchor)
    if relative.as_posix() == 'src/models/qwen3_5/program/program_impl.cpp':
        anchor = ('    if (pool == nullptr || !pool->mirror_captured()) {\n'
                  '        throw std::logic_error("overlay Vision weight pool has no captured window");\n'
                  '    }')
        if text.count(anchor) != 1:
            raise ValueError('expected one Vision KV arena admission point')
        text = text.replace(anchor, anchor + '\n'
            '    // A nullptr arena uses the existing exclusive weight-suffix tier;\n'
            '    // raw HIP VMM support does not qualify arbitrary KV-page aliasing.\n'
            '    if (!EvictableKVPool::noncontiguous_lending_qualified()) { return nullptr; }')
    if relative.as_posix() == 'src/ops/linear/common/route_table.cpp':
        # Shared by Attention/materialize and Linear. HIP's major/minor are not
        # NVIDIA compute capabilities, and no RX 9070 performance bands exist.
        text, count = re.subn(r'DeviceClass classify\(int device\) \{.*?\n\}',
            'DeviceClass classify(int device) {\n'
            '    (void)device;\n'
            '    return DeviceClass::Unmeasured;\n}', text, flags=re.S)
        if count != 1:
            raise ValueError('expected exactly one common route-table device classifier')
    if str(relative).replace('\\', '/') == 'src/ops/linear_attention/gated_delta_net/recurrent.cuh':
        # FoldAccess is a const value descriptor. Its methods use the fields,
        # never publish its parameter address across threads. HIP's ordinary
        # const value parameter preserves these semantics; no grid-address
        # optimization is claimed. Unqualified TMA descriptors are left alone.
        text = text.replace('const __grid_constant__ FoldAccess', 'const FoldAccess')
    if source.name == 'sampling_device.cuh':
        for scalar in ('unsigned long long', 'unsigned int'):
            text = text.replace(f'hipcub::WarpMergeSort<{scalar}, kSamplerItemsPerThread>',
                                f'hipcub::WarpMergeSort<{scalar}, kSamplerItemsPerThread, 32>')
    if str(relative).replace('\\', '/') == 'src/ops/linear/nvfp4/nvfp4_codec.cuh':
        start = text.index('__device__ __forceinline__ std::uint32_t pack_nvfp4_e2m1x8(')
        end = text.index('// Sixteen bf16 values already in registers', start)
        # Preserve E2M1 RNE/satfinite encoding and low-nibble-first order via
        # the SDK constructor already used by the original 16-value path.
        text = text[:start] + '''__device__ __forceinline__ std::uint32_t pack_nvfp4_e2m1x8(
    const float2 (&values)[4]) {
    std::uint32_t codes = 0;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        codes |= static_cast<std::uint32_t>(pack_e2m1x2(values[pair])) << (8 * pair);
    }
    return codes;
}

''' + text[end:]
    if source.name == 'ggml_bridge.cu':
        text = text.replace('facts[device].cc                     = 100 * properties.major + 10 * properties.minor;',
                            '''const std::string arch = properties.gcnArchName;
        if (!arch.starts_with("gfx1201")) { throw std::invalid_argument("gguf HIP: expected gfx1201"); }
        facts[device].cc = GGML_CUDA_CC_OFFSET_AMD + 0x1201;''')
        text = text.replace('properties.sharedMemPerBlockOptin', 'properties.sharedMemPerBlock')
    if str(relative).replace('\\', '/') == 'src/ops/common/math.cuh':
        for half in ('low', 'high'):
            text = text.replace(f'__half(__{half}2bfloat16(source))',
                                f'__float2half_rn(__bfloat162float(__{half}2bfloat16(source)))')
        text = re.sub(r'__device__ __forceinline__ float exp2_approx\(float x\) \{.*?\n\}',
                      '__device__ __forceinline__ float exp2_approx(float x) { return exp2f(x); }', text, flags=re.S)
        text = re.sub(r'__device__ __forceinline__ float log2_approx\(float x\) \{.*?\n\}',
                      '__device__ __forceinline__ float log2_approx(float x) { return log2f(x); }', text, flags=re.S)
        text = re.sub(r'__device__ __forceinline__ std::uint32_t pack_bf16x2\(float lo, float hi\) \{.*?\n\}',
                      '''__device__ __forceinline__ std::uint32_t pack_bf16x2(float lo, float hi) {
    return static_cast<std::uint32_t>(__builtin_bit_cast(unsigned short, __float2bfloat16(lo))) |
           (static_cast<std::uint32_t>(__builtin_bit_cast(unsigned short, __float2bfloat16(hi))) << 16);
}''', text, flags=re.S)
    if source.name == 'decode_graph.cpp':
        text = text.replace('hipGraphInstantiate(&exec, definition.graph_, 0)',
                            'hipGraphInstantiateWithFlags(&exec, definition.graph_, 0)')
        text = text.replace('cudaGraphExecUpdateResultInfo result{};',
                            'hipGraphExecUpdateResult result{};\n    hipGraphNode_t error_node{};')
        text = text.replace('hipGraphExecUpdate(exec_, definition.graph_, &result)',
                            'hipGraphExecUpdate(exec_, definition.graph_, &error_node, &result)')
        text = text.replace('result.result', 'result')
    if 'hipDeviceptr_t' in text:
        text = '#include "core/hip_vmm_address.h"\n' + text
        text = text.replace('hipDeviceptr_t', 'ninfer::amd::DeviceAddress')
        text = text.replace('hipMemAddressReserve(', 'ninfer::amd::reserve(')
    # HIP host callbacks use the platform default ABI on native Windows x64.
    text = text.replace('CUDART_CB ', '')
    if source.name == 'nvtx.h':
        enums = text[text.index('namespace ninfer::nvtx {'):text.index('[[nodiscard]] inline nvtxDomainHandle_t domain()')]
        names = re.search(r'(static constexpr auto names = std::to_array<const char\*>\(\{.*?\}\);)', text, re.S)[1]
        text = '''#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
''' + enums + '''
inline const char* registered_message(Name name) noexcept {
    ''' + names + '''
    static_assert(names.size() == static_cast<std::size_t>(Name::Count));
    return names[static_cast<std::size_t>(name)];
}
std::uint64_t trace_begin(const char*, Category, std::uint64_t) noexcept;
void trace_end(std::uint64_t) noexcept;
class ScopedRange {
public:
    explicit ScopedRange(Name name, Category category, std::uint64_t payload = 0) noexcept
        : id_(trace_begin(registered_message(name), category, payload)) {}
    ~ScopedRange() noexcept { trace_end(id_); }
    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
private:
    std::uint64_t id_;
};
class ScopedAsyncRange {
public:
    explicit ScopedAsyncRange(Name name, Category category, std::uint64_t payload = 0) noexcept
        : id_(trace_begin(registered_message(name), category, payload)) {}
    ~ScopedAsyncRange() noexcept { trace_end(id_); }
    ScopedAsyncRange(const ScopedAsyncRange&) = delete;
    ScopedAsyncRange& operator=(const ScopedAsyncRange&) = delete;
private:
    std::uint64_t id_;
};
} // namespace ninfer::nvtx
'''
    write(relative, text)

for filename in ('hip_vmm_address.h', 'hip_profiler.cpp', 'hip_intrinsics.h', 'pdl.cuh'):
    write(Path('src/core') / filename, (port / 'core' / filename).read_text(encoding='utf-8'))
for filename in ('memory.cuh', 'ldmatrix.cuh', 'mma.cuh', 'mma_f16_gfx12.cuh', 'mma_s8_gfx12.cuh',
                 'mma_fp8_gfx12.cuh', 'mma_nvfp4_gfx12.cuh'):
    write(Path('src/ops/common') / filename, (port / 'ops' / filename).read_text(encoding='utf-8'))
write('src/ops/common/mma_bf16_gfx12.h', (port / 'gfx12-mma/mma_bf16_gfx12.h').read_text(encoding='utf-8'))
write('HIPIFY-LICENSE.txt', table[:table.index('# IMPORTANT:')])
write('generation.json', json.dumps({'sdk_hipify_sha256': hashlib.sha256(hipify.read_bytes()).hexdigest(),
                                    'mapping_count': len(pairs), 'engine': str(engine),
                                    'scope': 'Core, source API translation and explicit HIP primitive overrides; production routes require their own qualification'}, indent=2))
print(f'Generated native HIP source view from {engine}')
