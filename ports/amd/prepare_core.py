"""Generate native HIP Core sources at build time; keep the CUDA tree intact.

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
for symbol in ('__CUDACC__', '__CUDA_ARCH__'):
    pairs.pop(symbol, None)
pairs['cudaHostAlloc'] = 'hipHostMalloc'
headers = {'cuda.h': 'hip/hip_runtime.h', 'cuda_runtime.h': 'hip/hip_runtime.h',
           'cuda_runtime_api.h': 'hip/hip_runtime_api.h', 'cuda_bf16.h': 'hip/hip_bf16.h',
           'cuda_fp16.h': 'hip/hip_fp16.h'}
word = re.compile(r'\b[A-Za-z_][A-Za-z_0-9]*\b')

def write(relative, text):
    destination = out / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or destination.read_text(encoding='utf-8') != text:
        destination.write_text(text, encoding='utf-8', newline='\n')

sources = list((engine / 'src/core').glob('*')) + list((engine / 'tests').glob('*'))
for source in sources:
    if not source.is_file() or source.suffix not in ('.h', '.cuh', '.cpp', '.cu'):
        continue
    relative = source.relative_to(engine)
    text = source.read_text(encoding='utf-8-sig')
    for original, replacement in headers.items():
        text = text.replace(f'<{original}>', f'<{replacement}>')
    text = word.sub(lambda m: pairs.get(m[0], m[0]), text)
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

for filename in ('hip_vmm_address.h', 'hip_profiler.cpp'):
    write(Path('src/core') / filename, (port / 'core' / filename).read_text(encoding='utf-8'))
write('HIPIFY-LICENSE.txt', table[:table.index('# IMPORTANT:')])
write('generation.json', json.dumps({'sdk_hipify_sha256': hashlib.sha256(hipify.read_bytes()).hexdigest(),
                                    'mapping_count': len(pairs), 'engine': str(engine),
                                    'scope': 'Core and original tests, not Engine or production Ops'}, indent=2))
print(f'Generated native HIP Core from {engine}')
