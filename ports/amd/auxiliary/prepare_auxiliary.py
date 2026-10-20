"""Prepare complete original auxiliary Op families for the native HIP build."""
import argparse
import json
from pathlib import Path
import re


parser = argparse.ArgumentParser()
parser.add_argument('--engine', required=True, type=Path)
parser.add_argument('--source-view', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
engine = args.engine.resolve(strict=True)
view = args.source_view.resolve(strict=True)
output = args.output.resolve()
owned = Path(__file__).resolve().parent
families = ('candidate_selector', 'dynamic_grouped_conv', 'kvmem', 'sparse_moe')
manifest = {'families': {}, 'changes': []}


def write(relative, text):
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.read_text(encoding='utf-8') != text:
        path.write_text(text, encoding='utf-8', newline='\n')


for family in families:
    directory = engine / 'src/ops' / family
    source_list = (directory / 'sources.cmake').read_text(encoding='utf-8-sig')
    sources = []
    for relative in re.findall(r'"\$\{CMAKE_CURRENT_LIST_DIR\}(/[^"\n]+\.(?:cu|cpp))"', source_list):
        # Resolve ../wrapper and any other manifest-relative path before
        # choosing its source-view location. Do not invent family-local files.
        original = (directory / relative.lstrip('/')).resolve(strict=True)
        original.relative_to(engine)
        path = original.relative_to(engine).as_posix()
        if not (view / path).is_file():
            raise FileNotFoundError(f'Missing translated original source: {path}')
        if path in sources:
            raise ValueError(f'Duplicate source in original manifest: {path}')
        sources.append(path)
        if view != output:
            write(path, (view / path).read_text(encoding='utf-8'))
    if not sources:
        raise ValueError(f'No original sources found for {family}')
    manifest['families'][family] = sources

relative = 'src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu'
source = (view / relative).read_text(encoding='utf-8')
def prefetch(match):
    pointer = match[1]
    lengths = {'shared_down_codes': 'shared_down_bytes', 'prefetch_data': 'prefetch_bytes'}
    if pointer not in lengths:
        raise ValueError(f'Unknown prefetch span owner: {pointer}')
    return (f'ninfer::amd::prefetch_for_read({pointer} + offset, '
            f'{lengths[pointer]} - offset);')


source, count = re.subn(
    r'asm volatile\("prefetch\.global\.L2 \[%0\];" ::"l"\((\w+) \+ offset\)\);',
    prefetch, source)
if count != 2:
    raise ValueError(f'Expected two real SparseMoE cache hints, found {count}')
source = '#include "ops/sparse_moe/amd_cache_prefetch.cuh"\n' + source
write(relative, source)
write('src/ops/sparse_moe/amd_cache_prefetch.cuh',
      (owned / 'cache_prefetch.cuh').read_text(encoding='utf-8'))
manifest['changes'].append({
    'mechanism': 'NVIDIA PTX L2 prefetch',
    'replacement': 'native gfx1201 s_prefetch_data, wave-coalesced 1--32 128B chunks with span-tail clamp',
    'guarantee': 'same Op arithmetic and caller-owned span; scalar/constant cache, not an explicit L2 residency guarantee',
    'source': 'https://clang.llvm.org/docs/AMDGPUBuiltinReference.html#builtin-amdgcn-s-prefetch-data',
    'sites': count,
})

# HIP's launch-bounds spelling is a macro: protect the comma in this dependent
# template expression so it remains one argument. The actual thread count and
# complete production template domain are unchanged.
relative = 'src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_partial.cu'
source = (view / relative).read_text(encoding='utf-8')
before = '__launch_bounds__(Tile<Rows, Columns>::threads, 1)'
if source.count(before) != 1:
    raise ValueError('Expected the original dependent launch-bounds expression')
write(relative, source.replace(before, '__launch_bounds__((Tile<Rows, Columns>::threads), 1)'))
manifest['changes'].append({'mechanism': 'dependent CUDA launch-bounds expression',
                            'replacement': 'parenthesized HIP macro argument',
                            'guarantee': 'identical thread count and production template domain'})
write('auxiliary-manifest.json', json.dumps(manifest, indent=2) + '\n')
print('Prepared original auxiliary families: ' + ', '.join(
    f'{name}={len(paths)}' for name, paths in manifest['families'].items()))
