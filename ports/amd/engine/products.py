"""Original product/CLI/serving/inference-benchmark source owners for HIP."""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().with_name('products')))
import platform_metadata

parser = argparse.ArgumentParser()
parser.add_argument('--engine', required=True, type=Path)
parser.add_argument('--source-view', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
engine = args.engine.resolve(strict=True)
view = args.source_view.resolve(strict=True)
output = args.output.resolve()
manifest = {'components': {}, 'executables': {}, 'tests': {}, 'inputs': [], 'changes': []}


def write(relative, content):
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.read_text(encoding='utf-8') != content:
        path.write_text(content, encoding='utf-8', newline='\n')


if view != output:
    # Standalone source-view checks need the same adjacent private app/bench
    # headers that common HIPIFY already provides in the actual build tree.
    for folder in ('apps', 'bench'):
        for path in (view / folder).rglob('*'):
            if path.is_file() and path.suffix in ('.h', '.hpp', '.cuh'):
                write(path.relative_to(view), path.read_text(encoding='utf-8'))


def commands(owner, command):
    manifest['inputs'].append(owner)
    source = re.sub(r'(?m)#.*$', '', (engine / owner).read_text(encoding='utf-8-sig'))
    for body in re.findall(r'\b' + re.escape(command) + r'\((.*?)\)', source, re.S):
        yield [quoted or plain for quoted, plain in re.findall(r'"([^"\n]*)"|([^\s()]+)', body)]


def relative_path(value, owner, copy=False):
    parent = (engine / owner).parent
    value = value.replace('${PROJECT_SOURCE_DIR}', engine.as_posix())
    value = value.replace('${CMAKE_CURRENT_SOURCE_DIR}', parent.as_posix())
    value = value.replace('${CMAKE_CURRENT_LIST_DIR}', parent.as_posix())
    if '$' in value:
        raise ValueError('Unresolved original product source: ' + value)
    original = Path(value) if Path(value).is_absolute() else parent / value
    relative = original.resolve(strict=True).relative_to(engine).as_posix()
    if copy:
        source = view / relative
        if not source.is_file():
            raise FileNotFoundError('Common HIPIFY source-view must include apps/bench: ' + relative)
        if view != output:
            write(relative, source.read_text(encoding='utf-8'))
    return relative


def sources(values, owner):
    return [relative_path(value, owner, True) for value in values
            if re.search(r'\.(?:cpp|cc|cu|c)$', value)]


for owner, selected in [('src/product/CMakeLists.txt', {
        'ninfer_media_acquire': 'media_acquire',
        'ninfer_product_prompt_input': 'product_prompt_input',
        'ninfer_product_logging': 'product_logging'}),
        ('src/serve/CMakeLists.txt', {'ninfer_serve': 'serve'})]:
    for values in commands(owner, 'add_library'):
        if values[0] in selected:
            manifest['components'][selected[values[0]]] = sources(values[1:], owner)
# Preserve the original explicit WebUI choice and both real source owners.
manifest['webui'] = {
    'none': relative_path('webui_none.cpp', 'src/serve/CMakeLists.txt', True),
    'embed': relative_path('webui/embed.cpp', 'src/serve/CMakeLists.txt', True)}

for owner in ['apps/CMakeLists.txt', 'bench/inference/benchmarks.cmake']:
    for values in commands(owner, 'add_executable'):
        manifest['executables'][values[0]] = sources(values[1:], owner)


def add_tests(owner, selected=None):
    for values in commands(owner, 'ninfer_add_test'):
        name = values[0]
        if selected is not None and name not in selected:
            continue
        record = {'sources': [], 'libraries': [], 'arguments': [], 'include_directories': [],
                  'needs_source_dir': 'NEEDS_SOURCE_DIR' in values}
        field = None
        for value in values[1:]:
            if value in ('NEEDS_SOURCE_DIR', 'STANDALONE'):
                continue
            if value in ('SOURCES', 'LIBRARIES', 'TEST_ARGS'):
                field = {'SOURCES': 'sources', 'LIBRARIES': 'libraries', 'TEST_ARGS': 'arguments'}[value]
                continue
            if field is None:
                raise ValueError('Unknown original product test token: ' + value)
            record[field].append(relative_path(value, owner, True) if field == 'sources' else value)
        if not record['sources']:
            raise ValueError('Original product test has no sources: ' + name)
        manifest['tests'][name] = record
    for values in commands(owner, 'target_include_directories'):
        if values[0] in manifest['tests']:
            manifest['tests'][values[0]]['include_directories'] += [relative_path(value, owner)
                for value in values[1:] if value not in ('PRIVATE', 'PUBLIC', 'INTERFACE')]


add_tests('tests/cmake/ProductTests.cmake')
add_tests('tests/cmake/BenchSupportTests.cmake', {'ninfer_bench_support_test'})


def edit(relative, transform):
    source = (view / relative).read_text(encoding='utf-8')
    marker = '// Native AMD product platform overlay.\n'
    if not source.startswith(marker):
        write(relative, marker + transform(source))


edit('bench/inference/ninfer_bench_support.h', platform_metadata.environment_header)
edit('bench/inference/ninfer_bench_support.cpp', platform_metadata.benchmark_support)
edit('bench/inference/ninfer_bench.cpp', platform_metadata.benchmark_main)
edit('src/serve/request_log.h', platform_metadata.environment_header)
edit('src/serve/request_log.cpp', platform_metadata.request_log)
edit('src/product/logging/startup_log.cpp', platform_metadata.startup_text)
edit('apps/calibrate/main.cpp', platform_metadata.calibrate_hardware_class)
write('src/product/amd_environment.h', (Path(__file__).with_name('products') /
      'native_environment.h').read_text(encoding='utf-8'))

# The original stamp script still generates the actual identity. Its compiled
# source roots must also include this port, which lives outside the CUDA tree.
build_id = (engine / 'cmake/GenerateBuildId.cmake').read_text(encoding='utf-8')
anchor = 'set(NINFER_PRODUCT_ROOTS src include apps cmake third_party)'
if build_id.count(anchor) != 1:
    raise ValueError('Expected one original build-id source-root authority')
build_id = build_id.replace(anchor,
    'set(NINFER_PRODUCT_ROOTS src include apps bench cmake third_party ../../ports/amd)')
write('cmake/GenerateBuildId.cmake', build_id)
manifest['inputs'].append('cmake/GenerateBuildId.cmake')
manifest['changes'] += [
    {'mechanism': 'CUDA-only product environment formatting',
     'replacement': 'actual HIP encoded versions and gcnArchName; native records omit CUDA/compute-capability fields',
     'guarantee': 'same original public Engine, CLI commands, benchmark measurements and OpenAI/Anthropic protocol semantics; original CUDA formatter fixtures untouched'},
    {'mechanism': 'NVIDIA-only calibration hardware-class formatting',
     'replacement': 'shared ninfer::amd::context_cost_hardware_class(properties.name, properties.gcnArchName)',
     'guarantee': 'same native identity as actual Engine route-profile lookup; original NVIDIA formatter/tests retained'},
    {'mechanism': 'build-time source identity',
     'replacement': 'original Git build-id script also tracks the actual compiled AMD port and benchmark roots'},
]

for name, paths in manifest['components'].items():
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Invalid original product owner: ' + name)
for name, paths in manifest['executables'].items():
    if not paths:
        raise ValueError('Original product executable has no sources: ' + name)
manifest['inputs'] = list(dict.fromkeys(manifest['inputs']))
write('products-manifest.json', json.dumps(manifest, indent=2) + '\n')
print('Original native products: ' + ', '.join(f'{name}={len(paths)}' for name, paths in manifest['components'].items()))
print('Original executables: ' + ', '.join(f'{name}={len(paths)}' for name, paths in manifest['executables'].items()))
print(f'Original product/benchmark helper tests: {len(manifest["tests"])}')
