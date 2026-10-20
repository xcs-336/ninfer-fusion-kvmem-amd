"""CPU-only, serial semantic checks for each actual auxiliary production TU."""
import argparse
import json
from pathlib import Path
import re
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--source-view', required=True, type=Path)
parser.add_argument('--engine', required=True, type=Path)
parser.add_argument('--compiler', required=True, type=Path)
parser.add_argument('--sdk', required=True, type=Path)
parser.add_argument('--logs', required=True, type=Path)
parser.add_argument('--timeout', type=float, default=120)
parser.add_argument('--tests', action='store_true')
args = parser.parse_args()
view = args.source_view.resolve(strict=True)
engine = args.engine.resolve(strict=True)
sdk = args.sdk.resolve(strict=True)
logs = args.logs.resolve()
logs.mkdir(parents=True, exist_ok=True)
manifest = json.loads((view / 'auxiliary-manifest.json').read_text(encoding='utf-8'))
sources = [path for paths in manifest['families'].values() for path in paths]
if args.tests:
    sources += ['tests/ops/test_candidate_selector.cpp',
                'tests/ops/test_dynamic_grouped_conv_prepare.cpp',
                'tests/ops/test_linear_dynamic_grouped_conv_add.cpp',
                'tests/ops/test_sparse_moe.cpp', 'tests/ops/test_sparse_moe_route_network.cu']
common = ['-fsyntax-only', '-std=c++20', '-ferror-limit=8', '-D__HIP_PLATFORM_AMD__=1',
          '-DNINFER_HIP=1', '-DNOMINMAX', '-DWIN32_LEAN_AND_MEAN',
          '-Wno-deprecated-declarations', '-Wno-unused-result']
for path in (sdk / 'include', view / 'src', view / 'include', view / 'tests',
             engine / 'src', engine / 'include', view / 'src/ops/kvmem',
             engine / 'src/ops/kvmem'):
    common += ['-I', str(path)]
device = ['-x', 'hip', '--cuda-device-only', '--offload-arch=gfx1201', '-mno-wavefrontsize64',
          f'--rocm-path={sdk}', f'--rocm-device-lib-path={sdk}/lib/llvm/amdgcn/bitcode',
          '-include', str(view / 'src/core/hip_intrinsics.h')]
records = []
for relative in sources:
    command = [str(args.compiler.resolve()), *common]
    if relative.endswith('.cu'):
        command += device
    command.append(str(view / relative))
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                encoding='utf-8', errors='replace', timeout=args.timeout)
        status, content = result.returncode, result.stdout
    except subprocess.TimeoutExpired as error:
        status = 124
        content = str(error)
    log = logs / (relative.replace('/', '__') + '.log')
    log.write_text(content, encoding='utf-8')
    records.append({'source': relative, 'returncode': status, 'log': str(log)})
    print(('PASS ' if status == 0 else 'FAIL ') + relative, flush=True)
    if status:
        for line in content.splitlines():
            if re.search(r'error:|fatal error:', line):
                print(line, flush=True)
result_path = logs / 'result.json'
result_path.write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
print(f'{sum(record["returncode"] == 0 for record in records)}/{len(records)} syntax checks passed')
raise SystemExit(any(record['returncode'] != 0 for record in records))
