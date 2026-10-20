"""Serial CPU frontend checks for the actual public Engine source owners."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


parser = argparse.ArgumentParser()
parser.add_argument('--source-view', required=True, type=Path)
parser.add_argument('--base-view', required=True, type=Path)
parser.add_argument('--engine', required=True, type=Path)
parser.add_argument('--compiler', required=True, type=Path)
parser.add_argument('--sdk', required=True, type=Path)
parser.add_argument('--ffmpeg', required=True, type=Path)
parser.add_argument('--logs', required=True, type=Path)
parser.add_argument('--components', nargs='*')
parser.add_argument('--tests', action='store_true')
parser.add_argument('--host-only', action='store_true')
parser.add_argument('--device-only', action='store_true')
parser.add_argument('--filter', default='')
parser.add_argument('--timeout', type=float, default=120)
parser.add_argument('--stop-on-error', action='store_true')
args = parser.parse_args()
view = args.source_view.resolve(strict=True)
base = args.base_view.resolve(strict=True)
engine = args.engine.resolve(strict=True)
sdk = args.sdk.resolve(strict=True)
logs = args.logs.resolve()
logs.mkdir(parents=True, exist_ok=True)
manifest = json.loads((view / 'engine-manifest.json').read_text(encoding='utf-8'))
sources = []
for name in args.components or manifest['components']:
    if name != 'media_native_png':
        sources += manifest['components'][name]
if args.tests:
    sources += [source for record in manifest['tests'].values() for source in record['sources']]
sources = list(dict.fromkeys(sources))
if args.host_only:
    sources = [source for source in sources if not source.endswith('.cu')]
if args.device_only:
    sources = [source for source in sources if source.endswith('.cu')]
if args.filter:
    sources = [source for source in sources if re.search(args.filter, source)]
common = ['-fsyntax-only', '-std=c++20', '-ferror-limit=8', '-D__HIP_PLATFORM_AMD__=1',
          '-DNINFER_HIP=1', '-DNOMINMAX', '-DWIN32_LEAN_AND_MEAN', '-DUTF8PROC_STATIC',
          '-Wno-deprecated-declarations', '-Wno-unused-result',
          '-DNINFER_SOURCE_DIR="' + engine.as_posix() + '"',
          '-DNINFER_PYTHON_EXECUTABLE="' + Path(__import__('sys').executable).as_posix() + '"']
for path in (sdk / 'include', view / 'src', view / 'include', view / 'tests',
             base / 'src', base / 'include', base / 'tests',
             base / 'src/models/qwen3_8_flash_next/export',
             base / 'src/models/qwen3_vision/export',
             engine / 'src', engine / 'include', engine / 'third_party',
             engine / 'third_party/llama-jinja', engine / 'third_party/xgrammar/include',
             engine / 'third_party/xgrammar/3rdparty/picojson',
             engine / 'third_party/xgrammar/3rdparty/dlpack/include',
             args.ffmpeg.resolve(strict=True) / 'include'):
    common += ['-I', str(path)]
device = ['-x', 'hip', '--cuda-device-only', '--offload-arch=gfx1201', '-mno-wavefrontsize64',
          f'--rocm-path={sdk}', f'--rocm-device-lib-path={sdk}/lib/llvm/amdgcn/bitcode',
          '-include', str(base / 'src/core/hip_intrinsics.h')]
records = []
for relative in sources:
    source = engine / relative if relative.startswith('third_party/') else view / relative
    command = [str(args.compiler.resolve()), *common]
    if relative.endswith('.cu'):
        command += device
    elif relative.endswith('.c'):
        command[0] = str(args.compiler.resolve().with_name('clang.exe'))
        command += ['-x', 'c', '-std=c11']
    command.append(str(source))
    started = time.monotonic()
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                encoding='utf-8', errors='replace', timeout=args.timeout)
        status, content = result.returncode, result.stdout
    except subprocess.TimeoutExpired as error:
        status, content = 124, str(error)
    log = logs / (relative.replace('/', '__') + '.log')
    log.write_text(content, encoding='utf-8')
    records.append({'source': relative, 'returncode': status,
                    'seconds': round(time.monotonic() - started, 3), 'log': str(log)})
    print(('PASS ' if status == 0 else 'FAIL ') + relative, flush=True)
    if status:
        for line in content.splitlines():
            if re.search(r'error:|fatal error:', line):
                print(line, flush=True)
        if args.stop_on_error:
            break
(logs / 'result.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
print(f'{sum(record["returncode"] == 0 for record in records)}/{len(records)} actual TU checks passed')
raise SystemExit(any(record['returncode'] != 0 for record in records))
