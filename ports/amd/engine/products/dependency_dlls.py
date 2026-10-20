"""Resolve actual Windows import dependencies without loading their code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--readobj', required=True, type=Path)
parser.add_argument('--entry', required=True, action='append', type=Path)
parser.add_argument('--root', required=True, action='append', type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
roots = [path.resolve(strict=True) for path in args.root]
system = Path(os.environ['SystemRoot']) / 'System32'
pending = [path.resolve(strict=True) for path in args.entry]
visited = {}
while pending:
    source = pending.pop()
    key = source.name.casefold()
    if key in visited:
        if visited[key]['source'] != str(source):
            raise ValueError('Ambiguous dependency DLL identity: ' + source.name)
        continue
    result = subprocess.run([str(args.readobj.resolve(strict=True)), '--coff-imports', str(source)],
                            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    imports = re.findall(r'^\s*Name:\s*(\S+\.dll)\s*$', result.stdout, re.M | re.I)
    record = {'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'imports': imports, 'system_imports': []}
    visited[key] = record
    for name in imports:
        if name.casefold().startswith(('api-ms-', 'ext-ms-')) or (system / name).is_file():
            record['system_imports'].append(name)
            continue
        dependency = next((root / name for root in roots if (root / name).is_file()), None)
        if dependency is None:
            raise FileNotFoundError(f'Actual imported DLL {name} required by {source} is not in the configured roots')
        pending.append(dependency.resolve(strict=True))

content = json.dumps({'dlls': sorted(visited.values(), key=lambda item: item['source'].casefold())}, indent=2) + '\n'
destination = args.output.resolve()
destination.parent.mkdir(parents=True, exist_ok=True)
if not destination.is_file() or destination.read_text(encoding='utf-8') != content:
    destination.write_text(content, encoding='utf-8', newline='\n')
print(f'Actual non-system DLL dependency closure: {len(visited)}')
