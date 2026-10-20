"""Publish fully adapted HIP sources without touching unchanged compiler inputs."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True, type=Path)
parser.add_argument('--destination', required=True, type=Path)
args = parser.parse_args()
source = args.source.resolve(strict=True)
destination = args.destination.resolve()
if source == destination or source in destination.parents or destination in source.parents:
    parser.error('source and destination must be separate directories')
changed = 0
for path in source.rglob('*'):
    if not path.is_file():
        continue
    target = destination / path.relative_to(source)
    data = path.read_bytes()
    if target.is_file() and target.read_bytes() == data:
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    changed += 1
print(f'Published HIP source view: {changed} changed files')
