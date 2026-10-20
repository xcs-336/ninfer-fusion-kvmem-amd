"""Exercise actual application parse/help exits before model or GPU creation."""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--cli', required=True, type=Path)
parser.add_argument('--bench', required=True, type=Path)
parser.add_argument('--perplexity', required=True, type=Path)
args = parser.parse_args()

cases = (
    ('cli', args.cli, ['profile-mode-test.ninfer', '--prompt', 'x'], 1),
    ('bench', args.bench, ['--weights', 'profile-mode-test.ninfer'], 2),
    ('perplexity', args.perplexity,
     ['profile-mode-test.ninfer', '--text', 'profile-mode-test.txt'], 1),
)
for name, executable, base, error_exit in cases:
    for arguments, expected, diagnostic in (
            (['--help'], 0, '--device-profile'),
            (base + ['--device-profile', 'pretend'], error_exit,
             '--device-profile must be auto, off or calibrate'),
            (base + ['--device-profile'], error_exit, '--device-profile')):
        result = subprocess.run([str(executable), *arguments], text=True,
                                encoding='utf-8', errors='replace',
                                stdin=subprocess.DEVNULL, capture_output=True, timeout=10)
        output = result.stdout + result.stderr
        if result.returncode != expected or diagnostic not in output:
            raise AssertionError(f'{name} {arguments}: expected exit {expected} and '
                                 f'{diagnostic!r}; got {result.returncode}: {output}')
print('Actual CLI/bench/perplexity profile help and rejection paths: PASS (9 cases)')
