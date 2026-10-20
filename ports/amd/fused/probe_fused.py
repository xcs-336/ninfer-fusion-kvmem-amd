"""Sequential, isolated production-frontend checks; never launches GPU code."""
import argparse
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--engine", type=Path, required=True)
parser.add_argument("--base-view", type=Path, required=True)
parser.add_argument("--fused-view", type=Path, required=True)
parser.add_argument("--sdk", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--families", nargs="+", default=["linear_add", "linear_swiglu", "rmsnorm_rope"])
parser.add_argument("--files", nargs="*")
parser.add_argument("--after")
args = parser.parse_args()
engine, base, fused, sdk = (p.resolve() for p in (args.engine, args.base_view, args.fused_view, args.sdk))
metadata = json.loads((fused / "fused-manifest.json").read_text(encoding="utf-8"))
sources = args.files if args.files else [p for family in args.families for p in metadata["families"][family]["native_sources"]]
if args.after:
    sources = sources[sources.index(args.after) + 1:]
flags = [str(sdk / "lib/llvm/bin/clang++.exe"), "-std=c++20", "-fsyntax-only", "-D__HIP_PLATFORM_AMD__=1",
         "-DNINFER_HIP=1", "-DNOMINMAX", "-DWIN32_LEAN_AND_MEAN"]
for directory in (fused / "src", fused / "include", base / "src", base / "include", base / "tests", engine / "src", engine / "include", sdk / "include"):
    flags += ["-I" + str(directory)]
results = []
for source in sources:
    command = flags.copy()
    if source.endswith(".cu"):
        command += ["-x", "hip", "--cuda-device-only", "--offload-arch=gfx1201", "-mno-wavefrontsize64",
                    "--rocm-path=" + str(sdk), "--rocm-device-lib-path=" + str(sdk / "lib/llvm/amdgcn/bitcode"),
                    "-include", str(base / "src/core/hip_intrinsics.h")]
    command += [str(fused / source)]
    started = time.monotonic()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    entry = {"source": source, "exit_code": result.returncode, "seconds": time.monotonic() - started,
             "output": result.stdout}
    results.append(entry)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(("PASS " if result.returncode == 0 else "FAIL ") + source, flush=True)
    if result.returncode:
        print(result.stdout, flush=True)
        break
raise SystemExit(0 if results and all(item["exit_code"] == 0 for item in results) else 1)
