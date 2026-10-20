"""Inspect the complete native Linear manifest sequentially, without linking or GPU execution.

Run from the selected MSVC 14.44 environment. This establishes frontend acceptance only: device
code generation, native library linkage and original mathematical tests remain separate gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", required=True, type=Path)
    parser.add_argument("--sdk", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--match", default=".")
    parser.add_argument("--timeout-seconds", default=120, type=int)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    view, sdk, output = (path.resolve() for path in (args.view, args.sdk, args.output))
    manifest = json.loads((view / "linear-manifest.json").read_text(encoding="utf-8"))
    sources = [name for name in manifest["native_sources"] if re.search(args.match, name)]
    if not sources:
        parser.error("the selected manifest subset is empty")
    output.mkdir(parents=True, exist_ok=True)
    common = [str(sdk / "lib/llvm/bin/clang++.exe"), "-std=c++20", "-fsyntax-only",
              "-D__HIP_PLATFORM_AMD__=1", "-DNINFER_HIP=1", "-DNOMINMAX", "-DWIN32_LEAN_AND_MEAN",
              f"-I{sdk / 'include'}", f"-I{view / 'src'}", f"-I{view / 'include'}",
              "-ferror-limit=5", "-Wno-deprecated-declarations"]
    results = []
    for index, name in enumerate(sources):
        arguments = list(common)
        if name.endswith(".cu"):
            arguments += ["-include", str(view / "src/core/hip_intrinsics.h"), "-x", "hip",
                          "--cuda-device-only", "--offload-arch=gfx1201", "-mno-wavefrontsize64",
                          f"--rocm-path={sdk}",
                          f"--rocm-device-lib-path={sdk / 'lib/llvm/amdgcn/bitcode'}"]
        arguments.append(str(view / name))
        log = output / (name.replace("/", "__") + ".log")
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as destination:
            try:
                completed = subprocess.run(arguments, stdin=subprocess.DEVNULL, stdout=destination,
                                           stderr=subprocess.STDOUT, timeout=args.timeout_seconds,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                status = completed.returncode
            except subprocess.TimeoutExpired:
                status = 124
        result = {"source": name, "returncode": status,
                  "seconds": round(time.monotonic() - started, 3), "log": str(log)}
        results.append(result)
        print(json.dumps({"unit": index + 1, "selected": len(sources), **result}), flush=True)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        if status != 0 and not args.keep_going:
            break
    failed = sum(result["returncode"] != 0 for result in results)
    print(json.dumps({"selected": len(sources), "inspected": len(results),
                      "failed": failed, "scope": "frontend-only; no link/codegen/GPU"}), flush=True)
    raise SystemExit(1 if failed or len(results) != len(sources) else 0)


if __name__ == "__main__":
    main()
