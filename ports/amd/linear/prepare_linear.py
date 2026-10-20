"""Prepare actual Linear sources for native gfx1201, retaining public dispatch and format math.

This view consumes the common HIP translation first. Its edits are platform mechanisms only:
native loads, a native LDS/WMMA replacement for BF16 TMA, native FP4 codec operations, and an
ordinary LDS/WMMA implementation of the NVFP4 A4 profile without NVIDIA-only TMA routes.
Stored weights keep their original packed representation. Public A4 numerical qualification
remains distinct from qualification of the underlying native matrix primitive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def definition(text: str, pattern: str, replacement: str) -> str:
    """Replace exactly one known definition, matching braces rather than a multiline wildcard."""
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    if len(matches) != 1:
        raise ValueError(f"expected one definition for {pattern!r}, found {len(matches)}")
    match = matches[0]
    opening = text.find("{", match.start())
    depth = 1
    position = opening + 1
    while depth and position < len(text):
        depth += (text[position] == "{") - (text[position] == "}")
        position += 1
    if depth:
        raise ValueError(f"unclosed definition for {pattern!r}")
    return text[:match.start()] + replacement + text[position:]


def body(text: str, pattern: str, replacement: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing function {pattern!r}")
    opening = text.find("{", match.start())
    # Keep the exact original declaration, replacing only its implementation.
    return definition(text, pattern, text[match.start():opening + 1] + "\n" + replacement + "\n}")


def native_global_loads(text: str) -> str:
    # These loads express cache intent and byte width, not format conversion. Native HIP vector
    # loads preserve byte interpretation; the CUDA cache modifier is not a performance claim here.
    begin = 0
    while True:
        start = text.find('asm volatile("ld.global.cg', begin)
        if start < 0:
            return text
        end = text.find(");", start) + 2
        assembly = text[start:end]
        outputs = re.findall(r'"=r"\(([^)]+)\)', assembly)
        pointer = re.search(r'"l"\(([^)]+)\)', assembly)
        if pointer is None or len(outputs) not in (1, 2, 4):
            raise ValueError(f"unsupported global load: {assembly}")
        if len(outputs) == 1:
            replacement = f"{outputs[0]} = load_vec<unsigned>({pointer[1]});"
        else:
            assignments = " ".join(f"{target} = native_words.{component};"
                                   for target, component in zip(outputs, "xyzw"))
            replacement = (f"{{ const auto native_words = load_vec<uint{len(outputs)}>("
                           f"{pointer[1]}); {assignments} }}")
        text = text[:start] + replacement + text[end:]
        begin = start + len(replacement)


def nvfp4_native_paths(text: str, relative: str) -> str:
    """Keep the original A4 math on native WMMA; TMA selection becomes ordinary LDS staging."""
    if "/nvfp4/shapes/" in relative:
        # The largest 128x128x256 two-stage tile stages 73,728 bytes. K128 keeps the same
        # scale-group/matrix mathematics and fits 36,864 bytes in RDNA4's 64 KiB workgroup LDS.
        text = re.sub(r"(Nvfp4(?:W4a4|A4)MmaSchedule<\s*\d+\s*,\s*\d+\s*,\s*)256(\s*,)",
                      r"\g<1>128\2", text)
        # Both tables already finish with ordinary A4 MMA routes. Large T continues through those
        # routes and their matching row-major scale layout, instead of selecting a TMA descriptor.
        text = re.sub(r"\s*if\s*\(tokens\s*>=\s*\d+\)\s*return\s+nvfp4_a4_tma_route<[^;]+;",
                      "", text)
    elif relative.endswith("nvfp4_launch.cuh"):
        text = text.replace('#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"\n', "")
        text = definition(text, r"template <Nvfp4GeometryId Geometry>\s*void\s+launch_nvfp4_a4_tma\([^{}]+\)\s*\{", "")
        text = definition(text, r"template <Nvfp4GeometryId Geometry>\s*constexpr\s+Nvfp4A4Route\s+nvfp4_a4_tma_route\(\)\s*\{", "")
    elif relative.endswith("nvfp4_launch_unified.cuh"):
        text = text.replace('#include "ops/linear/nvfp4/nvfp4_a4_tma_launch.h"\n', "")
        text = definition(text, r"template <Nvfp4GeometryId Geometry, Nvfp4ScaleLayout Layout>\s*void\s+nvfp4_linear_a4_tma\([^{}]+\)\s*\{", "")
        text = definition(text, r"template <Nvfp4GeometryId Geometry, Nvfp4ScaleLayout Layout = Nvfp4ScaleLayout::Tiled256>\s*constexpr\s+Nvfp4A4Route\s+nvfp4_a4_tma_route\(\)\s*\{", "")
    elif relative.endswith("nvfp4_schedule.cuh"):
        first = text.index("template <int BlockTokens, int Stages, int MinBlocksPerSm,\n          CUtensorMapL2promotion")
        next_definition = text.index("enum class Nvfp4MmaFragmentPipeline", first)
        text = text[:first] + text[next_definition:]
    elif relative.endswith("nvfp4_w4a4_mma.cuh"):
        # The original static storage is reused for the output after its last K consumer. With
        # K128 a one-stage tile can have less staging than output. Allocate the actual maximum
        # through a union instead of weakening the kernel's existing output-capacity assertion.
        pattern = r"struct Nvfp4W4a4SharedStorage \{(?P<fields>.*?)\n\};"
        match = re.search(pattern, text, re.DOTALL)
        if match is None:
            raise ValueError("missing original NVFP4 W4A4 staging layout")
        storage = ("union Nvfp4W4a4SharedStorage {\n    struct Staging {" + match["fields"] +
                   "\n    } staged;\n    alignas(16) std::uint8_t output[Schedule::kBlockM * "
                   "(Schedule::kBlockN + 8) * sizeof(__hip_bfloat16)];\n};")
        text = text[:match.start()] + storage + text[match.end():]
        for field in ("a_codes", "b_codes", "a_scale4", "b_scales"):
            text = text.replace(f"shared.{field}", f"shared.staged.{field}")
    return text


def transform(relative: str, text: str) -> str:
    text = native_global_loads(text)
    if "/bf16/" in relative:
        # These two original [256,5120] schedules request 98,304 bytes of dynamic LDS. RDNA4's
        # per-workgroup limit is 65,536 bytes; retain the same production mathematics with half
        # the K partition/stage width (49,152 bytes), rather than asking HIP for impossible opt-in.
        text = text.replace("Bf16N256K5120MmaSchedule<16, 8>",
                            "Bf16N256K5120MmaSchedule<8, 8>")
        text = text.replace("Bf16A16MmaR32T32K256S3", "AmdBf16A16MmaR32T32K128S3")
        text = text.replace("Bf16A16MmaSchedule<32, 32, 256, 16, 8, 3, 1,",
                            "Bf16A16MmaSchedule<32, 32, 128, 16, 8, 3, 1,")
    if relative == "src/ops/common/rowsplit_a8_mma.cuh":
        text = text.replace('#include "core/tensor.h"', '#include "core/tensor.h"\n'
                            '#include "ops/common/memory.cuh"\n#include "ops/common/mma.cuh"')
        text = body(text, r"void\s+mma_s8\([^{}]+\)\s*\{",
                    "    ::ninfer::ops::mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);")
        for function, scalar in (("lds128", "uint4"), ("lds16", "unsigned short"),
                                 ("lds8", "unsigned char")):
            text = body(text, rf"\b{function}\(const void\* p\)\s*\{{",
                        f"    return ::ninfer::ops::load_vec<{scalar}>(p);")
        text = body(text, r"void\s+cp_async\(void\* smem, const void\* gmem\)\s*\{",
                    "    ::ninfer::ops::cp_async<kBytes>(smem, gmem);")
        text = text.replace('asm volatile("cp.async.commit_group;");', '::ninfer::ops::cp_commit();')
        for pending in (0, 1):
            text = text.replace(f'asm volatile("cp.async.wait_group {pending};");',
                                f'::ninfer::ops::cp_wait<{pending}>();')
    if "/fp8/" in relative:
        # The SDK has no __ldg overload for its BF16 class. A native 16-bit typed load preserves
        # the exact stored scale; the NVIDIA readonly-cache hint does not change its mathematics.
        text = text.replace("__ldg(row_scales", "load_vec<__hip_bfloat16>(row_scales")
    if relative.endswith("fp8/fp8_a16_codec.cuh"):
        # FP16 -> FP32 widening is exact. Spell the RNE BF16 cast explicitly to resolve HIP's
        # ambiguous half-class constructor overload; do not reinterpret the two formats' bits.
        for half in ("__low2half(half_pair)", "__high2half(half_pair)"):
            text = text.replace(f"__hip_bfloat16({half})", f"__float2bfloat16(__half2float({half}))")
    if relative.endswith("t2/t2_prefill_i8.cuh"):
        # HIP spells __launch_bounds__ as a two-argument macro. Parentheses protect the commas
        # inside the original dependent schedule type; its selected values remain unchanged.
        for member in ("kThreads", "kMinBlocks"):
            value = f"std::conditional_t<kPtq1, T2PrefillI8Ptq1, T2PrefillI8>::{member}"
            text = text.replace(value, f"({value})")
    if relative.endswith("t2/t2_small_t_i8.cuh"):
        # This const descriptor reads fields and invokes its inline epilogue. It never publishes
        # its parameter address between lanes, so ordinary native HIP const-value passing suffices.
        text = text.replace("const __grid_constant__ T2I8Parents<Epilogue>",
                            "const T2I8Parents<Epilogue>")
    if relative.endswith("bf16/bf16_template_launch.cuh"):
        text = text.replace('#include "ops/linear/bf16/bf16_a16_tma_mma.cuh"\n', "")
    if relative.endswith("bf16/bf16_launch_unified.cuh"):
        text = body(text, r"void\s+launch_bf16_tma_mma\([^{}]+\)\s*\{",
                    "    // RDNA4 uses the actual native A16 WMMA kernel with ordinary LDS staging.\n"
                    "    // Retile to 32x32x64, two stages (16 KiB), without TMA producer warps.\n"
                    "    using NativeTile = Bf16A16MmaSchedule<32, 32, 64, 16, 16, 2, 1,\n"
                    "        Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong,\n"
                    "        Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64>;\n"
                    "    using Native = Bf16ScheduleInstance<NativeTile, Schedule::kStaticK>;\n"
                    "    launch_bf16_a16_mma<Native>(bf16_a16_operands(x, w),\n"
                    "        LinearBf16Output{static_cast<__hip_bfloat16*>(out.data), w.n},\n"
                    "        LinearIdentityEpilogue{}, stream);")
    if relative.endswith("nvfp4/nvfp4_codec_unified.cuh"):
        text = body(text, r"pack_nvfp4_e2m1x16\([^{}]+\)\s*\{",
                    "    codes_lo = 0; codes_hi = 0;\n#pragma unroll\n"
                    "    for (int pair = 0; pair < 8; ++pair) {\n"
                    "        const __hip_fp4x2_e2m1 packed(values[pair]);\n"
                    "        if (pair < 4) codes_lo |= unsigned(packed.__x) << (pair * 8);\n"
                    "        else codes_hi |= unsigned(packed.__x) << ((pair - 4) * 8);\n    }")
    if "/nvfp4/" in relative:
        text = nvfp4_native_paths(text, relative)
    return text


def manifests(engine: Path) -> tuple[list[str], list[str]]:
    sources: list[str] = []
    seen: set[Path] = set()
    def visit(manifest: Path) -> None:
        manifest = manifest.resolve()
        if manifest in seen:
            raise ValueError(f"duplicate manifest inclusion: {manifest}")
        seen.add(manifest)
        for raw in manifest.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0]
            for name in re.findall(r'"\$\{CMAKE_CURRENT_LIST_DIR\}/([^"\n]+)"', line):
                path = (manifest.parent / name).resolve()
                path.relative_to(engine.resolve())
                if path.suffix in (".cu", ".cpp"):
                    if not path.is_file():
                        raise ValueError(f"missing original source: {path}")
                    sources.append(path.relative_to(engine).as_posix())
                elif path.name == "sources.cmake":
                    visit(path)
    visit(engine / "src/ops/linear/sources.cmake")
    if len(set(sources)) != len(sources):
        raise ValueError("a Linear source was included twice")
    excluded_tma = [name for name in sources if name.endswith(("nvfp4_a4_tma.cu", "nvfp4_w4a4_tma.cu"))]
    return sources, excluded_tma


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--source-view", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    engine, source, output = (path.resolve() for path in (args.engine, args.source_view, args.output))
    all_sources, excluded_tma = manifests(engine)
    if source != output and source in output.parents:
        raise ValueError("Linear output must not be nested in its common source view")
    changes: list[str] = []
    for subtree in ("src/ops/linear", "tests/ops/linear", "src/ops/common/rowsplit_a8_mma.cuh"):
        selected = source / subtree
        paths = [selected] if selected.is_file() else sorted(selected.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix not in (".h", ".cuh", ".cpp", ".cu"):
                continue
            relative = path.relative_to(source).as_posix()
            text = path.read_text(encoding="utf-8")
            rewritten = transform(relative, text)
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rewritten, encoding="utf-8", newline="\n")
            if rewritten != text:
                changes.append(relative)
    # GGUF's full native dependency closure already belongs to BasicOps/Gguf.cmake. The shared
    # FP8 format descriptor and shared route policy likewise have a single owner in BasicOps;
    # the latter also serves attention/materialize when the Linear family is disabled.
    reused = [name for name in all_sources if "/linear/gguf/" in name or
              name.endswith(("fp8_format.cpp", "linear/common/route_table.cpp"))]
    native = [name for name in all_sources if name not in reused and name not in excluded_tma]
    metadata = {"all_original_sources": all_sources, "native_sources": native,
                "reused_basicops_sources": reused, "unsupported_tma_sources": excluded_tma,
                "transformed_sources": changes,
                "unimplemented_profiles": ["NVIDIA TMA tensor-map transfers (native LDS/MMA routes retained)"],
                "pending_original_tests": {"test_gguf.cpp": "requires complete public LinearAdd/LinearSwiGLU production families",
                                           "test_nvfp4_tma_staging_race.cpp": "requires native TMA and fused LinearSwiGLU"},
                "public_linear_source": "src/ops/linear/linear.cpp"}
    output.mkdir(parents=True, exist_ok=True)
    (output / "linear-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"original_sources": len(all_sources), "native_sources": len(native),
                      "reused_basicops_sources": len(reused), "unsupported_tma_sources": len(excluded_tma),
                      "transformed_sources": len(changes), "manifest": str(output / "linear-manifest.json")}))


if __name__ == "__main__":
    main()
