"""Adapt production fusion families without replacing public Ops or their oracles.

Consumes the common HIP view and the actual Linear dependency closure. Only native
cache loads, explicit casts, physical tile sizes and TMA route selection change.
NVFP4's non-TMA A4 launch definitions must be provided by the Linear preparation;
the register primitive is qualified separately before any A4 route is admitted.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re

_cooperative_spec = importlib.util.spec_from_file_location(
    "ninfer_amd_fused_cooperative", Path(__file__).with_name("cooperative.py"))
cooperative = importlib.util.module_from_spec(_cooperative_spec)
assert _cooperative_spec.loader is not None
_cooperative_spec.loader.exec_module(cooperative)
_resource_spec = importlib.util.spec_from_file_location(
    "ninfer_amd_fused_resources", Path(__file__).with_name("resources.py"))
resources = importlib.util.module_from_spec(_resource_spec)
assert _resource_spec.loader is not None
_resource_spec.loader.exec_module(resources)

FAMILIES = ("linear_add", "linear_swiglu", "gdn_input_proj", "gdn_gating_proj",
            "attn_input_proj", "linear_pair", "linear_topk", "rmsnorm_rope")
MARKER = "// Native gfx1201 fusion physical-policy overlay.\n"
_linear_path = Path(__file__).resolve().parents[1] / "linear" / "prepare_linear.py"
_spec = importlib.util.spec_from_file_location("ninfer_amd_linear_helpers", _linear_path)
_linear = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_linear)


def remove_definition(text: str, pattern: str) -> str:
    return _linear.definition(text, pattern, "")


def remove_tma_branch(text: str, predicate: str) -> str:
    pattern = rf"if\s*\(\s*{re.escape(predicate)}\s*\)\s*\{{"
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1: raise ValueError(f"expected one TMA branch for {predicate}")
    match = matches[0]
    opening = text.index("{", match.start())
    depth, end = 1, opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    tail = text[end:]
    # Some resolvers express their ordinary native cases as `else if` after
    # the TMA arm. Keep that complete chain as the new first native branch.
    following_else = re.match(r"(\s*)else\b\s*", tail)
    if following_else:
        tail = following_else[1] + tail[following_else.end():]
    return text[:match.start()] + tail


def transform(relative: str, text: str) -> str:
    text = cooperative.transform(relative, text, _linear)
    text = resources.transform(relative, text, _linear)
    if relative.endswith("rmsnorm_rope/d256.cuh"):
        before = "return ninfer_hip_floats2bfloat162_rn(second.x * c0 + first.x * s0, second.y * c1 + first.y * s1);"
        if text.count(before) != 1: raise ValueError("missing D256 second-half rotary contract")
        # The split RoPE's two-output helper reuses first*cos - second*sin;
        # Clang's exact actual O3 IR consequently orders its second-half add
        # as first*sin + second*cos. The fused lane-select form ordered it
        # oppositely, allowing a different FMA and rare BF16 tie crossings.
        # Select the split path's one-rounding FMA explicitly, without changing
        # normalization, coefficient generation or either output oracle.
        text = text.replace(before, """// Match the split RoPE instruction contract: round second*cos,
    // then one native FMA for first*sin plus that product.
    return ninfer_hip_floats2bfloat162_rn(fmaf(first.x, s0, second.x * c0),
                                       fmaf(first.y, s1, second.y * c1));""")
    if "/linear_topk/" in relative:
        # HIP exposes launch bounds as a macro; protect template commas while
        # retaining the original exact thread-count expression.
        text = re.sub(r"__launch_bounds__\((M64Schedule<[^>]+>::kThreads),",
                      r"__launch_bounds__((\1),", text)
        if relative.endswith("_m64.cu") or relative.endswith("_m64_unified.cu"):
            before = "    if constexpr (shared_bytes > 48 * 1024) {"
            if text.count(before) != 1: raise ValueError("missing M64 native LDS capacity anchor")
            text = text.replace(before, '''    static_assert(shared_bytes <= 64 * 1024,
        "Native gfx1201 TopK tile must fit the actual 64-KiB LDS limit");
''' + before)
    text = _linear.native_global_loads(text)
    if relative.endswith("q4a8/q4a8_linear_swiglu.cu"):
        text = text.replace('#include "core/device.h"', '#include "core/device.h"\n#include "ops/common/mma.cuh"\n#include "ops/common/memory.cuh"')
        text = _linear.body(text, r"__device__ __forceinline__ void mma_s8\([^{}]+\)\s*\{",
                            "    ::ninfer::ops::mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);")
        text = _linear.body(text, r"__device__ __forceinline__ uint4 lds128\([^{}]+\)\s*\{",
                            "    return ::ninfer::ops::load_vec<uint4>(p);")
    if relative.endswith("linear_swiglu/token_major_mma_epilogue.cuh"):
        before = '''        const auto synchronize = [] {
            if constexpr (producers)
                asm volatile("bar.sync 1, %0;" ::"r"(threads) : "memory");
            else
                __syncthreads();
        };'''
        if text.count(before) != 1: raise ValueError("missing SwiGLU consumer barrier definition")
        text = text.replace(before, '''        static_assert(producers == 0,
            "Native gfx1201 SwiGLU uses full-CTA ordinary LDS/MMA schedules");
        const auto synchronize = [] { __syncthreads(); };''')
    if "/fp8/" in relative:
        text = text.replace("__ldg(row_scales", "load_vec<__hip_bfloat16>(row_scales")
    # Half/BF16 wrapper types have several class conversions on HIP. Widening
    # to FP32 is exact; the explicit BF16 conversion retains native RNE.
    for value in ("__low2half(half_pair)", "__high2half(half_pair)"):
        text = text.replace(f"__hip_bfloat16({value})", f"__float2bfloat16(__half2float({value}))")
    if relative.endswith("bf16_linear_add_gemm_mma.cu"):
        text = text.replace("Bf16MmaSchedule<32, 32, 256, 16, 8, 3, 1,", "Bf16MmaSchedule<32, 32, 64, 16, 8, 2, 1,")
    if relative.endswith("bf16_linear_add_gemm_mma_unified.cu"):
        for function in ("bf16_linear_add_aggregate_mma_launch", "bf16_linear_add_mma_launch"):
            text = _linear.body(text, rf"void {function}\([^{{}}]+\)\s*\{{", """
    auto* data = static_cast<__hip_bfloat16*>(residual.data);
    using NativeTile = Bf16A16MmaSchedule<32, 32, 64, 16, 16, 2, 1,
        Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong,
        Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64>;
    launch_bf16_a16_mma<Bf16ScheduleInstance<NativeTile, 6144>>(
        bf16_a16_operands(x, weight), LinearBf16Output{data, weight.n},
        LinearResidualAddEpilogue{{data, weight.n}}, stream);""")
    if "/nvfp4/" in relative:
        # S2 128x128xK256 staging is 73,728 bytes. K128 keeps the same K64
        # fragment/scale mathematics with 36,864 bytes of staging; output LDS
        # is at most 34,816 bytes. All registered K values divide by128.
        text = re.sub(r"(Nvfp4(?:W4a4|A4)MmaSchedule<\s*\d+\s*,\s*\d+\s*,\s*)256(?=\s*,)",
                      r"\g<1>128", text)
        if relative.endswith(("nvfp4_linear_add_w4a4.cu", "nvfp4_gdn_input_w4a4.cu")):
            text = remove_tma_branch(text, "w4a4_tma_route(tokens)")
            text = remove_definition(text, r"constexpr bool w4a4_tma_route\([^{}]+\)\s*\{")
            text = text.replace("w4a4_tma_route(tokens)", "false")
        if relative.endswith(("nvfp4_linear_add_a4.cu", "nvfp4_gdn_input_a4.cu", "nvfp4_attn_input_a4.cu")):
            text = remove_tma_branch(text, "uses_tma(tokens)")
            text = remove_definition(text, r"constexpr bool uses_tma\([^{}]+\)\s*\{")
            text = text.replace("uses_tma(tokens)", "false")
        if relative.endswith("nvfp4_attn_input_w4a4.cu"):
            text = remove_tma_branch(text, "nvfp4_attn_input_tma_route(tokens)")
            text = text.replace("nvfp4_attn_input_tma_route(tokens)", "false")
        if relative.endswith("nvfp4_attn_input_fused_rmsnorm_w4a4.cu"):
            # The original row-major fused quantizer and banded native MMA
            # consumer already implement the complete public transformation.
            first = text.index("    if (!nvfp4_attn_input_tma_route(tokens)) {")
            opening = text.index("{", first)
            depth, end = 1, opening + 1
            while depth:
                depth += (text[end] == "{") - (text[end] == "}")
                end += 1
            native = text[opening + 1:end - 1]
            tail = text.index("\n}\n", end)
            text = text[:first] + native + text[tail:]
        if relative.endswith("nvfp4_linear_swiglu_plan.cpp"):
            text = _linear.body(text, r"Nvfp4LinearSwiGluRoute resolve_route\([^{}]+\)\s*\{", """
    if (tokens <= 0) throw std::invalid_argument("nvfp4 linear_swiglu: T must be positive");
    if (!valid_linear_policy(policy)) throw std::invalid_argument("nvfp4 linear_swiglu: invalid compute policy");
    if (!allows_a4(policy)) {
        if (tokens == 1) return Nvfp4LinearSwiGluRoute::DecodeFusedA16;
        if (tokens <= 16) return Nvfp4LinearSwiGluRoute::SmallTFusedA16;
        throw std::invalid_argument("nvfp4 linear_swiglu A16 is registered only through T=16");
    }
    if (tokens == 1) return Nvfp4LinearSwiGluRoute::DecodeFusedA16;
    if (tokens <= 4) return Nvfp4LinearSwiGluRoute::SmallTFusedA16;
    if (tokens <= kFusedMaxTokens) return Nvfp4LinearSwiGluRoute::FusedW4A4;
    return Nvfp4LinearSwiGluRoute::LinearW4A4Post;""")
            text = _linear.body(text, r"static std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes_own\([^{}]+\)\s*\{", """
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("nvfp4 linear_swiglu workspace: invalid token interval");
    (void)resolve_route(policy, min_tokens); (void)resolve_route(policy, max_tokens);
    if (!allows_a4(policy) || max_tokens <= 4) return 0;
    std::size_t maximum = 0;
    if (min_tokens <= kFusedMaxTokens && max_tokens >= 5)
        maximum = fused_workspace_bytes(std::min(max_tokens, kFusedMaxTokens));
    if (max_tokens > kFusedMaxTokens)
        maximum = std::max(maximum, baseline_workspace_bytes(max_tokens));
    return maximum;""")
            text = remove_definition(text, r"case Nvfp4LinearSwiGluRoute::TmaFusedW4A4:\s*\{")
            text = text.replace("    TmaFusedW4A4,\n", "")
        if relative.endswith("nvfp4_linear_swiglu_plan_unified.cpp"):
            text = remove_tma_branch(text, "tokens >= 256")
            text = remove_definition(text, r"case Nvfp4LinearSwiGluRoute::TmaFusedA4:\s*\{")
            text = text.replace("    TmaFusedA4,\n", "")
        # Every removed TMA call above selects the existing complete ordinary
        # LDS/MMA route. These platform headers are consequently not consumed.
        text = re.sub(r'^#include "[^"\n]*_tma_launch\.h"\n', "", text, flags=re.MULTILINE)
    if relative.endswith("bf16_attn_input_gemm_mma_unified.cu"):
        text = _linear.body(text, r"void bf16_attn_input_mma_launch\([^{}]+\)\s*\{", """
    const LinearBf16SegmentedOutput<6144, 1024, 6144, 1024> output{
        {static_cast<__hip_bfloat16*>(q.data), static_cast<__hip_bfloat16*>(k.data),
         static_cast<__hip_bfloat16*>(gate.data), static_cast<__hip_bfloat16*>(v.data)}};
    using NativeTile = Bf16A16MmaSchedule<32, 32, 64, 16, 16, 2, 1,
        Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong,
        Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64>;
    launch_bf16_a16_mma<Bf16ScheduleInstance<NativeTile, 5120>>(
        bf16_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);""")
    return text


def manifest(engine: Path, family: str) -> tuple[list[str], list[str]]:
    path = engine / "src/ops" / family / "sources.cmake"
    sources = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        for name in re.findall(r'"\$\{CMAKE_CURRENT_LIST_DIR\}/([^"\n]+\.(?:cu|cpp))"', raw.split("#", 1)[0]):
            source = (path.parent / name).resolve()
            if not source.is_file():
                raise ValueError(f"missing original fusion source: {source}")
            sources.append(source.relative_to(engine).as_posix())
    if len(sources) != len(set(sources)):
        raise ValueError(f"duplicate original sources in {family}")
    tma = [name for name in sources if name.endswith("_tma.cu")]
    return sources, tma


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--source-view", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    args = parser.parse_args()
    engine, source, output = (p.resolve() for p in (args.engine, args.source_view, args.output))
    catalog, changed = {}, []
    for family in args.families:
        original, tma = manifest(engine, family)
        catalog[family] = {"all_original_sources": original,
                           "native_sources": [name for name in original if name not in tma],
                           "replaced_tma_sources": tma}
        roots = [source / "src/ops" / family, source / "tests/ops" / family]
        paths = [p for root in roots if root.exists() for p in root.rglob("*")
                 if p.is_file() and p.suffix in (".h", ".cuh", ".cpp", ".cu")]
        wrapper = source / "src/ops/wrapper" / f"{family}.cpp"
        if wrapper.is_file(): paths.append(wrapper)
        flat_tests = {"gdn_input_proj": ("gdn_input_proj", "gdn_input_small_t", "gdn_input_proj_conv_record", "gdn_input_proj_conv_snapshot", "gdn_replay_fold"),
                      "attn_input_proj": ("attn_input_proj", "attn_input_small_t", "attn_input_proj_fused_rmsnorm"),
                      "gdn_gating_proj": ("gdn_gating_proj",), "rmsnorm_rope": ("rmsnorm_rope",),
                      "linear_topk": ("linear_topk",)}
        for test in flat_tests.get(family, ()):
            extension = "cu" if test == "linear_topk" else "cpp"
            test_path = source / "tests/ops" / f"test_{test}.{extension}"
            if not test_path.is_file(): raise ValueError(f"missing original fusion test: {test_path}")
            paths.append(test_path)
        if family in ("gdn_input_proj", "attn_input_proj"):
            # Both original tests directly call SmallTMma rather than relying
            # on the default plan to happen to select it. Preserve their full
            # element-by-element FP64 oracle, independently of route tables.
            paths.append(source / "tests/ops/small_t_oracle.h")
        for path in sorted(set(paths)):
            relative = path.relative_to(source).as_posix()
            text = path.read_text(encoding="utf-8")
            if text.startswith(MARKER):
                rewritten = text
            elif relative in tma:
                # Retain unused originals for audit. They are absent from the
                # native target and have no resolver-selected call remaining.
                rewritten = text
            else:
                rewritten = MARKER + transform(relative, text)
                if rewritten[len(MARKER):] != text: changed.append(relative)
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rewritten, encoding="utf-8", newline="\n")
    # Other public projections share these integer/BLAS implementation files,
    # but do not require the whole public LinearSwiGLU family to be selected.
    if any(family in ("linear_add", "gdn_input_proj", "attn_input_proj") for family in args.families) and "linear_swiglu" not in args.families:
        for directory in ("q4a8", "q4cublas"):
            for path in sorted((source / "src/ops/linear_swiglu" / directory).rglob("*")):
                if not path.is_file() or path.suffix not in (".h", ".cuh", ".cpp", ".cu"): continue
                relative = path.relative_to(source).as_posix()
                text = path.read_text(encoding="utf-8")
                rewritten = text if text.startswith(MARKER) else MARKER + transform(relative, text)
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(rewritten, encoding="utf-8", newline="\n")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"families": catalog, "transformed_sources": changed,
                "qualification": "production families require original mathematical suites; source preparation is not a pass"}
    (output / "fused-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"families": {name: len(data["native_sources"]) for name, data in catalog.items()},
                      "transformed_sources": len(changed), "manifest": str(output / "fused-manifest.json")}))


if __name__ == "__main__": main()
