"""Isolate the ACTUAL native public BF16 policy/launchers. Import writes nothing.

The frozen six-case generator is reused without editing it. This new generator
adds the actual public validation/resolver/workspace, dynamic B1 row, original
BF16 gate reducer, actual prompt post-gate, and original chunk slicing. It never
modifies the Core ABI, old source views, Models, original tests or old receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
NATIVE = "src/ops/softmax_attention/dense/causal_cache/"
EXTRA = {
    NATIVE + "launch.h": "f84dee43afc0bb8f8a14e83aede78c67a1bf8d5cd143819ccaf66e6a5408de11",
    NATIVE + "causal_softmax_attention.cpp": "cd9a95fe24491fb1798f61fcfa39d1e4362e46858e7f3fc2b9350437e1ac7a45",
    NATIVE + "small_t.cu": "26ce738ab290155a045844b64964f20ce5003a060984b882d37aa3e2e374df2f",
    NATIVE + "prompt.cu": "e7ac54a71480fb41952b4fb1f3b65569ccec727dcd71f5f766799b9ff64a1f70",
    "src/ops/kv_cache/append/launch.cu": "43aecba3a594e05cf8ba4b4ccaf12da4f83a18567edabfc8e9c1f71ac2e67a1d",
}
OWNED = ("public_bf16.h", "test_public_bf16.cu", "fixture_common.cuh", "verify_map.py", "PublicBF16.cmake")


def exact(text: str, old: str, new: str, count: int = 1) -> str:
    if text.count(old) != count:
        raise ValueError(f"public BF16 anchor changed ({text.count(old)} != {count}): {old[:100]!r}")
    return text.replace(old, new)


def end_brace(text: str, opening: int) -> int:
    """Match real braces while ignoring C++ strings/chars/comments."""
    if text[opening] != "{":
        raise ValueError("not a C++ opening brace")
    depth, i, state = 0, opening, "code"
    while i < len(text):
        c, pair = text[i], text[i:i + 2]
        if state == "line":
            if c == "\n": state = "code"
        elif state == "block":
            if pair == "*/": state, i = "code", i + 1
        elif state in ("\"", "'"):
            if c == "\\": i += 1
            elif c == state: state = "code"
        elif pair == "//": state, i = "line", i + 1
        elif pair == "/*": state, i = "block", i + 1
        elif c in ("\"", "'"): state = c
        elif c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0: return i + 1
        i += 1
    raise ValueError("unterminated C++ source region")


def remove_if(text: str, condition: str, count: int) -> str:
    anchor = "if (" + condition + ") {"
    if text.count(anchor) != count:
        raise ValueError(f"conditional source inventory changed: {condition}")
    for _ in range(count):
        start = text.index(anchor)
        end = end_brace(text, text.index("{", start))
        text = text[:start] + text[end:]
    return text


def bf16_else(text: str) -> str:
    anchor = "if (kv_cache_is_int8_family(cache.storage)) {"
    start = text.index(anchor)
    first_end = end_brace(text, text.index("{", start))
    tail = first_end
    while text[tail].isspace(): tail += 1
    if not text.startswith("else {", tail): raise ValueError("expected original BF16 else branch")
    opening = text.index("{", tail)
    last = end_brace(text, opening)
    return text[:start] + text[opening + 1:last - 1] + text[last:]


def function(text: str, marker: str, include_template: bool = False) -> str:
    start = text.index(marker)
    if include_template:
        start = text.rfind("template <", 0, start)
    opening = text.index("{", start)
    return text[start:end_brace(text, opening)]


def rename(text: str) -> str:
    # Rename only the public/launcher functions declared by the original headers,
    # never the original reducer or copied kernel arithmetic identifiers.
    names = [
        "causal_softmax_attention", "causal_softmax_attention_cached",
        "causal_softmax_attention_route_family", "causal_softmax_attention_workspace_capacity_bytes",
        "causal_softmax_attention_prompt_wave_tokens", "causal_softmax_attention_prompt_aligned_chunk",
        "causal_attention_split_capacity", "causal_attention_resolve_route", "causal_attention_route_name",
        "causal_attention_small_t_launch", "causal_attention_cached_small_t_launch",
        "causal_attention_prompt_launch", "causal_attention_prompt_attention_launch",
        "causal_attention_prompt_fast_kernel", "kv_cache_append_launch", "kv_cache_append_batch_launch",
        "CausalAttentionRoute", "CausalSmallTInvocation", "kPromptWaveRows", "proto_selected_blocks",
    ]
    return re.sub(r"\b(?:" + "|".join(map(re.escape, names)) + r")\b",
                  lambda m: "segmented_b1_" + m[0], text)


SCOPE = r'''
namespace ninfer::ops {
namespace {
void segmented_b1_profile(KvCacheStorage storage, CausalAttentionExecutionEnvelope envelope,
                          int batch, int lo, int hi) {
    const char* mask = std::getenv("NINFER_PROTO_SELECT");
    if (storage != KvCacheStorage::BFloat16 || batch != 1 || lo < 1 || hi < lo || hi > 512 ||
        envelope.min_visible_keys == 0 || envelope.max_visible_keys > 512 || hi > envelope.max_visible_keys ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        (mask && std::string_view(mask) != "all"))
        throw std::invalid_argument("segmented B1 candidate: only dense BF16 B1 widths/visible<=512");
}
void segmented_b1_scope(const Tensor& q, const Tensor& valid, const PagedKVBatchLayerView& c,
                        CausalAttentionExecutionEnvelope e) {
    segmented_b1_profile(c.storage, e, q.ne[3], q.ne[2], q.ne[2]);
    if (valid.data || c.selected_blocks || c.head_dim != 256 ||
        c.k_pages.ne[0] != 256 || c.k_pages.ne[1] != 64 ||
        c.v_pages.ne[0] != 256 || c.v_pages.ne[1] != 64 ||
        c.k_pages.ne[3] < c.block_tables.ne[0] || c.block_tables.ne[0] > 8 ||
        ((c.k_page_bases == nullptr) != (c.v_page_bases == nullptr)) ||
        (reinterpret_cast<std::uintptr_t>(c.k_page_bases) & 7u) ||
        (reinterpret_cast<std::uintptr_t>(c.v_page_bases) & 7u))
        throw std::invalid_argument("segmented B1 candidate: unsupported mask/layout/table inventory");
}
} // namespace
} // namespace ninfer::ops
'''


def transform(sources: dict[str, str], frozen) -> dict[str, str]:
    outputs = frozen.transform(sources)
    outputs["amd-memory/segmented_bf16_view.cuh"] = (ROOT.parent / "segmented_bf16_view.cuh").read_text(encoding="utf-8")
    # Same frozen resolver and math bodies; no independent attention kernel.
    launch = sources[NATIVE + "launch.h"]
    a = launch.index("inline const std::uint32_t* proto_selected_blocks() {")
    b = end_brace(launch, launch.index("{", a))
    launch = launch[:a] + "inline const std::uint32_t* proto_selected_blocks() { return nullptr; }" + launch[b:]
    launch = exact(launch, '#include "ninfer/ops/softmax_attention.h"',
                   '#include "amd-b1/public_bf16.h"')
    # Declare the separately isolated append launcher before Prompt uses it.
    launch += '\nnamespace ninfer::ops::detail {\nvoid kv_cache_append_batch_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, PagedKVBatchLayerView, hipStream_t);\n}\n'
    outputs["amd-b1/launch.h"] = rename(launch)

    small = sources[NATIVE + "small_t.cu"]
    small = exact(small, '#include "ops/softmax_attention/dense/causal_cache/launch.h"', '#include "amd-b1/launch.h"')
    small = exact(small, '#include "ops/softmax_attention/dense/causal_cache/small_t_bf16.cuh"', '#include "amd-memory/segmented_bf16_small_t.cuh"')
    small = exact(small, '#include "ops/softmax_attention/dense/causal_cache/small_t_i8_launch.h"', '')
    for storage in ("Fp8KeyNvfp4Value", "Fp8E4M3Row256", "Nvfp4Group16"):
        small = remove_if(small, "cache.storage == KvCacheStorage::" + storage, 2)
    small, changed = re.subn(r'            if \(kv_cache_is_int8_family\(cache.storage\)\) \{.*?            \} else \{',
                             '            {', small, count=1, flags=re.S)
    if changed != 1: raise ValueError("original small-T BF16 dispatch macro changed")
    small = exact(small, "causal_attention_small_t_tc_partial_bf16_kernel<", "segmented_causal_attention_small_t_tc_partial_bf16_kernel<")
    small = exact(small, "                                                 : proto_selected_blocks());",
                   "                                                 : proto_selected_blocks(),\n"
                   "            {cache.k_page_bases, cache.v_page_bases});")
    outputs["amd-b1/small_t.cu"] = rename(small)

    prompt = sources[NATIVE + "prompt.cu"]
    prompt = exact(prompt, '#include "ops/softmax_attention/dense/causal_cache/launch.h"', '#include "amd-b1/launch.h"')
    prompt = exact(prompt, '#include "ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh"', '#include "amd-memory/segmented_bf16_prompt.cuh"')
    prompt = exact(prompt, '#include "ops/softmax_attention/dense/causal_cache/prompt_i8.cuh"', '')
    prompt = exact(prompt, '#include "ops/kv_cache/append/launch.h"', '')
    prompt = exact(prompt, 'static_assert(kPromptWaveRows % kCausalPromptI8Br == 0);', '')
    for helper in ("bool prompt_pv_f16()", "bool prompt_pack_gqa()"):
        prompt = exact(prompt, function(prompt, helper), '')
    prompt = bf16_else(prompt)
    for storage in ("Fp8KeyNvfp4Value", "Fp8E4M3Row256", "Nvfp4Group16"):
        prompt = remove_if(prompt, "cache.storage == KvCacheStorage::" + storage, 2)
    prompt = exact(prompt, "causal_attention_prompt_bf16_kernel<", "segmented_causal_attention_prompt_bf16_kernel<", 2)
    prompt = exact(prompt, "static_cast<__hip_bfloat16*>(out.data), tokens);",
                    "static_cast<__hip_bfloat16*>(out.data), tokens,\n                {cache.k_page_bases, cache.v_page_bases});")
    outputs["amd-b1/prompt.cu"] = rename(prompt)

    append = sources["src/ops/kv_cache/append/launch.cu"]
    full = function(append, "void launch_full(", True)
    full = remove_if(full, "cache.storage == KvCacheStorage::Fp8E4M3Row256", 1)
    full = remove_if(full, "kv_cache_is_int8_family(cache.storage)", 1)
    full = exact(full, "kv_cache_append_full_bf16_kernel<", "segmented_kv_cache_append_full_bf16_kernel<")
    full = exact(full, "static_cast<CacheKey*>(cache_k.data), static_cast<CacheValue*>(cache_v.data), tokens);",
                 "static_cast<CacheKey*>(cache_k.data), static_cast<CacheValue*>(cache_v.data), tokens,\n"
                 "        {cache.k_page_bases, cache.v_page_bases});")
    functions = []
    for name in ("void kv_cache_append_launch(", "void kv_cache_append_batch_launch("):
        item = function(append, name)
        for storage in ("Fp8KeyNvfp4Value", "Nvfp4Group16"):
            item = remove_if(item, "cache.storage == KvCacheStorage::" + storage, 1)
        functions.append(item)
    outputs["amd-b1/append.cu"] = rename(
        '#include "amd-b1/launch.h"\n#include "amd-memory/segmented_bf16_append.cuh"\n'
        '#include "ops/kv_cache/plane_types.h"\n#include "core/device.h"\n'
        'namespace ninfer::ops::detail { namespace {\n' + full + '\n}\n' + '\n'.join(functions) + '\n}\n')

    public = sources[NATIVE + "causal_softmax_attention.cpp"]
    public = exact(public, '#include "ops/softmax_attention/dense/causal_cache/launch.h"',
                   '#include "amd-b1/launch.h"\n#include <string_view>\n#include <stdexcept>\n' + SCOPE)
    public = exact(public, '    auto scope = workspace.scope();\n    const detail::CausalAttentionRoute route =\n        detail::causal_attention_resolve_route(q.ne[1], width, batch, cache.storage, envelope);',
                   '    segmented_b1_scope(q, valid_columns, cache, envelope);\n'
                   '    auto scope = workspace.scope();\n    const detail::CausalAttentionRoute route =\n'
                   '        detail::causal_attention_resolve_route(q.ne[1], width, batch, cache.storage, envelope);', 1)
    # Cached public entry is outside the finite B1 append product slice. Keep its
    # original helpers inert/available but reject this entry before any allocation.
    entry = '    validate_attention_tensors(q, positions, out, geometry, cache, envelope, scale, op);'
    public = exact(public, entry, '    throw std::invalid_argument("segmented B1 candidate: cached public entry not admitted");\n' + entry)
    entry = '    require_causal_geometry(geometry, "causal_softmax_attention workspace");'
    public = exact(public, entry, '    segmented_b1_profile(cache_storage, envelope, batch_size, min_width, max_width);\n' + entry)
    outputs["amd-b1/public.cpp"] = rename(public)
    outputs["amd-b1/public_bf16.h"] = (ROOT / "public_bf16.h").read_text(encoding="utf-8")
    # Every original dynamic selector, column offset, gate and split capacity must
    # still reach its actual body after specialization. These are source guards.
    for token in ("invocation.table_rows", "invocation.column_begin", "invocation.full_width", "static_cast<const __hip_bfloat16*>(gate)"):
        if token not in outputs["amd-b1/small_t.cu"]: raise ValueError("lost actual public small-T contract: " + token)
    for token in ("segmented_b1_causal_attention_resolve_route", "segmented_b1_causal_attention_split_capacity", "if (gate != nullptr) { sigmoid_mul(*gate, out, stream); }"):
        if token not in outputs["amd-b1/public.cpp"]: raise ValueError("lost original public wrapper contract: " + token)
    return outputs


def frozen_module():
    path = ROOT.parent / "segmented_bf16_ops_overlay.py"
    spec = importlib.util.spec_from_file_location("frozen_segmented_bf16", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def external_inputs():
    spec = importlib.util.spec_from_file_location("segmented_b1_map_closure", ROOT / "verify_map.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.external_inputs()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    baseline, output = args.baseline.resolve(), args.output.resolve()
    if (output.exists() and not args.verify_existing) or output == baseline or baseline in output.parents or output in baseline.parents:
        raise ValueError("output must be a new isolated view, never a production or ancestor path")
    frozen = frozen_module()
    baselines = {**frozen.BASELINES, **EXTRA}
    if len(EXTRA) != 5: raise ValueError("five reviewed public native source fingerprints required")
    raw = {name: (baseline / name).read_bytes() for name in baselines}
    for name, data in raw.items():
        if hashlib.sha256(data).hexdigest() != baselines[name]: raise ValueError("reviewed native source changed: " + name)
    outputs = transform({name: data.decode("utf-8").replace("\r\n", "\n") for name, data in raw.items()}, frozen)
    owned = {str(Path(__file__).resolve()): Path(__file__).read_bytes(),
             str(ROOT.parent / "segmented_bf16_ops_overlay.py"): (ROOT.parent / "segmented_bf16_ops_overlay.py").read_bytes(),
             str(ROOT.parent / "segmented_bf16_view.cuh"): (ROOT.parent / "segmented_bf16_view.cuh").read_bytes()}
    for name in OWNED: owned[str(ROOT / name)] = (ROOT / name).read_bytes()
    external = external_inputs()
    receipt = {"schema": "ninfer.segmented_b1_public_source.v1", "status": "SOURCE_GENERATED_NOT_BUILT_NOT_QUALIFIED",
               "baseline": str(baseline), "native_inputs_sha256": baselines,
               "owned_inputs_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in owned.items()},
               "external_map_inputs_sha256": {str(name): hashlib.sha256(data).hexdigest() for name, data in external.items()},
               "outputs_sha256": {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in outputs.items()},
               "core_abi_modified": False, "production_or_engine_gate": False,
               "admission": "BF16 PageMajor D256 H24/KV4 or H16/KV2 B1 dense widths/visible<=512",
               "original_public_validation_resolver_workspace_gate": True}
    if output.exists():
        inventory = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
        if inventory != set(outputs) | {"source-receipt.json"}:
            raise ValueError("isolated source inventory has missing or extra files")
        if json.loads((output / "source-receipt.json").read_text(encoding="utf-8")) != receipt: raise ValueError("source receipt changed")
        for name, text in outputs.items():
            if (output / name).read_bytes() != text.encode("utf-8"): raise ValueError("isolated source changed: " + name)
        return
    output.mkdir(parents=True, exist_ok=False)
    for name, text in outputs.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream: stream.write(text.encode("utf-8"))
    with (output / "source-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
