"""SOURCE ONLY. Copy actual prepared native BF16 bodies into a NEW private include root.

No production source is changed, no CMake/Engine hook is installed, and importing this
module generates nothing. The isolated Core ABI from prepare_segmented_pool.py must
be first among Core includes. Never link published old-view Op/test-support archives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BASELINES = {
    "src/ops/kv_cache/append/kernel.cuh": "12b4b11da63062aed7c1257455ab42ae13ff2998cec1b8581e12a0ea74ab78b5",
    "src/ops/softmax_attention/dense/causal_cache/small_t_bf16.cuh": "733fb898c4eaa35f87b741df1c4fad2b334428bae52fc6b5b75514d647f861cb",
    "src/ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh": "e1028c350a401d245201b22ae821dde64004fcb637b0f3488d3f50da14dbb625",
    "src/ops/softmax_attention/dense/causal_cache/small_t.cuh": "8d8d192cf4b06f37fe25bd55d6d80b90ba7719ac9f9d429cfe30638b9c1a0b24",
    "src/ops/softmax_attention/dense/causal_cache/prompt_common.cuh": "ec78d722686121c4aecd0ddc2ee28c3cbcb6f1c397f053b365cf27cb1538b95d",
    "src/ops/kernel/paged_kv_address.cuh": "7d107f0bc9582f0e00ff357d8852ee84814633185a8ea54c23dd27b3236a6401",
    "src/ops/softmax_attention/dense/causal_cache/geometry.cuh": "cccf46c4e0704cfa5893a45f9bb9a0e51b7f300c921b873046671f9e44788368",
    "src/ops/kv_cache/append/geometry.cuh": "b04cc6a4305ef7a077460a0b9ce346eda32e23a36428e8afc01442c746df1c2f",
}
OWNED = ("segmented_bf16_view.cuh", "segmented_bf16_launch.cuh")
TABLES = "ninfer::amd::memory::SegmentedBF16Tables"


def exact(text: str, old: str, new: str, count: int = 1) -> str:
    if text.count(old) != count:
        raise ValueError(f"BF16 native source anchor changed ({text.count(old)} != {count}): {old[:100]!r}")
    return text.replace(old, new)


def address(text: str, letter: str, count: int) -> str:
    return exact(text, f"&cache_{letter}[off]",
                 f"ninfer::amd::memory::segmented_bf16_address<Geometry::KVHeads>(cache_{letter}, page_bases.{letter}, off)", count)


def transform(sources: dict[str, str]) -> dict[str, str]:
    source = sources["src/ops/kv_cache/append/kernel.cuh"]
    start = source.index("template <typename Geometry, typename Metadata>\n__global__ void kv_cache_append_full_bf16_kernel")
    end = source.index("\ntemplate <typename Geometry, typename Metadata>", start + 1)
    append = source[start:end]
    append = exact(append, "kv_cache_append_full_bf16_kernel", "segmented_kv_cache_append_full_bf16_kernel")
    append = exact(append, "std::int32_t width) {", f"std::int32_t width, {TABLES} page_bases) {{")
    for letter in "kv":
        append = exact(append, f"&cache_{letter}[cache_off]",
            f"ninfer::amd::memory::segmented_bf16_address<Geometry::KVHeads>(cache_{letter}, page_bases.{letter}, cache_off)")
    append = ("#pragma once\n#include \"ops/common/math.cuh\"\n"
              "#include \"ops/common/memory.cuh\"\n#include \"ops/common/warp.cuh\"\n"
              "#include \"ops/kernel/paged_kv_address.cuh\"\n"
              "#include \"ops/kv_cache/append/geometry.cuh\"\n"
              "#include \"amd-memory/segmented_bf16_view.cuh\"\n"
              "#include <hip/hip_bf16.h>\nnamespace ninfer::ops {\n" + append + "\n}\n")

    small = sources["src/ops/softmax_attention/dense/causal_cache/small_t_bf16.cuh"]
    small = exact(small, "#pragma once", '#pragma once\n#include "amd-memory/segmented_bf16_view.cuh"')
    small = exact(small, "causal_attention_small_t_tc_partial_bf16_kernel",
                  "segmented_causal_attention_small_t_tc_partial_bf16_kernel")
    small = exact(small, "const std::uint32_t* selected_blocks) {",
                  f"const std::uint32_t* selected_blocks, {TABLES} page_bases) {{")
    for letter in "kv":
        small = exact(small, f"&cache_{letter}[cache_off]",
            f"ninfer::amd::memory::segmented_bf16_address<Geometry::KVHeads>(cache_{letter}, page_bases.{letter}, cache_off)")
        small = address(small, letter, 2)

    prompt = sources["src/ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh"]
    prompt = exact(prompt, "#pragma once", '#pragma once\n#include "amd-memory/segmented_bf16_view.cuh"')
    prompt = exact(prompt, "causal_prompt_stage_kv", "segmented_causal_prompt_stage_kv", 4)
    prompt = exact(prompt, "causal_attention_prompt_bf16_kernel", "segmented_causal_attention_prompt_bf16_kernel")
    prompt = exact(prompt, "int max_query_abs, int physical_page, int tid) {",
                    "int max_query_abs, int physical_page, int tid, const std::uint64_t* page_bases) {")
    prompt = exact(prompt,
        "cache + paged_kv_element_offset<kCausalPromptHeadDim, Geometry::KVHeads>(\n"
        "                    physical_page, kv_head, k0 & kPagedKVPageMask, 0);",
        "ninfer::amd::memory::segmented_bf16_address<Geometry::KVHeads>(cache, page_bases,\n"
        "            paged_kv_element_offset<kCausalPromptHeadDim, Geometry::KVHeads>(\n"
        "                physical_page, kv_head, k0 & kPagedKVPageMask, 0));")
    prompt = exact(prompt, "std::int32_t width) {", f"std::int32_t width, {TABLES} page_bases) {{")
    prompt = exact(prompt, "max_query_abs, physical_page, tid);", "max_query_abs, physical_page, tid, page_bases.k);")
    prompt = exact(prompt, "physical_page,\n                                         tid);", "physical_page,\n                                         tid, page_bases.v);")
    prompt = exact(prompt, "physical_page, tid);", "physical_page, tid, page_bases.k);")
    return {"amd-memory/segmented_bf16_append.cuh": append,
            "amd-memory/segmented_bf16_small_t.cuh": small,
            "amd-memory/segmented_bf16_prompt.cuh": prompt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-existing", action="store_true",
                        help="Allow exact byte-identical existing output; never overwrite")
    args = parser.parse_args()
    baseline, output = args.baseline.resolve(), args.output.resolve()
    if (output.exists() and not args.verify_existing) or output == baseline or baseline in output.parents or output in baseline.parents:
        raise ValueError("output must be a NEW isolated include root, never a native production source tree")
    raw = {name: (baseline / name).read_bytes() for name in BASELINES}
    for name, data in raw.items():
        if hashlib.sha256(data).hexdigest() != BASELINES[name]:
            raise ValueError(f"reviewed native source changed: {name}")
    outputs = transform({name: data.decode("utf-8").replace("\r\n", "\n") for name, data in raw.items()})
    owned = Path(__file__).resolve().parent
    own_raw = {name: (owned / name).read_bytes() for name in OWNED}
    for name, data in own_raw.items():
        outputs["amd-memory/" + name] = data.decode("utf-8")
    recipe = Path(__file__).read_bytes()
    hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in outputs.items()}
    receipt = {"schema_version": 1, "status": "SOURCE_GENERATED_NOT_BUILT_NOT_QUALIFIED",
               "source_scope": "actual native BF16 kernel bodies; only addresses and extra kernel arguments changed",
               "baseline": str(baseline), "native_source_sha256": BASELINES,
               "owned_input_sha256": {**{k: hashlib.sha256(v).hexdigest() for k, v in own_raw.items()},
                                        Path(__file__).name: hashlib.sha256(recipe).hexdigest()},
               "outputs_sha256": hashes, "core_view_abi_required": "NINFER_AMD_SEGMENTED_POOL_ABI",
               "old_view_libraries_compatible": False, "public_op_or_engine_admission": False,
               "supported_candidate_scope": "PageMajor D256 BF16, H24/KV4 or H16/KV2, single unmasked request",
               "rejected": ["HeadMajor", "D128", "quantized KV", "scales", "batched/padded views", "selected_blocks"]}
    if output.exists():
        previous = json.loads((output / "segmented-bf16-source-receipt.json").read_text(encoding="utf-8"))
        if previous != receipt:
            raise ValueError("existing BF16 source receipt differs from this exact native/owned source closure")
        for name, text in outputs.items():
            if (output / name).read_bytes() != text.encode("utf-8"):
                raise ValueError(f"existing BF16 candidate was modified: {name}")
        return
    # Transform validation and all reads finish before the first exclusive write.
    output.mkdir(parents=True, exist_ok=False)
    for name, text in outputs.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(text.encode("utf-8"))
    with (output / "segmented-bf16-source-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
