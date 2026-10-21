"""Isolate a one-semantic-change BF16 table-load candidate; import writes nothing.

Only the frozen helper's volatile U64 read becomes an ordinary U64 read. The
address formula, copied kernels, launchers, original fixture and Core ABI stay
fixed. Running this generator is a future owner-controlled configure operation;
neither generation nor successful linking qualifies math, ISA or performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LIMIT = 512 * 1024
HELPER = "amd-memory/segmented_bf16_view.cuh"
FROZEN = {
    "segmented_bf16_view.cuh": "92ad00f333565e298f34020f71f3a163f2c4b22d1acc6056e80bc3ea47ad02f1",
    "segmented_bf16_launch.cuh": "196ef24290725625ab1c3cb76b809b8dd44cec21dd8f04e543d600c451a564df",
    "segmented_bf16_ops_overlay.py": "4520f1e53b3180be38739962f61a88e16420f7259e8a95f9954d74cb7174c5ed",
    "SegmentedBF16Ops.cmake": "3561acfc0c78c68e6113c2fd147e76674693762c460dc0e6862fe8eb96a81d38",
    "tests/kv_segmented_bf16_attention_test.cu": "25b9567d70959f14df2b3ba1d38ac0e9e1ce268a2e6966414353fbc1e2aa0196",
    "tests/verify_segmented_bf16_map.py": "0906d41f704bb95d162e4e8799eded354b719a83c12a5817b7ac1b2e6abfa582",
    "tests/verify_segmented_pool_map.py": "05fc5f91db3aab76707c36d08959f545d48228648d9ea094fbf32833d498e8e4",
}
OP_CHECK_SHA = "63c49ca0b934a8c007fe755eddf402887df72da9136511e9d5259baa1a8b3c27"
NATIVE = {
    "src/ops/kv_cache/append/kernel.cuh": "12b4b11da63062aed7c1257455ab42ae13ff2998cec1b8581e12a0ea74ab78b5",
    "src/ops/softmax_attention/dense/causal_cache/small_t_bf16.cuh": "733fb898c4eaa35f87b741df1c4fad2b334428bae52fc6b5b75514d647f861cb",
    "src/ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh": "e1028c350a401d245201b22ae821dde64004fcb637b0f3488d3f50da14dbb625",
    "src/ops/softmax_attention/dense/causal_cache/small_t.cuh": "8d8d192cf4b06f37fe25bd55d6d80b90ba7719ac9f9d429cfe30638b9c1a0b24",
    "src/ops/softmax_attention/dense/causal_cache/prompt_common.cuh": "ec78d722686121c4aecd0ddc2ee28c3cbcb6f1c397f053b365cf27cb1538b95d",
    "src/ops/kernel/paged_kv_address.cuh": "7d107f0bc9582f0e00ff357d8852ee84814633185a8ea54c23dd27b3236a6401",
    "src/ops/softmax_attention/dense/causal_cache/geometry.cuh": "cccf46c4e0704cfa5893a45f9bb9a0e51b7f300c921b873046671f9e44788368",
    "src/ops/kv_cache/append/geometry.cuh": "b04cc6a4305ef7a077460a0b9ce346eda32e23a36428e8afc01442c746df1c2f",
}
BF16_OUTPUTS = {
    "amd-memory/segmented_bf16_append.cuh": "bc5a22a00ef074042a0ff28a900bf0b0d905c0e22eedd4c05b5af9ce1009b238",
    "amd-memory/segmented_bf16_small_t.cuh": "538cf5c78ecdbec0d057507fa3f42d41c33d91db2d69858632d959bc36dcab31",
    "amd-memory/segmented_bf16_prompt.cuh": "174e53323323a0f8512a4f19085fbaa1e5a74936d4048f00a46e942337a30fc5",
    HELPER: FROZEN["segmented_bf16_view.cuh"],
    "amd-memory/segmented_bf16_launch.cuh": FROZEN["segmented_bf16_launch.cuh"],
}
CORE_INPUTS = {
    "src/core/paged_kv_cache.h": "86935cf1b3f20a324bc1c9d4cd53ca21e9d5914c01f30e0bbd998fad455dd789",
    "src/core/paged_kv_cache.cpp": "fb173794ce15a662472fe05e47255f08c5d88e9b86120ee94d71ae39d72a4a4a",
    "kv_segmented_storage.h": "31e31117f4e93620f523485abe83e86362e7ae718ec978a3f737d98f3fe2229a",
    "kv_segmented_storage.cpp": "29939c1400103e6161df9a26dadd20a9f6a478eb08b369eb70e35accef4d2c46",
}
CORE_OUTPUTS = {
    "src/core/paged_kv_cache.h": "8ba0b39fb9ad09cbc39dc11d779f413f0c6f68bde0d7e817599b389ef6fd9c61",
    "src/core/paged_kv_cache.cpp": "ac37b15cde2fd6e3239db08210abcd4367424ea8c328ae7334df125dabcec121",
    "amd-memory/kv_segmented_storage.h": "382df2b0cd3870e106f9044fa58f9a7ede9b3dc5bba5e24e9a34dbfb8ce56bf3",
    "amd-memory/kv_segmented_storage.cpp": "5b490b0cfee0e847cfbd061873c028b75da1be7b2e036de95d3cfd8378ac15da",
    "amd-memory/kv_segmented_pool_control.h": "d7fb9c918bcdbf87bd318ab0b68b339bbf89052dc962c9f2e82effb7f4603049",
    "amd-memory/kv_segmented_pool_owner.h": "83124366a905fc8f460379d531fa1ddbdb303df52727a7efa4b35ea54c900dc0",
    "amd-memory/kv_segmented_pool_owner.cpp": "486dfcaeaa6020c10d706fe5d9a053a49d272b1134b9645368e6d45cfa48c5e6",
    "amd-memory/kv_segmented_loan_plan.h": "589568a3050e9362bb02dfec2e8c4c9e769a09cea2322cf278adefe563451c0b",
    "amd-memory/kv_segmented_loan_plan.cpp": "ff7c29e6d985cbd7a21f5c2b592ce16c5eb8f6394b4158a5fb97ab31bc86a7cd",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bounded(path: Path) -> bytes:
    if not path.is_file() or not 0 < path.stat().st_size <= LIMIT:
        raise ValueError(f"missing/oversized small source: {path}")
    data = path.read_bytes()
    if not 0 < len(data) <= LIMIT:
        raise ValueError(f"small source changed extent: {path}")
    return data


def pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError(f"duplicate receipt key: {key}")
        value[key] = item
    return value


def invalid_constant(value):
    raise ValueError(f"nonfinite receipt value: {value}")


def receipt(data: bytes) -> dict:
    value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                       parse_constant=invalid_constant)
    if type(value) is not dict:
        raise ValueError("receipt is not an object")
    return value


def pinned(root: Path, expected: dict[str, str]) -> dict[str, bytes]:
    result = {name: bounded(root / name) for name in expected}
    for name, data in result.items():
        if digest(data) != expected[name]:
            raise ValueError(f"frozen source/view changed: {root / name}")
    return result


def transform_helper(source: bytes) -> bytes:
    if digest(source) != BF16_OUTPUTS[HELPER]:
        raise ValueError("volatile baseline helper changed")
    changes = (
        (b"static_cast<const volatile std::uint64_t*>(page_bases)[cell]", b"page_bases[cell]"),
        (b"// Core publishes/finishes table updates before a consumer starts or replays; volatile",
         b"// Core completes publication before use; table values stay fixed for this dispatch."),
        (b"// below is a compiler load requirement, NOT a substitute for that stream dependency.",
         b"// Ordinary loads rely on that dependency and retain the same address formula."),
    )
    result = source
    for old, new in changes:
        if result.count(old) != 1:
            raise ValueError("readonly helper anchor changed")
        result = result.replace(old, new)
    return result


def prepare_in_memory(native: Path, baseline: Path, pool: Path, engine: Path):
    """Read bounded pinned sources, validate receipts, transform bytes; never write."""
    memory = Path(__file__).resolve().parent.parent
    frozen = pinned(memory, FROZEN)
    native_bytes = pinned(native, NATIVE)
    pinned(native, {key: CORE_INPUTS[key] for key in CORE_INPUTS if key.startswith("src/")})
    if digest(bounded(engine / "tests/ops/op_check.h")) != OP_CHECK_SHA:
        raise ValueError("original mathematical criterion changed")
    source_receipt = bounded(baseline / "segmented-bf16-source-receipt.json")
    prior = receipt(source_receipt)
    if (prior.get("schema_version") != 1 or prior.get("native_source_sha256") != NATIVE or
        prior.get("outputs_sha256") != BF16_OUTPUTS or
        prior.get("owned_input_sha256") != {key: FROZEN[key] for key in
            ("segmented_bf16_view.cuh", "segmented_bf16_launch.cuh", "segmented_bf16_ops_overlay.py")} or
        Path(prior.get("baseline", "")).resolve() != native or
        prior.get("core_view_abi_required") != "NINFER_AMD_SEGMENTED_POOL_ABI" or
        prior.get("old_view_libraries_compatible") is not False or
        prior.get("public_op_or_engine_admission") is not False):
        raise ValueError("baseline BF16 source receipt is not the pinned candidate")
    pool_receipt = bounded(pool / "segmented-source-receipt.json")
    core = receipt(pool_receipt)
    if (core.get("schema_version") != 1 or core.get("reviewed_input_sha256") != CORE_INPUTS or
        core.get("output_sha256") != CORE_OUTPUTS or
        Path(core.get("baseline", "")).resolve() != native or
        core.get("core_view_abi_changed") is not True or
        core.get("old_abi_libraries_compatible") is not False or
        core.get("production_gate_changed") is not False):
        raise ValueError("Core source receipt is not the pinned isolated ABI")
    pinned(pool, CORE_OUTPUTS)
    outputs = pinned(baseline, BF16_OUTPUTS)
    original_helper = outputs[HELPER]
    outputs[HELPER] = transform_helper(original_helper)
    if {name for name in outputs if digest(outputs[name]) != BF16_OUTPUTS[name]} != {HELPER}:
        raise AssertionError("readonly candidate changed a second output")
    report = {
        "schema_version": 1,
        "status": "SOURCE_GENERATED_NOT_BUILT_NOT_QUALIFIED",
        "variant": "ordinary-readonly-table-load-v1",
        "sole_semantic_difference": "one volatile U64 table read becomes an ordinary U64 table read",
        "helper_comment_only_edits": 2,
        "address_formula_changed": False, "math_or_fixture_changed": False,
        "graph_or_submission_contract_changed": False,
        "table_epoch_immutable_for_entire_consumer_dispatch_required": True,
        "concurrent_publication_supported": False,
        "baseline_six_case_math_must_be_qualified_first": True,
        "production_or_engine_admission": False, "isa_or_performance_evidence": False,
        "native_root": str(native), "baseline_view": str(baseline), "pool_view": str(pool),
        "baseline_source_receipt_sha256": digest(source_receipt),
        "pool_source_receipt_sha256": digest(pool_receipt),
        "native_source_sha256": {key: digest(value) for key, value in native_bytes.items()},
        "frozen_input_sha256": {key: digest(value) for key, value in frozen.items()},
        "original_op_check_sha256": OP_CHECK_SHA,
        "core_output_sha256": CORE_OUTPUTS,
        "baseline_outputs_sha256": BF16_OUTPUTS,
        "outputs_sha256": {key: digest(value) for key, value in outputs.items()},
        "generator_sha256": digest(bounded(Path(__file__).resolve())),
        "candidate_leaf_sha256": digest(bounded(Path(__file__).with_name("SegmentedBF16Readonly.cmake"))),
    }
    report["recipe_sha256"] = digest(json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return outputs, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-root", "baseline-view", "pool-view", "engine-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    native, baseline, pool, engine, output = [getattr(args, name).resolve() for name in
        ("native_root", "baseline_view", "pool_view", "engine_root", "output")]
    protected = (native, baseline, pool, engine, Path(__file__).resolve().parent.parent)
    if any(output == root or output in root.parents or root in output.parents for root in protected):
        raise ValueError("output overlaps a frozen source/view root")
    if output.exists() and not args.verify_existing:
        raise ValueError("output must be new; verify-existing never permits overwrite")
    outputs, report = prepare_in_memory(native, baseline, pool, engine)
    report_name = "segmented-bf16-readonly-source-receipt.json"
    report_bytes = (json.dumps(report, indent=2) + "\n").encode("utf-8")
    if output.exists():
        if bounded(output / report_name) != report_bytes:
            raise ValueError("existing readonly receipt differs")
        for name, data in outputs.items():
            if bounded(output / name) != data:
                raise ValueError(f"existing readonly candidate modified: {name}")
        return
    # Every read, source pin and in-memory transform succeeds before first mkdir.
    output.mkdir(parents=True, exist_ok=False)
    for name, data in outputs.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    with (output / report_name).open("xb") as stream:
        stream.write(report_bytes)


if __name__ == "__main__":
    main()
