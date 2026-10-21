"""Isolated optional-capacity Core derivative; imports/transforms never write.

The default CLI is plan-only. Future --emit writes a new content-addressed full
Core view or verifies its complete existing inventory, never overwriting it.
This is not a Vision broker, an Engine provider, or production build wiring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
INPUT_RECEIPT = "controller-handoff-source-receipt.json"
OUTPUT_RECEIPT = "vision-segmented-capacity-source-receipt.json"
HEADER = "amd-memory/kv_segmented_storage.h"
SOURCE = "amd-memory/kv_segmented_storage.cpp"
PLANNER = "amd-memory/kv_segmented_loan_plan.cpp"
OWNED = ("prepare_vision_segmented_capacity.py", "manifest.json", "check_source.py", "README.md")
DECLARATION = ("    [[nodiscard]] Transaction begin_loan(std::uint64_t requested_bytes,\n"
               "                                       std::span<const hipStream_t> all_prior_users);")
OPTIONAL_DECLARATION = (
    "\n    // Only the frozen planner's insufficient-unpromised-capacity result is empty.\n"
    "    // Invalid input, controller/poison, HIP and allocator errors still propagate.\n"
    "    // One real allocator Freeze covers planning through publication, as above.\n"
    "    [[nodiscard]] std::optional<Transaction> try_begin_loan(std::uint64_t requested_bytes,\n"
    "                                                          std::span<const hipStream_t> all_prior_users);")
BEGIN_SIGNATURE = ("KVSegmentedStorage::Transaction KVSegmentedStorage::begin_loan(\n"
                   "    std::uint64_t requested,std::span<const hipStream_t> users) {")
TRY_SIGNATURE = ("std::optional<KVSegmentedStorage::Transaction> KVSegmentedStorage::try_begin_loan(\n"
                 "    std::uint64_t requested,std::span<const hipStream_t> users) {")
CAPACITY_ERROR = 'std::runtime_error("insufficient unpromised segmented KV capacity")'
CAPACITY_THROW = "    if(!planned)throw " + CAPACITY_ERROR + ";"
CAPACITY_EMPTY = "    if(!planned)return std::nullopt;"
WRAPPER = (BEGIN_SIGNATURE + "\n"
           "    auto loan=try_begin_loan(requested,users);\n"
           "    if(!loan)throw " + CAPACITY_ERROR + ";\n"
           "    return std::move(*loan);\n}\n\n")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw: bytes):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("non-finite JSON value: " + value)

    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)


def exact(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError("frozen optional-capacity anchor changed: " + repr(old[:96]))
    return text.replace(old, new, 1)


def transform(raw: dict[str, bytes]) -> tuple[dict[str, bytes], dict]:
    """Only two exact text changes; inverse proves the whole original body.

    Filesystem/recipe pins are checked separately by verify_controller(). This
    function is pure and can be reviewed/tested without generating any view.
    """
    original_h = raw[HEADER].decode("utf-8")
    original_c = raw[SOURCE].decode("utf-8")
    h = exact(original_h, "#include <memory>\n", "#include <memory>\n#include <optional>\n")
    h = exact(h, DECLARATION, DECLARATION + OPTIONAL_DECLARATION)
    c = exact(original_c, BEGIN_SIGNATURE, WRAPPER + TRY_SIGNATURE)
    c = exact(c, CAPACITY_THROW, CAPACITY_EMPTY)
    restored_h = exact(h, DECLARATION + OPTIONAL_DECLARATION, DECLARATION)
    restored_h = exact(restored_h, "#include <memory>\n#include <optional>\n", "#include <memory>\n")
    restored_c = exact(c, WRAPPER + TRY_SIGNATURE, BEGIN_SIGNATURE)
    restored_c = exact(restored_c, CAPACITY_EMPTY, CAPACITY_THROW)
    if restored_h != original_h or restored_c != original_c:
        raise ValueError("optional-capacity inverse no longer restores the full original sources")
    # The actual planner has exactly one nullopt. Its hash and the whole original
    # transaction body are pinned: no capacity pre-query, second Freeze, catch,
    # altered grant, synchronization, rollback, close, or destructor is added.
    if c.count("const auto planned=plan_kv_segmented_loan(") != 1:
        raise ValueError("one original planner invocation required")
    if not (c.index("auto frozen=s.freeze();") < c.index(CAPACITY_EMPTY)
            < c.index("    s.next=s.current;") < c.index("s.grant=frozen->lend_selected(")):
        raise ValueError("capacity refusal must precede all placement/grant mutation in the same Freeze")
    out = dict(raw)
    out[HEADER], out[SOURCE] = h.encode("utf-8"), c.encode("utf-8")
    return out, {"changed_outputs": [HEADER, SOURCE],
                 "whole_original_header_and_source_inverse_exact": True,
                 "same_original_freeze_planner_and_transaction_body": True,
                 "only_empty_result": "frozen planner's wanted > available_pages",
                 "throwing_entry_capacity_type": "std::runtime_error",
                 "throwing_entry_capacity_message": "insufficient unpromised segmented KV capacity",
                 "all_other_exceptions_propagate": True,
                 "no_new_payload_stream_event_or_table": True}


def relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("invalid inventory path")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or path.as_posix() != value or any(p in (".", "..") for p in path.parts):
        raise ValueError("noncanonical inventory path")
    return value


def reparse(path: Path) -> bool:
    info = path.lstat()
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0)
                                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def bounded_file(path: Path, limit: int = 2 * 1024**2) -> bytes:
    if reparse(path) or not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError("missing/non-small/reparse source: " + str(path))
    return path.read_bytes()


def verify_inventory(root: Path, expected: dict[str, str], receipt_name: str) -> dict[str, bytes]:
    if reparse(root) or not root.is_dir():
        raise ValueError("source view must be a real directory")
    if not isinstance(expected, dict) or not 10 <= len(expected) <= 512:
        raise ValueError("bounded complete Core inventory required")
    wanted = {relative_path(k) for k in expected} | {receipt_name}
    found = set()
    for path in root.rglob("*"):
        if reparse(path):
            raise ValueError("source inventory contains a reparse point")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name not in wanted:
                raise ValueError("unrecorded compiler-visible source: " + name)
            found.add(name)
    if found != wanted:
        raise ValueError("complete source inventory differs")
    outputs, total = {}, 0
    for name, digest in expected.items():
        data = bounded_file(root / name)
        total += len(data)
        if total > 16 * 1024**2 or sha(data) != digest:
            raise ValueError("changed/oversized source view: " + name)
        outputs[name] = data
    return outputs


def verify_controller(view: Path, manifest: dict) -> tuple[bytes, dict, dict[str, bytes]]:
    if reparse(view):
        raise ValueError("controller root is a reparse point")
    receipt_raw = bounded_file(view / INPUT_RECEIPT, 1024**2)
    if sha(receipt_raw) != manifest["controller_receipt_sha256"]:
        raise ValueError("not the exact frozen controller receipt; create a reviewed new revision")
    receipt = strict_json(receipt_raw)
    if (receipt.get("all_core_rebuild_required") is not True
            or receipt.get("old_core_or_old_candidate_archives_allowed") is not False
            or receipt.get("production_gate_changed") is not False):
        raise ValueError("not the full isolated, production-disabled controller Core")
    raw = verify_inventory(view, receipt["output_sha256"], INPUT_RECEIPT)
    if (len(raw) != manifest["source_view_outputs"]
            or receipt.get("owned_recipe_sha256") != manifest["controller_recipe_sha256"]):
        raise ValueError("named frozen controller inventory/recipe pin differs")
    for name, key in (("src/core/paged_kv_cache.h", "controller_header_sha256"),
                      (HEADER, "storage_header_sha256"), (SOURCE, "storage_source_sha256"),
                      (PLANNER, "planner_source_sha256")):
        if sha(raw[name]) != manifest[key]:
            raise ValueError("named frozen source pin differs: " + name)
    sources = receipt["compile_sources"]
    if (not isinstance(sources, list) or not sources or len(sources) != len(set(sources))
            or len(sources) != manifest["core_compile_sources"]
            or any(relative_path(s) not in raw or Path(s).suffix not in (".cpp", ".cu") for s in sources)):
        raise ValueError("invalid full Core compilation inventory")
    if b"#define NINFER_AMD_SEGMENTED_CONTROLLER_HANDOFF_ABI 1" not in raw["src/core/paged_kv_cache.h"]:
        raise ValueError("controller header ABI marker absent")
    planner = raw[PLANNER].decode("utf-8")
    if (planner.count("return std::nullopt;") != 1
            or "if (wanted > input.available_pages) return std::nullopt;" not in planner):
        raise ValueError("the frozen planner no longer has one capacity-only refusal")
    return receipt_raw, receipt, raw


def prepare(view: Path) -> tuple[dict[str, bytes], dict, str]:
    manifest = strict_json(bounded_file(ROOT / "manifest.json", 64 * 1024))
    previous_raw, previous, raw = verify_controller(view, manifest)
    outputs, proof = transform(raw)
    recipes = {name: sha(bounded_file(ROOT / name)) for name in OWNED}
    recipe = {"schema": "ninfer.vision_segmented_capacity.recipe.v1",
              "controller_receipt_sha256": sha(previous_raw), "owned_source_sha256": recipes}
    key = sha(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    receipt = {"schema_version": 1, "status": "SOURCE_GENERATED_NOT_BUILT_NOT_QUALIFIED",
               "recipe_sha256": key, "recipe": recipe, "controller_view": str(view.resolve()),
               "controller_receipt": previous, "compile_sources": previous["compile_sources"],
               "output_sha256": {name: sha(data) for name, data in outputs.items()},
               "mechanism_proof": proof, "all_core_rebuild_required": True,
               "old_core_or_old_candidate_archives_allowed": False,
               "production_gate_changed": False, "vision_broker_connected": False,
               "engine_provider_connected": False, "runtime_qualified": False}
    return outputs, receipt, key


def publish(output: Path, outputs: dict[str, bytes], receipt: dict, verify_existing: bool) -> None:
    receipt_raw = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if output.exists():
        if not verify_existing:
            raise ValueError("existing output requires explicit --verify-existing; never overwrite")
        old = bounded_file(output / OUTPUT_RECEIPT, 1024**2)
        if old != receipt_raw:
            raise ValueError("existing derivative receipt differs")
        verify_inventory(output, receipt["output_sha256"], OUTPUT_RECEIPT)
        return
    output.mkdir(parents=True, exist_ok=False)
    for name, data in outputs.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    with (output / OUTPUT_RECEIPT).open("xb") as stream:
        stream.write(receipt_raw)


def require_separate_output(path: Path, controller_view: Path) -> None:
    for source in (controller_view, ROOT.resolve()):
        if path == source or path in source.parents or source in path.parents:
            raise ValueError("derivative output must be separate from controller and recipe input trees")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-view", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--emit", action="store_true", help="write a NEW isolated view; default is plan-only")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if reparse(args.controller_view):
        raise ValueError("controller argument is a reparse point")
    view = args.controller_view.resolve()
    root = args.output_root.resolve()
    require_separate_output(root, view)
    outputs, receipt, key = prepare(view)
    output = root / key
    require_separate_output(output, view)
    if args.emit:
        publish(output, outputs, receipt, args.verify_existing)
    print(json.dumps({"status": "SOURCE_GENERATED_NOT_BUILT_NOT_QUALIFIED" if args.emit else "PLAN_ONLY_NOT_GENERATED",
                      "output": str(output), "recipe_sha256": key, "outputs": len(outputs),
                      "compile_sources": len(receipt["compile_sources"]),
                      "mechanism_proof": receipt["mechanism_proof"], "production_gate_changed": False}, indent=2))


if __name__ == "__main__":
    main()
