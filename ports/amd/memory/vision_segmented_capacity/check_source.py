"""Small source-only checks; never generates a view or executes native code."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-view", type=Path, required=True)
    args = parser.parse_args()
    path = ROOT / "prepare_vision_segmented_capacity.py"
    ast.parse(path.read_text(encoding="utf-8"))
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("vision_segmented_capacity_source_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.strict_json((ROOT / "manifest.json").read_bytes())
    receipt_raw, receipt, raw = module.verify_controller(args.controller_view, manifest)
    out, proof = module.transform(raw)
    changed = sorted(k for k in raw if raw[k] != out[k])
    if changed != sorted((module.HEADER, module.SOURCE)):
        raise AssertionError("only the two reviewed source outputs may differ")
    if len(out) != 67 or len(receipt["compile_sources"]) != 25:
        raise AssertionError("frozen full-Core inventory changed")
    for name, key in (("src/core/paged_kv_cache.h", "controller_header_sha256"),
                      (module.HEADER, "storage_header_sha256"),
                      (module.SOURCE, "storage_source_sha256"),
                      (module.PLANNER, "planner_source_sha256")):
        if module.sha(raw[name]) != manifest[key]:
            raise AssertionError("named manifest source pin differs")
    rejected = 0
    for name, old, new in ((module.HEADER, b"Transaction begin_loan", b"Transaction changed_loan"),
                           (module.SOURCE, module.CAPACITY_THROW.encode(), b"    if(!planned)changed();"),
                           (module.SOURCE, module.BEGIN_SIGNATURE.encode(), b"changed_signature {")):
        broken = dict(raw)
        broken[name] = broken[name].replace(old, new, 1)
        try:
            module.transform(broken)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("changed source anchor incorrectly admitted")

    # This deliberately tiny control model is not an implementation of the
    # planner/transaction and is not mathematical or runtime qualification.
    # It witnesses only caller classification and identity-preserving rethrow.
    events = []
    sentinel = object()

    class PlannerError(RuntimeError):
        pass

    def classify(planner):
        events.append("one_freeze")
        try:
            planned = planner()
            if planned is None:
                return None
            events.append("grant_after_plan")
            return sentinel
        finally:
            events.append("release_freeze")

    if classify(lambda: None) is not None or events != ["one_freeze", "release_freeze"]:
        raise AssertionError("refusal control mutated its grant")
    events.clear()
    if classify(lambda: object()) is not sentinel or events != ["one_freeze", "grant_after_plan", "release_freeze"]:
        raise AssertionError("admission control changed ticket identity")
    events.clear()
    error = PlannerError("noncapacity failure")

    def fail():
        raise error

    try:
        classify(fail)
    except PlannerError as caught:
        if caught is not error or events != ["one_freeze", "release_freeze"]:
            raise AssertionError("noncapacity error was changed")
    else:
        raise AssertionError("noncapacity failure incorrectly became empty")

    print(json.dumps({"status": "SOURCE_ONLY_IN_MEMORY_CHECKS_PASSED_NOT_RUNTIME_QUALIFIED",
                      "controller_receipt_sha256": module.sha(receipt_raw),
                      "input_outputs": len(raw), "unchanged_outputs": len(raw) - len(changed),
                      "changed_outputs": changed, "source_anchor_refusals": rejected,
                      "tiny_control_cases": 3, "mechanism_proof": proof,
                      "view_generated": False, "compiler_or_native_run": False,
                      "output_sha256_in_memory": {k: module.sha(out[k]) for k in changed}}, indent=2))


if __name__ == "__main__":
    main()
