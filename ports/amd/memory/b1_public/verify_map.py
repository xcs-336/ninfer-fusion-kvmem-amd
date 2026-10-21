"""Actual final map gate: new public/Core objects, no old by-value paged ABI."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
OLD_OP = re.compile(r"(?<![\w.-])ninfer_[^\s:()]*?(?:attention|kv_cache_append|test_support)[^\s:()]*?(?:\.lib)?[:(]", re.I)
PUBLIC_ARCHIVE = re.compile(r"(?<![\w.-])ninfer_segmented_b1_public_ops_amd(?:\.lib)?[:(]", re.I)
EXTERNAL_SHA256 = {
    "verify_segmented_pool_map.py": "05fc5f91db3aab76707c36d08959f545d48228648d9ea094fbf32833d498e8e4",
    "verify_segmented_bf16_map.py": "0906d41f704bb95d162e4e8799eded354b719a83c12a5817b7ac1b2e6abfa582",
}
REQUIRED = {
    "actual_public_policy": ("public.cpp.obj", "segmented_b1_causal_softmax_attention"),
    "actual_split_launcher": ("small_t.cu.obj", "segmented_b1_causal_attention_small_t_launch"),
    "actual_prompt_launcher": ("prompt.cu.obj", "segmented_b1_causal_attention_prompt_launch"),
    "actual_append_launcher": ("append.cu.obj", "segmented_b1_kv_cache_append_batch_launch"),
}


def external_inputs():
    raw = {(ROOT.parent / "tests" / name): (ROOT.parent / "tests" / name).read_bytes()
           for name in EXTERNAL_SHA256}
    for path, value in raw.items():
        if hashlib.sha256(value).hexdigest() != EXTERNAL_SHA256[path.name]:
            raise ValueError("reviewed external map parser changed: " + str(path))
    return raw


def check_lines(lines):
    external_inputs()
    path = ROOT.parent / "tests/verify_segmented_pool_map.py"
    spec = importlib.util.spec_from_file_location("segmented_core_map", path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    found, forbidden, main = {}, [], None

    def witness():
        nonlocal main
        for number, line in enumerate(lines, 1):
            if OLD_OP.search(line) and len(forbidden) < 8:
                forbidden.append({"line": number, "text": line.rstrip()[:2048]})
            if re.search(r"(?<![\w.-])test_public_bf16\.cu\.obj(?=\s|\)|$)",line) and re.search(r"\bmain\b", line):
                main = {"line": number, "text": line.rstrip()[:2048]}
            for key, (member, symbol) in REQUIRED.items():
                exact_member = re.search(r"(?<![\w.-])" + re.escape(member) + r"(?=\s|\)|$)", line)
                if key not in found and PUBLIC_ARCHIVE.search(line) and exact_member and symbol in line:
                    found[key] = {"line": number, "text": line.rstrip()[:2048]}
            yield line

    result = core.check_lines(witness())
    missing = sorted(set(REQUIRED) - found.keys())
    if forbidden or missing or not main:
        raise ValueError(json.dumps({"old_op_or_test_support_extracted": forbidden, "missing_new_public_members": missing, "fixture_main": main}))
    return {"status": "LINK_CLOSURE_ACCEPTED_NOT_FUNCTIONAL_OR_RESOURCE_QUALIFIED",
            "core": result, "new_public_objects": found, "fixture_main": main,
            "old_view_op_or_test_support_members": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    args = parser.parse_args()
    path = args.map.resolve()
    if not path.is_file() or not 0 < path.stat().st_size <= 64 * 1024 * 1024:
        raise ValueError("missing/empty/oversized map")
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        result = check_lines(stream)
    result["map"] = str(path)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
