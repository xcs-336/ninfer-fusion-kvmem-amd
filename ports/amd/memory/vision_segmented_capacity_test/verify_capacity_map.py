"""Strict new capacity-Core extraction gate; no runtime/math/resource admission."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

CORE = "ninfer_vision_segmented_capacity_core_amd"
ARCHIVE = re.compile(r"(?<![\w.-])(?P<name>ninfer_[\w.-]+)[:(]", re.I)
SELECTED = re.compile(r"(?<![\w.-])" + re.escape(CORE) + r"(?:\.lib)?[:(]", re.I)
PROTECTED = {"paged_kv_cache.cpp.obj", "kv_segmented_pool_owner.cpp.obj",
             "kv_segmented_storage.cpp.obj", "kv_segmented_loan_plan.cpp.obj"}
REQUIRED = {
    "real_original_allocator": ("paged_kv_cache.cpp.obj", "DeviceKVPagePool"),
    "real_table_view": ("paged_kv_cache.cpp.obj", "plane_view"),
    "real_single_payload_factory": ("kv_segmented_pool_owner.cpp.obj", "KVSegmentedNativePool"),
    "real_controller_freeze": ("kv_segmented_pool_owner.cpp.obj", "freeze"),
    "optional_entry": ("kv_segmented_storage.cpp.obj", "try_begin_loan"),
    "throwing_entry": ("kv_segmented_storage.cpp.obj", "begin_loan"),
    "original_close": ("kv_segmented_storage.cpp.obj", "close_after_streams"),
    "real_planner": ("kv_segmented_loan_plan.cpp.obj", "plan_kv_segmented_loan"),
}
FIXTURE = "test_vision_segmented_capacity.cu.obj"


def member(line: str, name: str) -> bool:
    return re.search(r"(?<![\w.-])" + re.escape(name) + r"(?=\s|\)|$)", line) is not None


def lld_symbol(line: str) -> str | None:
    match = re.match(r"^\s*[0-9A-Fa-f]{4}:[0-9A-Fa-f]{8,16}\s+(\S+)\s+"
                     r"[0-9A-Fa-f]{8,16}(?=\s|$)", line)
    return match.group(1) if match else None


def direct_fixture_main(line: str) -> bool:
    # LLD's complete owner field must be a direct object, never an archive
    # member, a path with the same basename, or a symbol containing that name.
    return re.fullmatch(
        r"\s*[0-9A-Fa-f]{4}:[0-9A-Fa-f]{8,16}\s+main\s+"
        r"[0-9A-Fa-f]{8,16}\s+(?:(?:f\s+i|i\s+f|f|i)\s+)?"
        + re.escape(FIXTURE) + r"\s*", line) is not None


def identifier(symbol: str, name: str) -> bool:
    if name in ("DeviceKVPagePool", "KVSegmentedNativePool"):
        return any(symbol.startswith(prefix + name + "@") for prefix in ("??0", "??1"))
    # The LLD symbol field names the function itself, not an exception/unwind
    # metadata symbol containing that name or try_begin_loan's suffix.
    return symbol == name or symbol.startswith("?" + name + "@")


def check_lines(lines) -> dict:
    found, forbidden, main = {}, [], None
    for number, line in enumerate(lines, 1):
        if len(line) > 1024**2:
            raise ValueError("oversized map row")
        symbol = lld_symbol(line)
        archives = [m.group("name").lower().removesuffix(".lib") for m in ARCHIVE.finditer(line)]
        reason = None
        if any(name != CORE for name in archives):
            reason = "other ninfer archive extracted; only one full new capacity Core is allowed"
        if any(member(line, name) for name in PROTECTED) and not SELECTED.search(line):
            reason = "Core/storage/planner member outside selected new Core archive"
        if reason and len(forbidden) < 12:
            forbidden.append({"line": number, "reason": reason, "text": line.rstrip()[:2048]})
        if symbol is not None and SELECTED.search(line):
            for key, (obj, name) in REQUIRED.items():
                if key not in found and member(line, obj) and identifier(symbol, name):
                    found[key] = {"line": number, "text": line.rstrip()[:2048]}
        if symbol == "main" and direct_fixture_main(line):
            main = {"line": number, "text": line.rstrip()[:2048]}
    missing = sorted(set(REQUIRED) - found.keys())
    if missing or forbidden or main is None:
        raise ValueError(json.dumps({"missing": missing, "foreign_extraction": forbidden, "fixture_main": main}))
    return {"status": "FULL_NEW_CAPACITY_CORE_MAP_ONLY", "selected_core": CORE,
            "positive_evidence": found, "fixture_main": main,
            "old_core_or_old_storage_extracted": False, "runtime_qualified": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path)
    args = parser.parse_args()
    if not args.map.is_file() or not 0 < args.map.stat().st_size <= 64 * 1024**2:
        raise ValueError("missing/oversized map")
    with args.map.open(encoding="utf-8", errors="strict") as stream:
        result = check_lines(stream)
    result["map_path"] = str(args.map.resolve())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
