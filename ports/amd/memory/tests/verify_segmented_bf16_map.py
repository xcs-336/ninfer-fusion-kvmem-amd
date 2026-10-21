"""Read-only map closure gate; successful link is not numerical/resource qualification."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_segmented_pool_map import check_lines as check_core_lines

FORBIDDEN = re.compile(
    r"\bninfer_[^\s:()]*?(?:attention|kv_cache_append|test_support)[^\s:()]*?(?:\.lib)?[:(]", re.I)
FIXTURE = "kv_segmented_bf16_attention_test.cu.obj"


def check_lines(lines) -> dict:
    bad, fixture = [], []

    def witnessed():
        for number, line in enumerate(lines, 1):
            if FORBIDDEN.search(line) and len(bad) < 8:
                bad.append({"line": number, "text": line.rstrip()[:2048]})
            if FIXTURE in line and re.search(r"\bmain\b", line) and not fixture:
                fixture.append({"line": number, "text": line.rstrip()[:2048]})
            yield line

    core = check_core_lines(witnessed())
    if bad or not fixture:
        raise ValueError(json.dumps({"forbidden_old_op_or_support_members": bad,
                                     "fixture_main_object_found": bool(fixture)}))
    return {"status": "LINK_CLOSURE_ACCEPTED_NOT_MATH_OR_RESOURCE_QUALIFIED",
            "core": core, "fixture_main": fixture[0],
            "old_op_or_test_support_members_extracted": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path)
    args = parser.parse_args()
    path = args.map.resolve()
    if not path.is_file() or not 0 < path.stat().st_size <= 64 * 1024 * 1024:
        raise ValueError("missing/empty/oversized diagnostic map")
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        receipt = check_lines(stream)
    receipt["map_path"] = str(path)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
