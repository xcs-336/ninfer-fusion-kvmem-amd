"""Read-only link-map gate for the isolated original-Core allocator fixture."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

# lld-link's real COFF maps omit .lib in archive:member records. MSVC/other
# map producers may retain it. Both spellings name the same required archive;
# retain the exact member and symbol checks, including old-Core rejection.
CANDIDATE = r"(?<![\w.-])ninfer_segmented_pool_candidate_amd(?:\.lib)?[:(]"
OLD_CORE = re.compile(r"(?<![\w.-])ninfer_core_amd(?:\.lib)?[:(][^\r\n]*paged_kv_cache\.cpp\.obj\b", re.I)
REQUIRED = {
    "new_original_core_member": (re.compile(CANDIDATE+r"[^\r\n]*paged_kv_cache\.cpp\.obj\b",re.I),"DeviceKVPagePool"),
    "new_original_core_plane_view": (re.compile(CANDIDATE+r"[^\r\n]*paged_kv_cache\.cpp\.obj\b",re.I),"plane_view"),
    "new_segmented_host_address_member": (re.compile(CANDIDATE+r"[^\r\n]*kv_segmented_storage\.cpp\.obj\b",re.I),"host_page_address"),
    "new_original_allocator_owner_member": (re.compile(CANDIDATE+r"[^\r\n]*kv_segmented_pool_owner\.cpp\.obj\b",re.I),"KVSegmentedPoolOwner"),
}

def check_lines(lines) -> dict:
    evidence={};old=[];line_count=0
    for line_count,line in enumerate(lines,1):
        if len(line)>1024*1024:
            raise ValueError("link-map line exceeds the metadata bound")
        if OLD_CORE.search(line):
            if len(old)<8:old.append({"line":line_count,"text":line.rstrip()[:2048]})
        for key,(member,symbol) in REQUIRED.items():
            if key not in evidence and member.search(line) and symbol in line:
                evidence[key]={"line":line_count,"text":line.rstrip()[:2048]}
    missing=[key for key in REQUIRED if key not in evidence]
    if old or missing:
        raise ValueError(json.dumps({"old_original_core_member_extracted":old,"missing_positive_witnesses":missing}))
    return {"status":"LINK_MAP_ACCEPTED_NOT_MATH_OR_RESOURCE_QUALIFIED","line_count":line_count,
            "positive_extraction_evidence":evidence,"old_original_core_member_extracted":False}

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map",required=True,type=Path)
    args=parser.parse_args();path=args.map.resolve()
    if not path.is_file() or not 0<path.stat().st_size<=64*1024*1024:
        raise ValueError("missing/empty/oversized diagnostic link map")
    with path.open("r",encoding="utf-8",errors="strict") as stream:
        receipt=check_lines(stream)
    receipt["map_path"]=str(path);receipt["map_bytes"]=path.stat().st_size
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=="__main__":main()
