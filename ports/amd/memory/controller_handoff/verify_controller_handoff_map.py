"""Small read-only map gate; link closure is not functional/Engine qualification."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

NEW=r"(?<![\w.-])ninfer_controller_handoff_core_amd(?:\.lib)?[:(]"
OLD=re.compile(r"(?<![\w.-])(?:ninfer_core_amd|ninfer_segmented_pool_candidate_amd)(?:\.lib)?[:(]",re.I)
REQUIRED={
    "new_paged_core":(re.compile(NEW+r"[^\r\n]*(?<![\w.-])paged_kv_cache\.cpp\.obj(?=[\s)]|$)",re.I),"DeviceKVPagePool"),
    "new_controller_view":(re.compile(NEW+r"[^\r\n]*(?<![\w.-])paged_kv_cache\.cpp\.obj(?=[\s)]|$)",re.I),"plane_view"),
    "release":(re.compile(NEW+r"[^\r\n]*(?<![\w.-])kv_segmented_pool_owner\.cpp\.obj(?=[\s)]|$)",re.I),"release_controller_for_worker"),
    "consume":(re.compile(NEW+r"[^\r\n]*(?<![\w.-])kv_segmented_pool_owner\.cpp\.obj(?=[\s)]|$)",re.I),"consume_controller_for_worker"),
    "retire":(re.compile(NEW+r"[^\r\n]*(?<![\w.-])kv_segmented_pool_owner\.cpp\.obj(?=[\s)]|$)",re.I),"retire_controller_for_teardown"),
    "new_storage":(re.compile(NEW+r"[^\r\n]*(?<![\w.-])kv_segmented_storage\.cpp\.obj(?=[\s)]|$)",re.I),"host_page_address"),
    "fixture":(re.compile(r"(?<![\w.-])test_controller_handoff\.cu\.obj(?=[\s)]|$)",re.I),"main"),
}

def check_lines(lines) -> dict:
    evidence={};old=[];count=0
    for count,line in enumerate(lines,1):
        if len(line)>1024**2:raise ValueError("oversized map line")
        if OLD.search(line) and len(old)<8:old.append({"line":count,"text":line.rstrip()[:2048]})
        for key,(member,symbol) in REQUIRED.items():
            if key not in evidence and member.search(line) and symbol in line:
                evidence[key]={"line":count,"text":line.rstrip()[:2048]}
    missing=[key for key in REQUIRED if key not in evidence]
    if old or missing:raise ValueError(json.dumps({"old_archives":old,"missing":missing}))
    return {"status":"NEW_FULL_CORE_LINK_CLOSURE_ONLY","positive_evidence":evidence,
            "old_core_or_old_candidate_extracted":False,"line_count":count,"runtime_qualified":False}

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--map",required=True,type=Path)
    args=parser.parse_args();path=args.map.resolve()
    if not path.is_file() or not 0<path.stat().st_size<=64*1024**2:raise ValueError("missing/oversized map")
    with path.open(encoding="utf-8",errors="strict") as stream:receipt=check_lines(stream)
    receipt["map_path"]=str(path);print(json.dumps(receipt))

if __name__=="__main__":main()
