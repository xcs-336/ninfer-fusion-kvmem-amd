"""Small strict Broker/new full Core link gate, never physical qualification."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

CORE="ninfer_segmented_vision_core_amd"
BROKER="ninfer_segmented_vision_broker_amd"
FIXTURE="test_segmented_vision_broker.cu.obj"
ARCHIVE=re.compile(r"(?<![\w.-])(?P<name>ninfer_[\w.-]+)[:(]",re.I)
def archive(name):return re.compile(r"(?<![\w.-])"+re.escape(name)+r"(?:\.lib)?[:(]",re.I)
CORE_RE=archive(CORE);BROKER_RE=archive(BROKER)
CORE_MEMBERS={"paged_kv_cache.cpp.obj","kv_segmented_pool_owner.cpp.obj","kv_segmented_storage.cpp.obj",
              "kv_segmented_loan_plan.cpp.obj","evictable_weight_pool.cpp.obj","device.cu.obj"}
REQUIRED={
 "Core_table":(CORE,"paged_kv_cache.cpp.obj","plane_view"),
 "Core_factory":(CORE,"kv_segmented_pool_owner.cpp.obj","KVSegmentedNativePool"),
 "Core_optional":(CORE,"kv_segmented_storage.cpp.obj","try_begin_loan"),
 "Core_throwing":(CORE,"kv_segmented_storage.cpp.obj","begin_loan"),
 "real_WeightTail":(CORE,"evictable_weight_pool.cpp.obj","evict"),
 "real_Weight_close":(CORE,"evictable_weight_pool.cpp.obj","close"),
 "Broker_select":(BROKER,"vision_broker.cpp.obj","enable_segmented_tier"),
 "Broker_typed_admission":(BROKER,"vision_broker.cpp.obj","try_acquire_segmented"),
 "Broker_original_fallback":(BROKER,"vision_broker.cpp.obj","acquire"),
 "Window_checked_return":(BROKER,"vision_broker.cpp.obj","close_checked"),
 "Window_ready":(BROKER,"vision_broker.cpp.obj","wait_ready"),
}
def member(line,name):return re.search(r"(?<![\w.-])"+re.escape(name)+r"(?=\s|\)|$)",line) is not None
def symbol(line):
 m=re.match(r"^\s*[0-9A-Fa-f]{4}:[0-9A-Fa-f]{8,16}\s+(\S+)\s+[0-9A-Fa-f]{8,16}(?=\s|$)",line)
 return m.group(1) if m else None
def direct_fixture_main(line):
 # A substring/basename suffix does not prove who owns this main. Consume the
 # complete LLD owner field: optional known f/i markers, then exactly one direct
 # fixture basename. Archive/member, paths, extra tokens and unknown flags fail.
 m=re.match(r"^\s*[0-9A-Fa-f]{4}:[0-9A-Fa-f]{8,16}\s+main\s+"
            r"[0-9A-Fa-f]{8,16}\s+(.*?)\s*$",line)
 if not m:return False
 fields=m.group(1).split()
 if not fields or fields[-1]!=FIXTURE:return False
 flags=fields[:-1]
 return len(flags)<=2 and len(set(flags))==len(flags) and all(f in ("f","i") for f in flags)
def function(value,name):
 if name=="KVSegmentedNativePool":return value.startswith("??0"+name+"@")
 return value==name or value.startswith("?"+name+"@")
def check_lines(lines):
 found={};bad=[];main=None
 for number,line in enumerate(lines,1):
  if len(line)>1024**2:raise ValueError("oversized map row")
  owners=[m.group("name").lower().removesuffix(".lib") for m in ARCHIVE.finditer(line)]
  reason=None
  if any(o not in (CORE,BROKER) for o in owners):reason="foreign ninfer archive (old Core/Models/Ops/support forbidden)"
  if any(member(line,n) for n in CORE_MEMBERS) and not CORE_RE.search(line):reason="Core/weight/storage member outside new full Core"
  if member(line,"vision_overlay.cpp.obj"):reason="unextracted old full Models Vision TU"
  if member(line,"vision_broker.cpp.obj") and not BROKER_RE.search(line):reason="Broker member outside new component archive"
  if reason and len(bad)<12:bad.append({"line":number,"reason":reason,"text":line.rstrip()[:2048]})
  value=symbol(line)
  if value:
   for key,(owner,obj,name) in REQUIRED.items():
    if key not in found and archive(owner).search(line) and member(line,obj) and function(value,name):
     found[key]={"line":number,"text":line.rstrip()[:2048]}
   if value=="main" and direct_fixture_main(line):main={"line":number,"text":line.rstrip()[:2048]}
 missing=sorted(set(REQUIRED)-found.keys())
 if missing or bad or main is None:raise ValueError(json.dumps({"missing":missing,"foreign":bad,"fixture_main":main}))
 return {"status":"REAL_BROKER_AND_SINGLE_NEW_CORE_MAP_ONLY","positive_evidence":found,"fixture_main":main,
         "old_Core_ModelRuntime_Ops_extracted":False,"physical_or_WeightTail_failure_recovery_qualified":False}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--map",required=True,type=Path);a=p.parse_args()
 if not a.map.is_file() or not 0<a.map.stat().st_size<=64*1024**2:raise ValueError("missing/oversized map")
 with a.map.open(encoding="utf-8",errors="strict") as f:result=check_lines(f)
 print(json.dumps(result))
if __name__=="__main__":main()
