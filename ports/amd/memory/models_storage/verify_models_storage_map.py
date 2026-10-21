"""Exact source-closure map witness, never mathematical/Engine qualification."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

def member(library:str,basename:str):
    return re.compile(r'(?<![\w.-])'+re.escape(library)+r'(?:\.lib)?[:(][^\r\n]*(?<![\w.-])'+
        re.escape(basename)+r'(?=[\s)]|$)',re.I)
MODELS='ninfer_models_storage_candidate_amd'
CORE='ninfer_controller_handoff_core_amd'
OLD=re.compile(r'(?<![\w.-])(?:ninfer_core_amd|ninfer_segmented_pool_candidate_amd|ninfer_engine_amd|ninfer_models_amd|ninfer_model_runtime_amd|ninfer_model_loading_amd)(?:\.lib)?[:(]',re.I)
REQUIRED={
 'new_models_constructor':(member(MODELS,'decoder_state.cpp.obj'),'PagedKVCache'),
 'new_split_planner':(member(MODELS,'decoder_state.cpp.obj'),'plan_segmented_decoder_state'),
 'new_models_layer_view':(member(MODELS,'decoder_state.cpp.obj'),'layer_view'),
 'new_core_pool':(member(CORE,'paged_kv_cache.cpp.obj'),'DeviceKVPagePool'),
 'new_core_plane_view':(member(CORE,'paged_kv_cache.cpp.obj'),'plane_view'),
 'new_native_owner':(member(CORE,'kv_segmented_pool_owner.cpp.obj'),'KVSegmentedPoolOwner'),
 'new_native_storage':(member(CORE,'kv_segmented_storage.cpp.obj'),'host_page_address'),
 'fixture':(re.compile(r'(?<![\w.-])test_models_storage\.cu\.obj(?=[\s)]|$)',re.I),'main'),
}
def check_lines(lines)->dict:
    evidence={};old=[];count=0
    for count,line in enumerate(lines,1):
        if len(line)>1024**2:raise ValueError('oversized map line')
        if OLD.search(line) and len(old)<8:old.append({'line':count,'text':line.rstrip()[:2048]})
        for key,(pattern,symbol) in REQUIRED.items():
            if key not in evidence and pattern.search(line) and symbol in line:
                evidence[key]={'line':count,'text':line.rstrip()[:2048]}
    missing=[key for key in REQUIRED if key not in evidence]
    if old or missing:raise ValueError(json.dumps({'forbidden_old_archives':old,'missing':missing}))
    return {'status':'NEW_MODELS_FULL_CORE_LINK_CLOSURE_ONLY','positive_evidence':evidence,
        'old_archive_members_extracted':False,'line_count':count,'runtime_qualified':False}
def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--map',required=True,type=Path)
    args=parser.parse_args();path=args.map.resolve()
    if not path.is_file() or not 0<path.stat().st_size<=64*1024**2:raise ValueError('missing/oversized map')
    with path.open(encoding='utf-8',errors='strict') as stream:report=check_lines(stream)
    report['map_path']=str(path);print(json.dumps(report))
if __name__=='__main__':main()
