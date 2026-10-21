"""Strict final worker Engine link evidence; no archive is admitted by link success."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
START=r'(?<![\w.-])';END=r'(?=[\s)]|$)'
def archive(name:str)->str:return START+re.escape(name)+r'(?:\.lib)?[:(]'
def member(name:str)->str:return START+re.escape(name)+END
def lld_symbol(line:str)->str|None:
    match=re.match(r'^\s*[0-9A-Fa-f]{4}:[0-9A-Fa-f]{8,16}\s+(\S+)\s+[0-9A-Fa-f]{8,16}(?=\s|$)',line)
    return match[1] if match else None
REQUIRED={
 'ninfer_models_engine_worker_core_amd':('paged_kv_cache.cpp.obj','decode_graph.cpp.obj',
    'kv_segmented_pool_owner.cpp.obj','kv_segmented_storage.cpp.obj','kv_segmented_loan_plan.cpp.obj'),
 'ninfer_models_engine_worker_model_runtime_amd':('decoder_state.cpp.obj','program_impl.cpp.obj',
    'program.cpp.obj','text.cpp.obj','graphs.cpp.obj','startup.cpp.obj'),
 'ninfer_models_engine_worker_model_loading_amd':('config.cpp.obj','model.cpp.obj','load.cpp.obj'),
 'ninfer_models_engine_worker_engine_amd':('engine.cpp.obj','model_instance.cpp.obj'),
}
FORBIDDEN=('ninfer_core_amd','ninfer_controller_handoff_core_amd','ninfer_segmented_pool_candidate_amd',
 'ninfer_models_storage_candidate_amd','ninfer_model_runtime_amd','ninfer_model_loading_amd','ninfer_engine_amd',
 'ninfer_segmented_b1_public_amd','ninfer_segmented_bf16_ops_amd','ninfer_attention_ops_amd',
 'ninfer_attention_context_ops_amd','ninfer_attention_storage_ops_amd','ninfer_attention_packed_ops_amd')
def check(text:str)->None:
    if not text or len(text)>128*1024**2:raise ValueError('empty/nonbounded actual map')
    lines=text.splitlines()
    for library,members in REQUIRED.items():
        for obj in members:
            if not any(re.search(archive(library)+r'[^\r\n]*'+member(obj),line) for line in lines):
                raise ValueError('missing exact actual provider: '+library+':'+obj)
    if not any(lld_symbol(line)=='main' and re.search(member('test_models_engine_worker.cpp.obj'),line) for line in lines):
        raise ValueError('missing real public Engine fixture object')
    for name in FORBIDDEN:
        if re.search(archive(name),text):raise ValueError('forbidden old ABI extraction: '+name)
    for symbol in ('release_startup_controller','accept_startup_controller','retire_worker_controller',
                   'segmented_storage_snapshot','segmented_b1_causal_softmax_attention'):
        if symbol not in text:raise ValueError('required real lifecycle/Op symbol absent: '+symbol)
    # Reuse the independent Ops-specific original-manifest member allowlist and
    # exact B1/cache producer proof. It allows ONLY this named full Core provider.
    path=Path(__file__).resolve().parent.parent/'models_engine_ops/verify_map.py'
    pins=json.loads((Path(__file__).resolve().parent/'dependency-pins.json').read_text(encoding='utf8'))['source_sha256']
    key='ports/amd/memory/models_engine_ops/verify_map.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=pins[key]:raise ValueError('independent Ops map gate changed since this freeze')
    scope={'__name__':'worker_map_dependency','__file__':str(path)}
    exec(compile(path.read_text(encoding='utf8'),str(path),'exec'),scope)
    scope['check_lines'](lines,fixture=False,core_archive='ninfer_models_engine_worker_core_amd',
        main_router_archive='ninfer_models_engine_worker_model_runtime_amd')
def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--map',required=True,type=Path);a=p.parse_args()
    if not 0<a.map.stat().st_size<=128*1024**2:raise ValueError('actual map bound')
    check(a.map.read_text(encoding='utf8',errors='strict'))
    print('MODELS_ENGINE_WORKER_EXACT_MAP_PASS')
if __name__=='__main__':main()
