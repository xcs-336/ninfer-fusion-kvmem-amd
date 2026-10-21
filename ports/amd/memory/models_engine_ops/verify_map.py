"""Strict full-Core/new-B1/new-ordinary-cache extraction gate, never math proof."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

CORE='ninfer_controller_handoff_core_amd'
B1='ninfer_models_engine_b1_ops_amd'
CACHE='ninfer_models_engine_cache_ops_amd'
ARCHIVE=lambda name:re.compile(r'(?<![\w.-])'+re.escape(name)+r'(?:\.lib)?[:(]',re.I)
CORE_RE,B1_RE,CACHE_RE=map(ARCHIVE,(CORE,B1,CACHE))
BASIC_RE=ARCHIVE('ninfer_basic_ops_amd')
OLD=re.compile(r'(?<![\w.-])(?:ninfer_core_amd|ninfer_segmented_pool_candidate_amd|'
    r'ninfer_segmented_b1_public_ops_amd|ninfer_attention[^\s:()]*|'
    r'ninfer_[^\s:()]*kv_cache_append[^\s:()]*|ninfer_[^\s:()]*test_support[^\s:()]*)(?:\.lib)?[:(]',re.I)
CORE_MEMBERS={'paged_kv_cache.cpp.obj','kv_segmented_pool_owner.cpp.obj','kv_segmented_storage.cpp.obj'}
CACHE_MEMBERS={'causal_softmax_attention.cpp.obj','small_t.cu.obj','small_t_fp8.cu.obj',
    'small_t_nvfp4.cu.obj','small_t_k8v4.cu.obj','prompt.cu.obj','prompt_fp8.cu.obj',
    'prompt_nvfp4.cu.obj','prompt_k8v4.cu.obj','kv_cache_append.cpp.obj','nvfp4_launch.cu.obj',
    'k8v4_launch.cu.obj','context_softmax_attention.cpp.obj','packed_softmax_attention.cpp.obj',
    'sliding_window_attention.cpp.obj','context_kv_materialize.cpp.obj','materialize.cu.obj',
    'selected_block_attention.cu.obj','kv_stream.cu.obj'}
CACHE_MEMBERS.update(f'small_t_i8_w{w}_h{h}_{mode}.cu.obj'
    for h in (16,24) for w in range(1,7 if h==16 else 9) for mode in ('append','cached'))
CACHE_SYMBOL=re.compile(r'(?:causal_(?:softmax_)?attention|kv_cache_append|context_(?:softmax_)?attention|'
    r'context_kv_materialize|sliding_window_attention|packed_(?:softmax_)?attention|selected_block_attention)')
REQUIRED={
    'full_core_paged_view':(CORE_RE,'paged_kv_cache.cpp.obj','plane_view'),
    'full_core_owner':(CORE_RE,'kv_segmented_pool_owner.cpp.obj','KVSegmentedNativePool'),
    'full_core_storage':(CORE_RE,'kv_segmented_storage.cpp.obj','host_page_address'),
    'new_b1_public':(B1_RE,'public.cpp.obj','segmented_b1_causal_softmax_attention'),
    'new_b1_small':(B1_RE,'small_t.cu.obj','segmented_b1_causal_attention_small_t_launch'),
    'new_b1_prompt':(B1_RE,'prompt.cu.obj','segmented_b1_causal_attention_prompt_launch'),
    'new_b1_append':(B1_RE,'append.cu.obj','segmented_b1_kv_cache_append_batch_launch'),
    'ordinary_causal_definition':(CACHE_RE,'causal_softmax_attention.cpp.obj','causal_softmax_attention'),
    'ordinary_append_definition':(CACHE_RE,'kv_cache_append.cpp.obj','kv_cache_append'),
    'ordinary_context_definition':(CACHE_RE,'context_softmax_attention.cpp.obj','context_softmax_attention'),
}


def has_member(line:str,name:str)->bool:
    return re.search(r'(?<![\w.-])'+re.escape(name)+r'(?=\s|\)|$)',line) is not None


def lld_symbol(line:str)->str|None:
    # Actual Windows LLD Publics-by-Value columns: segment:offset, symbol, VA.
    # Exception metadata containing main is not the main entry point.
    match=re.match(r'^\s*[0-9A-Fa-f]{4}:[0-9A-Fa-f]{8,16}\s+(\S+)\s+'
                   r'[0-9A-Fa-f]{8,16}(?=\s|$)',line)
    return match.group(1) if match else None


# Deliberately exact: these are the five diagnosed guard-argument constants in
# the direct fixture object, not a general exception for DATA or ?? metadata.
# A compiler/mangler change fails closed and needs a reviewed new evidence set.
FIXTURE_GUARD_LITERALS=frozenset((
    '??_C@_0BH@CCMJLNH@kv_cache_append_prefix?$AA@',
    '??_C@_0BK@NHAHFCAK@context_softmax_attention?$AA@',
    '??_C@_0BJ@NAHLBFNO@causal_softmax_attention?$AA@',
    '??_C@_0BA@CEMMIBIJ@kv_cache_append?$AA@',
    '??_C@_0CA@JKMMJNID@causal_softmax_attention_cached?$AA@',
))


def lld_record(line:str)->dict|None:
    match=re.match(r'^\s*([0-9A-Fa-f]{4}):([0-9A-Fa-f]{8,16})\s+(\S+)\s+'
                   r'([0-9A-Fa-f]{8,16})\s+(.+?)\s*$',line)
    if not match:return None
    return {'segment':match[1].lower(),'offset':int(match[2],16),
            'symbol':match[3],'absolute':int(match[4],16),'owner':match[5]}


def lld_segments(lines)->dict:
    by_segment={}
    for number,line in enumerate(lines,1):
        # Class is intentionally not limited to CODE/DATA: unknown/mixed class
        # declarations must prevent the literal exception, not disappear.
        match=re.match(r'^\s*([0-9A-Fa-f]{4}):([0-9A-Fa-f]{8,16})\s+'
                       r'([0-9A-Fa-f]{1,16})H\s+(\S+)\s+(\S+)\s*$',line)
        if not match:continue
        start,length=int(match[2],16),int(match[3],16)
        by_segment.setdefault(match[1].lower(),[]).append(
            {'start':start,'end':start+length,'name':match[4],
             'class':match[5],'line':number})
    result={}
    for segment,parts in by_segment.items():
        ordered=sorted(parts,key=lambda part:(part['start'],part['end']))
        valid=all(part['start']<part['end']<=0xffffffffffffffff for part in ordered)
        valid=valid and all(a['end']<=b['start'] for a,b in zip(ordered,ordered[1:]))
        classes={part['class'] for part in ordered}
        result[segment]={'valid':valid,'unique_DATA':classes=={'DATA'},
            'start':ordered[0]['start'],'end':max(part['end'] for part in ordered),
            'contributions':ordered}
    return result


def fixture_guard_literal(line:str,segments:dict)->dict|None:
    record=lld_record(line)
    if not record or record['symbol'] not in FIXTURE_GUARD_LITERALS:return None
    if record['owner']!='test_models_engine_ops.cu.obj':return None
    segment=segments.get(record['segment'])
    if not segment or not segment['valid'] or not segment['unique_DATA']:return None
    if not segment['start']<=record['offset']<segment['end']:return None
    if record['absolute']<=record['offset']:return None
    # This is a segment envelope, not individual subsection coverage. LLD can
    # omit a merged-string contribution in the gap between named DATA parts.
    return {**record,'DATA_segment_start':segment['start'],'DATA_segment_end':segment['end']}


def check_lines(lines,fixture:bool=True,core_archive:str=CORE,main_router_archive:str|None=None)->dict:
    if core_archive not in (CORE,'ninfer_models_engine_worker_core_amd'):
        raise ValueError('only the two separately reviewed new full-Core archives are selectable')
    if main_router_archive is not None and not re.fullmatch(r'ninfer_models_engine_worker_[A-Za-z0-9_]+',main_router_archive):
        raise ValueError('explicit new worker Models archive identity required')
    actual_core=ARCHIVE(core_archive)
    actual_router=ARCHIVE(main_router_archive) if main_router_archive else None
    required={key:(actual_core if key.startswith('full_core_') else archive,member,symbol)
              for key,(archive,member,symbol) in REQUIRED.items()}
    basic_allowed=set(json.loads((Path(__file__).resolve().parent/'external-pins.json').read_text(encoding='utf-8'))['basic_archive']['allowed_members'])
    lines=list(lines) # Both header consistency and all later records must be seen.
    segments=lld_segments(lines)
    found={};forbidden=[];fixture_main=None;literal_exemptions=[]
    for number,line in enumerate(lines,1):
        if len(line)>1024**2:raise ValueError('oversized map line')
        obj=re.search(r'(?<![\w.-])[\w.-]+\.obj(?=\s|\)|$)',line)
        reason=None
        if OLD.search(line):reason='old ABI/archive extraction'
        if core_archive!=CORE and CORE_RE.search(line):reason='parent controller archive extracted into worker-Core closure'
        if BASIC_RE.search(line) and obj and not any(has_member(line,name) for name in basic_allowed):
            reason='Basic member outside exact Tensor-only original manifest allowlist'
        if any(has_member(line,name) for name in CORE_MEMBERS) and not actual_core.search(line):
            reason='paged Core/owner/storage member outside new full Core'
        cache_member=any(has_member(line,name) for name in CACHE_MEMBERS)
        literal=fixture_guard_literal(line,segments)
        if literal:literal_exemptions.append({'line':number,**literal})
        main_router=(actual_router is not None and actual_router.search(line) and
            has_member(line,'text.cpp.obj') and
            re.search(r'(?<![\w])models_engine_main_causal_softmax_attention(?=[@(<\s]|$)',line))
        # Exempt only cache-family text in this proven fixture constant. Never
        # skip the line: old archives/Core/cache members/Basic rules still run.
        if (obj or lld_record(line)) and (cache_member or (not literal and CACHE_SYMBOL.search(line))) and not (B1_RE.search(line) or CACHE_RE.search(line) or main_router):
            reason='cache consumer member/symbol outside the two new ABI archives'
        if reason and len(forbidden)<12:
            forbidden.append({'line':number,'reason':reason,'text':line.rstrip()[:2048]})
        for key,(archive,member,symbol) in required.items():
            if key not in found and archive.search(line) and has_member(line,member) and symbol in line:
                found[key]={'line':number,'text':line.rstrip()[:2048]}
        if has_member(line,'test_models_engine_ops.cu.obj') and lld_symbol(line)=='main':
            fixture_main={'line':number,'text':line.rstrip()[:2048]}
    missing=sorted(set(required)-found.keys())
    if missing or forbidden or (fixture and not fixture_main):
        raise ValueError(json.dumps({'missing':missing,'forbidden':forbidden,'fixture_main':fixture_main}))
    return {'status':'FULL_NEW_CORE_AND_CACHE_ABI_MAP_ONLY','positive_evidence':found,
        'fixture_main':fixture_main,'old_core_partial_b1_cache_consumers_extracted':False,
        'selected_full_core_archive':core_archive,'explicit_main_router_archive':main_router_archive,
        'fixture_guard_literal_symbol_exemptions':literal_exemptions,
        'math_resource_engine_or_performance_qualified':False}


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map',required=True,type=Path)
    parser.add_argument('--engine-map',action='store_true',help='worker consumer supplies its own Engine main/member gates')
    parser.add_argument('--core-archive',default=CORE)
    parser.add_argument('--main-router-archive')
    args=parser.parse_args();path=args.map.resolve()
    if not path.is_file() or not 0<path.stat().st_size<=64*1024**2:raise ValueError('missing/oversized map')
    with path.open(encoding='utf-8',errors='strict') as stream:
        result=check_lines(stream,not args.engine_map,args.core_archive,args.main_router_archive)
    result['map']=str(path);print(json.dumps(result))


if __name__=='__main__':main()
