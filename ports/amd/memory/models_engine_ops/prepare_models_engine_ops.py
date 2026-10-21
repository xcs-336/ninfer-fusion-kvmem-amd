"""NEW full-controller-Core ABI Ops view. Import and transform never write.

The future explicit CLI copies real prepared original cache families, inserts
five early table refusals, and rebases the independently tested B1 transform.
It never changes published/prepared/original/old candidate inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
GUARDS = {
    'src/ops/kv_cache/append/kv_cache_append.cpp': [
        ('void kv_cache_append(', 'PagedKVLayerView cache', 'cache', 'kv_cache_append'),
        ('void kv_cache_append_prefix(', 'PagedKVBatchLayerView cache', 'cache', 'kv_cache_append_prefix'),
    ],
    'src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp': [
        ('void causal_softmax_attention(', 'PagedKVBatchLayerView cache', 'cache', 'causal_softmax_attention'),
        ('void causal_softmax_attention_cached(', 'const PagedKVLayerView& cache', 'cache', 'causal_softmax_attention_cached'),
    ],
    'src/ops/softmax_attention/dense/context/context_softmax_attention.cpp': [
        ('void context_softmax_attention(', 'const PagedKVBatchLayerView& context', 'context', 'context_softmax_attention'),
    ],
}
OWNED = ('ModelsEngineOps.cmake','prepare_models_engine_ops.py','public_bf16.h',
         'affine_guard.h','bridge_models_calls.py','test_models_engine_ops.cu',
         'verify_map.py','external-pins.json','manifest.json','README.md')


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pins() -> dict:
    return json.loads((ROOT/'external-pins.json').read_text(encoding='utf-8'))


def bounded_file(path: Path, limit: int = 2*1024**2) -> bytes:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError('missing/non-small/symlinked source: '+str(path))
    return path.read_bytes()


def verify_external_recipes(inventory: dict) -> dict[str, str]:
    result = {}
    for relative, expected in inventory['external_recipe_sha256'].items():
        actual = sha(bounded_file(REPO/relative))
        if actual != expected:
            raise ValueError('frozen external source changed: '+relative)
        result[relative] = actual
    return result


def insert_guard(text: str, parser, marker: str, parameter: str, cache: str, op: str) -> tuple[str, str]:
    # One explicit reviewed public overload, not a generic C++ signature rewrite.
    choices = []
    for match in re.finditer(r'(?m)^'+re.escape(marker), text):
        opening = text.index('{', match.start())
        signature = text[match.start():opening]
        if parameter in signature:
            choices.append(opening)
    if len(choices) != 1:
        raise ValueError('public paged overload anchor changed: '+op)
    opening = choices[0]
    statement = ('\n    ::ninfer::amd::models_engine_ops::reject_table_backed_affine_entry('
                 +cache+', "'+op+'");')
    changed = text[:opening+1]+statement+text[opening+1:]
    # Mechanism proof: removing exactly the inserted entry statement reconstructs
    # the original body, including validation, all formats, quantizers and kernels.
    if changed.count(statement) != 1 or changed.replace(statement,'',1) != text:
        raise ValueError('entry guard was not an exact body-preserving insertion')
    return changed, statement


def transform(native: dict[str,str], old_b1, frozen_bf16) -> tuple[dict[str,str],dict]:
    outputs = dict(native)
    guard_rows = []
    for path, entries in GUARDS.items():
        source = native[path]
        changed = source
        statements = []
        for marker, parameter, cache, operation in entries:
            changed, statement = insert_guard(changed,old_b1,marker,parameter,cache,operation)
            statements.append(statement)
            guard_rows.append({'source':path,'entry':operation,'parameter':parameter,
                               'position':'first statement before original validation/scope/launch'})
        include = '#include "amd-models-b1/affine_guard.h"\n'
        outputs[path] = include+changed
        restored = outputs[path].removeprefix(include)
        for statement in statements:
            restored = restored.replace(statement,'',1)
        if restored != source:
            raise ValueError('ordinary original source no longer exactly recoverable: '+path)

    b1 = old_b1.transform(native,frozen_bf16)
    for relative,text in b1.items():
        target = relative.replace('amd-b1/','amd-models-b1/')
        outputs[target] = text.replace('amd-b1/','amd-models-b1/')
    outputs['amd-models-b1/public_bf16.h'] = (ROOT/'public_bf16.h').read_text(encoding='utf-8')
    outputs['amd-models-b1/affine_guard.h'] = (ROOT/'affine_guard.h').read_text(encoding='utf-8')
    for name in ('fixture_common.cuh','test_public_bf16.cu'):
        original = (ROOT.parent/'b1_public'/name).read_text(encoding='utf-8')
        outputs['amd-models-b1/'+name] = original.replace('amd-b1/','amd-models-b1/')

    # Shared Graph classifications keep the complete original query. Prove its
    # exact resolver and route-label bodies, not sampled route coincidences.
    path = 'src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp'
    rebased = b1['amd-b1/public.cpp']
    resolver_proofs = []
    for marker in ('CausalAttentionRoute causal_attention_resolve_route(',
                   'int causal_softmax_attention_route_family('):
        original_body = old_b1.function(native[path],marker)
        renamed_body = old_b1.function(rebased,old_b1.rename(marker))
        if old_b1.rename(original_body) != renamed_body:
            raise ValueError('ordinary and B1 Graph resolver bodies differ: '+marker)
        resolver_proofs.append({'marker':marker,'original_body_sha256':sha(original_body.encode()),
                                'private_prefix_only_equivalent':True})
    expected_b1 = {key.replace('amd-b1/','amd-models-b1/') for key in b1}
    if not expected_b1 <= outputs.keys() or len(guard_rows)!=5:
        raise ValueError('incomplete actual public B1 or ordinary paged-entry closure')
    return outputs, {'ordinary_public_entry_guards':guard_rows,
        'ordinary_body_inverse_exact':True,'b1_original_transform_reused':True,
        'same_original_graph_resolver_bodies':resolver_proofs,
        'original_fixture_only_include_path_changed':True,
        'automatic_engine_or_math_admission':False}


def verify_controller(view: Path, inventory: dict) -> tuple[bytes,dict]:
    receipt_raw = bounded_file(view/'controller-handoff-source-receipt.json')
    receipt = json.loads(receipt_raw)
    external = inventory['external_recipe_sha256']
    expected = {
        'owned_recipe_sha256': external['ports/amd/memory/controller_handoff/prepare_controller_handoff.py'],
        'types_sha256': external['ports/amd/memory/controller_handoff/controller_handoff_types.h'],
        'methods_sha256': external['ports/amd/memory/controller_handoff/controller_handoff_methods.cpp.inc'],
    }
    if any(receipt.get(key)!=value for key,value in expected.items()):
        raise ValueError('not the exact full controller-handoff source receipt')
    if receipt.get('all_core_rebuild_required') is not True or receipt.get('old_core_or_old_candidate_archives_allowed') is not False:
        raise ValueError('incomplete full-Core ABI receipt')
    outputs = receipt.get('output_sha256',{})
    if not isinstance(outputs,dict) or not 10<=len(outputs)<=512:
        raise ValueError('invalid full Core inventory')
    total=0
    for relative,expected_sha in outputs.items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or '\\' in relative:
            raise ValueError('invalid controller source path')
        raw=bounded_file(view/path);total+=len(raw)
        if total>16*1024**2 or sha(raw)!=expected_sha:
            raise ValueError('changed/oversized controller view: '+relative)
    header=(view/'src/core/paged_kv_cache.h').read_text(encoding='utf-8')
    if '#define NINFER_AMD_SEGMENTED_CONTROLLER_HANDOFF_ABI 1' not in header:
        raise ValueError('full controller Core header marker absent')
    return receipt_raw,receipt


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for argument in ('baseline','engine','controller-view','output'):
        parser.add_argument('--'+argument,required=True,type=Path)
    parser.add_argument('--verify-existing',action='store_true')
    args=parser.parse_args()
    native,engine,controller,output=(getattr(args,name).resolve() for name in ('baseline','engine','controller_view','output'))
    for source in (native,engine,controller,ROOT):
        if source==output or source in output.parents or output in source.parents:
            raise ValueError('output must be a separate NEW sibling of every source tree')
    if output.exists() and not args.verify_existing:
        raise ValueError('new output required; refusing overwrite')
    inventory=pins();external=verify_external_recipes(inventory)
    manifests={}
    for relative,expected in inventory['manifest_sha256'].items():
        raw=bounded_file(engine/relative)
        if sha(raw)!=expected:raise ValueError('original manifest changed: '+relative)
        manifests[relative]=sha(raw)
    basic=inventory['basic_archive']
    if sha(bounded_file(engine/basic['manifest']))!=basic['manifest_sha256']:
        raise ValueError('Tensor-only Basic manifest changed')
    raw_sources={};total=0
    native_pins={**inventory['native_family_source_sha256'],**inventory['native_b1_extra_source_sha256']}
    for relative,expected in native_pins.items():
        raw=bounded_file(native/relative);total+=len(raw)
        if total>8*1024**2 or sha(raw)!=expected:raise ValueError('native family source changed: '+relative)
        raw_sources[relative]=raw
    b1=load_module('frozen_public_b1',ROOT.parent/'b1_public/prepare_public_bf16.py')
    bf16=load_module('frozen_native_bf16',ROOT.parent/'segmented_bf16_ops_overlay.py')
    # The 13 B1 native anchors are in the 91 original-family texts plus one exact
    # paged-address header. No substitute kernels or oracle are created.
    for relative,expected in {**bf16.BASELINES,**b1.EXTRA}.items():
        if sha(raw_sources[relative])!=expected:raise ValueError('original B1 body anchor changed')
    transformed,mechanism=transform({k:v.decode('utf-8').replace('\r\n','\n') for k,v in raw_sources.items()},b1,bf16)
    controller_raw,controller_receipt=verify_controller(controller,inventory)
    outputs={name:text.encode('utf-8') for name,text in transformed.items()}
    own_raw={name:bounded_file(ROOT/name) for name in OWNED}
    receipt={'schema':'ninfer.models_engine_ops.source.v1','status':'SOURCE_GENERATED_NOT_BUILT_NOT_RUN_NOT_QUALIFIED',
        'baseline':str(native),'controller_view':str(controller),
        'controller_receipt_sha256':sha(controller_raw),'controller_compile_sources':controller_receipt['compile_sources'],
        'external_recipe_sha256':external,'original_manifest_sha256':manifests,
        'basic_archive_manifest_and_members':inventory['basic_archive'],
        'native_source_sha256':native_pins,
        'owned_source_sha256':{name:sha(raw) for name,raw in own_raw.items()},
        'compile_sources':inventory['compile_sources'],'mechanism':mechanism,
        'outputs_sha256':{name:sha(raw) for name,raw in outputs.items()},
        'old_partial_core_or_b1_or_old_cache_ops_allowed':False,'engine_qualified':False}
    receipt_name='models-engine-ops-source-receipt.json'
    if output.exists():
        paths=list(output.rglob('*'))
        if any(path.is_symlink() for path in paths):raise ValueError('symlink in exact source view')
        found={path.relative_to(output).as_posix() for path in paths if path.is_file()}
        if found!=set(outputs)|{receipt_name}:raise ValueError('missing/extra source-view file')
        if json.loads((output/receipt_name).read_text(encoding='utf-8'))!=receipt:raise ValueError('existing source receipt differs')
        for name,raw in outputs.items():
            if (output/name).read_bytes()!=raw:raise ValueError('existing source bytes differ: '+name)
        return
    output.mkdir(parents=True,exist_ok=False)
    for relative,raw in outputs.items():
        path=output/relative;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(raw)
    with (output/receipt_name).open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2);stream.write('\n')


if __name__=='__main__':main()
