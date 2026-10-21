"""Partial source-only Engine candidate: real planner split + fail-closed Program gate.

No Program factory/worker/Op bridge is implemented. Main is a future explicit
source-generation action; import/transform are pure and never generate files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

PINNED={'src/models/qwen3_5/program/planning/startup.h': '996f2d0aad9aef3515c29c7f67c8094b060e0456ecd0b3c757d40536f0afa510', 'src/models/qwen3_5/program/planning/startup.cpp': '193d76f07db0b808be2f9d1f80e0a46e6edda0a05cad82041c15744ca1bb05fe', 'src/models/qwen3_5/program/program.cpp': 'e42742bbd86e76571c0c2967ea2291ff8880408dd4bf7fb6d90562cfc2665072'}
STARTUP='src/models/qwen3_5/program/planning/startup'
PROGRAM='src/models/qwen3_5/program/program.cpp'
def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()
def exact(source:str,old:str,new:str,count:int=1)->str:
    if source.count(old)!=count:raise ValueError('exact Engine planner anchor changed: '+old[:100])
    return source.replace(old,new)

def transform(header:str,source:str,program:str,checks:str)->tuple[str,str,str]:
    h=exact(header,'#include "models/qwen3_5/state/decoder_state.h"',
        '#include "models/qwen3_5/state/decoder_state.h"\n'
        '#ifndef NINFER_AMD_SEGMENTED_MODELS_STORAGE_ABI\n#error "Planner candidate requires the new two-file Models storage ABI"\n#endif\n'
        '#define NINFER_AMD_SEGMENTED_ENGINE_PLANNER_ABI 1')
    h=exact(h,'    std::size_t bytes            = 0;',
        '    std::size_t bytes            = 0;\n'
        '    std::size_t segmented_payload_bytes=0;\n    std::size_t segmented_pointer_table_bytes=0;')
    h=exact(h,'    const execution::Parameters* parameters = nullptr;',
        '    const execution::Parameters* parameters = nullptr;\n    bool segmented_text_storage=false;',2)
    declaration='''make_sequence_planner_impl(const execution::Parameters& parameters, DeviceContext& device,
                           const EngineOptions& options);'''
    h=exact(h,declaration,declaration+'''
// Explicit partial candidate. Ordinary create_program refuses this plan until
// the separate ownership/worker/Op bridge is implemented and qualified.
[[nodiscard]] std::unique_ptr<SequencePlannerImpl>
make_segmented_sequence_planner_impl(const execution::Parameters&,DeviceContext&,const EngineOptions&);''')
    c=exact(source,'PersistentLayout persistent_layout(const SequencePlanImpl& plan) {',
        checks+'\nPersistentLayout persistent_layout(const SequencePlanImpl& plan) {\n'
        '    if(plan.segmented_text_storage) {\n'
        '        require_segmented_engine_scope(*plan.parameters,plan.capacity,plan.max_concurrency,\n'
        '            plan.kv_storage,plan.features,plan.context_cache);\n'
        '        if(plan.speculative_backend!=SpeculativeBackend::None||plan.draft_window!=0||\n'
        '           plan.ngram_draft_window!=0||plan.lookup_ngram!=0||\n'
        '           plan.main_page_groups<(plan.capacity+63U)/64U)\n'
        '            throw std::invalid_argument("segmented planner requires no draft/copy and physical groups>=logical");\n'
        '    }')
    c=exact(c,'    out.decoder = qwen3_5::plan_decoder_state(\n        builder_pointers,\n        qwen3_5::DecoderStateSpec{',
        '    LayoutBuilder segmented_payload;\n    const qwen3_5::DecoderStateSpec decoder_spec{')
    c=exact(c,'            .mtp_layers                = 1,','            .mtp_layers                = plan.segmented_text_storage?0U:1U,')
    c=exact(c,'            .mtp_physical_page_groups  = mtp_physical_pages,',
        '            .mtp_physical_page_groups  = plan.segmented_text_storage?0U:mtp_physical_pages,')
    c=exact(c,'            .text_layer_rank           = std::move(attention_layer_rank),\n        });',
        '            .text_layer_rank           = std::move(attention_layer_rank),\n        };\n'
        '    out.decoder=plan.segmented_text_storage\n'
        '        ? qwen3_5::plan_segmented_decoder_state(segmented_payload,builder,decoder_spec)\n'
        '        : qwen3_5::plan_decoder_state(builder_pointers,decoder_spec);\n'
        '    if(plan.segmented_text_storage) {\n'
        '        out.segmented_payload_bytes=segmented_payload.finish(256);\n'
        '        const auto planes=out.decoder.text_kv.pages.planes.size();\n'
        '        const auto groups=out.decoder.text_kv.pages.spec.page_group_count;\n'
        '        if(planes>std::numeric_limits<std::size_t>::max()/(8ULL*groups))\n'
        '            throw std::overflow_error("segmented pointer table byte overflow");\n'
        '        out.segmented_pointer_table_bytes=8ULL*planes*groups;\n'
        '    }')
    c=exact(c,'    out.lendable_kv_end_bytes =\n        std::max(',
        '    out.lendable_kv_end_bytes = plan.segmented_text_storage?0U:\n        std::max(')
    c=exact(c,'    impl->parameters          = inputs.parameters;',
        '    impl->parameters          = inputs.parameters;\n    impl->segmented_text_storage=inputs.segmented_text_storage;')
    c=exact(c,'        checked_add(impl->persistent.bytes, impl->workspace.capacity, "sequence memory plan"),',
        '        checked_add(segmented_persistent_resident_bytes(impl->persistent), impl->workspace.capacity, "sequence memory plan"),')
    anchor='std::unique_ptr<SequencePlanImpl>\nfinalize_sequence_plan_impl('
    wrapper='''std::unique_ptr<SequencePlannerImpl>
make_segmented_sequence_planner_impl(const execution::Parameters& parameters,DeviceContext& device,
                                    const EngineOptions& options) {
    require_segmented_engine_scope(parameters,options.max_context,options.max_concurrency,
        options.kv_cache,models::load_options(options),options.context_cache);
    if(device.size()!=1||device.rank(0).device!=options.device)
        throw std::invalid_argument("segmented planner device inventory must be exactly its one configured device");
    // Existing planner may query device metadata; it creates no Program payload.
    auto planner=make_sequence_planner_impl(parameters,device,options);
    if(!planner->curve.extra_ranks.empty())throw std::logic_error("segmented planner received unexpected extra ranks");
    planner->inputs.segmented_text_storage=true;
    const auto minimum=planner->curve.minimum_main_page_groups;
    planner->minimum=build_sequence_candidate(planner->inputs,minimum);
    planner->curve.minimum_device_reservation_bytes=planner->minimum->device_reservation_bytes;
    planner->curve.bytes_per_additional_main_page_group=0;
    if(minimum<planner->curve.maximum_main_page_groups) {
        auto adjacent=build_sequence_candidate(planner->inputs,minimum+1U);
        if(adjacent->device_reservation_bytes<=planner->minimum->device_reservation_bytes)
            throw std::logic_error("segmented planner nonpositive capacity stride");
        planner->curve.bytes_per_additional_main_page_group=
            adjacent->device_reservation_bytes-planner->minimum->device_reservation_bytes;
    }
    return planner;
}

'''
    c=exact(c,anchor,wrapper+anchor)
    p=exact(program,'    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }',
        '    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }\n'
        '    if(plan.impl_->segmented_text_storage)\n'
        '        throw std::logic_error("segmented planner-only candidate: Program ownership/worker/Op bridge NOT_IMPLEMENTED");')
    return h,c,p

def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root',required=True,type=Path)
    parser.add_argument('--models-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--verify-existing',action='store_true')
    a=parser.parse_args();native=a.native_root.resolve();models=a.models_root.resolve();output=a.output.resolve()
    if a.output.is_symlink() or a.models_root.is_symlink():raise ValueError('symlink source/output root rejected')
    for parent in (native,models):
        if output==parent or output in parent.parents or parent in output.parents:raise ValueError('separate new output required')
    if output.exists() and not a.verify_existing:raise ValueError('never overwrite')
    model_receipt_path=models/'models-storage-source-receipt.json'
    if not 0<model_receipt_path.stat().st_size<=1024**2:raise ValueError('Models receipt exceeds bound')
    model_receipt=model_receipt_path.read_bytes();model_report=json.loads(model_receipt)
    expected_models={'src/models/qwen3_5/state/decoder_state.h','src/models/qwen3_5/state/decoder_state.cpp'}
    if set(model_report['output_sha256'])!=expected_models:raise ValueError('unexpected Models output inventory')
    entries=list(models.rglob('*'))
    if any(path.is_symlink() for path in entries):raise ValueError('symlink Models source view')
    actual={path.relative_to(models).as_posix() for path in entries if path.is_file()}
    if actual!=set(model_report['output_sha256'])|{'models-storage-source-receipt.json'}:
        raise ValueError('extra/missing Models source inventory')
    for name,digest in model_report['output_sha256'].items():
        path=models/name
        if not path.resolve().is_relative_to(models) or not 0<path.stat().st_size<=512*1024 or sha(path.read_bytes())!=digest:
            raise ValueError('Models input mismatch')
    header=(models/'src/models/qwen3_5/state/decoder_state.h').read_text(encoding='utf-8')
    if 'NINFER_AMD_SEGMENTED_MODELS_STORAGE_ABI 1' not in header:raise ValueError('wrong Models ABI')
    if any(not 0<(native/name).stat().st_size<=512*1024 for name in PINNED):raise ValueError('native source bound exceeded')
    inputs={name:(native/name).read_bytes() for name in PINNED}
    if {name:sha(data) for name,data in inputs.items()}!=PINNED:raise ValueError('real planner/Program source changed')
    own=Path(__file__).resolve().parent;checks=(own/'segmented_planner_checks.cpp.inc').read_bytes()
    h,c,p=transform(inputs[STARTUP+'.h'].decode().replace('\r\n','\n'),
        inputs[STARTUP+'.cpp'].decode().replace('\r\n','\n'),inputs[PROGRAM].decode().replace('\r\n','\n'),checks.decode())
    outputs={STARTUP+'.h':h.encode(),STARTUP+'.cpp':c.encode(),PROGRAM:p.encode()}
    report={'schema_version':1,'status':'PARTIAL_PLANNER_SOURCE_ONLY_ENGINE_BRIDGE_NOT_IMPLEMENTED',
        'input_sha256':PINNED,'models_root':str(models),'models_receipt_sha256':sha(model_receipt),
        'recipe_sha256':sha(Path(__file__).read_bytes()),'checks_sha256':sha(checks),
        'output_sha256':{name:sha(data) for name,data in outputs.items()},
        'program_creation_of_segmented_plan':'EXPLICITLY_REJECTED_BEFORE_PROGRAM_ALLOCATION',
        'worker_bridge_implemented':False,'engine_integrated':False,'production_changed':False}
    if output.exists():
        entries=list(output.rglob('*'))
        if any(path.is_symlink() for path in entries):raise ValueError('symlink in existing source view')
        actual={path.relative_to(output).as_posix() for path in entries if path.is_file()}
        if actual!=set(outputs)|{'engine-storage-plan-source-receipt.json'}:raise ValueError('extra/missing source inventory')
        if json.loads((output/'engine-storage-plan-source-receipt.json').read_text(encoding='utf-8'))!=report:raise ValueError('existing receipt differs')
        for name,data in outputs.items():
            if (output/name).read_bytes()!=data:raise ValueError('existing source differs')
        return
    output.mkdir(parents=True,exist_ok=False)
    for name,data in outputs.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(data)
    with (output/'engine-storage-plan-source-receipt.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
if __name__=='__main__':main()
