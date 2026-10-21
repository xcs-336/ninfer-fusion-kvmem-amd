"""Unwired candidate: Program single native factory and one-time worker controller.

Import/transform are pure. Explicit main generation is deferred to a root window.
Every output is new, exact verified, or refused; this does not build an Engine.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True

NATIVE_PINS = {'src/models/qwen3_5/program/planning/startup.h': '996f2d0aad9aef3515c29c7f67c8094b060e0456ecd0b3c757d40536f0afa510',
 'src/models/qwen3_5/program/planning/startup.cpp': '193d76f07db0b808be2f9d1f80e0a46e6edda0a05cad82041c15744ca1bb05fe',
 'src/models/qwen3_5/program/program.h': 'f7dc5a2df1c87509caacaf11e502e0b6f2a1f688024f0c1ff26eb90117216e80',
 'src/models/qwen3_5/program/program.cpp': 'e42742bbd86e76571c0c2967ea2291ff8880408dd4bf7fb6d90562cfc2665072',
 'src/models/qwen3_5/program/program_impl.h': '01c38ff6af466618be2109f150466081cf2d5b324764a8d04089d2e771cc73c0',
 'src/models/qwen3_5/program/program_impl.cpp': 'aceaadb9f85ae7b762d88a9ece90d277f2a37df1889f2995a6270155543417e4',
 'src/runtime/engine/engine_core.h': 'c3b7bb1c088a686d50cd8f93f98d07f7ac765c383c43b419111636a5beb66b29',
 'src/runtime/engine/causal_score_core.h': 'aabb915d712391651817d0a7c87f119b4468b1212e60d5ed2b862fff0285bcba',
 'src/runtime/engine/model_instance.h': '66373cea71209b30503a1a78c4fd08aa5bb314e74758cad5a81d7be758d0d1a9',
 'src/runtime/engine/model_instance.cpp': '536676efee26fb5638308837479cf5a5e3c592613d4a93529d0b0be7071936ad',
 'include/ninfer/engine.h': '3eca4314efe6fda0dc67990b38b0aa613ec1c24495e2afa9207bcff19bdda9c2',
 'src/runtime/engine/engine.cpp': '6801038fb226fc96c85aef13f4ce85b6901f256cd20c335795e1089315e5ba13',
 'src/core/decode_graph.cpp': 'd7fbd0123e97a4abb88b3213576e40f73f2e84a485b850975ea7d653485c13db',
 'src/models/qwen3_5/execution/text.cpp': '6bc2636496786bc2564d7ba8d35c7f307109c02b97fbaf59c7b477ee892c7aa3',
 'src/models/qwen3_5/program/planning/graph_profiles.cpp': '5106682e09f9a2b0930d350317e4b14b43eea6fbe6923e24b5d1a608a9bee4d1'}
ENGINE_MANIFEST_SHA = '75ac09a9dfd5b26e492b89124174e550bf301e9d5cb30de385cd4dc8d02b8457'
OWN=Path(__file__).resolve().parent
PLAN=OWN.parent/'models_engine'
MODELS=OWN.parent/'models_storage'
def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def exact(text:str,old:str,new:str,count:int=1)->str:
    if text.count(old)!=count:raise ValueError('worker exact source anchor changed: '+old[:120])
    return text.replace(old,new)
def module(path:Path):
    scope={'__name__':'source_only_dependency','__file__':str(path)}
    exec(compile(path.read_text(encoding='utf8'),str(path),'exec'),scope)
    return scope
def owned(name:str)->str:return (OWN/name).read_text(encoding='utf8').replace('\r\n','\n')
def validate_dependencies()->dict[str,str]:
    expected=json.loads((OWN/'dependency-pins.json').read_text(encoding='utf8'))['source_sha256']
    repo=OWN.parents[3]
    for name,digest in expected.items():
        path=repo/name
        if path.is_symlink() or not 0<path.stat().st_size<=2*1024**2 or sha(path.read_bytes())!=digest:
            raise ValueError('frozen external recipe changed: '+name)
    return expected

def transform(native:dict[str,str],bridge=None)->dict[str,str]:
    """Return only small derived source strings; no filesystem writes or HIP."""
    previous=module(PLAN/'prepare_engine_storage_plan.py')
    startup='src/models/qwen3_5/program/planning/startup'
    program='src/models/qwen3_5/program/program'
    impl='src/models/qwen3_5/program/program_impl'
    sh,sc,pc=previous['transform'](native[startup+'.h'],native[startup+'.cpp'],
        native[program+'.cpp'],(PLAN/'segmented_planner_checks.cpp.inc').read_text(encoding='utf8'))
    # Preserve the immutable earlier barrier; remove it ONLY in this new candidate.
    pc=exact(pc,'    if(plan.impl_->segmented_text_storage)\n'
        '        throw std::logic_error("segmented planner-only candidate: Program ownership/worker/Op bridge NOT_IMPLEMENTED");','')
    sh=exact(sh,'// Explicit partial candidate. Ordinary create_program refuses this plan until\n'
        '// the separate ownership/worker/Op bridge is implemented and qualified.',
        '// Explicit candidate planner; all Program and worker consumers require the new full ABI.')
    declarations='''
void validate_segmented_preallocation(const Config&,const EngineOptions&,DeviceContext&);
void validate_segmented_sequence_plan(const execution::Parameters&,const SequencePlanImpl&,DeviceContext&);
'''
    sh=exact(sh,'// Explicit candidate planner;',declarations+'\n// Explicit candidate planner;')
    # Insertion is outside the anonymous namespace (same placement as the original public make).
    anchor='std::unique_ptr<SequencePlannerImpl>\nmake_segmented_sequence_planner_impl('
    sc=exact(sc,anchor,owned('validation.cpp.inc')+'\n'+anchor)
    sc=exact(sc,'make_segmented_sequence_planner_impl(const execution::Parameters& parameters,DeviceContext& device,\n'
        '                                    const EngineOptions& options) {',
        'make_segmented_sequence_planner_impl(const execution::Parameters& parameters,DeviceContext& device,\n'
        '                                    const EngineOptions& options) {\n'
        '    validate_segmented_preallocation(parameters.model.config(),options,device);')
    ph=exact(native[program+'.h'],'#include "ninfer/types.h"',
        '#include "ninfer/types.h"\n#include "ninfer/segmented_engine_candidate.h"')
    ph=exact(ph,'class Program {\npublic:',
        '#define NINFER_AMD_SEGMENTED_ENGINE_WORKER_ABI 1\nclass Program {\npublic:\n'+owned('program_capability.h.inc'))
    declare='''[[nodiscard]] SequencePlanner make_segmented_sequence_planner(
    const execution::Parameters&,DeviceContext&,const EngineOptions&);
'''
    ph=exact(ph,'[[nodiscard]] SequencePlanner make_sequence_planner(',declare+'\n[[nodiscard]] SequencePlanner make_sequence_planner(')
    pc=exact(pc,'#include <stdexcept>','#include <stdexcept>\n#include \"amd-engine-worker/worker_graph_observer.h\"')
    pc=exact(pc,'namespace ninfer::models::qwen3_5 {\n\n\nSequencePlan::SequencePlan(',
        'namespace ninfer::models::qwen3_5 {\n'+owned('program_capability.cpp.inc')+'\nSequencePlan::SequencePlan(')
    pc=exact(pc,'SequencePlanner make_sequence_planner(',
        'SequencePlanner make_segmented_sequence_planner(const execution::Parameters& p,DeviceContext& d,const EngineOptions& o) {\n'
        '    return SequencePlanner(detail::make_segmented_sequence_planner_impl(p,d,o));\n}\n\nSequencePlanner make_sequence_planner(')
    ih=exact(native[impl+'.h'],'#include "core/arena.h"',
        '#include "core/arena.h"\n#include "kv_segmented_pool_owner.h"\n#include <mutex>\n'
        '#ifndef NINFER_AMD_SEGMENTED_CONTROLLER_HANDOFF_ABI\n#error "Worker Program requires full controller Core ABI"\n#endif')
    ih=exact(ih,'    std::unique_ptr<EvictableKVPool> kv_arena;',
        '    // Declared before every payload/table borrower: destroyed LAST among them.\n'
        '    std::unique_ptr<amd::memory::KVSegmentedNativePool> segmented_native_pool;\n'
        '    const std::size_t segmented_resident_device_bytes;\n'
        '    mutable std::mutex segmented_graph_mutex;\n'
        '    SegmentedEngineStorageSnapshot segmented_graph_observation;\n'
        '    SegmentedWorkerStartupTest segmented_startup_test=SegmentedWorkerStartupTest::None;\n'
        '    void record_segmented_graph(std::uintptr_t,std::uint64_t,std::uint64_t) noexcept;\n'
        '    [[nodiscard]] std::vector<hipStream_t> segmented_controller_streams() const;\n'
        '    std::unique_ptr<EvictableKVPool> kv_arena;')
    ic=exact(native[impl+'.cpp'],'#include "core/device.h"',
        '#include "core/device.h"\n#include "amd-engine-worker/worker_graph_observer.h"')
    ic=exact(ic,'std::uint32_t normalized_private_capacity(const ContextCacheOptions& options);',
        owned('program_native.cpp.inc')+'\nstd::uint32_t normalized_private_capacity(const ContextCacheOptions& options);')
    ic=exact(ic,'      kv_arena(make_kv_arena(device_in, parameters_in, plan)),',
        '      segmented_native_pool(make_segmented_program_storage(parameters_in,plan,device_in)),\n'
        '      segmented_resident_device_bytes(segmented_program_device_bytes(segmented_native_pool.get())),\n'
        '      kv_arena(make_kv_arena(device_in, parameters_in, plan)),')
    ic=exact(ic,'    decoder = std::make_unique<qwen3_5::DecoderState>(backings, plan.persistent.decoder);',
        '    if(segmented_native_pool) {\n'
        '        decoder=std::make_unique<qwen3_5::DecoderState>(qwen3_5::segmented_decoder_storage,\n'
        '            backing,plan.persistent.decoder,*segmented_native_pool);\n'
        '    } else {\n    decoder = std::make_unique<qwen3_5::DecoderState>(backings, plan.persistent.decoder);\n    }')
    ic=exact(ic,'ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};',
        'ArenaMemorySummary{persistent.capacity()+segmented_resident_device_bytes,\n'
        '            persistent.used()+segmented_resident_device_bytes,\n'
        '            persistent.peak_used()+segmented_resident_device_bytes};')
    streams='''void ProgramImpl::record_segmented_graph(std::uintptr_t exec,std::uint64_t generation,std::uint64_t revision) noexcept {
    std::lock_guard lock(segmented_graph_mutex);
    auto& observation=segmented_graph_observation;
    if(!exec || !generation){observation.graph_observation_overflow=true;return;}
    for(std::uint32_t i=0;i<observation.graph_exec_count;++i) {
        if(observation.graph_execs[i]==exec && observation.graph_instance_generations[i]==generation &&
           observation.graph_update_revisions[i]==revision) {
            if(observation.successful_graph_launches[i]==std::numeric_limits<std::uint64_t>::max())
                observation.graph_observation_overflow=true;
            else ++observation.successful_graph_launches[i];
            return;
        }
    }
    if(observation.graph_exec_count==observation.graph_execs.size()) {
        observation.graph_observation_overflow=true;return;
    }
    const auto i=observation.graph_exec_count++;
    observation.graph_execs[i]=exec;observation.graph_instance_generations[i]=generation;
    observation.graph_update_revisions[i]=revision;observation.successful_graph_launches[i]=1;
}
std::vector<hipStream_t> ProgramImpl::segmented_controller_streams() const {
    if(!segmented_native_pool)return {};
    if(device.size()!=1 || hybrid_ || context_cache.enabled || disk_kv)
        throw std::logic_error("segmented worker stream inventory is not the admitted single-rank no-prefix/no-disk set");
    std::vector<hipStream_t> out;
    const auto add=[&](hipStream_t s){if(std::find(out.begin(),out.end(),s)==out.end())out.push_back(s);};
    const auto& rank=device.rank(0);
    add(rank.stream);add(rank.transfer_stream);
    if(rank.vision_stream)add(rank.vision_stream);
    add(nullptr); // original default-stream helper work is covered, aliases deduplicated
    return out;
}
'''
    ic=exact(ic,'ProgramImpl::~ProgramImpl() noexcept {',streams+'\nProgramImpl::~ProgramImpl() noexcept {')
    out={startup+'.h':sh,startup+'.cpp':sc,program+'.h':ph,program+'.cpp':pc,impl+'.h':ih,impl+'.cpp':ic}
    for path,engine in [('src/runtime/engine/engine_core.h',True),('src/runtime/engine/causal_score_core.h',False)]:
        c=exact(native[path],'#include "core/device.h"','#include "core/device.h"\n#include "runtime/engine/segmented_worker_controller.h"\n#include "runtime/engine/diagnostics.h"')
        typename='Program' if engine else 'typename ModelContract::Program'
        c=exact(c,'        worker_                   = std::thread([this, startup = std::move(startup)]() mutable {',
            '        auto controller=segmented_worker_detail::Controller<'+typename+'>(*instance_.program);\n'
            '        worker_ = std::thread([this,startup=std::move(startup),controller=std::move(controller)]() mutable {\n'
            '            segmented_worker_detail::Lifetime lifetime(controller.tracked());')
        c=exact(c,'                device_.bind_to_current_thread();\n                startup.set_value();',
            '                device_.bind_to_current_thread();\n                controller.consume();\n'
            '                controller.before_ready();\n                startup.set_value();')
        c=exact(c,'                startup.set_exception(std::current_exception());\n                return;',
            '                auto startup_error=std::current_exception();\n'
            '                try {controller.retire();} catch(...) {\n'
            '                    auto retirement=std::current_exception();remember_controller_failure(retirement);\n'
            '                    startup_error=std::make_exception_ptr(SegmentedWorkerStartupFailure(startup_error,retirement));\n'
            '                }\n'
            '                startup.set_exception(startup_error);\n                return;')
        c=exact(c,'            worker_loop();\n        });',
            '            worker_loop();\n'
            '            try {controller.retire();} catch(...) {remember_controller_failure(std::current_exception());}\n        });')
        cls='EngineCore' if engine else 'CausalScoreCore'
        old='''    ~'''+cls+'''() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
    }'''
        observer='diagnostics_' if engine else 'DiagnosticObserver{}'
        new='''    // Explicit shutdown may be called by the owner before destroying this core.
    // Join happens before reading the retained checked-retirement error.
    void checked_shutdown_worker() {
        {std::lock_guard lock(queue_mutex_);stopping_=true;}
        queue_cv_.notify_all();
        if(worker_.joinable())worker_.join();
        std::exception_ptr error;
        {std::lock_guard lock(queue_mutex_);error=worker_controller_error_;}
        if(error)std::rethrow_exception(error);
    }
    [[nodiscard]] std::exception_ptr worker_controller_error() const {
        std::lock_guard lock(queue_mutex_);return worker_controller_error_;
    }
    ~'''+cls+'''() noexcept {
        try {checked_shutdown_worker();}
        catch(...) {publish_diagnostic('''+observer+''',DiagnosticLevel::Error,
            "checked worker controller retirement/join failed; shutdown error retained until core destruction");}
    }'''
        c=exact(c,old,new)
        remember='''    void remember_controller_failure(std::exception_ptr error) noexcept {
        {std::lock_guard lock(queue_mutex_);
         if(!worker_controller_error_)worker_controller_error_=std::move(error);
         stopping_=true;'''+('failed_=true;' if engine else '')+'''}
        publish_diagnostic('''+observer+''',DiagnosticLevel::Error,
            "checked retirement failed; worker stopped and error retained; controller status must be queried");
    }
'''
        c=exact(c,'    void worker_loop() noexcept {',remember+'\n    void worker_loop() noexcept {')
        c=exact(c,'    std::thread worker_;','    std::exception_ptr worker_controller_error_;\n    std::thread worker_;')
        out[path]=c
    # An explicit tagged INTERNAL runtime factory, not a new undocumented EngineOptions field.
    path='src/runtime/engine/model_instance.h'
    out[path]=exact(native[path],'[[nodiscard]] ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device);',
        'struct SegmentedEngineStorageTag {explicit SegmentedEngineStorageTag()=default;};\n'
        'inline constexpr SegmentedEngineStorageTag segmented_engine_storage{};\n'
        '[[nodiscard]] ConstructedModel construct_model(SegmentedEngineStorageTag,const EngineOptions&,DeviceContext&);\n'
        '[[nodiscard]] ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device);')
    path='src/runtime/engine/model_instance.cpp';c=native[path]
    c=exact(c,'#include "models/qwen3_5/load.h"','#include "models/qwen3_5/load.h"\n#include "models/qwen3_5/program/planning/startup.h"')
    c=exact(c,'ConstructedModel assemble_fusion(const EngineOptions& requested, DeviceContext& device,\n                                 artifact::Reader& reader, Clock::time_point start) {',
        'ConstructedModel assemble_fusion(const EngineOptions& requested, DeviceContext& device,\n                                 artifact::Reader& reader, Clock::time_point start,bool segmented=false) {')
    c=exact(c,'    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options));',
        '    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options));\n'
        '    if(segmented)models::qwen3_5::detail::validate_segmented_preallocation(plan.config(),options,device);')
    c=exact(c,'    auto planner = models::qwen3_5::make_sequence_planner(\n        loaded->parameters, device,\n        artifact_scoped_disk_tier(options, loaded->model->info().artifact_id));',
        '    auto planner = segmented ? models::qwen3_5::make_segmented_sequence_planner(loaded->parameters,device,options)\n'
        '        : models::qwen3_5::make_sequence_planner(\n        loaded->parameters, device,\n'
        '        artifact_scoped_disk_tier(options, loaded->model->info().artifact_id));')
    c=exact(c,'ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device) {',
        'ConstructedModel construct_model_internal(const EngineOptions& options,DeviceContext& device,bool segmented) {')
    c=exact(c,'    install_device_route_profile_for(options, device);',
        '    if(segmented && (options.devices.size()>1 || options.max_context==0 || options.max_context>512 ||\n'
        '        options.max_concurrency!=1 || options.kv_cache!=KvCacheStorage::BFloat16 ||\n'
        '        options.enable_vision || options.speculative.backend!=SpeculativeBackend::None ||\n'
        '        options.context_cache.enabled || options.device_profile!="off"))\n'
        '        throw std::invalid_argument("segmented runtime requires its explicit bounded no-prefix/no-spec profile before loading");\n'
        '    // off must really clear any prior process route profile; this branch cannot calibrate.\n'
        '    install_device_route_profile_for(options, device);')
    c=exact(c,'        return assemble_flash_next(options, device, reader, start);',
        '        if(segmented)throw std::invalid_argument("segmented runtime does not admit FlashNext");\n'
        '        return assemble_flash_next(options, device, reader, start);')
    c=exact(c,'    return assemble_fusion(options, device, reader, start);',
        '    return assemble_fusion(options, device, reader, start,segmented);')
    c=exact(c,'\n} // namespace ninfer::runtime',
        '\nConstructedModel construct_model(const EngineOptions& o,DeviceContext& d) {return construct_model_internal(o,d,false);}\n'
        'ConstructedModel construct_model(SegmentedEngineStorageTag,const EngineOptions& o,DeviceContext& d) {return construct_model_internal(o,d,true);}\n'
        '\n} // namespace ninfer::runtime')
    out[path]=c
    out['src/runtime/engine/segmented_worker_controller.h']=owned('worker_controller_bridge.h')
    # Real public Engine candidate constructor; ordinary entry is still false.
    path='include/ninfer/engine.h';eh=native[path]
    eh=exact(eh,'#include "ninfer/types.h"','#include "ninfer/types.h"\n#include "ninfer/segmented_engine_candidate.h"\n#include <exception>')
    eh=exact(eh,'    explicit Engine(EngineOptions options);',
        '    explicit Engine(EngineOptions options);\n'
        '    Engine(SegmentedEngineStorageTag,EngineOptions,SegmentedWorkerStartupTest=SegmentedWorkerStartupTest::None);\n'
        '    void checked_shutdown();\n'
        '    [[nodiscard]] std::exception_ptr shutdown_error() const;\n'
        '    [[nodiscard]] SegmentedEngineStorageSnapshot segmented_storage_snapshot() const;')
    out[path]=eh
    path='src/runtime/engine/engine.cpp';ec=native[path]
    ec=exact(ec,'#include <memory>', '#include <memory>\n#include <mutex>') if '#include <memory>' in ec else exact(ec,'#include <limits>','#include <limits>\n#include <mutex>')
    ec=exact(ec,'    explicit Impl(EngineOptions engine_options)',
        '    explicit Impl(EngineOptions engine_options,bool segmented=false,\n'
        '        SegmentedWorkerStartupTest startup_test=SegmentedWorkerStartupTest::None)')
    ec=exact(ec,': options(runtime::normalize_engine_options(std::move(engine_options))),',
        ': segmented_storage_requested(segmented),\n          options(runtime::normalize_engine_options(std::move(engine_options))),')
    ec=exact(ec,'options(runtime::normalize_engine_options(std::move(engine_options))),',
        'options(runtime::normalize_engine_options(segmented\n'
        '              ? validated_segmented_engine_options(std::move(engine_options)) : std::move(engine_options))),')
    ec=exact(ec,'        auto constructed  = runtime::construct_model(options, device);',
        '        auto constructed = segmented_storage_requested\n'
        '            ? runtime::construct_model(runtime::segmented_engine_storage,options,device)\n'
        '            : runtime::construct_model(options,device);')
    ec=exact(ec,'        active            = std::move(constructed.active);',
        '        active            = std::move(constructed.active);\n'
        '        if(segmented_storage_requested) {\n'
        '            auto* selected=active_fusion_program(active);\n'
        '            if(!selected)throw std::logic_error("tagged Engine did not construct the concrete Qwen Program");\n'
        '            selected->set_segmented_startup_test(startup_test);\n        }')
    ec=exact(ec,'            if (options.purpose == EnginePurpose::CausalScoring) { throw; }',
        '            // Released one-shot controller cannot be retried by the ordinary fallback.\n'
        '            if(segmented_storage_requested)throw;\n'
        '            if (options.purpose == EnginePurpose::CausalScoring) { throw; }')
    ec=exact(ec,'    ~Impl() noexcept {',
        '    void checked_shutdown() {\n'
        '        std::lock_guard shutdown_lock(controller_shutdown_mutex);\n'
        '        try {std::visit([](auto& state){\n'
        '            using State=std::remove_cvref_t<decltype(state)>;\n'
        '            if constexpr(!std::is_same_v<State,std::monostate>)if(state)state->checked_shutdown_worker();\n'
        '        },core);} catch(...) {\n'
        '            if(!controller_shutdown_error)controller_shutdown_error=std::current_exception();\n'
        '            throw;\n        }\n    }\n'
        '    ~Impl() noexcept {')
    ec=exact(ec,'        core.emplace<std::monostate>();',
        '        if(segmented_storage_requested) {\n'
        '            try {checked_shutdown();} catch(...) {runtime::publish_diagnostic(options.diagnostic_observer,\n'
        '                DiagnosticLevel::Error,"segmented public Engine retained a checked worker shutdown failure");}\n'
        '        }\n        core.emplace<std::monostate>();')
    ec=exact(ec,'    EngineOptions options;','    const bool segmented_storage_requested;\n'
        '    mutable std::mutex controller_shutdown_mutex;\n'
        '    std::exception_ptr controller_shutdown_error;\n    EngineOptions options;')
    ec=exact(ec,'Engine::~Engine()                            = default;',
        'Engine::Engine(SegmentedEngineStorageTag,EngineOptions options,SegmentedWorkerStartupTest startup_test) {\n'
        '    StartupObserver observer=options.startup_observer;\n'
        '    StartupPhaseScope phase(observer,StartupPhase::EngineStartup);\n'
        '    impl_=std::make_shared<Impl>(std::move(options),true,startup_test);phase.complete();\n}\n'
        'void Engine::checked_shutdown() {\n'
        '    if(!impl_)throw std::logic_error("Engine is moved from");impl_->checked_shutdown();\n}\n'
        'std::exception_ptr Engine::shutdown_error() const {\n'
        '    if(!impl_)return {};std::lock_guard lock(impl_->controller_shutdown_mutex);\n'
        '    return impl_->controller_shutdown_error;\n}\n'
        'SegmentedEngineStorageSnapshot Engine::segmented_storage_snapshot() const {\n'
        '    if(!impl_)throw std::logic_error("Engine is moved from");\n'
        '    const auto* program=active_fusion_program(impl_->active);\n'
        '    return program?program->segmented_storage_accounting():SegmentedEngineStorageSnapshot{};\n}\n'
        'Engine::~Engine()                            = default;')
    out[path]=ec
    # Full Core list is copied from the verified controller view by main. Only this
    # exact launch statement is changed; its API status still reaches CUDA_CHECK.
    path='src/core/decode_graph.cpp';cc=native[path]
    cc=exact(cc,'#include "core/decode_graph.h"','#include "core/decode_graph.h"\n#include "amd-engine-worker/worker_graph_observer.h"')
    cc=exact(cc,'CUDA_CHECK(hipGraphLaunch(exec_, stream));',
        'CUDA_CHECK(amd::models_engine_worker::observed_graph_launch(exec_, stream));')
    cc=exact(cc,'log_cuda_error("hipGraphExecDestroy", hipGraphExecDestroy(exec));',
        'log_cuda_error("hipGraphExecDestroy", amd::models_engine_worker::observed_graph_destroy(exec));')
    cc=exact(cc,'    exec_ = exec;\n}',
        '    exec_ = exec;\n    amd::models_engine_worker::registered_graph_exec(exec_);\n}')
    cc=exact(cc,'}\n\nvoid DecodeGraphExecutable::upload(hipStream_t stream) {',
        '    amd::models_engine_worker::updated_graph_exec(exec_);\n}\n\nvoid DecodeGraphExecutable::upload(hipStream_t stream) {')
    out[path]=cc
    out['include/ninfer/segmented_engine_candidate.h']=owned('segmented_engine_candidate.h')
    out['amd-engine-worker/worker_graph_observer.h']=owned('worker_graph_observer.h')
    if bridge is not None:
        out[startup+'.cpp']=bridge['transform_startup'](out[startup+'.cpp'])
        path='src/models/qwen3_5/execution/text.cpp'
        out[path]=bridge['transform_text'](native[path])
        path='src/models/qwen3_5/program/planning/graph_profiles.cpp'
        out[path]=bridge['transform_graph_profiles'](native[path])
    return out

def inventory(root:Path,receipt_name:str)->dict:
    if root.is_symlink():raise ValueError('symlink view root')
    raw=(root/receipt_name).read_bytes()
    if not 0<len(raw)<=2*1024**2:raise ValueError('source receipt bound')
    report=json.loads(raw);expected=report.get('output_sha256',report.get('outputs_sha256'))
    if not isinstance(expected,dict) or not expected:raise ValueError('source output inventory missing')
    entries=list(root.rglob('*'))
    if any(p.is_symlink() for p in entries):raise ValueError('symlink source inventory')
    if {p.relative_to(root).as_posix() for p in entries if p.is_file()}!=set(expected)|{receipt_name}:
        raise ValueError('untracked/missing source include')
    for name,digest in expected.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or not 0<p.stat().st_size<=2*1024**2 or sha(p.read_bytes())!=digest:
            raise ValueError('source closure changed: '+name)
    return report

def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-root',required=True,type=Path)
    p.add_argument('--models-root',required=True,type=Path)
    p.add_argument('--controller-root',required=True,type=Path)
    p.add_argument('--ops-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--verify-existing',action='store_true')
    a=p.parse_args();native=a.native_root.resolve();models=a.models_root.resolve();core=a.controller_root.resolve();ops=a.ops_root.resolve();output=a.output.resolve()
    dependency_pins=validate_dependencies()
    if a.output.is_symlink():raise ValueError('symlink output')
    for source in (native,models,core,ops):
        if output==source or source in output.parents or output in source.parents:raise ValueError('new separate sibling output required')
    if output.exists() and not a.verify_existing:raise ValueError('never overwrite existing output')
    model_report=inventory(models,'models-storage-source-receipt.json')
    core_report=inventory(core,'controller-handoff-source-receipt.json')
    ops_report=inventory(ops,'models-engine-ops-source-receipt.json')
    core_receipt_sha=sha((core/'controller-handoff-source-receipt.json').read_bytes())
    if model_report['core_receipt_sha256']!=core_receipt_sha or ops_report['controller_receipt_sha256']!=core_receipt_sha:
        raise ValueError('Models and Ops must bind the exact same controller Core recipe')
    expected_models={
        'src/models/qwen3_5/state/decoder_state.h':'9dc9e4a53f6c743d3cd1fd689b6568160acdd9fa1074c1b9877a05d35d978924',
        'src/models/qwen3_5/state/decoder_state.cpp':'70402fa5fa8b56d1b248392c7645451b039852396ebd7573b013ecdb01377b07'}
    if model_report['output_sha256']!=expected_models:raise ValueError('not the frozen two-file Models storage implementation')
    if 'NINFER_AMD_SEGMENTED_CONTROLLER_HANDOFF_ABI 1' not in (core/'src/core/paged_kv_cache.h').read_text(encoding='utf8'):
        raise ValueError('not the complete new controller Core ABI')
    data={}
    for name,digest in NATIVE_PINS.items():
        path=native/name
        if path.is_symlink() or not 0<path.stat().st_size<=512*1024:raise ValueError('small native text required')
        raw=path.read_bytes()
        if sha(raw)!=digest:raise ValueError('exact native source changed: '+name)
        data[name]=raw.decode('utf8').replace('\r\n','\n')
    ops_owned=OWN.parent/'models_engine_ops'
    bridge_path=ops_owned/'bridge_models_calls.py'
    for name,digest in ops_report['owned_source_sha256'].items():
        if sha((ops_owned/name).read_bytes())!=digest:raise ValueError('Ops bridge recipe changed since its source view')
    bridge=module(bridge_path)
    result=transform(data,bridge)
    manifest_raw=(native/'engine-manifest.json').read_bytes()
    if sha(manifest_raw)!=ENGINE_MANIFEST_SHA:raise ValueError('actual original Engine manifest changed')
    engine_manifest=json.loads(manifest_raw)
    components={name:engine_manifest['components'][name] for name in ('model_loading','model_runtime','engine','calibration')}
    # Future explicit generation: bounded full small-source shadow, never an old
    # layout archive or a few substituted objects in an existing Engine library.
    outputs={};snapshot_sha={};total=0
    for prefix in ('src/models','src/runtime','include/ninfer'):
        for path in sorted((native/prefix).rglob('*')):
            if path.is_symlink():raise ValueError('symlink in native model/runtime/public include closure')
            if not path.is_file() or path.suffix not in {'.h','.hpp','.cuh','.inc','.cpp','.cu'}:continue
            if not 0<path.stat().st_size<=2*1024**2:raise ValueError('non-small model/runtime source')
            raw=path.read_bytes();total+=len(raw)
            if total>32*1024**2 or len(snapshot_sha)>=4096:raise ValueError('bounded native text shadow exceeded')
            name=path.relative_to(native).as_posix();outputs[name]=raw;snapshot_sha[name]=sha(raw)
    for component in components.values():
        for name in component:
            if name.startswith('third_party/'):raise ValueError('unexpected third-party code in owned model/runtime components')
            if name not in outputs:
                raw=(native/name).read_bytes()
                if not 0<len(raw)<=2*1024**2:raise ValueError('component source bound')
                outputs[name]=raw;snapshot_sha[name]=sha(raw)
    for root,report,field in ((core,core_report,'output_sha256'),(models,model_report,'output_sha256'),(ops,ops_report,'outputs_sha256')):
        for name in report[field]:outputs[name]=(root/name).read_bytes()
    if core_report['output_sha256']['src/core/decode_graph.cpp']!=NATIVE_PINS['src/core/decode_graph.cpp']:
        raise ValueError('new Graph observer requires exact original controller decode Graph body')
    for name,text in result.items():outputs[name]=text.encode('utf8')
    core_identity={name:sha(outputs[name])==digest for name,digest in core_report['output_sha256'].items()}
    if {name for name,equal in core_identity.items() if not equal}!={'src/core/decode_graph.cpp'}:
        raise ValueError('only observed decode_graph.cpp may differ from the controller full Core')
    own={x.name:sha(x.read_bytes()) for x in OWN.iterdir() if x.is_file()}
    dependencies=dependency_pins
    report={'schema_version':1,'status':'SOURCE_ONLY_PROGRAM_WORKER_BRIDGE_NOT_BUILT',
        'native_input_sha256':NATIVE_PINS,'recipe_input_sha256':own,'dependency_sha256':dependencies,
        'models_receipt_sha256':sha((models/'models-storage-source-receipt.json').read_bytes()),
        'controller_receipt_sha256':core_receipt_sha,
        'ops_receipt_sha256':sha((ops/'models-engine-ops-source-receipt.json').read_bytes()),
        'controller_output_sha256':core_report['output_sha256'],
        'controller_abi_identity_proof':core_identity,'core_compile_sources':core_report['compile_sources'],
        'components':components,'native_snapshot_sha256':snapshot_sha,'engine_manifest_sha256':ENGINE_MANIFEST_SHA,
        'output_sha256':{name:sha(raw) for name,raw in outputs.items()},
        'ops_bridge_composed':True,'engine_activated':False,'runtime_qualified':False}
    if output.exists():
        actual=inventory(output,'models-engine-worker-source-receipt.json')
        if actual!=report:raise ValueError('existing candidate recipe differs')
        return
    output.mkdir(parents=True,exist_ok=False)
    for name,raw in outputs.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(raw)
    with (output/'models-engine-worker-source-receipt.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2);f.write('\n')
if __name__=='__main__':main()
