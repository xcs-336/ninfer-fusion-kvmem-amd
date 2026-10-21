// Real PUBLIC .ninfer Engine integration smoke. SOURCE ONLY, not a math/quality
// replacement. No Engine construction in default or --host-guards branches.
#include "ninfer/engine.h"
#include "nlohmann/json.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Json=nlohmann::json;
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
ninfer::EngineOptions options(const std::filesystem::path& artifact,bool score) {
    ninfer::EngineOptions out;
    out.artifact_path=artifact;out.device=0;out.max_context=512;out.max_concurrency=1;
    out.max_pending_requests=1;out.kv_cache=ninfer::KvCacheStorage::BFloat16;
    out.kv_capacity=ninfer::KvCapacityPolicy::explicit_capacity(512);
    out.purpose=score?ninfer::EnginePurpose::CausalScoring:ninfer::EnginePurpose::Generation;
    out.gdn_state_fp16=false;out.gdn_state_storage=ninfer::GdnStateStorage::FP32;
    out.prefill_a8=false;out.mlp_a8_decode=false;out.prefill_cublas=false;
    out.prefill_cublas_projections=false;out.fast_prefill_kernel=false;
    out.context_cache.enabled=false;out.enable_vision=false;out.speculative={};
    out.device_profile="off";out.use_cuda_graph=!score;
    out.prefill_chunk=score?1024:128;
    return out;
}
std::vector<ninfer::TokenId> ids(const std::filesystem::path& path) {
    const auto size=std::filesystem::file_size(path);
    need(size>0 && size<=65536,"bounded actual Frontend IDs JSON required");
    std::ifstream in(path,std::ios::binary);
    need(static_cast<bool>(in),"token input open failed");
    const std::string raw{std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
    need(!in.bad() && raw.size()==size,"token input read changed/truncated");
    const Json value=Json::parse(raw); // strict full-document parse, including trailing data
    need(value.is_array() && value.size()>=2 && value.size()<=504,"token array length outside smoke scope");
    std::vector<ninfer::TokenId> out;
    for(const auto& item:value) {
        need(item.is_number_integer(),"tokens must be integer IDs, never strings or floating values");
        const auto token=item.get<std::int64_t>();
        need(token>=0 && token<=std::numeric_limits<ninfer::TokenId>::max(),"token ID outside its public type");
        out.push_back(static_cast<ninfer::TokenId>(token));
    }
    return out; // No tokenizer call, chat rendering, inserted EOS or truncation.
}
void active(const ninfer::SegmentedEngineStorageSnapshot& s) {
    need(s.enabled && !s.controller_disabled && s.controller_phase==3 && s.handoff_generation==1 &&
        s.cold_device_syncs==1 && s.payload_device_bytes>0 && s.pointer_table_device_bytes>0,
        "public Engine did not observe the real Worker controller and one native payload/table");
    need(!s.graph_observation_overflow,"Graph registry/observation overflow or unknown identity");
}
bool same_graph(const ninfer::SegmentedEngineStorageSnapshot& first,
                const ninfer::SegmentedEngineStorageSnapshot& second) {
    for(std::uint32_t a=0;a<first.graph_exec_count;++a)
        for(std::uint32_t b=0;b<second.graph_exec_count;++b)
            if(first.graph_execs[a] && first.graph_execs[a]==second.graph_execs[b] &&
               first.graph_instance_generations[a]>0 &&
               first.graph_instance_generations[a]==second.graph_instance_generations[b] &&
               first.graph_update_revisions[a]==second.graph_update_revisions[b] &&
               first.successful_graph_launches[a]>0 &&
               second.successful_graph_launches[b]>first.successful_graph_launches[a])return true;
    return false;
}
std::vector<ninfer::TokenId> generate(ninfer::Engine& engine,const std::vector<ninfer::TokenId>& input) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens=4;request.execution.allow_prefix_reuse=false;
    request.execution.sampling.temperature=0.0f;request.execution.sampling.top_k=1;
    request.execution.sampling.top_p=1.0f;request.execution.sampling.min_p=0.0f;
    request.execution.sampling.presence_penalty=0.0f;request.execution.sampling.frequency_penalty=0.0f;
    request.execution.sampling.seed=17;
    request.stop.include_model_defaults=false;request.output.raw=true;request.output.preserve_special_tokens=true;
    auto result=engine.generate(engine.prepare_tokens(input,false),request);
    need(result.generated_token_ids.size()==4,"fixed-token smoke did not produce all four IDs");
    need(result.reused_prompt_tokens==0,"NoPrefix smoke reused previous prompt tokens");
    return result.generated_token_ids;
}
Json snapshot(const ninfer::SegmentedEngineStorageSnapshot& s) {
    return {{"enabled",s.enabled},{"phase",s.controller_phase},{"disabled",s.controller_disabled},
        {"generation",s.handoff_generation},{"cold_device_syncs",s.cold_device_syncs},
        {"payload_device_bytes",s.payload_device_bytes},{"pointer_table_device_bytes",s.pointer_table_device_bytes},
        {"pinned_pointer_shadow_bytes",s.pinned_pointer_shadow_bytes},{"slot_maps_host_bytes",s.slot_maps_host_bytes},
        {"graph_exec_count",s.graph_exec_count},{"execs",s.graph_execs},
        {"instance_generations",s.graph_instance_generations},{"update_revisions",s.graph_update_revisions},
        {"successful_launches",s.successful_graph_launches},{"overflow",s.graph_observation_overflow}};
}
std::uint32_t host_guards() {
    auto base=options({},false);ninfer::validate_segmented_engine_options(base);
    std::uint32_t rejected=0;
    for(unsigned test=0;test<15;++test) {
        auto bad=base;
        switch(test) {
        case 0:bad.max_context=0;break;case 1:bad.max_context=513;break;
        case 2:bad.max_concurrency=2;break;case 3:bad.devices={0,0};break;
        case 4:bad.kv_cache=ninfer::KvCacheStorage::Int8Group64;break;
        case 5:bad.enable_vision=true;break;case 6:bad.gdn_state_fp16=true;break;
        case 7:bad.speculative.backend=ninfer::SpeculativeBackend::Mtp;break;
        case 8:bad.speculative.lookup_ngram=1;break;case 9:bad.context_cache.enabled=true;break;
        case 10:bad.context_cache.disk_kv_path="fixture-only";break;
        case 11:bad.context_cache.disk_kv_capacity_bytes=1;break;
        case 12:bad.context_cache.disk_kv_restore=true;break;
        case 13:bad.context_cache.disk_kv_directstorage=true;break;
        case 14:bad.device_profile="auto";break;
        }
        bool refused=false;
        try{ninfer::validate_segmented_engine_options(bad);}catch(const std::invalid_argument&){refused=true;}
        need(refused,"unsupported early option was accepted");++rejected;
    }
    return rejected;
}
} // namespace
int main(int argc,char** argv) {
    try {
        std::string mode="plan";std::filesystem::path artifact,tokens;
        for(int i=1;i<argc;++i) {
            const std::string_view flag=argv[i];
            if(flag=="--host-guards")mode="host-guards";
            else if(flag=="--run-generation")mode="generation";
            else if(flag=="--run-causal")mode="causal";
            else if(flag=="--run-startup-before")mode="startup-before";
            else if(flag=="--run-startup-after")mode="startup-after";
            else if((flag=="--artifact"||flag=="--tokens")&&i+1<argc) {
                if(flag=="--artifact")artifact=argv[++i];else tokens=argv[++i];
            } else throw std::invalid_argument("unknown or incomplete candidate fixture option");
        }
        if(mode=="plan") {
            std::cout<<Json{{"status","PLAN_ONLY_NO_ENGINE_EXECUTION"},
                {"engine_constructed",false},{"math_or_quality_grant",false},
                {"run_modes",{"--run-generation","--run-causal","--run-startup-before","--run-startup-after"}},
                {"requires","explicit root resource/byte0/job guard, matching .ninfer and actual Frontend IDs"}}.dump(2)<<'\n';return 0;
        }
        if(mode=="host-guards") {
            std::cout<<Json{{"status","HOST_OPTIONS_BOUNDARIES_PASSED"},{"refused",host_guards()},
                {"Engine_constructed",false},{"HIP_linked_registration_before_main_possible",true}}.dump(2)<<'\n';return 0;
        }
        need(!artifact.empty() && std::filesystem::is_regular_file(artifact),"real .ninfer artifact required");
        if(mode=="startup-before" || mode=="startup-after") {
            const auto before=ninfer::segmented_worker_lifecycle_counts();bool refused=false;
            const bool after=mode=="startup-after";
            try {
                ninfer::Engine selected(ninfer::segmented_engine_storage,options(artifact,false),
                    after?ninfer::SegmentedWorkerStartupTest::AfterConsume:ninfer::SegmentedWorkerStartupTest::BeforeConsume);
            } catch(const std::runtime_error& error) {
                refused=std::string_view(error.what())==(after?"segmented fixture startup failure after consume":
                    "segmented fixture startup failure before consume");
            }
            const auto end=ninfer::segmented_worker_lifecycle_counts();
            need(refused && end.starts==before.starts+1 && end.exits==before.exits+1 &&
                end.checked_retires==before.checked_retires+(after?1U:0U),
                "real startup exception/promise/join/retire path did not match the explicit fixture control");
            std::cout<<Json{{"status","REAL_STARTUP_CONTROL_PATH_PASSED"},{"mode",mode},
                {"fault","explicit host fixture throw; not a HIP/driver failure"},{"all_workers_joined",true}}.dump(2)<<'\n';return 0;
        }
        need(!tokens.empty(),"actual Frontend integer token array required");const auto input=ids(tokens);
        const bool causal=mode=="causal";auto requested=options(artifact,causal);
        std::vector<ninfer::TokenId> reference_ids;std::vector<float> reference_scores;
        // Separate lifetimes: never keep two model payloads resident at once.
        {
            ninfer::Engine ordinary(requested);
            need(!ordinary.segmented_storage_snapshot().enabled,"ordinary constructor selected the candidate");
            if(causal)reference_scores=ordinary.score_tokens(input,1);else reference_ids=generate(ordinary,input);
            ordinary.checked_shutdown();need(!ordinary.shutdown_error(),"ordinary checked shutdown retained an error");
        }
        const auto before=ninfer::segmented_worker_lifecycle_counts();
        Json report;
        {
            ninfer::Engine selected(ninfer::segmented_engine_storage,requested);
            active(selected.segmented_storage_snapshot());
            if(causal) {
                auto actual=selected.score_tokens(input,1);
                need(actual.size()==input.size()-1 && actual.size()==reference_scores.size(),"full causal score count differs");
                for(std::size_t i=0;i<actual.size();++i)
                    need(std::isfinite(actual[i]) && std::isfinite(reference_scores[i]) &&
                        std::bit_cast<std::uint32_t>(actual[i])==std::bit_cast<std::uint32_t>(reference_scores[i]),
                        "candidate vs ordinary exact teacher-forced score smoke differs");
                report={{"scores",actual},{"criterion","complete finite FP32 bit equality; no relaxed model-quality threshold"}};
            } else {
                const auto first=generate(selected,input);const auto first_state=selected.segmented_storage_snapshot();
                const auto second=generate(selected,input);const auto second_state=selected.segmented_storage_snapshot();
                active(first_state);active(second_state);
                need(first==reference_ids && second==reference_ids,"candidate/ordinary/repeated greedy token smoke differs");
                need(same_graph(first_state,second_state),"two real public requests did not launch the same live unrevised Graph executable");
                report={{"first_ids",first},{"second_ids",second},{"first",snapshot(first_state)},
                    {"second",snapshot(second_state)},{"same_live_unrevised_exec_observed",true}};
            }
            selected.checked_shutdown();need(!selected.shutdown_error(),"candidate checked shutdown retained an error");
            const auto retired=selected.segmented_storage_snapshot();
            need(retired.controller_disabled && retired.controller_phase==4 && retired.cold_device_syncs==2,
                "worker did not retire/drain before owner teardown");report["retired"]=snapshot(retired);
        }
        const auto end=ninfer::segmented_worker_lifecycle_counts();
        need(end.starts==before.starts+1 && end.exits==before.exits+1 && end.checked_retires==before.checked_retires+1,
            "native selected worker did not start/retire/exit exactly once");
        report["status"]="PUBLIC_ENGINE_BOUNDED_INTEGRATION_SMOKE_PASSED";report["mode"]=mode;
        report["input_ids"]=input;report["broad_quality_grant"]=false;report["performance_sample"]=false;
        report["memory_savings_claim"]=false;report["production_gate"]=false;
        std::cout<<report.dump(2)<<'\n';return 0;
    } catch(const std::exception& error) {
        std::cerr<<"models_engine_worker fixture failure: "<<error.what()<<'\n';return 1;
    }
}
