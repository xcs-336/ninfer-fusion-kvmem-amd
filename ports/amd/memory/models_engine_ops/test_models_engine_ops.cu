// NEW fixture wrapper. The copied, SHA-pinned independent 24-replay B1 fixture
// supplies all existing FP64/cache/gate/guards without changing its criteria.
#include "amd-models-b1/public_bf16.h"
#include "amd-models-b1/affine_guard.h"
#include "ninfer/ops/kv_cache_append.h"
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#define main models_engine_rebased_b1_main
#include "amd-models-b1/test_public_bf16.cu"
#undef main

namespace {
template<class F> void require_affine_refusal(F&& invoke, const char* operation) {
    bool rejected = false;
    try { invoke(); }
    catch (const std::invalid_argument& error) {
        const std::string_view text(error.what());
        rejected = text.starts_with(operation) &&
            text.find("table-backed cache is outside this affine entry") != text.npos;
    }
    need(rejected, "ordinary paged public entry failed its early table-backed refusal");
}

void host_entry_guards() {
    // Borrowed non-owning arena constructor/destructor only; the guarded entries
    // must throw before any alloc/scope/launch. This storage is never a GPU pointer.
    alignas(256) std::array<std::byte,256> unused{};
    WorkspaceArena workspace(DeviceSpan{unused.data(),unused.size()});
    Tensor empty, out;
    const auto* sentinel = reinterpret_cast<const std::uint64_t*>(std::uintptr_t{8});
    for (int mask : {1,2,3}) {
        PagedKVLayerView direct;
        PagedKVBatchLayerView batch;
        if (mask & 1) direct.k_page_bases = batch.k_page_bases = sentinel;
        if (mask & 2) direct.v_page_bases = batch.v_page_bases = sentinel;
        require_affine_refusal([&]{ops::kv_cache_append(empty,empty,empty,direct,nullptr);},"kv_cache_append");
        require_affine_refusal([&]{ops::kv_cache_append_prefix(empty,empty,empty,empty,empty,{},batch,nullptr);},"kv_cache_append_prefix");
        require_affine_refusal([&]{ops::causal_softmax_attention(empty,empty,empty,empty,empty,empty,
            {256,24,4},0.0625f,batch,{1,512},workspace,out,nullptr);},"causal_softmax_attention");
        require_affine_refusal([&]{ops::causal_softmax_attention_cached(empty,empty,{256,24,4},0.0625f,
            direct,{1,512},workspace,out,nullptr);},"causal_softmax_attention_cached");
        require_affine_refusal([&]{ops::context_softmax_attention(empty,empty,empty,empty,empty,empty,
            {128,32,8},0.08838834764831844f,batch,{},workspace,out,nullptr);},"context_softmax_attention");
        need(workspace.used()==0 && workspace.peak_used()==0,"refusal allocated caller workspace");
    }
    // Positive host-only null-table check: preserve ordinary route labels for the
    // complete original API and compare the admitted B1 geometry's labels.
    for (auto g : {ops::AttentionHeadGeometry{256,24,4},ops::AttentionHeadGeometry{256,16,2}})
        for (int width : {1,3,8,16,35,128,512}) {
            const auto original=ops::causal_softmax_attention_route_family(g,KvCacheStorage::BFloat16,{1,512},1,width);
            const auto rebased=ops::segmented_b1_causal_softmax_attention_route_family(g,KvCacheStorage::BFloat16,{1,512},1,width);
            need(original==rebased,"same-source Graph route labels changed across ABI rebase");
        }
}
} // namespace

int main(int argc,char** argv) {
    if(argc==1) return models_engine_rebased_b1_main(argc,argv);
    if(argc!=2 || (std::string_view(argv[1])!="--run" && std::string_view(argv[1])!="--host-guards")) return 2;
    try {
        host_entry_guards();
        if(std::string_view(argv[1])=="--host-guards") {
            std::cout<<"{\"status\":\"HOST_ENTRY_GUARDS_ONLY_PASS\",\"table_guard_cases\":15,\"graph_label_checks\":14,\"explicit_gpu_api_called\":false,\"engine_qualified\":false}\n";
            return 0;
        }
        return models_engine_rebased_b1_main(argc,argv);
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
