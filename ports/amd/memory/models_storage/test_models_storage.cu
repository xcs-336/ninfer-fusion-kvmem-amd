// SOURCE ONLY. Real isolated DecoderState/PagedKVCache, no Engine or Attention math.
#include "models/qwen3_5/state/decoder_state.h"
#include "kv_segmented_pool_owner.h"
#include "core/host_kv_arena.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef NINFER_AMD_SEGMENTED_MODELS_STORAGE_ABI
#error "Compile the two new Models files and all Core against the same isolated ABI"
#endif
namespace {
using namespace ninfer;
using namespace ninfer::models::qwen3_5;
using namespace ninfer::amd::memory;
constexpr std::uint32_t Layers=2,Planes=2*Layers,Groups=4,Logical=3;
constexpr std::size_t Guard=256;
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void checked(hipError_t value,const char* message){if(value!=hipSuccess)throw std::runtime_error(std::string(message)+": "+hipGetErrorString(value));}
template<class Fn>void reject(Fn&& fn,const char* message){bool refused=false;try{fn();}catch(const std::invalid_argument&){refused=true;}need(refused,message);}
DecoderStateSpec spec(int heads) {
    DecoderStateSpec result;result.full_attention_layers=Layers;result.capacity=129;
    result.kv_heads=heads;result.attention_head_dim=256;result.kv_table_rows=1;
    result.text_physical_page_groups=Groups;return result;
}
struct Streams {
    std::array<hipStream_t,2> values{};
    Streams(){try{for(auto& stream:values)checked(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"Models stream");}
        catch(...){for(auto stream:values)if(stream)(void)hipStreamDestroy(stream);throw;}}
    ~Streams(){for(auto stream:values)if(stream){(void)hipStreamSynchronize(stream);(void)hipStreamDestroy(stream);}}
};
struct Graph {
    hipGraph_t graph=nullptr;hipGraphExec_t exec=nullptr;hipStream_t stream=nullptr;
    ~Graph(){if(stream)(void)hipStreamSynchronize(stream);if(exec)(void)hipGraphExecDestroy(exec);if(graph)(void)hipGraphDestroy(graph);}
};
struct Drain {
    Streams& streams;
    ~Drain(){for(auto stream:streams.values)if(stream)(void)hipStreamSynchronize(stream);}
};
// Independent physical-page byte observer, using only the public Models view.
__global__ void observe(PagedKVLayerView view,std::uint16_t* output,std::uint32_t words) {
    for(std::uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<2*Logical*words;i+=blockDim.x*gridDim.x) {
        const auto kind=i/(Logical*words),logical=(i/words)%Logical,word=i%words;
        const auto group=static_cast<const std::int32_t*>(view.block_table.data)[logical];
        const auto* table=kind==0?view.k_page_bases:view.v_page_bases;
        const auto* affine=static_cast<const std::uint16_t*>(kind==0?view.k_pages.data:view.v_pages.data);
        const auto* page=table?reinterpret_cast<const std::uint16_t*>(table[group]):affine+static_cast<std::size_t>(group)*words;
        output[i]=page[word];
    }
}
void planner_rejections() {
    auto good=spec(2);
    for(unsigned which=0;which<9;++which) {
        auto bad=good;
        switch(which) {
        case 0:bad.capacity=513;break;case 1:bad.kv_heads=1;break;case 2:bad.attention_head_dim=128;break;
        case 3:bad.kv_storage=KvCacheStorage::Fp8E4M3Row256;break;case 4:bad.kv_table_rows=2;break;
        case 5:bad.enable_mtp=true;bad.mtp_layers=1;break;case 6:bad.text_physical_page_groups=1;break;
        case 7:bad.text_layer_rank={0,1};break;case 8:bad.mtp_physical_page_groups=1;break;
        }
        LayoutBuilder payload,ordinary;
        reject([&]{(void)plan_segmented_decoder_state(payload,ordinary,bad);},"unsupported Models storage spec accepted");
        need(payload.finish()==0&&ordinary.finish()==0,"unsupported spec mutated layout builders");
    }
    LayoutBuilder one;
    reject([&]{(void)plan_segmented_decoder_state(one,one,good);},"aliased payload/metadata builder accepted");
    LayoutBuilder nonfresh,ordinary;(void)nonfresh.add(1,1,"prior payload");
    reject([&]{(void)plan_segmented_decoder_state(nonfresh,ordinary,good);},"nonzero payload builder accepted");
}

void ordinary_owned(int heads) {
    LayoutBuilder builder;const auto layout=plan_decoder_state(builder,spec(heads));
    DeviceArena backing(builder.finish(256));
    DecoderState state(DeviceSpan{backing.base(),backing.capacity()},layout);
    need(state.text_kv.owns_page_pool()&&!state.mtp_cache(),"ordinary path did not construct its actual owned pool");
    auto& pool=state.text_kv.page_pool();auto reservation=pool.reserve(Logical);
    need(reservation.has_value(),"ordinary original reserve failed");
    std::vector<DeviceKVPageLease> leases;leases.reserve(Logical);pool.materialize(*reservation,Logical,leases);
    auto row=state.text_kv.execution_tables().acquire(0);
    state.text_kv.execution_tables().publish(row.handle(),0,leases);
    checked(hipStreamSynchronize(nullptr),"ordinary table publication");
    const auto view=state.text_kv.execution_view(row).layer_view(0);
    const auto batch=state.text_kv.batch_layer_view(0);
    need(view.k_page_bases==nullptr&&view.v_page_bases==nullptr&&batch.k_page_bases==nullptr&&
         batch.v_page_bases==nullptr&&view.k_pages.data==pool.plane(0).data,
         "ordinary affine view/table contract changed");
}

unsigned external_selected(int heads) {
    LayoutBuilder payload,ordinary;
    const auto prefix=ordinary.add(37,16,"non-KV prefix storage marker");
    const auto layout=plan_segmented_decoder_state(payload,ordinary,spec(heads));
    const auto suffix=ordinary.add(11,16,"non-KV suffix storage marker");
    const auto& cache=layout.text_kv;const auto slot=static_cast<std::uint64_t>(256)*64*heads*2;
    need(payload.finish(256)==cache.payload_bytes()&&cache.pages.planes.front().storage.region.offset==0&&
         cache.execution_tables.front().block_tables.region.offset>=prefix.offset+prefix.bytes&&
         suffix.offset>=cache.execution_tables.front().block_tables.region.offset+cache.execution_tables.front().metadata_bytes(),
         "split planning lost fresh payload or ordinary metadata offsets");
    KVSegmentedStorage::Config cfg{0,Planes,Groups,slot,0,Planes*slot,256,2};
    KVSegmentedNativePool native(cfg,cache.pages); // exactly one payload + original ledger
    DeviceArena metadata(ordinary.finish(256));
    checked(hipMemset(metadata.base(),0x6c,metadata.capacity()),"metadata markers");
    checked(hipStreamSynchronize(nullptr),"order default metadata initialization before nonblocking users");
    DecoderState state(segmented_decoder_storage,DeviceSpan{metadata.base(),metadata.capacity()},layout,native);
    need(!state.text_kv.owns_page_pool() && &state.text_kv.page_pool()==&native.page_pool()&&
         !state.mtp_cache(),"tag constructor allocated a second original pool or MTP cache");
    reject([&]{DecoderState invalid(DeviceSpan{metadata.base(),metadata.capacity()},layout);},"external layout bound through ordinary owned constructor");
    const std::array<std::uint32_t,1> selected{~0U};
    reject([&]{state.text_kv.install_selection(selected);},"external storage installed selected/ring history");
    auto& pool=state.text_kv.page_pool();auto reservation=pool.reserve(Logical);
    need(reservation.has_value(),"selected original reserve failed");
    std::vector<DeviceKVPageLease> leases;leases.reserve(Logical);pool.materialize(*reservation,Logical,leases);
    const std::array<DeviceKVPageHandle,Logical> ids{leases[2].handle(),leases[0].handle(),leases[1].handle()};
    Streams streams;
    const auto host_layout=plan_host_kv_page_layout(pool.geometry());
    const std::array<HostKVPageLayout,1> supported{host_layout};
    HostKVArena host(Logical*host_layout.page_stride,supported);auto image=host.allocate(host_layout,Logical);
    need(image.has_value(),"Models pinned source allocation failed");
    Drain drain_before_pinned_destruction{streams};
    const auto words=static_cast<std::uint32_t>(slot/2);
    std::vector<std::uint16_t> expected(Planes*Logical*words);
    auto host_view=host.writable_view(*image);
    for(std::size_t p=0;p<Planes;++p)for(std::size_t page=0;page<Logical;++page) {
        auto* destination=reinterpret_cast<std::uint16_t*>(host_view.data()+page*host_layout.page_stride+host_layout.planes[p].offset);
        for(std::size_t word=0;word<words;++word) {
            const auto value=static_cast<std::uint16_t>(17+131*p+43*page+7*word);
            expected[(p*Logical+page)*words+word]=value;destination[word]=value;
        }
    }
    pool.copy_from_host(host.view(*image),ids,RankStreams{streams.values[0]});
    auto row=state.text_kv.execution_tables().acquire(0);
    state.text_kv.execution_tables().publish(row.handle(),0,ids,RankStreams{streams.values[0]});
    checked(hipStreamSynchronize(streams.values[0]),"selected upload/publication");
    const auto direct=state.text_kv.execution_view(row);
    std::array<PagedKVLayerView,Layers> views;
    for(std::size_t layer=0;layer<Layers;++layer) {
        views[layer]=direct.layer_view(layer);const auto batch=state.text_kv.batch_layer_view(layer);
        need(views[layer].k_page_bases==pool.plane_view(2*layer).page_bases&&
             views[layer].v_page_bases==pool.plane_view(2*layer+1).page_bases&&
             batch.k_page_bases==views[layer].k_page_bases&&batch.v_page_bases==views[layer].v_page_bases&&
             views[layer].selected_blocks==nullptr&&batch.selected_blocks==nullptr,
             "Models layer/batch lost the selected plane-specific device table");
    }
    DeviceBuffer out(expected.size()*2+2*Guard);Graph graph;graph.stream=streams.values[0];
    checked(hipStreamBeginCapture(graph.stream,hipStreamCaptureModeGlobal),"Models capture");
    for(std::size_t layer=0;layer<Layers;++layer)
        hipLaunchKernelGGL(observe,dim3(32),dim3(256),0,graph.stream,views[layer],
            reinterpret_cast<std::uint16_t*>(static_cast<std::byte*>(out.p)+Guard)+layer*2*Logical*words,words);
    checked(hipGetLastError(),"Models observer capture launch");
    checked(hipStreamEndCapture(graph.stream,&graph.graph),"Models capture end");
    checked(hipGraphInstantiate(&graph.exec,graph.graph,nullptr,nullptr,0),"Models graph instantiate");
    const auto original=graph.exec;
    const auto verify=[&] {
        need(graph.exec==original,"Models replaced its captured Graph exec");
        checked(hipMemsetAsync(out.p,0xa5,out.bytes,graph.stream),"Models fresh observer payload/guards");
        checked(hipGraphLaunch(graph.exec,graph.stream),"Models same-Graph replay");
        checked(hipStreamSynchronize(graph.stream),"Models observer finish");
        std::vector<std::byte> actual(out.bytes);out.copy_to_host(actual.data(),actual.size());
        need(std::all_of(actual.begin(),actual.begin()+Guard,[](auto x){return x==std::byte{0xa5};})&&
             std::all_of(actual.end()-Guard,actual.end(),[](auto x){return x==std::byte{0xa5};})&&
             std::equal(actual.begin()+Guard,actual.end()-Guard,reinterpret_cast<const std::byte*>(expected.data())),
             "Models actual table/ledger bytes or observer guards changed");
    };
    verify();
    // The window covers all four physical cells of plane0. Its three live
    // cells must move to the only free-group donors in other semantic planes.
    auto loan=native.storage().begin_loan(Planes*slot,streams.values);
    need(loan.copied_live_bytes()==Logical*slot,"Models fixture did not relocate the three live first-plane cells");
    loan.wait_ready(streams.values[1]);
    checked(hipMemsetAsync(loan.leased().data,0xd3,loan.leased().bytes,streams.values[1]),"Models borrower overwrite");
    checked(hipStreamSynchronize(streams.values[1]),"Models borrower finish");verify();
    need(loan.close_after_streams(streams.values),"Models actual original grant return failed");verify();
    need(pool.allocated_pages()==Logical&&pool.lent_pages()==0&&pool.available_pages()==Groups-Logical,
         "Models selected original allocator ledger changed");
    std::vector<std::byte> marker(metadata.capacity());
    checked(hipMemcpy(marker.data(),metadata.base(),marker.size(),hipMemcpyDeviceToHost),"ordinary metadata marker readback");
    for(const auto region:{prefix,suffix})
        need(std::all_of(marker.begin()+region.offset,marker.begin()+region.offset+region.bytes,[](auto b){return b==std::byte{0x6c};}),
             "execution tables damaged other ordinary persistent bytes");
    return 3;
}
} // namespace
int main(int argc,char** argv) {
    if(argc==1){std::cout<<"{\"status\":\"PLAN_ONLY_NOT_QUALIFIED\",\"main_explicit_hip_calls\":false,\"engine_integrated\":false}\n";return 0;}
    if(argc!=2||std::string_view(argv[1])!="--run")return 2;
    try {
        planner_rejections();checked(hipSetDevice(0),"Models explicit device bind");unsigned replays=0;
        for(int heads:{2,4}){ordinary_owned(heads);replays+=external_selected(heads);}
        std::cout<<"{\"status\":\"FUNCTIONAL_MODELS_STORAGE_PASS\",\"same_graph_replays\":"<<replays
                 <<",\"geometries\":2,\"attention_math_qualified\":false,\"engine_integrated\":false,\"production_gate\":false}\n";return 0;
    }catch(const std::exception& error){std::cerr<<"MODELS_STORAGE_NOT_QUALIFIED: "<<error.what()<<'\n';return 1;}
}
