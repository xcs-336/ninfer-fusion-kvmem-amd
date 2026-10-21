"""SOURCE ONLY: isolated opt-in Core ABI. Never edits the published source view.

Generation itself is deferred to the root's explicit build window. Inputs must match
the reviewed small-source fingerprints; a changed upstream/native view fails closed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

BASELINES = {
    "src/core/paged_kv_cache.h": "86935cf1b3f20a324bc1c9d4cd53ca21e9d5914c01f30e0bbd998fad455dd789",
    "src/core/paged_kv_cache.cpp": "fb173794ce15a662472fe05e47255f08c5d88e9b86120ee94d71ae39d72a4a4a",
}
STORAGE_BASELINES = {
    "kv_segmented_storage.h": "31e31117f4e93620f523485abe83e86362e7ae718ec978a3f737d98f3fe2229a",
    "kv_segmented_storage.cpp": "29939c1400103e6161df9a26dadd20a9f6a478eb08b369eb70e35accef4d2c46",
}

def exact(text: str, old: str, new: str, count: int = 1) -> str:
    if text.count(old) != count:
        raise ValueError(f"source anchor count changed: {old[:90]!r}")
    return text.replace(old, new)

def prefixes(text: str, name: str, prefix: str, expected: int = 1) -> str:
    found = list(re.finditer(r"\bDeviceKVPagePool::" + re.escape(name) + r"\s*\(", text))
    if len(found) != expected:
        raise ValueError(f"{name}: expected {expected} actual definitions, got {len(found)}")
    for match in reversed(found):
        brace = text.index("{", match.end())
        text = text[:brace + 1] + "\n" + prefix + text[brace + 1:]
    return text

ZERO = r'''
    if (segmented_native_.storage) {
        segmented_native_.require_submission_thread();
        validate_distinct_pages(pages, "segmented zero destination contains duplicate pages");
        for (std::size_t p=0;p<planes_.size();++p) {
            const auto stream=streams[0]; ninfer_segmented_non_capturing(stream);
            for (std::size_t begin=0;begin<pages.size();) {
                auto* base=static_cast<std::byte*>(segmented_native_.storage->host_page_address(p,physical_index(pages[begin])));
                const auto width=static_cast<std::size_t>(planes_[p].nb[3]);
                std::size_t end=begin+1;
                while(end<pages.size() &&
                      segmented_native_.storage->host_page_address(p,physical_index(pages[end]))==base+(end-begin)*width)++end;
                CUDA_CHECK(hipMemsetAsync(base,0,(end-begin)*width,stream));begin=end;
            }
        }
        return;
    }
'''
COPY = r'''
    if (segmented_native_.storage) {
        segmented_native_.require_submission_thread();
        const auto from=physical_index(source),to=physical_index(destination);
        if(from==to)return;
        const auto stream=streams[0];ninfer_segmented_non_capturing(stream);
        for(std::size_t p=0;p<planes_.size();++p)
            CUDA_CHECK(hipMemcpyAsync(segmented_native_.storage->host_page_address(p,to),
                       segmented_native_.storage->host_page_address(p,from),
                       static_cast<std::size_t>(planes_[p].nb[3]),hipMemcpyDeviceToDevice,stream));
        return;
    }
'''
HOST = r'''
    if (segmented_native_.storage) {
        segmented_native_.require_submission_thread();
        if(first<0 || static_cast<std::uint32_t>(first)>capacity_pages() ||
           count>capacity_pages()-static_cast<std::uint32_t>(first))
            throw std::out_of_range("segmented host run outside logical page groups");
        const auto stream=streams[0];ninfer_segmented_non_capturing(stream);
        const bool to_host=kind==hipMemcpyDeviceToHost;
        if(!to_host && kind!=hipMemcpyHostToDevice)throw std::logic_error("invalid segmented host copy direction");
        for(std::size_t p=plane_begin;p<plane_end;++p) {
            const auto width=static_cast<std::size_t>(planes_[p].nb[3]);
            if(host.planes[p].page_payload_bytes!=width)throw std::logic_error("segmented host payload differs from BF16 cell");
            for(std::size_t begin=0;begin<count;) {
                auto* device=static_cast<std::byte*>(segmented_native_.storage->host_page_address(p,first+begin));
                std::size_t end=begin+1;
                while(end<count && segmented_native_.storage->host_page_address(p,first+end)==device+(end-begin)*width)++end;
                auto* host_ptr=host_base+host.planes[p].offset+begin*host_pitch;
                if(to_host)CUDA_CHECK(hipMemcpy2DAsync(host_ptr,host_pitch,device,width,width,end-begin,hipMemcpyDeviceToHost,stream));
                else CUDA_CHECK(hipMemcpy2DAsync(device,width,host_ptr,host_pitch,width,end-begin,hipMemcpyHostToDevice,stream));
                begin=end;
            }
        }
        return;
    }
'''
RUNS = r'''
    if (segmented_native_.storage) {
        segmented_native_.require_submission_thread();
        if(pages.empty())return 0;
        std::uint32_t runs=1;
        for(std::size_t i=0;i<pages.size();++i) {
            const auto current=physical_index(pages[i]);
            if(i==0)continue;
            const auto previous=physical_index(pages[i-1]);bool contiguous=true;
            for(std::size_t p=0;p<planes_.size();++p) {
                auto* before=static_cast<std::byte*>(segmented_native_.storage->host_page_address(p,previous));
                if(segmented_native_.storage->host_page_address(p,current)!=before+planes_[p].nb[3]){contiguous=false;break;}
            }
            if(!contiguous)++runs;
        }
        return runs;
    }
'''

def transform_core(header: str, source: str) -> tuple[str, str]:
    header = exact(header, '#include "core/arena.h"',
        '#define NINFER_AMD_SEGMENTED_POOL_ABI 1\n#include "kv_segmented_pool_control.h"\n#include "core/arena.h"')
    header = exact(header, "    const std::uint32_t* selected_blocks = nullptr;",
        "    const std::uint32_t* selected_blocks = nullptr;\n"
        "    const std::uint64_t* k_page_bases = nullptr;\n"
        "    const std::uint64_t* v_page_bases = nullptr;", 2)
    header = exact(header, "    [[nodiscard]] const Tensor& plane(std::size_t index) const;",
        "    [[nodiscard]] const Tensor& plane(std::size_t index) const;\n"
        "    [[nodiscard]] DeviceKVPlaneView plane_view(std::size_t index) const;\n"
        "    [[nodiscard]] const std::uint64_t* plane_page_bases(std::size_t index) const;")
    header = exact(header,"\nclass DeviceKVPagePool;\n", "\nstruct DeviceKVPlaneView {\n"
        "    Tensor storage;\n    const std::uint64_t* page_bases=nullptr;\n};\n\nclass DeviceKVPagePool;\n", 1)
    header = exact(header,"    [[nodiscard]] std::span<const KVPageRun> free_runs() const noexcept;",
        "    [[nodiscard]] std::span<const KVPageRun> free_runs() const;")
    header = exact(header, "    std::uint32_t lent_pages_               = 0;",
        "    std::uint32_t lent_pages_               = 0;\n"
        "    friend class amd::memory::KVSegmentedPoolOwner;\n"
        "    amd::memory::KVSegmentedPoolControl segmented_native_;")
    source = exact(source, '#include "core/paged_kv_cache.h"',
        '#include "core/paged_kv_cache.h"\n#include "kv_segmented_storage.h"')
    source = exact(source, "namespace ninfer {", r'''
namespace {
void ninfer_segmented_non_capturing(hipStream_t stream) {
    hipStreamCaptureStatus state=hipStreamCaptureStatusNone;
    CUDA_CHECK(hipStreamIsCapturing(stream,&state));
    if(state!=hipStreamCaptureStatusNone)throw std::logic_error("Core segmented raw-address transfers cannot be captured");
}
}
namespace ninfer {''')
    source = exact(source, "        .selected_blocks = cache.selected_blocks,",
        "        .selected_blocks = cache.selected_blocks,\n"
        "        .k_page_bases = cache.k_page_bases,\n        .v_page_bases = cache.v_page_bases,")
    source = exact(source,"DeviceKVPagePool::free_runs() const noexcept", "DeviceKVPagePool::free_runs() const")
    mutators = ("lend_pages", "return_pages", "reserve", "resize_reservation", "materialize",
                "materialize_one", "dematerialize", "dematerialize_one", "release_page",
                "release_reservation", "consume_free_run", "release_free_page")
    for name in mutators:
        source = prefixes(source, name,
            "    amd::memory::KVSegmentedPoolMutation native_mutation(segmented_native_);\n")
    for name in ("allocated_pages", "reserved_pages", "usable_pages", "lent_pages", "available_pages",
                 "free_runs", "plane", "plane_page_range", "contiguous_run_count", "can_resize_reservation",
                 "valid_handle", "physical_index", "validate_distinct_pages", "zero_pages", "copy_page",
                 "copy_host_run", "copy_to_host", "copy_from_host", "copy_to_host_records", "copy_from_host_records"):
        source = prefixes(source,name,"    std::lock_guard native_lock(segmented_native_.mutex);\n",
                          2 if name == "copy_from_host_records" else 1)
    # Reject legacy untyped lending. Typed owner commits the original free-run ledger
    # atomically under that SAME mutex and binds a unique capability instead.
    for name in ("lend_pages", "return_pages"):
        source = prefixes(source,name,
            '    if(segmented_native_.owner)throw std::logic_error("segmented loan requires its typed owner capability");\n')
    source = prefixes(source,"plane_page_range",
        '    if(segmented_native_.storage)throw std::logic_error("segmented pages have no unconditional affine raw range");\n')
    source = prefixes(source,"free_runs",
        '    if(segmented_native_.owner)throw std::logic_error("bound segmented allocator requires a value-owned typed frozen snapshot, not a free-runs span");\n')
    source = prefixes(source,"plane",
        '    if(segmented_native_.storage)throw std::logic_error("bound segmented plane requires explicit plane_view with its device page table");\n')
    # Insert after the read lock, not ahead of it: all host address resolution/submission
    # must be serialized against relocation/publication, including compound host runs.
    for name, branch in (("zero_pages",ZERO),("copy_page",COPY),("copy_host_run",HOST),("contiguous_run_count",RUNS)):
        start=source.index("DeviceKVPagePool::"+name+"(")
        lock=source.index("    std::lock_guard native_lock(segmented_native_.mutex);\n",start)
        end=lock+len("    std::lock_guard native_lock(segmented_native_.mutex);\n")
        source=source[:end]+branch+source[end:]
    source = exact(source,"std::uint32_t DeviceKVPagePool::capacity_pages() const noexcept",r'''
DeviceKVPlaneView DeviceKVPagePool::plane_view(std::size_t index) const {
    std::lock_guard lock(segmented_native_.mutex);
    const auto& metadata=planes_.at(index);
    if(!segmented_native_.storage)return {metadata,nullptr};
    segmented_native_.require_submission_thread();
    return {metadata,segmented_native_.storage->device_plane_page_bases(static_cast<std::uint32_t>(index))};
}
const std::uint64_t* DeviceKVPagePool::plane_page_bases(std::size_t index) const {
    std::lock_guard lock(segmented_native_.mutex);
    if(index>=planes_.size())throw std::out_of_range("segmented plane index outside pool");
    if(!segmented_native_.storage)return nullptr;
    segmented_native_.require_submission_thread();
    return segmented_native_.storage->device_plane_page_bases(static_cast<std::uint32_t>(index));
}
std::uint32_t DeviceKVPagePool::capacity_pages() const noexcept''')
    return header, source

def transform_storage(header: str, source: str) -> tuple[str,str]:
    header=exact(header,"    [[nodiscard]] KVSegmentedSpan raw_payload_for_initialization() const;",
        "    [[nodiscard]] KVSegmentedSpan raw_payload_for_initialization() const;\n"
        "    void seal_initialization() noexcept;\n"
        "    // Only for serialized Core host submissions, never a captured GPU operand.\n"
        "    [[nodiscard]] void* host_page_address(std::uint32_t plane,std::uint32_t group) const;")
    source=exact(source,"    bool active=false,bad=false;", "    bool active=false,bad=false,initialization_sealed=false;")
    source=exact(source,'require(!impl_->active,"cannot initialize a borrowed segmented payload");',
        'require(!impl_->active&&!impl_->initialization_sealed,"segmented raw initialization is sealed or borrowed");')
    source=exact(source,"KVSegmentedStorage::Transaction KVSegmentedStorage::begin_loan(",r'''
void KVSegmentedStorage::seal_initialization() noexcept { impl_->initialization_sealed=true; }
void* KVSegmentedStorage::host_page_address(std::uint32_t plane,std::uint32_t group) const {
    const auto& s=*impl_;s.check_usable();
    require(plane<s.config.planes&&group<s.config.page_groups,"segmented logical cell outside pool");
    const auto slot=s.current[static_cast<std::size_t>(plane)*s.config.page_groups+group];
    return static_cast<std::byte*>(s.payload)+static_cast<std::uint64_t>(slot)*s.config.slot_bytes;
}
KVSegmentedStorage::Transaction KVSegmentedStorage::begin_loan(''')
    return header,source

def read_checked(path: Path, digest: str) -> str:
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:
        raise ValueError(f"reviewed input changed: {path}; do not silently apply to a new ABI")
    return raw.decode("utf-8").replace("\r\n","\n")

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--verify-existing",action="store_true",
                        help="Reuse only if every existing output byte matches; never overwrite")
    args=parser.parse_args()
    baseline=args.baseline.resolve();output=args.output.resolve();owned=Path(__file__).resolve().parent
    if (output.exists() and not args.verify_existing) or output==baseline or baseline in output.parents or output in baseline.parents:
        raise ValueError("output must be a new isolated sibling tree, never the production/generated tree")
    checked={name:read_checked(baseline/name,digest) for name,digest in BASELINES.items()}
    frozen={name:read_checked(owned/name,digest) for name,digest in STORAGE_BASELINES.items()}
    core=transform_core(checked["src/core/paged_kv_cache.h"],checked["src/core/paged_kv_cache.cpp"])
    storage=transform_storage(frozen["kv_segmented_storage.h"],frozen["kv_segmented_storage.cpp"])
    outputs={"src/core/paged_kv_cache.h":core[0],"src/core/paged_kv_cache.cpp":core[1],
             "amd-memory/kv_segmented_storage.h":storage[0],"amd-memory/kv_segmented_storage.cpp":storage[1]}
    for name in ("kv_segmented_pool_control.h","kv_segmented_pool_owner.h","kv_segmented_pool_owner.cpp",
                 "kv_segmented_loan_plan.h","kv_segmented_loan_plan.cpp"):
        outputs["amd-memory/"+name]=(owned/name).read_text(encoding="utf-8")
    if output.exists():
        receipt=json.loads((output/"segmented-source-receipt.json").read_text(encoding="utf-8"))
        expected={name:hashlib.sha256(text.encode("utf-8")).hexdigest() for name,text in outputs.items()}
        if receipt.get("output_sha256")!=expected or receipt.get("reviewed_input_sha256")!={**BASELINES,**STORAGE_BASELINES}:
            raise ValueError("existing source receipt differs from the exact reviewed recipe")
        for name,text in outputs.items():
            if (output/name).read_bytes()!=text.encode("utf-8"):
                raise ValueError(f"existing isolated output was changed: {name}")
        return
    # Every transform and source read completes before creating any output. No source
    # generation is performed merely by importing this module.
    output.mkdir(parents=True,exist_ok=False)
    hashes={}
    for name,text in outputs.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
        raw=text.encode("utf-8")
        with path.open("xb") as stream:stream.write(raw)
        hashes[name]=hashlib.sha256(raw).hexdigest()
    receipt={"schema_version":1,"status":"GENERATED_ONLY_NOT_COMPILED_NOT_QUALIFIED",
             "baseline":str(baseline),"reviewed_input_sha256":{**BASELINES,**STORAGE_BASELINES},
             "output_sha256":hashes,"core_view_abi_changed":True,"old_abi_libraries_compatible":False,
             "production_gate_changed":False,"attention_closure_complete":False}
    with (output/"segmented-source-receipt.json").open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2);stream.write("\n")

if __name__=="__main__":main()
