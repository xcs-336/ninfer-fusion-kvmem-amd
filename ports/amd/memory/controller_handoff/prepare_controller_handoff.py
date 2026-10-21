"""Default-unwired SOURCE candidate: full prepared-native Core shadow + one handoff.

Never edits the frozen base, its preparer, published native view or old leaf. Main
is deferred to an explicit root owner window. Imports/in-memory transforms do not
write. Output is new, or exact byte-verified; no overwrite is implemented.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

BASE = {
    "src/core/paged_kv_cache.h": "8ba0b39fb9ad09cbc39dc11d779f413f0c6f68bde0d7e817599b389ef6fd9c61",
    "src/core/paged_kv_cache.cpp": "ac37b15cde2fd6e3239db08210abcd4367424ea8c328ae7334df125dabcec121",
    "amd-memory/kv_segmented_storage.h": "382df2b0cd3870e106f9044fa58f9a7ede9b3dc5bba5e24e9a34dbfb8ce56bf3",
    "amd-memory/kv_segmented_storage.cpp": "5b490b0cfee0e847cfbd061873c028b75da1be7b2e036de95d3cfd8378ac15da",
    "amd-memory/kv_segmented_pool_control.h": "d7fb9c918bcdbf87bd318ab0b68b339bbf89052dc962c9f2e82effb7f4603049",
    "amd-memory/kv_segmented_pool_owner.h": "83124366a905fc8f460379d531fa1ddbdb303df52727a7efa4b35ea54c900dc0",
    "amd-memory/kv_segmented_pool_owner.cpp": "486dfcaeaa6020c10d706fe5d9a053a49d272b1134b9645368e6d45cfa48c5e6",
    "amd-memory/kv_segmented_loan_plan.h": "589568a3050e9362bb02dfec2e8c4c9e769a09cea2322cf278adefe563451c0b",
    "amd-memory/kv_segmented_loan_plan.cpp": "ff7c29e6d985cbd7a21f5c2b592ce16c5eb8f6394b4158a5fb97ab31bc86a7cd",
}
BASE_FIXTURE_SHA = "91d6be40cf648e2b8e80b9a439c476279b4a0358cc9824cfe479672aba756960"

def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()

def exact(text: str, old: str, new: str, count: int = 1) -> str:
    if text.count(old)!=count: raise ValueError(f"handoff base anchor changed: {old[:100]!r}")
    return text.replace(old,new)

def transform(base: dict[str,str], types: str, methods: str) -> dict[str,str]:
    out=dict(base)
    h=out["src/core/paged_kv_cache.h"]
    h=exact(h,"#define NINFER_AMD_SEGMENTED_POOL_ABI 1",
            "#define NINFER_AMD_SEGMENTED_POOL_ABI 1\n#define NINFER_AMD_SEGMENTED_CONTROLLER_HANDOFF_ABI 1")
    out["src/core/paged_kv_cache.h"]=h
    c=out["amd-memory/kv_segmented_pool_control.h"]
    c=exact(c,"#include <thread>", '#include <thread>\n#include "controller_handoff_types.h"')
    c=exact(c,"    std::thread::id submission_thread{};", "    std::thread::id submission_thread{};\n"
        "    KVControllerPhase controller_phase=KVControllerPhase::Startup;\n"
        "    bool controller_disabled=false;\n    std::uint64_t handoff_generation=0;\n"
        "    std::uint32_t completed_cold_device_syncs=0;\n"
        "    std::uint32_t active_controller_freezes=0;")
    c=exact(c,"        if(owner&&submission_thread!=std::this_thread::get_id())",
        "        std::lock_guard lock(mutex);\n"
        "        if(owner&&(controller_disabled||submission_thread!=std::this_thread::get_id()))")
    out["amd-memory/kv_segmented_pool_control.h"]=c
    h=out["amd-memory/kv_segmented_pool_owner.h"]
    h=exact(h,'#include "kv_segmented_storage.h"', '#include "kv_segmented_storage.h"\n#include "controller_handoff_types.h"')
    h=exact(h,"#ifndef NINFER_AMD_SEGMENTED_POOL_ABI", "#ifndef NINFER_AMD_SEGMENTED_CONTROLLER_HANDOFF_ABI")
    h=exact(h,"    [[nodiscard]] std::unique_ptr<KVSegmentedAllocatorFreeze> freeze() override;",
        "    [[nodiscard]] std::unique_ptr<KVSegmentedAllocatorFreeze> freeze() override;\n"
        "    [[nodiscard]] KVControllerHandoffToken release_controller_for_worker(std::span<const hipStream_t>);\n"
        "    void consume_controller_for_worker(KVControllerHandoffToken&&,std::span<const hipStream_t>);\n"
        "    void retire_controller_for_teardown(std::span<const hipStream_t>);\n"
        "    [[nodiscard]] KVControllerStatus controller_status() const;")
    h=exact(h,"    KVSegmentedStorage::Config config_;", "    std::shared_ptr<KVControllerIdentity> controller_identity_=std::make_shared<KVControllerIdentity>();\n"
            "    KVSegmentedStorage::Config config_;")
    h=exact(h,"    [[nodiscard]] KVSegmentedStorage& storage() noexcept{return *storage_;}",
        "    [[nodiscard]] KVSegmentedStorage& storage() noexcept{return *storage_;}\n"
        "    [[nodiscard]] KVControllerHandoffToken release_controller_for_worker(std::span<const hipStream_t> s){return owner_.release_controller_for_worker(s);}\n"
        "    void consume_controller_for_worker(KVControllerHandoffToken&& t,std::span<const hipStream_t> s){owner_.consume_controller_for_worker(std::move(t),s);}\n"
        "    void retire_controller_for_teardown(std::span<const hipStream_t> s){owner_.retire_controller_for_teardown(s);}\n"
        "    [[nodiscard]] KVControllerStatus controller_status() const{return owner_.controller_status();}\n"
        "    [[nodiscard]] std::unique_ptr<KVSegmentedAllocatorFreeze> freeze_allocator(){return owner_.freeze();}")
    out["amd-memory/kv_segmented_pool_owner.h"]=h
    c=out["amd-memory/kv_segmented_pool_owner.cpp"]
    c=exact(c,"#include <stdexcept>","#include <stdexcept>\n#include <string>")
    c=exact(c,"    explicit Freeze(KVSegmentedPoolOwner& owner):actor(owner),lock(owner.control_mutex()){}",
        "    explicit Freeze(KVSegmentedPoolOwner& owner):actor(owner),lock(owner.control_mutex()){\n"
        "        auto& count=actor.pool_->segmented_native_.active_controller_freezes;\n"
        "        need(count!=std::numeric_limits<std::uint32_t>::max(),\"controller freeze counter exhausted\");\n"
        "        ++count;\n    }\n"
        "    ~Freeze() override {--actor.pool_->segmented_native_.active_controller_freezes;}")
    c=exact(c,"KVSegmentedPoolOwner::~KVSegmentedPoolOwner()=default;",
        "KVSegmentedPoolOwner::~KVSegmentedPoolOwner(){controller_identity_->alive.store(false,std::memory_order_release);}")
    c=exact(c,"KVSegmentedNativePool::KVSegmentedNativePool(",methods+"\nKVSegmentedNativePool::KVSegmentedNativePool(")
    out["amd-memory/kv_segmented_pool_owner.cpp"]=c
    # These publicly exposed device/host views must not bypass controller affinity.
    # Freeze only obtains the original guarded lock; it does not take an allocator
    # snapshot, allocate device bytes, or reject a valid current-controller loan.
    c=out["amd-memory/kv_segmented_storage.cpp"]
    c=exact(c,"const std::uint64_t* KVSegmentedStorage::device_plane_page_bases(std::uint32_t plane) const {",
        "const std::uint64_t* KVSegmentedStorage::device_plane_page_bases(std::uint32_t plane) const {\n"
        "    auto controller_freeze=impl_->owner.freeze();")
    c=exact(c,"void* KVSegmentedStorage::host_page_address(std::uint32_t plane,std::uint32_t group) const {",
        "void* KVSegmentedStorage::host_page_address(std::uint32_t plane,std::uint32_t group) const {\n"
        "    auto controller_freeze=impl_->owner.freeze();")
    out["amd-memory/kv_segmented_storage.cpp"]=c
    out["amd-memory/controller_handoff_types.h"]=types
    return out

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-view",required=True,type=Path)
    parser.add_argument("--native-view",required=True,type=Path)
    parser.add_argument("--native-sources",required=True,type=Path)
    parser.add_argument("--base-fixture",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--verify-existing",action="store_true")
    args=parser.parse_args();base=args.base_view.resolve();native=args.native_view.resolve();output=args.output.resolve()
    for source in (base,native):
        if output==source or source in output.parents or output in source.parents:
            raise ValueError("handoff output must be a separate new sibling, never an input tree")
    if output.exists() and not args.verify_existing: raise ValueError("new output required")
    receipt_raw=(base/"segmented-source-receipt.json").read_bytes()
    previous=json.loads(receipt_raw)
    if previous.get("output_sha256")!=BASE: raise ValueError("not the exact frozen isolated Core base")
    base_raw={k:(base/k).read_bytes() for k in BASE}
    for name,data in base_raw.items():
        if sha(data)!=BASE[name]: raise ValueError(f"frozen base changed: {name}")
    owned=Path(__file__).resolve().parent
    types_raw=(owned/"controller_handoff_types.h").read_bytes();methods_raw=(owned/"controller_handoff_methods.cpp.inc").read_bytes()
    transformed=transform({k:v.decode("utf-8").replace("\r\n","\n") for k,v in base_raw.items()},
                          types_raw.decode("utf-8"),methods_raw.decode("utf-8"))
    core=native/"src/core";native_inputs={};outputs={};total=0
    # Future explicit generation is bounded to small prepared-native Core texts.
    paths=sorted(p for p in core.rglob("*") if p.is_file() and p.suffix in {".h",".hpp",".cuh",".cpp",".cu"})
    if not paths or len(paths)>512: raise ValueError("unexpected native Core inventory")
    for path in paths:
        if path.is_symlink() or not 0<path.stat().st_size<=2*1024**2: raise ValueError("non-small Core source")
        data=path.read_bytes();total+=len(data)
        if total>16*1024**2: raise ValueError("Core text snapshot exceeds bound")
        name=path.relative_to(native).as_posix();outputs[name]=data;native_inputs[str(path)]=sha(data)
    if not 0<args.native_sources.stat().st_size<=64*1024: raise ValueError("Core source manifest exceeds bound")
    source_list=args.native_sources.read_text(encoding="utf-8").splitlines()
    if not 1<=len(source_list)<=256: raise ValueError("bounded actual Core source manifest required")
    compile_sources=[]
    for value in source_list:
        path=Path(value).resolve()
        if not path.is_file() or path.suffix not in {".cpp",".cu"} or path.stat().st_size>2*1024**2:
            raise ValueError("invalid actual Core compilation source")
        data=path.read_bytes();native_inputs[str(path)]=sha(data)
        if path.is_relative_to(core): name=path.relative_to(native).as_posix()
        else:
            name="src/core/"+path.name
            if name in outputs and outputs[name]!=data: raise ValueError("external Core source basename collision")
            if name not in outputs:
                total+=len(data)
                if total>16*1024**2: raise ValueError("Core text snapshot exceeds bound")
            outputs[name]=data
        if name in compile_sources: raise ValueError("duplicate Core source")
        compile_sources.append(name)
    for name,text in transformed.items():outputs[name]=text.encode("utf-8")
    compile_sources += ["amd-memory/kv_segmented_loan_plan.cpp","amd-memory/kv_segmented_storage.cpp","amd-memory/kv_segmented_pool_owner.cpp"]
    if len(set(compile_sources))!=len(compile_sources): raise ValueError("duplicate final Core source")
    fixture=args.base_fixture.read_bytes()
    if sha(fixture)!=BASE_FIXTURE_SHA: raise ValueError("frozen original-Core fixture changed")
    outputs["test-support/original_pool_fixture.cu"]=fixture
    expected={name:sha(data) for name,data in outputs.items()}
    receipt={"schema_version":1,"status":"SOURCE_GENERATED_NOT_BUILT_NOT_QUALIFIED",
        "base_view":str(base),"base_receipt_sha256":sha(receipt_raw),"frozen_base_sha256":BASE,
        "native_core_inputs":native_inputs,"actual_core_source_list":source_list,
        "compile_sources":compile_sources,"output_sha256":expected,
        "owned_recipe_sha256":sha(Path(__file__).read_bytes()),"types_sha256":sha(types_raw),
        "methods_sha256":sha(methods_raw),"original_fixture_sha256":BASE_FIXTURE_SHA,
        "all_core_rebuild_required":True,"old_core_or_old_candidate_archives_allowed":False,
        "production_gate_changed":False,"engine_integration_qualified":False}
    if output.exists():
        if json.loads((output/"controller-handoff-source-receipt.json").read_text())!=receipt:
            raise ValueError("existing handoff receipt differs; never overwrite")
        # An unrecorded header beneath the BEFORE include root can change the
        # compiled closure even when every expected file still matches. A
        # symlinked directory is likewise compiler-visible without belonging to
        # this immutable snapshot. Accept only the complete recorded inventory.
        expected_files=set(outputs)|{"controller-handoff-source-receipt.json"}
        actual_files=set()
        for path in output.rglob("*"):
            if path.is_symlink():
                raise ValueError("existing handoff view contains a symlink")
            if path.is_file():
                name=path.relative_to(output).as_posix()
                if name not in expected_files:
                    raise ValueError(f"unexpected handoff output: {name}")
                actual_files.add(name)
        if actual_files!=expected_files:
            raise ValueError("existing handoff output inventory differs")
        for name,data in outputs.items():
            if (output/name).read_bytes()!=data: raise ValueError(f"existing handoff output changed: {name}")
        return
    output.mkdir(parents=True,exist_ok=False)
    for name,data in outputs.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as stream:stream.write(data)
    with (output/"controller-handoff-source-receipt.json").open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2);stream.write("\n")

if __name__=="__main__":main()
