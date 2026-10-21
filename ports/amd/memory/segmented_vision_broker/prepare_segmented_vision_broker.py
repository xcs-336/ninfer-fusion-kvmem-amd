"""Pure component derivative of the real prepared VisionWindow/Broker production body.

Default CLI is plan-only. No Session, Engine feature gate, Vision encoder, or
old model_runtime object supplies any symbol in this component closure.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
NAMESPACE="namespace ninfer::models::qwen3_5::execution {"
HPATH="src/models/qwen3_5/execution/vision_overlay.h"
CPATH="src/models/qwen3_5/execution/vision_overlay.cpp"
OUT_H="amd-segmented-vision/vision_broker.h"
OUT_C="amd-segmented-vision/vision_broker.cpp"
RECEIPT="segmented-vision-broker-source-receipt.json"
OWNED=("prepare_segmented_vision_broker.py","broker_methods.cpp.inc","manifest.json","README.md",
       "test_segmented_vision_broker.cu","SegmentedVisionBroker.cmake","verify_segmented_vision_map.py")
CAPACITY_PREPARER_SHA="101de66a965b34a2fc2f48de12e1e55a39dd8076916653a532f1eb4a08cd30d1"

def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()

def exact(text:str,old:str,new:str)->str:
    if text.count(old)!=1:raise ValueError("production component anchor changed: "+repr(old[:100]))
    return text.replace(old,new,1)

def section(text:str,begin:str,end:str)->str:
    if text.count(begin)!=1 or text.count(end)!=1:raise ValueError("component boundary changed")
    a=text.index(begin);b=text.index(end,a)
    return text[a:b]

def component(raw:dict[str,bytes],methods:str)->tuple[dict[str,bytes],dict]:
    h_original=section(raw[HPATH].decode("utf-8"),"\nclass VisionResidencyBroker;\n",
        "// Pinned slots receiving one item's merged embeddings each.")
    c_original=section(raw[CPATH].decode("utf-8"),"VisionWindow::~VisionWindow()",
        "// ---------------------------------------------------------------------------------------------\n// PinnedResultPool")
    h=h_original;c=c_original;h_edits=[];c_edits=[]
    def eh(old,new):
        nonlocal h
        h=exact(h,old,new);h_edits.append((old,new))
    def ec(old,new):
        nonlocal c
        c=exact(c,old,new);c_edits.append((old,new))
    eh("{ KvGranules, WeightTail }","{ KvGranules, WeightTail, KvSegmented }")
    eh("    [[nodiscard]] DeviceSpan memory() const noexcept {\n",
       "    [[nodiscard]] DeviceSpan memory() const noexcept {\n"
       "        if(close_failed_||!broker_)return {};\n"
       "        if(tier_==Tier::KvSegmented) {\n"
       "            const auto span=segmented_.leased();return {span.data,static_cast<std::size_t>(span.bytes)};\n        }\n")
    for marker,statement in (("double open_seconds()", "return segmented_open_seconds_;"),
                             ("double close_seconds()", "return segmented_close_seconds_;"),
                             ("std::size_t borrowed_bytes()", "return segmented_bytes_;")):
        old="    [[nodiscard]] "+marker+" const noexcept {\n"
        eh(old,old+"        if(tier_==Tier::KvSegmented)"+statement+"\n")
    eh("    // Returns the borrowed memory. Never throws: a failure poisons the owning pool.",
       "    // Returns memory without throwing. Segmented completion failure retains its\n"
       "    // lent grant/poison; revision errors are sticky broker failures. WeightTail\n"
       "    // retains the original best-effort close contract.")
    eh("    void close() noexcept;", "    void close() noexcept;\n"
       "    // Checked completion/unique return applies to KvSegmented only.\n"
       "    // WeightTail remains the original best-effort restore contract.\n"
       "    [[nodiscard]] bool close_checked() noexcept;\n"
       "    [[nodiscard]] bool close_failed() const noexcept{return close_failed_;}\n"
       "    void wait_ready(hipStream_t stream) const;")
    eh("    std::vector<KVPageRun> runs_;", "    std::vector<KVPageRun> runs_;\n"
       "    void close_legacy() noexcept;\n"
       "    ::ninfer::amd::memory::KVSegmentedStorage::Transaction segmented_;\n"
       "    std::array<hipStream_t,4> segmented_users_{};\n"
       "    std::size_t segmented_user_count_=0,segmented_bytes_=0;\n"
       "    double segmented_open_seconds_=0,segmented_close_seconds_=0;\n"
       "    bool close_failed_=false;")
    eh("    // Free KV granules only; empty when they cannot fund the window.",
       "    void enable_segmented_tier(::ninfer::amd::memory::KVSegmentedStorage& storage,\n"
       "        std::size_t maximum_request_bytes,std::span<const hipStream_t> all_prior_users,\n"
       "        std::span<const hipStream_t> all_borrower_users,\n"
       "        std::function<bool()> can_lend,std::function<void()> on_change);\n"
       "    [[nodiscard]] bool sticky_failure() const noexcept{return sticky_failure_;}\n"
       "    [[nodiscard]] std::string_view failure_reason() const noexcept{return failure_reason_;}\n"
       "    [[nodiscard]] std::exception_ptr failure_exception() const noexcept{return failure_exception_;}\n\n"
       "    // Free KV granules only; empty when they cannot fund the window.")
    eh("    std::function<void()> on_change_;", "    std::function<void()> on_change_;\n"
       "    [[nodiscard]] std::optional<VisionWindow> try_acquire_segmented(std::size_t);\n"
       "    void remember_segmented_failure(const char*,std::exception_ptr={}) noexcept;\n"
       "    ::ninfer::amd::memory::KVSegmentedStorage* segmented_storage_=nullptr;\n"
       "    std::size_t segmented_max_request_=0,segmented_prior_count_=0,segmented_borrower_count_=0;\n"
       "    std::array<hipStream_t,4> segmented_prior_{},segmented_borrowers_{};\n"
       "    bool sticky_failure_=false;\n"
       "    const char* failure_reason_=\"\";\n"
       "    std::exception_ptr failure_exception_;")
    ec("kv_(std::move(other.kv_)), runs_(std::move(other.runs_)) {",
       "kv_(std::move(other.kv_)), runs_(std::move(other.runs_)),\n"
       "      segmented_(std::move(other.segmented_)),segmented_users_(other.segmented_users_),\n"
       "      segmented_user_count_(other.segmented_user_count_),segmented_bytes_(other.segmented_bytes_),\n"
       "      segmented_open_seconds_(other.segmented_open_seconds_),segmented_close_seconds_(other.segmented_close_seconds_),\n"
       "      close_failed_(other.close_failed_) {")
    ec("        close();\n        broker_", "        if(!close_checked())return *this;\n        broker_")
    ec("        runs_         = std::move(other.runs_);", "        runs_         = std::move(other.runs_);\n"
       "        segmented_=std::move(other.segmented_);segmented_users_=other.segmented_users_;\n"
       "        segmented_user_count_=other.segmented_user_count_;segmented_bytes_=other.segmented_bytes_;\n"
       "        segmented_open_seconds_=other.segmented_open_seconds_;segmented_close_seconds_=other.segmented_close_seconds_;\n"
       "        close_failed_=other.close_failed_;")
    ec("void VisionWindow::close() noexcept {", "void VisionWindow::close() noexcept { (void)close_checked(); }\n\n"
       "void VisionWindow::close_legacy() noexcept {")
    ec("    kv_arena_  = &arena;", "    require_closed();\n"
       "    if(segmented_storage_)throw std::logic_error(\"cannot mix segmented and legacy raw Vision tiers\");\n"
       "    if(!EvictableKVPool::noncontiguous_lending_qualified())\n"
       "        throw std::logic_error(\"raw noncontiguous KV Vision lending is not qualified\");\n"
       "    kv_arena_  = &arena;")
    ec("    require_closed();\n    if (kv_arena_", "    require_closed();\n"
       "    if(segmented_storage_)return try_acquire_segmented(bytes);\n    if (kv_arena_")
    ec("    return pool_.poisoned() || (kv_arena_ != nullptr && kv_arena_->poisoned());",
       "    return sticky_failure_||pool_.poisoned() || (kv_arena_ != nullptr && kv_arena_->poisoned())||\n"
       "        (segmented_storage_&&segmented_storage_->poisoned());")
    ec("    return pool_.transaction_open() || (kv_arena_ != nullptr && kv_arena_->lease_open());",
       "    return sticky_failure_||pool_.transaction_open() || (kv_arena_ != nullptr && kv_arena_->lease_open())||\n"
       "        (segmented_storage_&&segmented_storage_->lease_open());")
    ec("void VisionResidencyBroker::require_closed() const {\n", "void VisionResidencyBroker::require_closed() const {\n"
       "    if(sticky_failure_||(segmented_storage_&&(segmented_storage_->poisoned()||pool_.poisoned())))\n"
       "        throw std::runtime_error(\"segmented Vision broker/pool has a sticky failure\");\n")
    # Inverse of EVERY changed extracted byte, not a sampled method comparison.
    restored_h=h;restored_c=c
    for old,new in reversed(h_edits):restored_h=exact(restored_h,new,old)
    for old,new in reversed(c_edits):restored_c=exact(restored_c,new,old)
    if restored_h!=h_original or restored_c!=c_original:raise ValueError("production component inverse failed")
    includes='''#pragma once
#include "core/arena.h"
#include "core/device.h"
#include "core/evictable_kv_pool.h"
#include "core/evictable_weight_pool.h"
#include "core/paged_kv_cache.h"
#include "kv_segmented_storage.h"
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <vector>
'''
    output_h=includes+"\n"+NAMESPACE+"\n"+h+"\n} // namespace\n"
    output_c=('#include "amd-segmented-vision/vision_broker.h"\n#include "core/kv_loan.h"\n'
        '#include <algorithm>\n#include <stdexcept>\n#include <string>\n#include <utility>\n\n'
        +NAMESPACE+"\n"+c+"\n"+methods+"\n} // namespace\n")
    return {OUT_H:output_h.encode(),OUT_C:output_c.encode()}, {
        "original_header_component_sha256":sha(h_original.encode()),
        "original_cpp_component_sha256":sha(c_original.encode()),
        "full_original_components_inverse_exact":True,
        "header_edit_count":len(h_edits),"cpp_edit_count":len(c_edits),
        "original_acquire_WeightTail_body_unmodified":True,
        "original_raw_bodies_retained_but_admission_gate_false":True,
        "Session_encoder_parameters_and_Engine_not_extracted":True,
        "legacy_weight_failure_is_not_segmented_checked_restore":True}

def load_capacity():
    path=ROOT.parent/"vision_segmented_capacity/prepare_vision_segmented_capacity.py"
    # Check executable recipe bytes before importing, not after its module body.
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=256*1024:
        raise ValueError("missing/non-small capacity preparer")
    if sha(path.read_bytes())!=CAPACITY_PREPARER_SHA:raise ValueError("frozen capacity v2 preparer changed")
    spec=importlib.util.spec_from_file_location("segmented_vision_frozen_capacity",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def prepare(controller:Path,native:Path):
    cap=load_capacity();manifest=cap.strict_json(cap.bounded_file(ROOT/"manifest.json",64*1024))
    for relative,digest in manifest["capacity_source_sha256"].items():
        if sha(cap.bounded_file(ROOT.parent/"vision_segmented_capacity"/relative))!=digest:
            raise ValueError("frozen capacity v2 recipe changed: "+relative)
    core,core_receipt,_=cap.prepare(controller)
    if len(core)!=manifest["core_outputs"] or len(core_receipt["compile_sources"])!=manifest["core_compile_sources"]:
        raise ValueError("full original 25-TU/67-output capacity Core snapshot required")
    sources={name:cap.bounded_file(native/name) for name in (HPATH,CPATH)}
    for name,raw in sources.items():
        if sha(raw)!=manifest["prepared_source_sha256"][name]:raise ValueError("prepared Vision source changed: "+name)
    own={name:sha(cap.bounded_file(ROOT/name)) for name in OWNED}
    added,proof=component(sources,(ROOT/"broker_methods.cpp.inc").read_text(encoding="utf-8"))
    outputs=dict(core);outputs.update(added)
    recipe={"owned_source_sha256":own,"prepared_source_sha256":manifest["prepared_source_sha256"],
        "capacity_recipe":core_receipt["recipe"],"schema":"ninfer.segmented_vision_broker.recipe.v1"}
    key=sha(json.dumps(recipe,sort_keys=True,separators=(",",":")).encode())
    receipt={"schema_version":1,"status":"SOURCE_GENERATED_NOT_BUILT_NOT_QUALIFIED", "recipe_sha256":key,
        "recipe":recipe,"capacity_core_receipt":core_receipt,"compile_sources":core_receipt["compile_sources"],
        "broker_compile_sources":[OUT_C],"output_sha256":{n:sha(r) for n,r in outputs.items()},
        "production_component_proof":proof,"single_full_Core_rebuild_required":True,
        "old_Core_ModelRuntime_Ops_archives_allowed":False,"Session_or_Engine_connected":False,
        "production_gate_changed":False,"physical_grants_or_restore_qualified":False}
    return outputs,receipt,key

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-view",required=True,type=Path)
    parser.add_argument("--native-view",required=True,type=Path)
    parser.add_argument("--output-root",required=True,type=Path)
    parser.add_argument("--emit",action="store_true");parser.add_argument("--verify-existing",action="store_true")
    args=parser.parse_args();cap=load_capacity()
    for path in (args.controller_view,args.native_view):
        if cap.reparse(path):raise ValueError("input root is a reparse point")
    controller=args.controller_view.resolve();native=args.native_view.resolve();root=args.output_root.resolve()
    def separate(path):
        for source in (controller,native,ROOT.resolve(),ROOT.parent/"vision_segmented_capacity"):
            if path==source or path in source.parents or source in path.parents:raise ValueError("output overlaps recipe/input tree")
    separate(root);outputs,receipt,key=prepare(controller,native);target=root/key;separate(target)
    if args.emit:
        if target.exists():
            if not args.verify_existing:raise ValueError("new output required; never overwrite")
            actual=cap.strict_json(cap.bounded_file(target/RECEIPT))
            if actual!=receipt:raise ValueError("existing broker source receipt differs")
            cap.verify_inventory(target,receipt["output_sha256"],RECEIPT)
        else:
            target.mkdir(parents=True,exist_ok=False)
            for name,raw in outputs.items():
                p=target/name;p.parent.mkdir(parents=True,exist_ok=True)
                with p.open("xb") as f:f.write(raw)
            with (target/RECEIPT).open("x",encoding="utf-8") as f:json.dump(receipt,f,indent=2);f.write("\n")
    print(json.dumps({"status":"SOURCE_GENERATED_NOT_BUILT_NOT_QUALIFIED" if args.emit else "PLAN_ONLY_NOT_GENERATED",
        "output":str(target),"recipe_sha256":key,"outputs":len(outputs),"production_gate_changed":False},indent=2))

if __name__=="__main__":main()
