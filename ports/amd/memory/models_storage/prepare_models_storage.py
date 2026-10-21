"""Two-file, default-unwired Models storage candidate; no production writes.

Import/transform is pure. Main requires a future explicit resource window and a
real new controller full-Core view; existing output is byte-verified, not replaced.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

PINNED = {'.h': '8d78255b0494c69b369bf0ece6f7d8bc61280eba74bc656658b6898dc9a0ca99', '.cpp': 'f22193dd1ffee770f7f9f392a43bc8bcf37324b9b4f3c61b064dfdd38c98c404'}
REL='src/models/qwen3_5/state/decoder_state'
def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def exact(text:str,old:str,new:str,count:int=1)->str:
    if text.count(old)!=count:raise ValueError('Models storage exact anchor changed: '+old[:100])
    return text.replace(old,new)

def require_exact_inventory(root,expected:set[str])->None:
    if root.is_symlink():raise ValueError('source view root is a symlink')
    entries=list(root.rglob('*'))
    if any(path.is_symlink() for path in entries):raise ValueError('source view contains symlink')
    actual={path.relative_to(root).as_posix() for path in entries if path.is_file()}
    if actual!=expected:raise ValueError('source view inventory differs')

def transform(header:str,source:str,implementation:str)->tuple[str,str]:
    h=exact(header,'namespace ninfer::models::qwen3_5 {',
        '#ifndef NINFER_AMD_SEGMENTED_CONTROLLER_HANDOFF_ABI\n#error "Rebuild all candidate Core/Models against the new full Core ABI"\n#endif\n'
        '#define NINFER_AMD_SEGMENTED_MODELS_STORAGE_ABI 1\n'
        'namespace ninfer::amd::memory {class KVSegmentedNativePool;}\n\nnamespace ninfer::models::qwen3_5 {\n'
        'struct SegmentedDecoderStorageTag {explicit constexpr SegmentedDecoderStorageTag()=default;};\n'
        'inline constexpr SegmentedDecoderStorageTag segmented_decoder_storage{};')
    h=exact(h,'    KvCacheStorage storage    = KvCacheStorage::BFloat16;',
        '    KvCacheStorage storage    = KvCacheStorage::BFloat16;\n'
        '    bool external_segmented_payload=false; // private isolated layout, never ordinary backing')
    h=exact(h,'    PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout);',
        '    PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout);\n'
        '    PagedKVCache(SegmentedDecoderStorageTag,DeviceSpan metadata,const PagedKVCacheLayout&,\n'
        '                 amd::memory::KVSegmentedNativePool&);')
    h=exact(h,'return pages_;','return *pages_;',2)
    h=exact(h,'    DeviceKVPagePool pages_;',
        '    std::optional<DeviceKVPagePool> owned_pages_;\n    DeviceKVPagePool* pages_=nullptr;')
    h=exact(h,'    const std::uint32_t* selected_blocks_ = nullptr;',
        '    bool external_segmented_pool_=false;\n    const std::uint32_t* selected_blocks_ = nullptr;')
    h=exact(h,'    [[nodiscard]] PagedKVBatchLayerView batch_layer_view(std::uint32_t layer) const;',
        '    [[nodiscard]] PagedKVBatchLayerView batch_layer_view(std::uint32_t layer) const;\n'
        '    [[nodiscard]] bool owns_page_pool()const noexcept{return owned_pages_.has_value();}')
    h=exact(h,'struct DecoderState {',
        '[[nodiscard]] DecoderStateLayout plan_segmented_decoder_state(LayoutBuilder& fresh_payload,\n'
        '    LayoutBuilder& ordinary_metadata,const DecoderStateSpec&);\n\nstruct DecoderState {')
    h=exact(h,'    DecoderState(DeviceSpan backing, const DecoderStateLayout& layout);',
        '    DecoderState(DeviceSpan backing, const DecoderStateLayout& layout);\n'
        '    DecoderState(SegmentedDecoderStorageTag,DeviceSpan metadata,const DecoderStateLayout&,\n'
        '                 amd::memory::KVSegmentedNativePool&);')
    c=exact(source,'#include "ops/kv_cache/d256_profile.h"',
        '#include "ops/kv_cache/d256_profile.h"\n#include "kv_segmented_pool_owner.h"\n#include <array>')
    c=exact(c,'                              std::uint32_t physical_page_groups) {',
        '                              std::uint32_t physical_page_groups,\n'
        '                              LayoutBuilder* external_payload_builder=nullptr) {')
    old='''    PagedKVCacheLayout out{
        .pages = plan_device_kv_page_pool(
            builders, plane_rank,
            DeviceKVPagePoolSpec{.page_group_count = physical_page_groups,
                                 .allow_logical_reservation = logical_pages > physical_page_groups,
                                 .geometry         = std::move(geometry)}),'''
    new='''    DeviceKVPagePoolSpec page_spec{.page_group_count=physical_page_groups,
        .allow_logical_reservation=logical_pages>physical_page_groups,.geometry=std::move(geometry)};
    auto planned_pages=external_payload_builder
        ? plan_device_kv_page_pool(*external_payload_builder,page_spec)
        : plan_device_kv_page_pool(builders,plane_rank,page_spec);
    PagedKVCacheLayout out{
        .pages = std::move(planned_pages),'''
    c=exact(c,old,new)
    c=exact(c,'        .storage        = storage,','        .storage        = storage,\n'
        '        .external_segmented_payload=external_payload_builder!=nullptr,')
    c=exact(c,'PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)',
        implementation+'\nPagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)')
    c=exact(c,'    : pages_(backings, layout.pages), execution_tables_(backings, layout.execution_tables, pages_),',
        '    : owned_pages_(std::in_place,backings,require_owned_layout(layout)),pages_(&*owned_pages_),\n'
        '      execution_tables_(backings, layout.execution_tables, *pages_),')
    c=exact(c,'pages_.','pages_->',6)
    c=exact(c,'    return PagedKVLayerView{\n        .k_pages       = pages_->plane(base),\n        .v_pages       = pages_->plane(base + 1),',
        '    const auto k=pages_->plane_view(base),v=pages_->plane_view(base+1);\n'
        '    return PagedKVLayerView{\n        .k_pages       = k.storage,\n        .v_pages       = v.storage,')
    c=exact(c,'        .selected_blocks = selected_blocks_,',
        '        .selected_blocks = selected_blocks_,\n        .k_page_bases=k.page_bases,\n        .v_page_bases=v.page_bases,')
    c=exact(c,'        .selected_blocks = direct.selected_blocks,',
        '        .selected_blocks = direct.selected_blocks,\n        .k_page_bases=direct.k_page_bases,\n'
        '        .v_page_bases=direct.v_page_bases,')
    c=exact(c,'void PagedKVCache::install_selection(std::span<const std::uint32_t> words) {',
        'void PagedKVCache::install_selection(std::span<const std::uint32_t> words) {\n'
        '    if(external_segmented_pool_&&!words.empty())\n'
        '        throw std::invalid_argument("segmented Models storage does not admit selection/ring");')
    return h,c

def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-root',required=True,type=Path);p.add_argument('--core-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--verify-existing',action='store_true')
    a=p.parse_args()
    if a.output.is_symlink():raise ValueError('output root must not be a symlink')
    native=a.native_root.resolve();core=a.core_root.resolve();output=a.output.resolve()
    for path in (native,core):
        if path==output or path in output.parents or output in path.parents:raise ValueError('separate new output required')
    if output.exists() and not a.verify_existing:raise ValueError('never overwrite existing view')
    core_receipt_path=core/'controller-handoff-source-receipt.json'
    if not 0<core_receipt_path.stat().st_size<=1024**2:raise ValueError('Core receipt exceeds bound')
    core_receipt_raw=core_receipt_path.read_bytes();receipt=json.loads(core_receipt_raw)
    if not receipt.get('all_core_rebuild_required') or receipt.get('old_core_or_old_candidate_archives_allowed') is not False:
        raise ValueError('actual full new-Core source receipt required')
    expected=receipt['output_sha256']
    if not isinstance(expected,dict) or not 1<=len(expected)<=768:raise ValueError('Core source inventory exceeds bound')
    require_exact_inventory(core,set(expected)|{'controller-handoff-source-receipt.json'})
    total=0
    for name,digest in expected.items():
        path=core/name
        if not path.resolve().is_relative_to(core) or not 0<path.stat().st_size<=2*1024**2:
            raise ValueError('invalid Core source path/size')
        data=path.read_bytes();total+=len(data)
        if total>16*1024**2 or sha(data)!=digest:raise ValueError('Core source content differs')
    header=(core/'src/core/paged_kv_cache.h').read_bytes()
    if sha(header)!=expected['src/core/paged_kv_cache.h'] or b'#define NINFER_AMD_SEGMENTED_CONTROLLER_HANDOFF_ABI 1' not in header:
        raise ValueError('wrong Core header or ABI')
    inputs={ext:(native/(REL+ext)).read_bytes() for ext in ('.h','.cpp')}
    if {ext:sha(data) for ext,data in inputs.items()}!=PINNED:raise ValueError('real generated DecoderState source changed')
    own=Path(__file__).resolve().parent;impl=(own/'decoder_storage_impl.cpp.inc').read_bytes()
    profile=native/'src/ops/kv_cache/d256_profile.h'
    if not 0<profile.stat().st_size<=2*1024**2:raise ValueError('native BF16 profile header exceeds bound')
    profile_sha=sha(profile.read_bytes())
    h,c=transform(inputs['.h'].decode().replace('\r\n','\n'),inputs['.cpp'].decode().replace('\r\n','\n'),impl.decode())
    outputs={REL+'.h':h.encode(),REL+'.cpp':c.encode()}
    report={'schema_version':1,'status':'SOURCE_GENERATED_NOT_BUILT_NOT_QUALIFIED','core_root':str(core),
        'core_receipt_sha256':sha(core_receipt_raw),'native_root':str(native),'input_sha256':PINNED,
        'recipe_sha256':sha(Path(__file__).read_bytes()),'implementation_sha256':sha(impl),
        'native_profile_header_sha256':profile_sha,
        'output_sha256':{k:sha(v) for k,v in outputs.items()},'production_changed':False,'engine_integrated':False}
    if output.exists():
        require_exact_inventory(output,set(outputs)|{'models-storage-source-receipt.json'})
        if json.loads((output/'models-storage-source-receipt.json').read_text())!=report:raise ValueError('existing receipt changed')
        for name,data in outputs.items():
            if (output/name).read_bytes()!=data:raise ValueError('existing output changed')
        return
    output.mkdir(parents=True,exist_ok=False)
    for name,data in outputs.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(data)
    with (output/'models-storage-source-receipt.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
if __name__=='__main__':main()
