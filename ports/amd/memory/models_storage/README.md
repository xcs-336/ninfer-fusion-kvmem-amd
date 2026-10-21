# Two-file Models storage candidate

Status, 2026-10-21: **ACTUAL TWO-GEOMETRY / SIX-REPLAY STORAGE FUNCTIONAL PASS;
DEFAULT OFF; NOT ENGINE INTEGRATED.** Generated two-file Models storage and full
new-Core linkage passed the actual 3,055-line map and guarded byte/Graph execution.
Only the real prepared-native `models/qwen3_5/state/decoder_state.h` and `.cpp`
are transformed into a new isolated source view. No production Models, Program,
Core, existing candidate, or Engine file is changed. This is a storage composition
candidate, not an Attention implementation or a new working Engine mode.

## Concrete storage changes

The isolated `PagedKVCache` stores `optional<DeviceKVPagePool> owned_pages_` and
a nonowning selected `DeviceKVPagePool* pages_`. Its ordinary constructor genuinely
constructs the optional original pool from the original backing/layout and selects
that object. Its real original execution-table pool binds that selected pool. The
ordinary ring/selection/quantized planning paths remain present; this source-only
change does not requalify them.

The explicit `segmented_decoder_storage` tag constructor takes ordinary metadata,
the separately planned layout, and a `KVSegmentedNativePool&`. It leaves the optional
owner empty, validates the actual native pool under its real allocator Freeze, and
selects exactly `native.page_pool()`. It does not instantiate another pool, payload,
DeviceArena, or allocation ledger. `KVExecutionTablePool` remains the original object,
with its real pinned shadow and metadata regions, addressing that same selected pool.
The native factory and its original pool must outlive the cache, table row leases,
page leases/reservations, transfers and Graph consumers. The tag borrows the factory;
it neither owns it nor extends its lifetime.

Both direct and batch layer views obtain the selected K/V `plane_view()` geometry
and plane-specific stable device U64 pointers. Ordinary unbound pools return null
tables with the original affine geometry. The tagged BF16 path does not call the
sealed `plane()`. The direct view retains the actual acquired execution-table row;
the batch view retains the real one-row execution-table matrix. Table entries must
be read at GPU execution time by separately qualified table-aware Ops. Copying these
fields alone does not make an old by-value Op or old Models archive ABI-compatible.

## Explicit admitted storage scope

Planning rejects unsupported inputs before touching either layout builder:

- BF16, PageMajor, D256, uniform KV-head count 2 or 4, positive Main layers.
- One rank and one execution-table row, capacity 1..512, physical groups at least
  `ceil(capacity/64)`. Empty rank map or an exact all-zero Main layer map.
- No MTP enable/layers/physical groups, quantization/scales, physical ring or selection.

The tagged constructor repeats layout admission, checks exact native geometry,
capacity, planes/ranks/strides, fresh zero-based payload offsets, real table shape and
metadata bounds, payload/table footprint, and rejects metadata aliasing the native
payload or native pointer table. It acquires the genuine controller Freeze, rejects
an open grant/loan or poisoned storage, and uses actual `plane_view()` controller
checks. Nonempty `install_selection()` explicitly fails for tagged caches. An ordinary
constructor refuses a layout marked as separately stored, before constructing an
owned pool. A tagged DecoderState refuses an MTP layout before constructing tables.

B1 here is a **storage/table-row** restriction. These two files cannot inspect a
future Op's device row selector, positions or history. The future consumer must
enforce B1, valid nonnegative published group IDs, consecutive positions and a complete
dense resident history within the admitted capacity. They do not admit true batching,
masked verification, selected history, DFlash, MTP, multirank pipelines or any old
untouched Attention TU. There is no permissive fallback for an explicit tagged request.

## Fresh planning, no offset subtraction

`plan_segmented_decoder_state(fresh_payload, ordinary_metadata, spec)` requires
different builders and `fresh_payload.finish()==0`. The real original page-pool
planner writes only the new zero-based payload builder. The real execution-table
planner writes only the ordinary builder, which may already contain other regions.
The returned DecoderState layout retains the real page layout and its logical
payload byte count; it is marked for the tagged constructor. No old offset is reduced,
no dummy affine pool is used, and no second physical payload is planned in the
ordinary builder.

This does **not** modify `program/planning/startup.cpp`, persistent-layout ownership,
replay/GDN/round/logits/workspace regions, Program construction, logical address-space
stores, memory accounting or the old lendable-prefix field. A future Program must
explicitly call this new planner, replan the remaining ordinary state through the
ordinary builder, allocate the one native factory, charge its payload/table once,
and keep that factory before all consumers in declaration/destruction order. The
existing ordinary startup path remains unchanged. The separate new controller
handoff must be integrated at Engine/scoring worker startup; it is not called here.

## Future isolated build and link gate

`ModelsStorage.cmake` is unwired and defaults `NINFER_AMD_MODELS_SEGMENTED_STORAGE`
to OFF. Both targets are `EXCLUDE_FROM_ALL`. It requires the new full-Core controller
candidate and uses only that same ABI/source root. Root may later include this leaf
after the controller leaf, then explicitly build:

```text
ninfer_amd_models_segmented_storage_test
```

The library `ninfer_models_storage_candidate_amd` contains only the transformed real
DecoderState TU. It links the full `ninfer_controller_handoff_core_amd`, never an old
Models/Engine/Core or partial segmented archive. The mandatory map gate checks exact
archive and object basenames (bare, `.lib:` and `.lib(member)`), real member-plus-symbol
witnesses, and rejects extracted old Core, partial segmented, Models or Engine members.
No whole-archive/force/test-support shortcut is present. An actual map pass is not
runtime or Engine qualification.

The preparer pins the two real native inputs, checks the entire supplied new Core
source inventory and hashes, and writes only two Models files plus a receipt into a
new view. Existing output must contain exactly those files and receipt, with no
symlinks or extra headers. It never overwrites an existing view. The native BF16
profile header is used only for the original inline geometry/codec description;
there is no original Op library link or new device mathematics in this target.

## Planned source fixture

Default mode reports PLAN_ONLY with no explicit HIP call in `main`. A HIP-linked
binary may still load/register runtime state before `main`; absence of DLL/runtime
activity is not claimed. Only `--run` performs the test, under root's future lock,
host guard, deadline and actual-PID monitor. No automatic GPU CTest is registered.

The proposed fixture has two sequential KV2/KV4 geometries, two Main layers, context129,
four physical groups and three nonmonotonic published logical IDs. It checks the
ordinary actual-owned path, tagged pointer identity and empty optional owner, builder
separation, selection/ordinary-constructor refusal, real HostKVArena upload and actual
execution-table publication. A fresh-output guarded independent byte observer reads
the Models views through one Graph/exec before, during and after a real native loan:
six total planned replays, no captured host-resolved page address. Full logical bytes
and non-KV metadata markers must survive borrower overwrite/relocation and return.
The requested window spans all four physical cells of the first semantic plane;
its three live cells must move into the free group's cells in other planes. The
fixture requires the reported copied bytes to equal those three full cells, then
overwrites the old window and verifies all semantic planes through the same Graph.
The byte pattern is arbitrary uint16 storage, not an Attention FP64/numerical oracle.

Planned negative specifications cover capacity513, KV1, D128, FP8, two rows, MTP,
undersized physical ring, nonzero rank, and MTP physical storage, plus aliased or
nonfresh payload builders. They do not exercise every malformed fabricated layout,
concurrent allocation, pending-work control, device faults, Engine handoff or all
original DecoderState tests. No test has been compiled or run.

Largest explicit device allocation in the proposed KV4 case: 2,097,152 payload +
128 device pointer table + 512 ordinary metadata + 1,573,376 guarded observer =
**3,671,168 bytes**. Explicit pinned payload is 1,572,864 HostKVArena + 128 pointer
shadow + 12 execution shadow = **1,573,004 bytes**. Runtime/Graph/driver allocations,
vector capacities, metadata and stacks are excluded. A conservative future host
admission is fixture0.05 GiB + runtime1 GiB + remaining floor2 GiB, with actual monitoring.
No memory saving, pure-text benefit, quality pass, throughput or E2E advantage is claimed.


Core now includes the leaf, default OFF. Actual standalone build/map/runtime
passed with the same selected original pool, metadata tables and physical
relocation. This does not close Program state planning/billing, runtime worker
hooks or Attention composition. The actual host guard minimum4.863GiB is an
admission observation, not a continuous peak or resource-saving measurement.
