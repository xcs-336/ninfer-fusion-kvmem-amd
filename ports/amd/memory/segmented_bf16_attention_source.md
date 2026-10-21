# Experimental table-aware BF16 consumer

Status, 2026-10-21: **SIX-CASE NATIVE FUNCTIONAL PASS; NOT ENGINE QUALIFIED.**
On Windows RX 9070 / gfx1201 wave32 / AMD clang 23 / MSVC 14.44, the actual
original-body candidate built and passed all six cases, 18 same-instance Graph
replays and 1,259,520 output checks. Independent FP64 criteria, complete cache
bytes, relocation bit parity, fresh output/append destinations and the raw borrower
span all passed. The production allocator/Attention/Engine gate remains unchanged;
these cases add zero to the original-registration count. Machine-readable scope is
in [tests/segmented_bf16_actual_scope.json](tests/segmented_bf16_actual_scope.json).

`Core.cmake` includes the separate `SegmentedPool.cmake` and
`SegmentedBF16Ops.cmake` leaves after native Core creation. Both options default
OFF; the BF16 leaf requires explicit `NINFER_AMD_SEGMENTED_POOL=ON` as well as
`NINFER_AMD_SEGMENTED_BF16_OPS=ON`. Their generated views are isolated from the
ordinary production Core and Ops sources.

## Address and ABI contract

Memory's isolated Core header appends `k_page_bases` and `v_page_bases` to
`PagedKVLayerView`/`PagedKVBatchLayerView`, under
`NINFER_AMD_SEGMENTED_POOL_ABI`. `DeviceKVPlaneView` pairs its geometry Tensor with
the stable per-plane device `U64[physical_group_count]` allocation. A table entry is
the actual address of one semantic plane/group cell, not an offset or a logical
block-table index. Original logical block tables still resolve logical page to
physical group. Storage can move a K cell into an originally V-addressed free slot;
the two semantic pointer tables remain independent.

For PageMajor `[256,64,KVHeads]`, the original element offset is
`d + 256*(page_offset + 64*(kv_head + KVHeads*physical_group))`. The helper divides
this exact original offset by `256*64*KVHeads` to obtain the physical group and
adds the remainder to the entry loaded on device. With an empty table it evaluates
the exact original `affine_base + original_element_offset` expression.

The table allocation stays stable across Graph capture/replay and is never part of
the raw borrowed span. Its contents change only after prior consumers finish and
before ready publication. A volatile device U64 load enforces an execution-time
compiler read; it does not provide GPU synchronization. The actual Core begin/wait/
close transaction supplies ordering and serializes host submissions. No page base
is read on host and embedded into the persistent Attention Graph.

The wrappers accept **only PageMajor D256 BF16**, H24/KV4 or H16/KV2, one dense
unmasked request, no scales and no selected-block bitmap. They take explicit
`PagedKVPlaneOrder`, rejecting HeadMajor. There is no overload for batched/padded
views, offsets, masked columns, selected blocks, gate epilogues, D128, quantized KV,
prefix/ring, sliding, or context-materialization routes. None is dispatched to an
unmodified old ABI TU. Before future public composition, each such consumer must be
ported or rejected by its actual public wrapper.

Device-data preconditions are explicit: positions are consecutive, nonnegative,
within logical capacity; dense history through the final query position is valid;
each block-table entry names a live minted physical group; each pointer table
contains the complete physical inventory with aligned, distinct live cells. Host
shape checks do **not** read or verify these device contents. In particular prompt
staging computes a page base before its row-tail predicate, so invalid sparse
history cannot be admitted merely because its invalid row would later be masked.

All view consumers must be rebuilt against the candidate header. Appending fields
does not alter C++ type names/mangled symbols, so linking an old public Op archive
can silently discard the table. The final actual link map must prove that the
isolated Core `paged_kv_cache.cpp` object is selected and the old Core object and
all original append/Attention/test-support members are absent. Normal archive
resolution is used; there is no whole-archive/force/duplicate-test-support workaround.

## Actual native bodies and precise access sites

Paths below are relative to the existing `ports/amd/build/generated/src/` native
view; original CUDA bodies have the corresponding `src-tree/fusion-engine-src/src/`
paths. The AMD view, not stale original tile comments, is the source baseline.

| Source and symbol | Exact access affected | Candidate change |
|---|---|---|
| `ops/kernel/paged_kv_address.cuh:54-80` | logical page (`position>>6`) to physical group; PageMajor affine offset | Preserve both formulas and the original block table |
| `ops/kv_cache/append/kernel.cuh:70-104`, `kv_cache_append_full_bf16_kernel` | two `store_vec(&cache_{k,v}[cache_off], int4)` at 102/103 | Redirect these two 16B destinations through the corresponding stable table |
| `ops/softmax_attention/dense/causal_cache/small_t_bf16.cuh:170-175` | owning split's inline-append K/V stores | Redirect two destinations; preserve direct new-input reads and split ownership |
| Same body, `:251-260` | cache branch of append and cached-only branch: K/V `cp_async<16>` source pairs | Redirect all four source expressions; preserve per-key validity/zero-fill |
| Same body, `:145-157,217-230` | staged physical-group IDs and page refresh | Preserve. Group ID is resolved to a pointer at the final access, not replaced by a host address |
| `ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh:31-63`, `causal_prompt_stage_kv` | one affine `cache_block` then 16B row-vector reads | Resolve this block base through its semantic K or V pointer table |
| Same body, `:193,204,382-383` | K prologue, V current tile, K next tile | Thread the appropriate table into all three existing stage calls |
| `small_t.cuh:213-324`, original reducer | FP32 split partials/statistics to BF16 output | Unmodified original reducer and matching active-split policy |
| `prompt_common.cuh:20-26` | actual native Br32/Bc32, 64 threads, 49,152 dynamic LDS bytes | Unmodified native configuration; configure actual kernel attribute before capture |
| `append/kernel.cuh:391-415`, D128 prefix helper | `physical_page + physical_pages*kv_head` HeadMajor offset | Explicitly excluded; never run this offset through the D256 helper |

The small-T Bc32 and native prompt Bc32 tiles start at multiples of 32. Their
page-local start is 0 or 32; all 32 rows and their 16B vectors remain within one
64-token cell. Append vectors contain eight BF16 values and remain within a D256
row. Heads and groups are preserved by the unchanged offset remainder. The helper
does not split a vector at a page boundary or introduce a second payload/cache.

`segmented_bf16_ops_overlay.py` pins eight small native input headers with raw-byte
SHA256 and exact anchors. It extracts the complete original native BF16 append
kernel and copies the complete native small-T/prompt bodies. Changes are limited
to private symbol names, added pointer-table arguments/includes, and the listed
address expressions. All Q/K/P/V WMMA, FMA, online softmax, reductions, predicates,
barriers and BF16 boundaries remain from those bodies. The generator is import-safe,
writes only a new isolated include root, and supports byte-exact `--verify-existing`
without overwriting files. The actual generated view was compiled and executed in
the six-case functional run; the original mathematical bodies and criteria were
not relaxed.

## Candidate fixture and oracle

`tests/kv_segmented_bf16_attention_test.cu` directly uses the actual isolated
`KVSegmentedNativePool`, original reserve/materialize/lease ownership, and original
`KVExecutionTablePool.publish` with real handles ordered 2,0,1. It never calls the
private physical-index API or reopens the sealed startup raw span. Initial valid
history is written with the actual copied BF16 append body. There is one physical
storage payload and a stable pointer table, not an affine mirror of live KV.

Six bounded geometry/mode cases are prepared: H24/KV4 and H16/KV2, each with
T3 append-then-cached-small at positions 63..65, T3 inline-small-append at 127..129,
and T35 prompt append at 95..129. They cover 63/64/127/128, multiple heads, three
logical page groups and a nonmonotonic logical block table. Each captures once and
replays the **same exec/table allocation** before relocation, while a raw-span
borrower overwrites the old storage, and after completion/return: 18 replays total.
An empty-table affine control runs only before any relocation and is never replayed
against moved storage.

Before **each** same-Graph replay, real Core `copy_from_host_records` transfers
canonical pinned Host records for only the two affected pages through the current
mapping. All non-target history retains its expected bytes; only appended K/V rows
become `0xffff` BF16 NaNs. The fixture reads every poisoned row back independently
to confirm fresh destinations, and sets only the output payload to NaNs. Guards
are untouched. This prevents a no-op Graph/output/store or a wrong affine append
from passing with correct stale data. During the loan it additionally verifies that
every byte of the active raw borrower span remains `0xd3` after the Graph. That raw
capability is discarded immediately after close and is not used after return.

The CPU oracle independently decodes represented BF16 bits and computes full D256
QK dot products, stable FP64 softmax, and FP64 PV over every causal key. It reuses
the original `op_check.h` reduction evaluator and the **unchanged original BF16
Attention criterion** from `tests/ops/softmax_attention/causal_cache.cpp:98-102`:
relative L2 <= 2.8e-3 **and** maximum absolute error <=
`1.0e-3 + (2/256)*maximum_absolute_reference`, with finite outputs. The host oracle
locally disables contraction with `#pragma clang fp contract(off)`; no whole-TU
`-ffp-contract=off` is applied to the actual device bodies.

The fixture also checks every live allocated cache BF16 word against an independent
CPU scatter, full output bits against the affine control across all relocations,
q/new-k/new-v/positions/block-table preservation, and prefix/suffix guards on inputs,
output and partial buffers. The six cases contain 1,259,520 checked output values
across the 18 replays. This is not all widths, all head geometries, batched/masked
routes, all original BF16 registrations, allocator-concurrency qualification, or a
forced-pending ordering test. The earlier pending Core fixture remains separate.

## Explicit build/run and actual evidence

The two leaves are included after native Core creation. Configuration uses both
`-DNINFER_AMD_SEGMENTED_POOL=ON` and
`-DNINFER_AMD_SEGMENTED_BF16_OPS=ON`; build target:

```text
cmake --build ports/amd/build --target ninfer_amd_segmented_bf16_attention_test -j1
```

This build target was actually built and run. The leaf gives the candidate Core
include root explicit precedence over native/original headers, compiles the real
device TU through `ninfer_native_hip`, uses the actual native FP32 divide/sqrt flag,
links only the new Core candidate/normal SDK closure, and emits an actual link map.
Its post-link gate reuses the isolated Core map witnesses, requires this fixture's
main object, and rejects old public Attention/append/test-support extraction.
It does not register an automatic GPU CTest. Its no-argument executable mode is
PLAN_ONLY; `--run` is required for any HIP call. Inspect the actual map, then run
under the byte0/job/deadline/host-resource protocol; no source
receipt or successful compilation substitutes for GPU oracle/guard evidence.
`--run` also rejects `NINFER_HOST_PAGEABLE` when it would bypass Core's actual
`PinnedHostBuffer` pinning path; default plan still makes no HIP call.

The first post-link gate failed because LLD's actual archive records omit `.lib`.
The parser now accepts bare, `.lib:` and `.lib(` archive formats while requiring
exact archive/member identities and rejecting prefixed impostors and old members.
An independent audit accepted the actual 3,086-line BF16 map, five positive
formats and rejected 27 negative controls. The gate was repaired, not bypassed.
The dedicated original-Core fixture separately built and passed its 3,101-line
map gate and subsequently passed its actual nine-replay byte/fragmented-transfer
fixture. Its separate runtime receipt is included in the combined scope below.

Native log: `ninfer-segmented-bf16-run-first-20261021.log`, exit0,
SHA256 `2fe38709550a789446dc0862ff416b9bb886f49f76cded7f3f97cfdd55845f35`.
The host guard observed minimum available RAM 5.074 GiB during that run, using
0.05 GiB fixture + 1 GiB runtime reserve + 2 GiB floor admission. This is a guard
sample, not a continuous process/adapter peak or a measured resource saving.

Static explicit sizes are small: the largest storage payload is 2 MiB and its
stable table 128 B. The largest small-T partial accumulator is 8,355,840 B
(H16/KV2 T3/170 splits); prompt avoids allocating a large unused partial arena.
Per-seed pinned canonical Host records are at most 524,288 B, plus the storage
pointer shadow and original execution-table Host shadow. Exact driver Graph/event/
stream and runtime overhead is unmeasured. These are source sizes, not sampled
resource minima or continuous peaks. There is no timing, performance advantage,
VRAM saving, production allocator/Attention admission, text-only Bonsai benefit, or
end-to-end claim. Public B1 dynamic rows/gates, controller transfer, Models/real
borrower lifetime and concurrent allocator mutations require separate validation.
