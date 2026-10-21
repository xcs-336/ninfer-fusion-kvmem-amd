# Partial Engine storage preparation, with an explicit execution barrier

Status: **SOURCE ONLY / PARTIAL IMPLEMENTATION / NOT GENERATED / NOT BUILT /
NOT ENGINE READY**. Ownership is only this new directory. No Core, CMake, frozen
Models candidate, live native source or existing receipt is modified. There is no
new working Engine option or build target.

## Implemented source candidate

`prepare_engine_storage_plan.py` pins three real prepared-native sources and can,
in a future authorized source-generation window, derive `planning/startup.h`,
`planning/startup.cpp`, and `program.cpp`. Import/transform performs no writes.
Existing output must match bytes, receipt and complete inventory; symlinks or
extra headers fail. The input Models view must contain exactly the two validated
Models storage outputs and its receipt. This is not a full compiler/header proof.

The planner adds a default-false internal `segmented_text_storage` bit to planning
inputs and the finalized implementation. Existing public creation still selects
false; ordinary storage/precision/policy stays unchanged. An explicit new **internal**
`make_segmented_sequence_planner_impl(parameters, device, options)` entry derives a
candidate plan. It does not expose or parse an Engine CLI/HTTP option.

For that entry, the actual original `persistent_layout()` calls the new Models
planner with a fresh zero-based text-payload builder and the ordinary rank0 builder
for execution metadata. It then executes all the original subsequent StateImage,
FP32 GDN, round, hidden, score-hidden and sampling planning in order on the ordinary
builder. Offsets are genuinely replanned, not shortened/subtracted from old offsets.
MTP physical/layer fields are zeroed only for the explicit candidate. The original
logical `kv_payload_bytes` still includes Main payload. `lendable_kv_end_bytes` is
zero for the ordinary arena, never a VMM/free-suffix assertion.

New persistent fields record external payload and `8 * planes * physical_groups`
device-table bytes. Original sequence reservation and its measured adjacent-page
capacity slope include ordinary remaining state + external payload + device table
+ workspace + Graph allowance exactly once. Overflow uses checked sums and a
checked table product. No driver-residency/heap-rounding saving is claimed. The
existing planner may query device metadata; no C++ planner execution was performed.

This initial explicit candidate also rejects unsupported configuration before any
**new Program resident allocation**: one actual/configured device/rank, B1, context
1..512, BF16 PageMajor D256, H16/KV2 or H24/KV4, no MTP/DFlash/copy proposer/Vision,
FP32 GDN, physical>=logical, and no ring/score/shadow/reuse mode. It additionally
requires context cache explicitly disabled and disk tier absent. It never silently
changes these options. Weights are materialized before the existing sequence planner;
rejecting a new option before weight upload would need an additional earlier runtime
gate, which is not implemented here.

Most importantly, the derived `Program::create_program()` rejects a flagged plan
**before** `make_unique<ProgramImpl>`. The single-factory Program ownership, public
opaque token bridge, worker handshake/exit and uniform-ABI Op closure below are
not implemented. The guard prevents this partial planner output from accidentally
allocating an ordinary duplicate payload or running through old worker/Op paths.
No numerical stub, alternative inference route, or fallback is installed.

## Verified real object and lifetime chain

Current native source paths below are under `src/` of the generated view. The JSON
patch-point companion records exact current lines and source SHA values.

1. `runtime/engine/model_instance.cpp:471–475` constructs the Qwen Program before
   either runtime worker exists. `program/program.cpp:658–669` owns its PImpl and
   calls the real `ProgramImpl` constructor. Program is not a shared virtual interface
   across targets: `ModelInstance` and `FlashNextInstance` own different concrete
   Program types (`runtime/engine/model_instance.h:49–88`).
2. `ProgramImpl` initializes persistent/workspace arenas at `program_impl.cpp:164–173`,
   binds the decoder at:261 and creates logical stores from that exact decoder pool
   at:264–268. StateImage/GDN and other regions bind the planned ordinary backing.
   Startup initializes buffers, synchronizes, prepares every Graph, synchronizes
   again, then creates the hybrid cache at:550–569.
3. `EngineCore` worker startup binds `DeviceContext` then immediately resolves its
   startup promise (`runtime/engine/engine_core.h:120–134`). CausalScoreCore does the
   same at:30–41. Neither presently transfers the Core startup-bound controller.
4. Normal Engine shutdown calls `fail_all_locked(...Shutdown)` on the worker before
   return (`engine_core.h:2370–2380`), then the destructor joins at:144–151. Scoring
   returns its worker on the stopping/no-job boundary, then its destructor joins.
   Neither invokes checked controller retirement today. Program destruction occurs
   later and currently calls `flush_disk_tier()` then stream synchronization
   (`program_impl.cpp:577–583`).

## Concrete next Program patch, deliberately not applied

Add `unique_ptr<KVSegmentedNativePool>` **before** `kv_arena`/persistent/decoder and
all logical stores/Graph users in `ProgramImpl` declaration order. Construct it only
after the finalized flagged plan passes actual device/geometry/no-backend admission,
and before any resident arena allocation. The unflagged path must leave it null.
Use its real zero-based `decoder.text_kv.pages` layout, one device, `2*MainLayers`
homogeneous slots of `256*64*KVHeads*2` bytes and the resolved physical groups. The
existing native factory owns the single payload and original ledger. No temporary
ordinary decoder/page pool, second DeviceArena payload or post-hoc aliasing is legal.

Keep `persistent` owning only newly planned ordinary bytes; `persistent_by_rank` must
be empty. At:261 select the tagged DecoderState constructor with rank0 metadata and
that factory, otherwise keep the original ordinary constructor. Logical page/address
stores at:264–268 must continue borrowing that same selected original pool and tables.
Program memory summary at:795–797 must add external payload **and device table** to
resident sequence capacity/used/peak once; loaning does not free either allocation.
Pinned shadow, slot maps, owner/grant/planner metadata and stream/event overhead need
separate accounting, not mislabeling as Host KV contents. The existing public summary
has no dedicated segmented-metadata field; an explicit telemetry/schema decision is
still required. The ordinary arena's old lendable-prefix field remains zero.

## Concrete public Program / worker token bridge, not implemented

Keep Core token representation out of runtime templates. Add a move-only **opaque**
Qwen Program worker-transfer type in `program/program.h`, with out-of-line lifetime
and a PImpl holding the actual Core token plus Program identity. Program methods
release/accept/retire delegate to its PImpl/native factory; an ordinary unbound Program
has no transfer, while a flagged Program must require its valid token. Do not add an
unchecked `rebind_thread()` or silently accept a missing token for the flagged path.

The runtime has two actual target types. Use explicit target-aware overloads or a
compile-time Qwen branch so unchanged FlashNext does not need fake token methods.
Release once on the startup controller **before spawning** the worker, move the
opaque token into the actual thread closure, bind the same `DeviceContext`, consume
the token, and only then set the existing startup promise. Wrong owner/thread/token
fails closed; thread-creation/accept failure must destroy that unusable instance,
not retry/re-enable it. The Core token's separate identity/generation permits legal
original CPU RAII changes while in transit.

Install a thread-entry exit guard around every return from `worker_loop()`, covering
startup-promise failure after acceptance and fatal exits. Normal worker shutdown
must finish/abort requests, close loans, complete all stream work, then call checked
Core retire before join. A retire synchronization error must be retained as a sticky
worker failure/diagnostic and leave Core disabled; a noexcept destructor cannot just
discard it. Parent-side teardown after join may destroy Graphs and release original
CPU leases, but must not submit new pool GPU copies/views after retirement. The
initial no-disk/no-hybrid/no-Vision scope removes the known post-join GPU borrowers;
their later admission needs independent ordering/affinity closure.

## Actual stream inventory and two real additional blockers

For this finite scope, collect rank0 compute/transfer/any nonnull Vision stream from
the real DeviceContext, plus default stream for original helper work, and deduplicate
aliases. `RankStreams::compute/transfer` at `program_impl.cpp:191–192` refer to those
rank streams, not additional streams. DeviceContext aliases are not a second inventory.
The native backend's private transfer stream is internally noncapturing/ready-synced;
the cold DeviceSync covers it. No other host submitter or Core-reentrant pending
host callback may cross the cold mutex-held synchronization.

HybridPrefixCache is different: `prefix/hybrid_cache.cpp:86` creates a private
nonblocking `restore_stream_`, stored privately at `hybrid_cache.h:288`. It has no
public stream getter. A claimed complete inventory must expose it through an explicit
Program-owned inventory accessor or reject Hybrid before allocation. This candidate
does the latter by requiring context cache disabled. Disk workers and Vision borrowing
must also remain absent initially.

`storage/context.cpp:1637` still calls the bound pool's rejected `free_runs()` during
Host restoration. Do not unseal that span. If retaining/context pressure is later
admitted, replace this inspection with a proper real allocator snapshot under Freeze,
respecting promised/allocated/lent capacity, or prove the specific call unreachable.
Disabling retained context is a finite admission restriction, not that replacement.

## Actual public B1 symbols and no mixed ABI link shortcut

The independent candidate's existing header
`memory/b1_public/public_bf16.h` declares:

- `segmented_b1_causal_softmax_attention`
- `segmented_b1_causal_softmax_attention_workspace_capacity_bytes`
- `segmented_b1_causal_softmax_attention_route_family`

It is a distinct public wrapper, not an automatic override of the original symbol.
The two Main full-attention calls at `execution/text.cpp:983,992` must explicitly
select that symbol for table-backed views, preserving device table rows, envelope,
column offsets, and exact gate/rotated-output handling at:999–1005. Do not replace the
gate or return ungated bytes. Matching workspace/profile queries in startup/graph
planning must use the matching candidate's actual capability, including declarations
for renamed prompt helpers if used; the current independent header declares only
the three functions above, so a broad prefix regex is not a legal API migration.

Unexecuted MTP/DFlash/cached/quantized branches in `text.cpp` and workspace planning
still reference original cache-Op symbols. Runtime admission does not remove those
references from an object. Rebuild all affected Models/runtime/actual cache Ops with
the same new full-Core headers; preserve ordinary original implementations where
needed, or provide a separately reviewed finite closed TU set. Do not satisfy them
by extracting existing old-header archives merely because no branch executes today.
The existing B1 leaf currently depends on the older partial segmented Core; rebase
its target/includes/source provenance to the selected full Core before Engine use.

Actual archive owners are `ninfer_model_runtime_amd` and `ninfer_model_loading_amd`,
`ninfer_engine_amd`, `ninfer_attention_*_ops_amd` / `ninfer_attention_ops_amd`, and
`ninfer_basic_ops_amd` for append. Future actual link maps must identify the selected
new members and exclude old affected members in bare/`.lib:`/parenthesized forms,
with exact library/object boundaries. No whole-archive/FORCE/multiple definition
or old test-support workaround is allowed. BasicOps whose signatures use only
unchanged Tensor types require a separately justified reuse boundary, not a blanket
assertion that every old archive is safe.

## Remaining acceptance order

Complete actual Core/Models storage tests first. Then independently review and
implement the Program factory/accounting + opaque worker bridge + uniform-ABI public
B1 closure. Test the same Graph across startup/worker and real relocation, both
gate/rotation branches, raw/byte + original FP64 criteria, queue/fatal/shutdown paths,
and one public ordinary none/BF16/FP32-GDN product. Only after those pass run the
preserved quality and fair E2E/resource comparisons. This directory supplies no
runtime, mathematics, resource-saving, pure-text benefit or superiority-over-llama claim.
