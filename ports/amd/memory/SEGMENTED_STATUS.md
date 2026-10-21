# Segmented KV and Vision integration checkpoint

Updated 2026-10-21. Windows 11, Radeon RX 9070 (`gfx1201`, wave32),
AMD clang 23, MSVC 14.44. The installed SDK directory is named `10.0.0`;
the reported HIP runtime API is `70260201`. Directory names are not HIP versions.

All new candidate options remain **OFF by default** and their targets are
excluded from the default build. The original production noncontiguous-lending
qualification remains false. There is no complete Session/Engine qualification,
resource saving, asynchronous-overlap speedup, or formal local-workflow migration
claim in this checkpoint.

## Implemented and physically checked components

The substitute backend uses one ordinary HIP payload allocation, stable device
page tables and bounded D2D relocation. It does not depend on successful VMM alias
mapping. Existing original allocator, BF16 consumer, B1, controller handoff and
Models component qualifications remain limited to their documented cases.

Two additional component fixtures have now been built and executed:

| Component | Actual evidence | Remaining boundary |
|---|---|---|
| Typed capacity admission | All 25 new Core translation units, strict actual link-map ownership, native exit 0; two empty-capacity attempts, seven noncapacity refusals, one unique grant, 131072 copied live bytes and seven same-Graph replays | Pending ordering and physical HIP fault recovery remain unqualified |
| Real production Vision Broker functions | All 25 new Core translation units plus the production Broker translation unit, strict actual link-map ownership, native exit 0; four same-Graph replays, typed admission, normal WeightTail byte restoration and semantic-close failure retaining lent/poison | Real Session/Engine connection and checked legacy WeightTail failure recovery remain unqualified |

These short native processes finished before the resource monitor obtained a
sample. Both guarded runners therefore returned **125 / RESOURCE_UNQUALIFIED**,
despite native exit 0. Zero samples do not establish zero memory use, continuous
peaks, resource savings or performance. The copied-byte count is a correctness
workload, not a saving measurement. These are additional component fixtures;
they do not increase the count of passed original test registrations.

`vision_segmented_capacity` adds an optional capacity refusal in the same
allocator Freeze before placement, grant or serial mutation. Noncapacity errors
remain errors. The original throwing entry retains its exact capacity failure.

`segmented_vision_broker` reuses the original production Broker functions and
ordinary WeightTail fallback. It tracks a unique segmented transaction and a
complete stream inventory. It returns a grant only after checked completion;
completion/return failure retains the lent/poison state without a retry. Future
Session integration must wait for readiness **before constructing WeightStream
or starting its H2D transfers**. Passing the isolated component does not prove
that this complete Session lifecycle is connected.

## Complete Models / Engine candidate

`models_engine`, `models_engine_ops` and `models_engine_worker` compose the
planner, full controller-Core ABI, all 51 ordinary cache translation units,
B1 table-aware consumers, startup/worker ownership and public Engine route.
The initial candidate remains dense BF16 D256 with KV2/4, one rank, at most
512 tokens, without Vision, prefix/disk cache or speculation. Those explicit
admission limits are not a long-context or production qualification.

The current normal configuration succeeds. All 51 cache translation units and
the B1 public entry compiled; a subsequent heavy translation unit was stopped
when available physical RAM fell to 1.973 GiB, below the unchanged 2 GiB floor.
This is a resource abort, not a C++ diagnostic or mathematical failure. Final
combined links and native tests remain pending. Compilation uses the original
O3 numerical flags; no test domain, oracle or math tolerance was weakened.

Keep the **complete ordinary AMD source-preparation pipeline enabled**, even
when selecting only a small candidate build target. Disabling preparation
stages changes the public generated source view. The local view has been
restored and all 337 native Engine inputs checked against the frozen snapshot;
normal dependency recompilation may still be required. Do not modify timestamps
or copy object files to bypass those dependencies.

Further work is complete Engine math/lifecycle qualification, actual Session
H2D/encode/exit integration, resource-qualified workload measurements, quantized
KV code/scale closure and repeated fair comparisons with the existing local
llama.cpp/KVMem workflow. Existing local optimizations will be formally migrated
only after quality, valid full-history behavior and stability satisfy the
agreed requirements and a material measured benefit is established.
