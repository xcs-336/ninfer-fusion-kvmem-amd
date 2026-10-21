# Isolated Models / public Engine worker candidate

This is **SOURCE ONLY: NOT GENERATED, NOT CONFIGURED, NOT BUILT, NOT RUN**.
It is a new default-OFF leaf; it changes no frozen candidate, published allocator,
ordinary Engine default, production gate, or root CMake file. The earlier four-file
`models_engine` partial planner and its always-rejecting Program gate remain intact.
This composer derives that planner in memory, then replaces the gate with actual
single-factory Program construction and actual worker lifecycle code.

## Implemented source

`prepare_models_engine_worker.py` pins fifteen real prepared native files. Its
pure `transform` returns eighteen small source strings (including the unchanged
Graph-profile identity result), including:

* One `KVSegmentedNativePool` declared before all Program borrowers and allocated
  before every other Program resident allocation. The tagged DecoderState selects
  that exact original allocator; ordinary metadata stays in the freshly replanned
  persistent arena. No dummy pool, second payload, second ledger, old offset
  subtraction, or ordinary arena backing for the external payload is constructed.
* Program `MemorySummary.sequence` adds the actual native payload plus nonlendable
  device table once to capacity/used/peak. `segmented_storage_accounting()` separately
  exposes pinned table and host slot-map bytes. Those are named arrays, not a claim
  to include vector capacities, allocator/control heaps, driver or Graph overhead.
* Public Program opaque, move-only startup capability with out-of-line incomplete
  PImpl ownership. The private State holds the true Core token and the originating
  ProgramImpl identity. Core additionally validates the unique live owner marker,
  device and one-shot generation. Dropped/foreign/consumed tokens never reenable a
  disabled owner. A token destructor never dereferences its borrowed Program pointer.
* Both real `EngineCore` and `CausalScoreCore`: release on startup controller;
  worker bind, consume, then startup promise; checked retirement at every normal
  worker-loop exit and after a post-consume startup exception. Retirement errors
  are retained in a mutex-protected `exception_ptr`, mark the core unavailable and
  produce diagnostics. `checked_shutdown_worker()` joins before rethrowing it.
  The public Engine candidate exposes checked shutdown and retained error access.
  If startup and its checked retirement both fail, the delivered typed exception
  retains both `exception_ptr`s; the constructor still joins before rethrowing it.
* A concrete FlashNext template branch remains ordinary and never fabricates or
  consumes a Qwen Core token. Unknown Program types fail at compile time. Tagged
  loading rejects FlashNext; the ordinary FlashNext source path is retained.
* A real public `Engine(segmented_engine_storage, options)` constructor routes to
  the new internal tagged factory and explicit segmented planner. The ordinary
  public constructor still selects `false`. Tagged startup does not retry the
  original fallback core after a one-shot release. Ordinary fallback is retained.
* The frozen Ops v2 bridge changes only Main Text's two attention calls to the
  table/null router and Main's workspace query to the explicit plan flag. MTP's
  unexecuted references retain real definitions in the new full cache closure.
  Graph route labels retain the original resolver; Ops independently proves its
  renamed B1 resolver and label function equal after removing the private prefix.

The main future-generation action produces one immutable source shadow: all actual
Models/runtime/public headers and compile-source files, the complete controller Core
snapshot, frozen two-file Models storage, and frozen full cache/B1 Ops snapshot.
All original model-loading, model-runtime, Engine and calibration component TUs
compile anew against that root. It never substitutes a few objects into an old
model/runtime archive. Every controller output remains byte-identical **except**
`src/core/decode_graph.cpp`; a receipt records the individual equality results.
Every Core header, owner layout and forced `hip_intrinsics.h` is identical.

## Closed admission and lifetime boundaries

The explicit public constructor's pure option check runs before DeviceContext
construction. Real `LoadPlan::config()` geometry validation runs before weight
materialization in the tagged runtime factory. Program revalidates before its
native payload allocation. This does **not** claim that existing ordinary weight
upload occurs after the old partial planner gate or that all HIP/device-context
allocations are prevented by the geometry check.

Initial admission is Qwen3.5 dense B1 context 1..512, D256, H16/KV2 or H24/KV4,
BF16 PageMajor KV, physical groups at least logical, one actual device/rank,
FP32 GDN, no MTP/DFlash/copy/archive proposer, Vision, prefix/Host restore,
disk, ring, selected history, retrieval score/shadow or automatic KV precision.
`NINFER_PROTO_SELECT` is accepted only when absent or exactly `all` (including an
explicit empty string is rejected). Options and these environment policies must
remain fixed for the Engine lifetime. The profile is explicitly `off`: the real
original installer still runs and clears any previously installed process route
table with `nullptr`; skipping the installer would not establish this property.
No weights are rounded/requantized by this layer. Existing A16/A8 projection
options retain their meanings; the new integration fixture explicitly uses A16.

The complete admitted stream inventory is rank-0 compute, transfer, non-null
Vision stream if it exists, and default stream, with aliases deduplicated. Context
cache/disk are rejected so no private Hybrid restore or disk submitter is hidden.
The native storage's private stream is internal/noncapturable and its completed
ready boundaries plus checked cold device synchronization cover it. There is no
automatic thread rebinding. Original mutex-protected CPU lease/reservation release
callbacks remain legal while the token is in transit and after retirement.

The sole factory outlives decoder, execution tables, leases, state stores, Graphs
and all GPU consumers by declaration order. The worker drains/retires before join;
the parent joins before Program/factory teardown. An owner must not concurrently
destroy/move Engine while another thread accesses it. Checked public shutdown is
serialized, retains its error, and may be explicitly called before destruction.
The noexcept destructor logs errors rather than converting them to success.
Core retirement precondition failures occur before disabling the valid controller;
only an error after the disable/quiesce commit guarantees disabled/Failed. A failed
worker stops and retains the error in either case. No diagnostic infers controller
status from an exception, retries retirement, or reports the failed path qualified.

## Real Graph witness and integration fixture

`worker_graph_observer.h` is a host-only observer of actual Core APIs. Successful
instantiate gets a monotonic instance generation; successful update advances its
revision; successful destroy removes it. This rejects pointer-address reuse as
evidence of the same executable. Successful GraphLaunch supplies the actual exec
pointer/generation/revision, and only then increments its launch count. There is
no device payload, fake Graph, extra synchronization or scheduling change.

The callback is installed **on the worker after consume**, not on the parent TLS.
It is cleared on both retirement success and failure. It is noexcept, bounded,
does not allocate or call HIP, and its observations are protected separately from
Core owner locks. The process registry has 256 slots (6,144 bytes of entries plus
mutex/counters/alignment); per-Program snapshot has 64 identities/counts (2,048
bytes of arrays plus fixed fields/mutex). Unknown identities, overflow, failed
cleanup or counter exhaustion fail observation admission; they do not become a
Graph proof. This instrumented artifact is not a performance binary.

`test_models_engine_worker.cpp` defaults to a plan without constructing Engine.
`--host-guards` exercises the exact shared pure option guard only. HIP linkage and
generated registration can run before main, so neither mode claims a DLL-free
executable. Explicit future run modes require a real `.ninfer` and external
byte-0/job/resource/deadline owner guard:

```text
ninfer_amd_models_engine_worker_test.exe --host-guards
ninfer_amd_models_engine_worker_test.exe --run-generation --artifact MODEL.ninfer --tokens FRONTEND_IDS_ARRAY.json
ninfer_amd_models_engine_worker_test.exe --run-causal --artifact MODEL.ninfer --tokens FRONTEND_IDS_ARRAY.json
ninfer_amd_models_engine_worker_test.exe --run-startup-before --artifact MODEL.ninfer
ninfer_amd_models_engine_worker_test.exe --run-startup-after --artifact MODEL.ninfer
```

The ordinary control is destroyed before allocating the selected model. The
generation scenario makes two real public requests with the same unretokenized
IDs, exact four-token greedy results, prefix reuse disabled, and stop detection
disabled for this fixed-token smoke. It demands an actual live unchanged exec
generation/revision whose successful launch count increases between requests.
It checks Worker status before calls and Retired/disabled plus exactly two cold
controller syncs after checked shutdown. Those counters exclude other original
startup synchronization calls. The causal mode compares every finite FP32 score
bit to the ordinary control, with no new relaxed tolerance. These deliberately
strict integration checks do not replace original FP64 BF16 Attention or full
quality suites, and a future mismatch is retained rather than weakening criteria.

Two explicit test-only startup controls throw on the worker before consume or
after consume/before promise. They exercise the actual exception promise, join,
and zero-versus-one retirement distinction; they are not simulated HIP/driver
failures and are never enabled by an environment variable or ordinary constructor.
They do not qualify out-of-memory, driver faults or arbitrary concurrent shutdown.

## Opt-in build and remaining qualification

Root must include this leaf only after the actual Engine and its dependencies,
the independent controller/Models/Ops leaves. Root owns the include; this folder
does not modify `Core.cmake`.

* Option: `NINFER_AMD_MODELS_ENGINE_WORKER=OFF` by default.
* Full Core target: `ninfer_models_engine_worker_core_amd`.
* Rebuilt components: `ninfer_models_engine_worker_{model_loading,model_runtime,engine,calibration}_amd`.
* Link interface: `ninfer_models_engine_worker_amd`.
* Explicit fixture: `ninfer_amd_models_engine_worker_test` (`EXCLUDE_FROM_ALL`).
* Snapshot root variable: `NINFER_AMD_MODELS_ENGINE_WORKER_VIEW`.

The two independent Ops archives are linked by TARGET_FILE, not their PUBLIC
parent-controller interface. The exact controller/worker header equality proof
permits this same-layout boundary; the final map forbids parent-controller
extraction. Tensor-only static dependencies are resolved through the real target
graph as archive files, never through old Core/Models/Attention PUBLIC fallbacks.
Unreviewed target types or generator expressions refuse configuration. That code
has not run: actual configure may expose an additional dependency expression and
must be fixed explicitly, not bypassed or counted as a passed link.

The strict final map requires the unique new full Core, changed Models/public
Engine providers, actual main address-column symbol and independent cache/B1
provider checks. It rejects bare, `.lib:` and `.lib(member)` old providers; fake
prefix/suffix object names and `$ip2state$main` do not prove extraction or entry.
Synthetic parser checks are not actual archive extraction evidence.

Still pending: independent final source review, actual source generation,
configure, compilation, actual post-link map, resource-qualified host/GPU runs of
every fixture mode, original math requalification across this ABI, and product
quality/performance evaluation. No original registration count is incremented.
No Vision borrower, table-aware full Engine qualification, live relocation in an
Engine, allocator-concurrency/driver-fault proof, pure-text memory benefit,
resource savings or advantage over llama/Prism is claimed.
