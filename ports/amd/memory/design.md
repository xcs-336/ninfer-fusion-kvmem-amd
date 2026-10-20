# Native HIP memory borrowing

The weight implementation in this directory uses one ordinary `hipMalloc` allocation and lends
the actual weight suffix in place. It avoids the Windows HIP VMM remap/alias path that failed
resident-byte isolation. It is intended as the AMD implementation of the existing Core physical
primitive, with the address contract below. Its host source and additional device test compile
with the current Windows clang 23 / MSVC 14.44 SDK. On RX 9070 / gfx1201, the unchanged upstream
full-arena byte test and the additional graph/lifetime test both pass under the shared GPU lock
(2026-10-20). This establishes the behavior below, not throughput or whole-Engine qualification.

## Weight suffix: preserve the original dirty-byte contract

For chunk size `C = 16 MiB`, configured weight bytes `A`, and evictable tail bytes `T`:

```
R = ceil(A / C) * C             allocated device backing, including final padding
tail_begin = ceil((A - T) / C) * C
W = ceil(max_window / C) * C    maximum admitted rounded loan
mirror_begin = R - W
M = max(0, A - mirror_begin)    pinned host mirror bytes
E = ceil(request / C) * C       this transaction's rounded loan
loan = [home + R - E, home + R)
restore = [home + R - E, home + A)
```

`W` must be positive and no larger than `R - tail_begin`. Requests must be positive and round to
at most `W`; overflow is rejected. The mirror captures only actual weight bytes after upload.
Each close uploads only the transaction's restore interval. Bytes before it, including bytes
inside the maximum window but outside this smaller transaction, remain unchanged.

This is required by `tests/test_evictable_weight_pool.cu`: its explicit corruption at chunk 4
must survive a one-chunk loan of chunk 5. Lending the prefix of a fixed maximum window, or restoring
the whole maximum window on every close, would violate that test and the original suffix semantics.
Neither change is used here. The unchanged original test is the full-arena independent byte oracle.

The device allocation has `R` payload bytes, exactly the chunk-aligned physical payload of the
original pool. There is no second device workspace allocation. Pinned host payload is `M`, also
matching the original mirror range. Allocator metadata and driver residency overhead are runtime
properties and have not been measured here. Opening and closing transactions allocate no buffers.

The owner device and its compute/transfer streams are captured when the pool is constructed.
`evict` drains both streams before granting exclusive access. The transaction's supplied stream
orders its writes before the restore copy; callers must drain or order writers from any other
stream first. `VisionOverlaySession::complete_item` already drains the encode completion and
the transfer stream before closing. Close remains idempotent and nonthrowing; a restore failure
poisons the pool and rejects subsequent requests. Constructor and mirror-capture failures release
partial allocations, with mirror metadata published only after the capture completes.

## Addresses and captured graphs

The original VMM pool exposes a fixed overlay base, while selecting different physical suffixes
for different requests. A normal allocation cannot provide both properties simultaneously.
This HIP implementation keeps the correct suffix and makes the lease pointer depend on `E`.

* Weight/home pointers are fixed for the pool lifetime. A text graph captured before a loan can
  read the same restored addresses after close. It must not execute while weights are borrowed.
* For a particular rounded extent, the lease pointer is fixed. A workspace graph can be captured
  for that extent and replayed only while a transaction of that same extent is live.
* Across different rounded extents, callers must consume the current `leased()` pointer. A graph
  containing a previous extent's pointer cannot be replayed as though it were a universal overlay.

The actual consumer satisfies the changing-address rule: `VisionOverlaySession::begin` obtains
`window_.memory().data`, reconstructs `VisionWeightStream`, rebases `window_parameters_`, binds
workspace from the new span, and invokes `VisionContext::encode` for that item. The item plan is
deliberately sized per item. There is no cached Vision overlay graph or saved overlay base in this
path. Core owns the memory/restore mechanism; the model retains mathematical and lifetime ownership.

Integration must overlay both `core/evictable_weight_pool.h` and its implementation in the generated
HIP source view. Do not link this implementation alongside the original VMM implementation.
Qualify it with the unchanged upstream weight-pool test plus `tests/memory_weight_graph.cu`.
The latter checks repeated same-extent and changed-extent addresses, transaction moves, four
captured workspace writes, stable captured home reads, and preservation of undirtied markers.

## KV: exact limits without remapping

The existing KV pool can lend arbitrary granules such as `{1, 3, 4}` and present them as one
contiguous window, while unrelated home addresses remain live on the decode stream. A single
ordinary allocation cannot concatenate those bytes into one pointer without an address remap.
Weight-style in-place borrowing cannot reproduce that physical contract for noncontiguous inputs.
Free KV pages need no content restore, because append writes them before any read; that removes
copy-back cost but does not solve contiguity.

The current native weight implementation therefore does **not** silently substitute a KV backend
or advertise noncontiguous KV lending. There are three explicit possible policies:

| Policy | Admission requirement | Additional device payload | Decode overlap |
|---|---|---:|---|
| Direct contiguous free KV span | A physically adjacent run of fully free granules covers the rounded window; its owning logical pages are not promised to reservations | 0 | Possible for disjoint live KV |
| Exclusive weight suffix | The requested rounded window fits the mirrored evictable weight tail | 0 | Other weight users wait |
| Dedicated startup workspace | The planner budgets the full maximum workspace while leaving required headroom, reducing KV capacity if necessary | Maximum workspace bytes | Possible |

The direct KV policy needs a distinct supported-capability/admission rule. It may not accept all
requests for which the original sum-of-free-granules policy succeeds. It must enumerate candidate
physical spans only from page-pool `free_runs()`, prove that each full granule lies inside free
pages of its plane, merge physically adjacent candidates, and select a run long enough for the
request. For every selected granule it must retain the witness logical page range; union those
ranges before testing that their page count is at most `available_pages()` and before lending.
`free_runs` alone is insufficient: free pages may already be promised to active reservations.
All witnessed page runs must be removed transactionally with rollback on a refusal, and returned
only after every writer of the borrowed span is complete. Resource revision and sealed-plan
checks from `VisionResidencyBroker` remain mandatory. No stream-wide device synchronization is
needed to protect disjoint live KV, but that concurrency needs its own runtime qualification.

Selecting only one physical run usually gives less yield than joining free storage across many
planes; aggregate free bytes are not a sufficient admission metric. A failed contiguous admission
can fall back to the exclusive weight suffix without additional VRAM. This is the recommended first
portable policy once the weight primitive has passed its runtime tests.

A dedicated workspace does not actually reuse free KV physical storage. It needs no KV copy:
the borrowed pages were free and contained no live payload. Its maximum workspace must be allocated
at startup and charged to the Program's capacity calculation, not hidden in a runtime kernel or
reported as a zero-cost KV loan. The decision requires measured device headroom and the user's
current memory budget. No dedicated allocation or corresponding admission reduction is implemented
by these files.

Full original noncontiguous KV physical equivalence remains unqualified on this Windows HIP driver.
Rewriting every consumer to accept a scatter workspace would be a larger interface change; merely
returning a list of pointers cannot satisfy the existing contiguous `DeviceSpan` contract.
