# Real Core optional-capacity fixture — source candidate

SOURCE ONLY / NOT_GENERATED / NOT_BUILT / NOT_RUN. This new directory does not
change the frozen four-file API derivative, controller, worker, Core.cmake,
Vision or production gate. Root has not included this default-OFF leaf.

`VisionSegmentedCapacityTest.cmake` defines
`NINFER_AMD_VISION_SEGMENTED_CAPACITY_TEST=OFF`. Future explicit opt-in generates
the immutable capacity view with `--emit --verify-existing`, rebuilds all 25 real
Core TUs into **one** `ninfer_vision_segmented_capacity_core_amd`, and compiles the
separate `ninfer_amd_vision_segmented_capacity_test`. Both targets are also
EXCLUDE_FROM_ALL. It links no old Core, old storage, support archive, Op library,
whole-archive or replacement allocator/provider. Unchanged existing generation
only verifies bytes; it does not rewrite outputs or a source-list file.

The exact original-pool fixture is included as a helper TU fragment from the new
view, with its old main renamed and never called. Its real Core layout, streams,
Host arena, byte observer and tagged host fixture compile against the **new**
header root. This is source reuse, not an old object/archive fallback.

## Physical checks encoded in the fixture

Geometry is P=2, B=8, S=32768 bytes, 64 tokens, BF16 D256/KV1. Actual
`reserve(LIVE=4)` plus materialize/lease uses the original allocator; another
real reservation of 3 leaves available=1. Request 4*S genuinely needs 2 page
IDs. Two optional attempts must be empty and leave every real allocation flag,
free-ID list, reserved/allocated/lent/available/usable counter, mutation epoch,
device table allocation and values, host resolved addresses, and **all 524288
bytes** of the entire physical class unchanged. The old throwing entry must
throw the exact dynamic `std::runtime_error` type and original message, with the
same complete invariance. Snapshot/free-ID copying is through a real typed
Freeze, not an external dummy owner.

Clearing the real reservation permits one genuine grant of two IDs. The test
requires 131072 live bytes relocated, a changed mapping and stable table
allocation. The borrower waits ready and overwrites its entire fixed window.
The same previously instantiated Graph reads all live K/V bytes through the
actual device table and original execution-table IDs; every replay first resets
the entire output and guards to 0xa5. Full live bytes must equal independent
host tags. Closing against both actual user streams returns the grant once;
the second idempotent close must change no epoch/capacity/table/bytes. Physical
placement remains moved after return. This uses two streams functionally; it
does **not** prove a deliberately pending wait boundary or new concurrency.

Finite nonhardware error cases require exceptions, never empty: zero or too
large request, absent or oversized inventory made of **valid** stream handles,
an actual active capture (ended/destroyed without launch), invalid alignment on
a separate real storage construction, and real poison created by abandoning an
actual unique ticket. Poison retains its excluded grant and epoch after retry;
no HIP error code/invalid device pointer/invalid stream handle or driver failure
is fabricated. Poison teardown is not successful grant return or fault recovery.

The serial counter is private: state invariance of that counter is established
by the source inverse/branch position, not falsely advertised as a public runtime
getter. The fixture measures the public epoch, ledger, mapping and full bytes.

## Resource bound and command

Three worlds run sequentially. Each has one 524288-byte physical payload, a
128-byte device table, 128-byte pinned pointer shadow, one 262656-byte observed
output, and a small real execution-table arena (record actual capacity at run).
The real Host image is 262144 bytes plus its original metadata/alignment; four
execution-table host IDs add 16 pinned bytes. Core maps are 128 host bytes, with
snapshot/planner/grant vectors separate. One storage owns one transfer stream
and nine events; the test adds two nonblocking streams and one observer Graph.
The helper's normal destructors may perform a final device drain; this is not
proof of cleanup after an arbitrary driver fault.

The first world retains `before`, `ready` and later `closed` full snapshots;
each is 524288 bytes plus O(P*B) mapping/ledger metadata. A temporary fourth
snapshot during equality adds another 524288 bytes. Independent live expected
and Graph readback add 262144 and 262656 bytes; the borrower check vector adds
131072 bytes. Conservatively budget **4 MiB** explicit host memory and **1 MiB**
device memory above HIP/runtime overhead. Use the root's existing byte0/host and
device-floor wrapper, not this source budget alone, for actual admission.

Future guarded executable argv: `ninfer_amd_vision_segmented_capacity_test --run`.
Default no-argv returns PLAN_ONLY before main explicitly calls HIP. HIP-linked
TU registration/DLL initialization before main is not excluded by that wording.
No actual command was executed for this source revision.

## Link gate and missing evidence

`verify_capacity_map.py` requires real LLD symbol-field evidence for both
`try_begin_loan` and `begin_loan` (the latter cannot be satisfied by the former),
real allocator/table/factory/freeze/planner/close members from the selected new
archive, and an exact `main` from the new fixture object. Any other ninfer
archive or protected Core/storage member outside the selected archive refuses;
fake basename prefixes/suffixes and exception metadata main strings refuse.
Accepted archive syntax includes bare `archive:member`, `.lib:member`, and
`.lib(member)`. A map result is link closure only, never physical qualification.

Pending: independent source review; actual generation/no-op-mtime check; genuine
25-TU compilation/link and map evidence; the complete physical fixture under
resource guard. Vision broker, worker/provider integration, async overlap,
heterogeneous quantized KV and performance remain separate future closures.
