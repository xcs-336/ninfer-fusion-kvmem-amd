# Optional capacity refusal, isolated Core source candidate

SOURCE ONLY. No output view, compile, native execution or Vision connection has
been qualified for this revision. The production gate and every frozen Core,
controller, worker, Vision and source generator remain unchanged. There is no
CMake target or automatic generation hook here.

`KVSegmentedStorage::try_begin_loan(bytes, prior_users)` returns
`std::optional<Transaction>`. Empty means only that the actual frozen pure
planner found `wanted > available_pages`, after validating its entire snapshot.
Invalid geometry, invalid arguments, controller/thread rejection, active capture,
an existing grant, poison, owner errors, allocation failures and HIP errors keep
their original exception paths. A broker must not catch `runtime_error` and treat
it as a capacity refusal.

The old `begin_loan` calls this implementation once and throws the same
`std::runtime_error("insufficient unpromised segmented KV capacity")` for empty.
Success moves the same unique ticket out through its existing noexcept move.
No ledger query before the original Freeze, second planning pass, extra Freeze,
catch, ledger, payload, stream or event is added. The full original transaction
body, including ready completion, rollback and poison retention, is recoverable
byte for byte by the generator's inverse. `close_after_streams`, abandonment and
grant return are unchanged.

Empty is decided before `s.next` changes, `lend_selected`, GPU publication or
serial advancement. The earlier original device binding/capture checks and host
snapshot/planner allocations still happen. This is not a promise of no HIP API
calls, no temporary host allocation, or a lock-free capacity query. The single
real allocator Freeze lives through planning and, on success, publication; an
empty return destroys it normally. Reservations remain part of available quota.

## Exact source and generation boundary

`manifest.json` pins the current full controller receipt, whose 67 output hashes
bind the complete source snapshot and 25 real compilation sources. Verification
checks every recorded file and rejects extra files, missing files and reparse
points, including directories. The actual frozen planner has exactly one
`return std::nullopt`, and the expected transaction branch is replaced exactly
once. Only the storage header and source are changed; no class data layout or
original mathematical kernel is edited.

`prepare_vision_segmented_capacity.py` is importable without writing. `transform`
is a pure byte-map transformation. The default CLI is plan-only; future explicit
`--emit` writes to `output-root/recipe-SHA`, or with `--verify-existing` verifies
the complete identical old output. It cannot overwrite or emit into an input
tree. The new receipt is `vision-segmented-capacity-source-receipt.json`, not an
old controller receipt claiming unchanged storage bytes. Any later build must
use the full new Core source inventory and a single selected Core archive; old
controller/worker/Core archives cannot supply these definitions. All affected
header consumers require a matching rebuilt view. Actual map validation is a
future build obligation, not supplied by a source inverse.

The prior controller input is currently:

`ports/amd/build/controller-handoff-candidate/3ada9fee6cbd7754532b7cf653d0118c26bf88fc080e46f3c25fdf834a61844c`.

Future owner-only command, **not executed for this revision**:

```text
python -B prepare_vision_segmented_capacity.py --controller-view <exact-controller-view> --output-root <new-separate-root>
```

Add `--emit` only in an explicitly owned generation/build window. A changed
controller receipt requires a reviewed new pin revision, not a relaxed check.

## Checks and still pending

`check_source.py --controller-view <exact-controller-view>` only reads small
source files and runs in-memory inverse/anchor checks. Its tiny injected planner
control distinguishes empty, admitted and throwing outcomes; this is a control
model and does not execute the C++ transaction, an allocator or HIP. No source
check is a runtime capacity, exception recovery, Vision, correctness or resource
qualification. Its stdout may be saved by the owner as a separate research
receipt; the check itself creates no files.

Next physical tests must verify unchanged epoch/mapping/serial/available capacity
on a genuine reservation-limited empty result, old throwing compatibility,
noncapacity failures propagating, and one real unique grant on success with the
same byte/Graph/pending-stream oracles. Only then can a new Vision broker tier
adopt this API. It must use the new ticket as the sole return authority, never
also call legacy `return_pages`, wait ready before any upload and close against
all encode/transfer users. Close failure retains lent IDs and poison; it cannot
be followed by a resource-revision return or fallback.

The current worker explicitly rejects Vision and supports only homogeneous
BF16/D256 page-major table-backed KV. Quantized Vision weight staging and
heterogeneous quantized KV data/scale planes are separate representation issues.
Neither Vision/Engine support nor memory savings/performance is claimed here.
