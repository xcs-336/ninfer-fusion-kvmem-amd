# Isolated, one-time Core controller handoff

2026-10-21: the full isolated Core was generated, built and passed its actual
3,463-line link map and four-replay functional handoff test. Core includes this
leaf, default OFF. There is no Engine integration, production gate or benefit claim.

The candidate addresses one concrete Engine integration boundary: a pool created
on the startup thread must explicitly transfer GPU submission control after the
Engine worker binds the same HIP device. Removing affinity checks or rebinding on
each operation would leave the pool's ordering contract undefined.

## Protocol

`release_controller_for_worker(all_program_streams)` runs on the legal startup
controller under the original pool's recursive mutex. It rejects a wrong thread,
prior handoff, live allocator Freeze, active grant, open loan, poison, exhausted
epoch, wrong current device, or active capture **before disabling** the legitimate
controller. The supplied distinct stream inventory must be complete, including the
default stream if the Program uses it; its bounded size is 1..8. The API checks the
provided streams, not the truth of that caller inventory or stream-device origin.
All streams must belong to the configured device and remain alive through the call.
No other host submitter may enqueue work across this boundary.
Already queued host callbacks must not require the pool mutex to finish: the cold
synchronization deliberately holds it. This fixture contains no such callback;
the API does not inspect queued Graph/callback bodies to establish that contract.

After validation, the method disables submission, advances the allocator epoch,
and performs one checked cold `hipDeviceSynchronize()` on the explicitly current
device. On success it returns a move-only token and remains disabled. A synchronization
error propagates and leaves phase `Failed`, disabled, with no token. Dropping the
token never returns control. There is no retry or automatic recovery API.

The token carries a shared allocation-identity marker, generation, and device. The
marker contains no Core/payload pointer and prevents an old token matching a new
factory at a reused address. A token may outlive the factory safely as an invalid
marker-only object; it does not extend the factory or any device allocation's life.
Copying is deleted. Moving over an older token drops that older capability without
reactivating its owner.

The worker explicitly binds the configured HIP device, then calls
`consume_controller_for_worker(std::move(token), all_worker_streams)`. It must be
a different thread, use the correct owner/identity/generation/device, and satisfy
the same grant/loan/Freeze/capture conditions. Only then does the mutex-protected
commit set the new thread, enable it, advance epoch, and consume the token.
Failed preconditions leave the target disabled and the supplied token intact.

The token deliberately does **not** require the old `mutation_epoch`: original
CPU-only reservation/materialization/RAII release remains mutex-serialized and
legal while control is in transit. Such changes invalidate allocator snapshots,
not the handoff generation. Actual original Freeze objects gain a clone-only live
counter so a recursive Freeze capability cannot survive across release/accept/retire.
Freeze still has its original same-thread mutex lifetime contract. Both direct
storage page-address getters also acquire that guarded Freeze; an old thread cannot
bypass `plane_view()` through the public storage getters. Already returned raw
pointers are not revocable capabilities: the host submission boundary remains an
explicit caller obligation, including direct HIP launches or Graph launches.

On the successful shutdown path the worker finishes requests, closes all loans,
drains its stream work, then calls `retire_controller_for_teardown(streams)`. This
permanently disables it and performs a second checked cold device synchronization.
The parent joins the worker before any factory, lease, pinned input, execution table,
or Graph destruction. Parent-side original CPU RAII releases remain legal; parent
GPU view/transfer access never silently reactivates. The factory outlives every
worker and every original pool lease/reservation. Final destruction requires the
proper device and no live users. Failure/poison teardown is conservative destruction,
not successful retirement or qualified Engine recovery. The existing backend
destructor has its own conservative synchronization; the status counter counts only
the two explicit handoff hooks, not all synchronization in a fixture or destructor.

## Isolated ABI and future build

`prepare_controller_handoff.py` pins all nine files of the existing isolated
segmented view and the original real-Core byte fixture. It snapshots the actual
prepared-native Core text inventory and actual Core source manifest into a **new**
content-addressed tree, then applies strict-count source transformations in memory.
It does not rerun the production preparation pipeline. Existing output must match
exactly; no overwrite exists. This changes embedded Core control and owner layouts:
all actual Core sources and owner/storage/plan bodies rebuild together in
`ninfer_controller_handoff_core_amd`.

The leaf is included by `Core.cmake`, default OFF. In an explicit resource
window, configure
`NINFER_AMD_CONTROLLER_HANDOFF=ON` with an absolute
`NINFER_AMD_CONTROLLER_HANDOFF_BASE_VIEW`. Both targets are `EXCLUDE_FROM_ALL`; the
default option is OFF. The explicit build target is:

```text
ninfer_amd_controller_handoff_test
```

The executable's default invocation reports `PLAN_ONLY_NOT_QUALIFIED`; only `--run`
executes the fixture. A linked HIP binary may load/register runtime state before
`main`; default mode does not claim absence of pre-main runtime activity. There is
no automatic GPU CTest or Engine hook. Future execution must use the existing root
byte-zero/job, host guard, cutoff, and actual-PID monitor protocol.

The target links only the new full Core archive and normal SDK runtime/system
dependencies. The forced HIP header is the new shadow's header, not an old-source
force-include. There is no whole-archive/force/test-support archive. The mandatory
post-link map gate requires member-plus-symbol evidence for the new paged Core,
view, owner handoff methods, and storage. It rejects **any** extracted member of
`ninfer_core_amd` or `ninfer_segmented_pool_candidate_amd`. It recognizes all actual
LLD forms: `archive:member`, `archive.lib:member`, and `archive.lib(member)`.
An actual map pass proves link closure only.

Core and the byte-only observer use the existing experimental Core optimization
flags. This target includes no Attention mathematical body; it cannot extend the
separate six-case BF16 Attention qualification or other original Op registrations.

## Fixture scope and resource budget

The source fixture reuses the pinned actual original-Core helper/observer source,
renaming but never calling its old `main`. It retains actual original reservation,
materialization, generation handles, HostKVArena upload, execution-table publication,
and full byte oracle. One captured Graph/exec and stable device table are used for
four fresh-output guarded reads: startup, accepted worker, active real loan after
borrower overwrite/relocation, and returned loan. The observer independently indexes
device pointer-table entries and published group IDs. A fifth materialized page,
absent from the Graph, is released by original CPU RAII during disabled transit;
the worker must still accept the token after the epoch changes.

Negative source cases cover unauthorized early thread, old-thread view/freeze/transfer,
token reuse and second handoff, foreign owner, lost token, factory-outliving marker,
live Freeze, grant/open loan, poisoned storage, and actual capture. Copy rejection is
a compile-time type assertion. The worker explicitly selects and queries device 0.
Wrong-device runtime behavior and injected device-synchronization failure are not
tested. There is no independent pending-work or concurrent-allocation witness and
no full Engine startup/shutdown test.

The largest explicit device-payload phase is two simultaneous factories for the
foreign-owner negative control: `2 * (524288 payload + 128 pointer table)` =
**1,048,832 bytes**. The success world uses 524288 payload + 128 table + 256 execution
metadata + 262656 guarded observer = **787,328 bytes**. Maximum explicit pinned
payload is 262144 HostKVArena + 128 pointer shadow + 16 execution shadow =
**262,288 bytes**. These do not include HIP/Graph/driver allocations, vector capacities,
thread stacks, or allocator metadata. A conservative future admission is 0.02 GiB
fixture + 1 GiB runtime reserve + 2 GiB remaining host floor; fresh actual monitoring
is still required. No resource saving, performance, or E2E advantage is established.


The actual test passed CPU RAII during transit, wrong owner/thread/reuse rejection,
live Freeze/open grant/poison/capture rejection and explicit device0 binding.
Wrong-device runtime and device-sync failure injection remain untested. The
existing-output generator gate also rejects extra headers and symlink files or
directories. Native host guard minimum4.772GiB is an observation, not a peak/saving.
