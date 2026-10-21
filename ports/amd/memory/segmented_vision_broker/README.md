# Segmented Vision Broker component candidate

SOURCE ONLY. No view generation, CMake configure, compiler/native/HIP execution,
Session or Engine integration is qualified. This directory changes no frozen
Core/capacity/worker/Ops recipe or production gate.

The generator extracts the **real** prepared production header's VisionWindow
and VisionResidencyBroker declarations and the contiguous C++ implementation
region ending before PinnedResultPool. It retains the real original acquire
WeightTail function body and raw tier implementation. All existing-component
edits are exactly reversed and compared to both complete original substrings.
It drops unrelated header dependencies and never compiles Session, Vision Ops,
weight rebasing/quantizers, Models or Engine. There are no stubs or old
model_runtime objects providing missing methods. The existing class namespace
is retained, so no old definition may be linked beside the derivative.

The input is the exact capacity-v2 full controller derivative: the 67-file source
snapshot / 25 real Core TUs. The future immutable output adds only two broker
files. No actual output is generated this turn. The default generator CLI is
plan-only; future explicit --emit is content addressed, exact-inventory checked,
symlink/reparse rejecting, and never overwrites. Output roots must be separate
from controller, native and owned recipe input trees.

The actual storage, DeviceContext and weight pool outlive the broker; the broker
and all stream handles outlive every window and its submitted users. All methods
and callback publication use one accepted Core host controller/submission thread.
The broker's sticky fields are not a separate cross-thread scheduler or mutex.
Moving a DeviceContext or replacing its stream handles after enabling this tier
violates the declared immutable inventory. Controller handoff/teardown is a later
Models obligation, not provided by this standalone component.

## New tier and true completion boundary

`enable_segmented_tier` selects one actual storage and maximum supported request,
and two immutable complete stream inventories. Each includes actual Context
compute, transfer, Vision and default streams (duplicates deduplicated); at most
four distinct streams are supported. No hidden restore/reader stream or separate
host scheduler is implicitly covered. The owner/controller guard is checked,
but the caller must supply the same device and max_request <= that storage's
configured fixed window. Wrong caller bounds still propagate Core's real error;
they are not translated to capacity refusal.

New admission uses the real `try_begin_loan`: only its typed empty result means
physical insufficient-unpromised-capacity. Existing `can_lend` refusal or a
request outside this tier's declared supported maximum also returns empty as a
documented policy veto. Invalid requests, poison, controller, allocator and HIP
errors propagate. Original `acquire` then takes the real WeightTail path on an
empty result. The raw VMM activation method now also requires the existing
noncontiguous_lending_qualified gate, which remains false. It cannot be mixed
with the selected segmented tier. No raw diagnostics are rerun.

Every segmented window owns exactly one Transaction; legacy runs remain empty.
`wait_ready(stream)` only accepts an inventoried user and forwards the real ready
event dependency. A future Session must call it before constructing
VisionWeightStream (whose constructor immediately enqueues transfer uploads)
and before encoding. No Session call site is changed here.

`close_checked` calls the real Transaction::close_after_streams with every
immutable borrower user. The Core records and completes its own events, returns
its unique grant once, and **only then** invokes the original revision callback.
Failure leaves the backend's grant lent/poisoned, retains an open/failed window
and sticky broker diagnosis, forbids new acquisition/fallback, and never retries
close or reissues legacy return_pages. A noexcept destructor cannot manufacture
a successful return. `memory()` is empty after close/failure even when `open()`
still signals an unresolved failed close. Move construction preserves the capability. Move assignment
into a failed-to-close destination leaves both operands unconsumed rather than
overwriting a failed owner. Callers must observe close_checked's result.

The actual Program revision callback is nonthrowing. The std::function API is
retained: if a supplied callback throws **after** a successful physical return,
the broker records a separate sticky error/exception_ptr; it does not claim the
already returned IDs are still lent or retry the callback. If an opening callback
throws, normal local-window unwinding executes the actual close protocol. Any
callback failure blocks further use, preserving diagnostic failure rather than
pretending a Program revision was published.

## WeightTail scope is intentionally different

WeightTail remains the real native backend: original evict body and best-effort
close/restore, with its existing caller obligation to finish all other users.
The new close_checked wrapper only observes pool.poisoned after that original
call. It is **not** proof of checked borrower completion or a new checked restore
for WeightTail. No HIP last-error check or added sync is used to invent that
evidence. A known pool poison becomes sticky and prevents a false healthy closed
state. The backend may clear its lent_extent even on restore failure; segmented
failure retention must not be generalized to this legacy fallback.

An independently implemented Core checked-restore/retained-ticket contract is
still needed for stronger fallback failure handling. Any future normal fallback
test must use the actual pool, real pinned mirror and byte-exact restored weights,
and clearly require all its users completed; no fault recovery can be inferred.

## Remaining closures

The new bounded fixture, default-OFF `SegmentedVisionBroker.cmake` and strict
map checker are source candidates, not executed evidence. Future targets rebuild
exactly one full25 capacity Core, compile the actual extracted broker against
the same root, and reject every other ninfer archive/old Core/ModelRuntime/Ops/
support. There is no Core.cmake include or auto CTest execution.

`ninfer_amd_segmented_vision_broker_test --run` is a future guarded command only.
The actual-code test reserves three real IDs after four live pages, witnesses
typed empty with no revision, then uses original acquire/actual WeightTail
16-MiB arena + pinned mirror. It dirties the borrowed suffix, **explicitly
finishes all legacy users**, closes normally and checks every restored weight
byte in bounded 1-MiB readback chunks. It does not infer checked restore or any
WeightTail fault handling from that normal path. After releasing reservations,
it checks actual segmented grant/ready/move/poison/return, two real revision
notifications and four fresh-output replays of one existing Graph reading every
live KV byte. A second sequential world closes while an actual valid stream is
capturing: the finite precondition refusal must retain lent/poison/sticky/open,
never notify return, retry, or choose fallback. No invalid handles or fabricated
HIP faults are submitted. Pending-event causality remains unqualified.

Explicit peak device payload is <=18 MiB: one 16-MiB real weight arena, one
512-KiB KV class, 128-B device table, 256.5-KiB observer, <=4-KiB execution
metadata; each world runs sequentially. Pinned weights are 16 MiB, Host KV image
256 KiB, pointer shadow128B and execution-ID shadow16B; 1-MiB readback and live
host oracle/output fit a conservative24-MiB explicit host budget plus runtime.
DeviceContext adds three real Context streams; Core adds one transfer stream
and nine events. Runtime/driver/Graph overhead requires actual parent guards.
Default fixture argv returns PLAN_ONLY before main explicitly calls HIP; linked
offload registration/DLL initialization before main is not excluded.

Later Model planning needs actual Vision staging + workspace size, single owner
shutdown drain and revision gates. Current worker explicitly excludes Vision,
B1/C1 excludes cross-lane overlap, and quantized KV needs all heterogeneous
code/scale planes and every corresponding consumer. Quantized Vision weight
staging is not quantized KV. None of these gates is opened by this component.
