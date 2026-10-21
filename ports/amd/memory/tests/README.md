# Experimental segmented KV tests

This checkpoint contains the actual pure planner, isolated HIP Core byte test,
and separate controlled-pending fixture. All three test sources are byte-identical
copies of research fixtures that passed direct-command testing on native Windows
RX 9070/gfx1201. Their delivery comments are retained for that byte identity.

All three repository CMake targets have now actually built and run successfully
on Windows gfx1201 under the shared byte-0 lock and host guard. The first two
executables reuse their earlier unchanged CMake build; the third was newly built
in session67105. These actual repository runs have their own immutable receipt,
`research/ninfer-kv-repo-v2-actual-summary.json`, separate from the earlier direct
build/run records. This is isolated Core functional qualification only.

The tests do not adapt DeviceKVPagePool, change the original VMM gate, supply a
table-aware production Attention implementation, integrate Engine, or imply a
pure-text Bonsai benefit. No actual resource savings or full-model quality result
has been established.

## Recorded direct-command results

On 2026-10-20 the owner built each fixture with AMD Clang/Windows MSVC 14.44 and ran
the resulting executables under the shared byte-0 lock and host guard. All three
direct builds and runs exited zero. The original evidence remains in workspace research:

| Fixture | Actual result |
| --- | --- |
| Pure C++ planner | 21,698 cases; 7,954 admitted; 13,744 capacity refusals; 6,882 unaligned admitted; 19 invalid rejections; stale-plan negative control observed |
| Isolated HIP Core | One Graph capture, four replays of that same instance; 1 MiB payload; 65,536 live bytes copied; full live-word/output-guard checks; abandon and atomic host return refusal retain IDs |
| Controlled pending HIP Core | Two prior and two borrower independent events NotReady at their boundaries; two successful completion queries before owner ledger return; four independent events; zero callback fallbacks; full original byte/Graph/guard checks |

Original logs: `ninfer-kv-segmented-{plan,storage}-{build,run}-first.log` and
`ninfer-kv-segmented-storage-pending-{build,run}-first.log`.
The host guard observed minima 5.325 GiB for the CPU run and 5.563 GiB for the HIP
v1 run, and 4.322 GiB for the pending-v2 run. These are sampled guard observations,
not continuous process/adapter peaks. The HIP v1 result explicitly has
`pending_ordering_witnessed=false`. The separate pending-v2 result has that field
true and exercises readiness at exposure and completion before its test-owner
ledger return for controlled workloads. It does not prove every internal wait
uniquely necessary under every driver schedule. Both HIP fixtures have
`concurrent_allocator_mutation_tested=false`; neither qualifies actual driver-error
rollback, original Attention math or production integration. CPU callback holds
are test controls, not performance samples.

The first two CMake targets linked in the successful build recorded by
`ninfer-t2-vmem-build-v6-resource-retry.log` (session49127, child exit0).
The third linked in `ninfer-t2-timing-build-v6-first.log` (session67105, child
exit0). All three CMake-produced executables then ran with child exit0 in
`ninfer-kv-repo-v2-{plan,storage,pending}-run-first.log`: full21698 planner cases,
one capture/four same-instance replays in each HIP fixture, and the same four
NotReady/zero-fallback pending boundaries. Host guard minima were respectively
5.157/5.095/5.130GiB. They are not sampled continuous resource peaks or measured
savings. Old receipts which said NOT_BUILT/NOT_RERUN remain historical snapshots.

## Opt-in CMake targets

The parent must include `memory/SegmentedStorage.cmake` after defining
`ninfer_native_hip` and the normal GPU-test helper. This leaf is default OFF and
does not edit Core.cmake. With that include present, an existing native port
configuration can enable only this test closure:

Run these commands from the repository root with the installed AMD compiler and
Windows MSVC environment selected:

```text
cmake -S ports/amd -B ports/amd/build -DNINFER_AMD_CORE=ON -DNINFER_AMD_SEGMENTED_STORAGE_TESTS=ON
cmake --build ports/amd/build --target ninfer_amd_kv_segmented_plan_test ninfer_amd_kv_segmented_storage_test ninfer_amd_kv_segmented_storage_pending_test -j1
ctest --test-dir ports/amd/build -R "^native_kv_segmented_plan$" --output-on-failure
```

Only in an explicitly owned GPU/resource window, run the GPU registration:

```text
ctest --test-dir ports/amd/build -R "^native_kv_segmented_storage$" --output-on-failure
ctest --test-dir ports/amd/build -R "^native_kv_segmented_storage_pending$" --output-on-failure
```

The CPU target compiles only host C++20 and links no HIP runtime or large Core
archive. Each HIP target links its own transport/planner and the normal SDK import
library through `ninfer_native_hip`; the helper selects real gfx1201/wave32 HIP
compilation and the standard generated compatibility header. It does not link
`ninfer_core_amd`. A link map is requested on the Windows toolchain. All three
repository build/run paths have passed at the narrow functional scope above;
this does not qualify an original allocator or production Attention consumer.

The GPU registration uses the repository's byte-0/process-tree cleanup wrapper,
has a 120-second native timeout, prepends the configured SDK bin directory, and
passes explicit `--run`. Calling the HIP executable without arguments is plan
only and must never be counted as a mathematical test pass. Lock contention or
skips are not PASS. Use the owner's host guard and actual-PID resource monitor
around an actual qualification run; the CTest helper alone does not provide that
resource evidence. Neither configure nor build automatically runs a test.

## Resources and boundaries

The largest case explicitly requests 2,097,928 device bytes: 1,048,576 payload,
256 pointer table, 8 allocation mask, and 1,049,088 guarded output. The pinned
shadow is 256 bytes. Explicit coexisting main host byte buffers reach 2,228,736
bytes, including the poison readback that remains alive across later Graph
readbacks. Actual vector capacities, allocator snapshots/planner temporaries,
one private plus five test streams, 17 storage events and driver/Graph/runtime
overhead are additional and are not asserted to be zero. Pending-v2 adds four
test-owned events and at most two simultaneous host helper threads; their stacks,
mutexes, condition variables and runtime overhead are additional. It adds no
device payload or GPU spin kernel.

The stable-table consumer reads full arbitrary BF16 bits through the table at
execution; it is not an affine production KV kernel. All Graph replays finish
before returning a loan. The backend owns its payload, so future integration
must replace/exclude the old payload allocation instead of keeping both.

The controlled pending boundary test has direct-run evidence at its narrow scope.
Future qualification still needs concurrent allocator mutations, real driver-error
and copy/table rollback, actual production Attention, complete resource receipts
and separate performance evidence. The original
raw-VMM diagnostics and their failed production gate remain unchanged.


## Original pool and original-body BF16 consumer checkpoint (2026-10-21)

Two additional leaves are included from Core.cmake, with both options default OFF.
NINFER_AMD_SEGMENTED_POOL isolates the actual original allocator with typed ownership,
mutation epochs and mapped Core transfers. NINFER_AMD_SEGMENTED_BF16_OPS adds the
actual original BF16 append, small-T and prompt bodies with final-address indirection.
Neither changes the production Engine or its non-contiguous-loan qualification gate.

The BF16 executable actually passed six geometry/mode cases: H24/KV4 and H16/KV2,
each cached small-T, inline-append small-T and prompt. Eighteen replays of the same
Graph instances checked 1,259,520 outputs against the independent original FP64
criterion, complete live KV bytes, relocation bit parity, input/guard preservation,
fresh output and append writes on every replay, and the borrowed raw span. These
cases use the real allocator and execution tables, including actual cross-plane
physical relocation. They add zero original-registration passes. They do not test
forced pending ordering or concurrent allocator mutation. The separate earlier
pending-v2 test retains its own scope and cannot grant those properties here.

The dedicated ninfer_amd_segmented_pool_test separately built and passed its actual
link-map gate; it has NOT RUN. BF16 and Core maps reject old ABI Core/Op members.
The LLD bare-archive-format parser correction was independently checked against the
actual BF16 map, five positive formats and 27 negative controls. The mathematical
bodies and error criteria were not changed to repair the map gate.

Enable with NINFER_AMD_SEGMENTED_POOL=ON and NINFER_AMD_SEGMENTED_BF16_OPS=ON;
explicit targets are ninfer_amd_segmented_pool_test and
ninfer_amd_segmented_bf16_attention_test. Both executables require --run for HIP;
the BF16 leaf adds no automatic GPU CTest. Use an explicitly owned resource window,
the byte-0/process-job wrapper and host-budget/deadline admission. Exact scope,
resource formulas and commands are in
[the consumer document](../segmented_bf16_attention_source.md) and
[the structured actual receipt](segmented_bf16_actual_scope.json).

Public B1 dynamic rows/gates, worker/controller transfer, Models integration, the
real borrower lifetime, complete resource accounting and E2E benefit remain open.
There is no measured performance/resource-saving claim or formal workflow switch.
